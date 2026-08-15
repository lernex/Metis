from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from .ngram_canonical import (
    CANONICAL_IDS_BINARY,
    CANONICAL_IDS_MANIFEST,
    build_canonical_id_sidecar,
    canonicalize_decoded_token,
)
from .state import atomic_json, utc_now


def iter_jsonl_text(paths: Iterable[Path]) -> Iterator[str]:
    for path in paths:
        opener = path.open
        if path.suffix == ".zst":
            import io
            import zstandard as zstd

            def opener(mode: str = "r", encoding: str = "utf-8"):  # type: ignore[no-redef]
                raw = path.open("rb")
                stream = zstd.ZstdDecompressor().stream_reader(raw)
                return io.TextIOWrapper(stream, encoding=encoding)
        with opener("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row.get("text", ""))
                if text:
                    yield text


def train_tokenizer(
    texts: Iterable[str],
    *,
    output_dir: str | Path,
    vocabulary_size: int,
    special_tokens: list[str],
    minimum_frequency: int = 2,
    split_digits: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    if split_digits:
        # The GPT-2 byte-level regex matches runs of digits, so 147832 becomes
        # one pre-token and BPE is free to merge it into a single id. A model
        # then has no positional handle on the digits and place value has to be
        # memorised per literal rather than learned once. Splitting them first
        # is the current standard and costs only the tokens it stops merging.
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), byte_level]
        )
    else:
        tokenizer.pre_tokenizer = byte_level
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocabulary_size,
        min_frequency=minimum_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=None)
    eos = special_tokens[0]
    eos_id = tokenizer.token_to_id(eos)
    if eos_id is None:
        raise RuntimeError(f"EOS token {eos!r} was not assigned")
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    vocab = tokenizer.get_vocab()
    if len(vocab) > vocabulary_size:
        raise RuntimeError(f"Tokenizer emitted {len(vocab):,} entries, over target {vocabulary_size:,}")
    if vocabulary_size == 65_536 and len(vocab) != vocabulary_size:
        raise RuntimeError(f"Production tokenizer must contain exactly 65,536 entries, got {len(vocab):,}")
    if max(vocab.values(), default=0) >= 65_536:
        raise RuntimeError("Tokenizer IDs do not fit uint16")

    sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    canonical_ids = build_canonical_id_sidecar(
        tokenizer,
        tokenizer_path=tokenizer_path,
        output_dir=output,
    )
    release = {
        "schema": "metis.tokenizer-release/v1",
        "created_at": utc_now(),
        "algorithm": "byte_level_bpe",
        "vocabulary_size": len(vocab),
        "requested_vocabulary_size": vocabulary_size,
        "special_tokens": {token: tokenizer.token_to_id(token) for token in special_tokens},
        "eos_token": eos,
        "eos_token_id": eos_id,
        "tokenizer_sha256": sha256,
        "uint16_safe": max(vocab.values(), default=0) < 65_536,
        "ngram_canonicalization_algorithm": canonical_ids["algorithm"],
        "ngram_canonical_ids_manifest": CANONICAL_IDS_MANIFEST,
        "ngram_canonical_ids_manifest_sha256": canonical_ids["manifest_sha256"],
        "ngram_canonical_ids_binary": CANONICAL_IDS_BINARY,
        "ngram_canonical_ids_sha256": canonical_ids["binary_sha256"],
        "ngram_canonical_vocabulary_size": canonical_ids[
            "canonical_vocabulary_size"
        ],
    }
    atomic_json(output / "TOKENIZER_RELEASE.json", release)
    (output / "vocab.json").write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return release


def validate_tokenizer(tokenizer_path: str | Path, samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    domains: dict[str, dict[str, float]] = defaultdict(lambda: {"characters": 0, "tokens": 0, "documents": 0})
    failures: list[dict[str, str]] = []
    for sample in samples:
        text = str(sample.get("text", ""))
        domain = str(sample.get("category", sample.get("source_id", "unknown")))
        encoding = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        if decoded != text:
            failures.append({"domain": domain, "input": text[:200], "decoded": decoded[:200]})
        domains[domain]["characters"] += len(text)
        domains[domain]["tokens"] += len(encoding.ids)
        domains[domain]["documents"] += 1
    for values in domains.values():
        values["characters_per_token"] = values["characters"] / max(1, values["tokens"])
        values["tokens_per_document"] = values["tokens"] / max(1, values["documents"])
    return {
        "ok": not failures,
        "roundtrip_failures": failures[:100],
        "roundtrip_failure_count": len(failures),
        "domains": dict(domains),
    }
