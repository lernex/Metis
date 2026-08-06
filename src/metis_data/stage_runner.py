from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import lzma
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
import numpy as np
from tokenizers import Tokenizer

from .config import load_profile, load_yaml, repository_root
from .download import run_download_task
from .manifest import validate_manifest
from .quality import evaluate_quality, priority_score
from .stage_code import stage_code_sha256
from .state import StateStore, atomic_json, utc_now
from .tokenizer import train_tokenizer, validate_tokenizer
from .selection import build_selection, hamilton_apportion, replay_quotas, unique_quotas
from .replacement import allocate_replacements
from .download import sha256_file
from .code_dedup import code_hygiene_reason
from .final_dedup import content_sha256
from .build_inputs import build_input_count
from .normalization_evidence import (
    derive_normalization_evidence,
    extract_training_text,
    final_common_crawl_opt_out_reason,
    load_frozen_common_crawl_opt_out,
)
from .ngram_canonical import (
    CANONICAL_IDS_BINARY,
    CANONICAL_IDS_MANIFEST,
    validate_canonical_id_sidecar,
)
from .context_extension import (
    CONTEXT_PACK_PLAN_SCHEMA,
    build_context_pack_plan,
    build_context_selection,
    context_group_id,
    initialize_context_arrays,
    pack_context_evaluation,
    pack_context_task,
    structural_evidence,
    validate_context_pack_receipt,
    validate_context_selection,
    verify_and_seal_context_release,
)


def _paths(profile: dict[str, Any]) -> tuple[Path, StateStore]:
    root = Path(profile["storage"]["lustre_root"])
    state = StateStore(root / profile["storage"]["directories"]["state"])
    return root, state


def _stage_temporary_directory(
    profile: dict[str, Any], root: Path, stage: str
) -> Path:
    """Return durable-enough stage scratch, preferring scheduler-local storage."""

    runtime = profile.get("runtime", {})
    configured = str(runtime.get("node_local_temp_dir") or "").strip()
    if not configured or configured.lower() == "auto":
        configured = str(os.environ.get("SLURM_TMPDIR") or "").strip()
    if not configured:
        configured = str(runtime.get("temp_dir", "cache/tmp"))
    temporary = Path(configured).expanduser()
    if not temporary.is_absolute():
        temporary = root / temporary
    destination = temporary.resolve() / "metis-1.6" / stage
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _require_safety_space(profile: dict[str, Any], stage: str) -> None:
    root = Path(profile["storage"]["lustre_root"])
    safety_bytes = int(float(profile["storage"].get("safety_free_tb", 0)) * 1_000_000_000_000)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < safety_bytes:
        raise RuntimeError(
            f"Refusing stage {stage}: {free_bytes:,} free bytes is below the configured "
            f"{safety_bytes:,}-byte safety reserve"
        )


def _manifest(profile: dict[str, Any]) -> dict[str, Any]:
    path = Path(profile["manifest"])
    if not path.is_absolute():
        path = repository_root() / path
    return validate_manifest(path).require_valid()


def _normalization_evidence(
    row: dict[str, Any],
    source: dict[str, Any],
    file_record: dict[str, Any],
    text: str,
) -> dict[str, Any]:
    return derive_normalization_evidence(row, source, file_record, text)


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_dir():
        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            if child.name.startswith(".") or child.name == "ACQUISITION_RECEIPT.json":
                continue
            yield from _iter_rows(child)
        return
    name = path.name.lower()
    if name.endswith(".parquet"):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512):
            yield from batch.to_pylist()
        return
    if name.endswith(".jsonl.zst"):
        import zstandard as zstd

        raw = path.open("rb")
        handle = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
    elif name.endswith(".jsonl.gz") or name.endswith(".json.gz"):
        handle = gzip.open(path, "rt", encoding="utf-8")
    elif name.endswith(".jsonl.xz") or name.endswith(".json.xz"):
        handle = lzma.open(path, "rt", encoding="utf-8")
    elif name.endswith(".jsonl") or name.endswith(".json"):
        handle = path.open("r", encoding="utf-8")
    elif name.endswith(".txt") or name.endswith(".md"):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            yield {"id": path.stem, "text": text, "source_file": path.name}
        return
    else:
        return
    with handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _text_from_row(row: dict[str, Any]) -> str:
    return extract_training_text(row)


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# Fields that say who produced an artifact and when, rather than what it is.
# Hashing them made a re-resolve or a rehandoff invalidate finished stage work
# even when the resolved data was byte-identical.
_PROVENANCE_KEYS: dict[str, tuple[str, ...]] = {
    "sources.lock.json": (
        "repository_commit",
        "repository_dirty",
        "resolved_at",
        "lock_sha256",
        "resolver_runtime",
        "resolver_version",
    ),
    "build.inputs.json": ("created_at",),
    "ACQUISITION_READY.json": (
        "created_at",
        "handoff_sha256",
        "repository",
        "acquisition_runtime",
        "source_lock_sha256",
        "lustre_root",
    ),
}

# Scheduler keys that change how work is divided or buffered but cannot change
# what the work produces. Everything else in the scheduler block stays bound,
# including finder_tasks and bucket counts, which decide how records shard and
# therefore which duplicate survives.
_THROUGHPUT_ONLY_SCHEDULER_KEYS = frozenset(
    {
        "tasks_per_job",  # Slurm packaging
        "time",  # Slurm wall clock
        "maximum_open_files",  # writer handle pool; every record still routes to its own bucket
        "external_sort_max_open_runs",  # merge fan-in; the merge is total either way
    }
)


def _content_identity(value: Any, provenance: tuple[str, ...]) -> Any:
    """Strip provenance from a state artifact so identity follows content."""

    if not isinstance(value, dict):
        return value
    stripped = {key: item for key, item in value.items() if key not in provenance}
    artifacts = stripped.get("artifacts")
    if isinstance(artifacts, list):
        # mtime_ns is rewritten by any re-download of identical bytes.
        stripped["artifacts"] = [
            {key: item for key, item in record.items() if key != "mtime_ns"}
            if isinstance(record, dict)
            else record
            for record in artifacts
        ]
    return stripped


def _output_relevant_scheduler(scheduler: Any) -> Any:
    if isinstance(scheduler, dict):
        return {
            key: _output_relevant_scheduler(value)
            for key, value in scheduler.items()
            if key not in _THROUGHPUT_ONLY_SCHEDULER_KEYS
        }
    if isinstance(scheduler, list):
        return [_output_relevant_scheduler(item) for item in scheduler]
    return scheduler


def _stage_execution_contract(
    profile: dict[str, Any], state: StateStore, stage: str
) -> str:
    """Bind resumable CPU work to the inputs and policies that shape its output.

    Deliberately excluded: the repository commit, artifact timestamps, and
    scheduler knobs that only affect throughput. Those made unrelated edits
    invalidate finished work -- a fix to a dedup writer discarded a verified
    normalize pass over 1,862 shards. Code identity is still bound, but per
    stage, through stage_code_sha256.
    """

    manifest = _manifest(profile)
    state_artifacts: dict[str, str] = {}
    require_handoff = bool(profile.get("gates", {}).get("require_acquisition_handoff", False))
    for name in ("sources.lock.json", "build.inputs.json", "ACQUISITION_READY.json"):
        path = state.path(name)
        if not path.is_file():
            if name == "build.inputs.json" or require_handoff:
                raise RuntimeError(f"Stage {stage} requires immutable state artifact {name}")
            state_artifacts[name] = "absent-by-test-profile-contract"
        else:
            state_artifacts[name] = _json_sha256(
                _content_identity(
                    json.loads(path.read_text(encoding="utf-8")), _PROVENANCE_KEYS[name]
                )
            )
    return _json_sha256(
        {
            "schema": "metis.cpu-stage-execution/v2",
            "stage": stage,
            "release": manifest.get("release"),
            "manifest_contract_sha256": _manifest_contract_sha256(manifest),
            "state_artifacts": state_artifacts,
            "stage_code_sha256": stage_code_sha256(stage),
            "scheduler": _output_relevant_scheduler(profile.get("scheduler", {})),
            "gates": profile.get("gates", {}),
        }
    )


def _manifest_contract_sha256(manifest: dict[str, Any]) -> str:
    def public(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): public(item)
                for key, item in value.items()
                if not str(key).startswith("_")
            }
        if isinstance(value, list):
            return [public(item) for item in value]
        return value

    return _json_sha256(public(manifest))


def _production_tokenizer_contract(
    profile: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and fingerprint every tokenizer artifact used by packing."""

    manifest = manifest or _manifest(profile)
    root = Path(profile["storage"]["lustre_root"])
    tokenizer_root = root / profile["storage"]["directories"]["tokenizer"]
    paths = {
        "tokenizer": tokenizer_root / "tokenizer.json",
        "vocab": tokenizer_root / "vocab.json",
        "release": tokenizer_root / "TOKENIZER_RELEASE.json",
        "validation": tokenizer_root / "TOKENIZER_VALIDATION.json",
        "ngram_canonical_manifest": tokenizer_root / CANONICAL_IDS_MANIFEST,
        "ngram_canonical_ids": tokenizer_root / CANONICAL_IDS_BINARY,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Tokenizer release artifacts are missing: {missing}")
    if profile["storage"].get("final_token_dtype") != "uint16":
        raise RuntimeError("Metis-1.6 production packing requires final_token_dtype=uint16")

    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    vocab = tokenizer.get_vocab()
    expected_size = int(
        manifest["tokenizer"]["vocabulary_size_including_special_tokens"]
    )
    ids = list(vocab.values())
    if len(vocab) != expected_size or len(set(ids)) != expected_size:
        raise RuntimeError(
            f"Production tokenizer must have {expected_size:,} unique IDs, got {len(vocab):,}"
        )
    if set(ids) != set(range(expected_size)) or max(ids, default=-1) >= 65_536:
        raise RuntimeError("Production tokenizer IDs must be contiguous 0..65,535")
    expected_special_tokens = {
        str(token): tokenizer.token_to_id(str(token))
        for token in manifest["tokenizer"]["special_tokens"]
    }
    for token, token_id in expected_special_tokens.items():
        if token_id is None:
            raise RuntimeError(f"Production tokenizer is missing special token {token!r}")

    release = json.loads(paths["release"].read_text(encoding="utf-8"))
    tokenizer_sha = sha256_file(paths["tokenizer"])
    if (
        release.get("schema") != "metis.tokenizer-release/v1"
        or int(release.get("vocabulary_size", -1)) != expected_size
        or release.get("uint16_safe") is not True
        or release.get("tokenizer_sha256") != tokenizer_sha
        or release.get("special_tokens") != expected_special_tokens
    ):
        raise RuntimeError("TOKENIZER_RELEASE.json does not describe tokenizer.json")
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    if validation.get("ok") is not True:
        raise RuntimeError("TOKENIZER_VALIDATION.json is absent or did not pass")
    saved_vocab = json.loads(paths["vocab"].read_text(encoding="utf-8"))
    if saved_vocab != vocab:
        raise RuntimeError("vocab.json does not match tokenizer.json")
    canonical_descriptor, _canonical_ids = validate_canonical_id_sidecar(
        manifest_path=paths["ngram_canonical_manifest"],
        binary_path=paths["ngram_canonical_ids"],
        tokenizer_path=paths["tokenizer"],
        expected_vocabulary_size=expected_size,
        expected_manifest_sha256=release.get(
            "ngram_canonical_ids_manifest_sha256"
        ),
        expected_binary_sha256=release.get("ngram_canonical_ids_sha256"),
        recompute_from_tokenizer=True,
    )
    if (
        release.get("ngram_canonical_ids_manifest") != CANONICAL_IDS_MANIFEST
        or release.get("ngram_canonical_ids_binary") != CANONICAL_IDS_BINARY
        or release.get("ngram_canonicalization_algorithm")
        != canonical_descriptor["algorithm"]
        or int(release.get("ngram_canonical_vocabulary_size", -1))
        != int(canonical_descriptor["canonical_vocabulary_size"])
    ):
        raise RuntimeError(
            "TOKENIZER_RELEASE.json does not attest the canonical-ID sidecar"
        )

    eos_token = str(manifest["tokenizer"]["special_tokens"][0])
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise RuntimeError(f"Tokenizer is missing EOS token {eos_token!r}")
    contract: dict[str, Any] = {
        "schema": "metis.tokenizer-contract/v1",
        "tokenizer_sha256": tokenizer_sha,
        "vocab_sha256": sha256_file(paths["vocab"]),
        "tokenizer_release_sha256": sha256_file(paths["release"]),
        "tokenizer_validation_sha256": sha256_file(paths["validation"]),
        "ngram_canonical_map_manifest_sha256": sha256_file(
            paths["ngram_canonical_manifest"]
        ),
        "ngram_canonical_map_self_sha256": canonical_descriptor[
            "manifest_sha256"
        ],
        "ngram_canonical_ids_sha256": sha256_file(paths["ngram_canonical_ids"]),
        "ngram_canonicalization_algorithm": canonical_descriptor["algorithm"],
        "ngram_canonical_entry_count": int(canonical_descriptor["entry_count"]),
        "ngram_canonical_vocabulary_size": int(
            canonical_descriptor["canonical_vocabulary_size"]
        ),
        "ngram_canonical_dtype": canonical_descriptor["dtype"],
        "ngram_canonical_endianness": canonical_descriptor["endianness"],
        "vocabulary_size": expected_size,
        "minimum_id": min(ids),
        "maximum_id": max(ids),
        "token_dtype": "uint16",
        "endianness": "little",
        "eos_token": eos_token,
        "eos_token_id": int(eos_id),
    }
    contract["contract_sha256"] = _json_sha256(contract)
    return contract


def _file_inventory(path: Path, *, relative_to: Path) -> dict[str, Any]:
    resolved = path.resolve()
    base = relative_to.resolve()
    try:
        relative = resolved.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"Artifact escapes its immutable root: {path}") from exc
    return {
        "path": str(relative),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _require_inventory_file(root: Path, record: dict[str, Any], label: str) -> Path:
    candidate = (root / str(record.get("path", ""))).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes its declared root: {record}") from exc
    if (
        not candidate.is_file()
        or candidate.stat().st_size != int(record.get("size", -1))
        or sha256_file(candidate) != record.get("sha256")
    ):
        raise RuntimeError(f"{label} is missing or changed: {candidate}")
    return candidate


def _completion_inventory(
    state: StateStore,
    stage: str,
    expected_tasks: int,
    *,
    expected_execution_contract_sha256: str,
) -> dict[str, Any]:
    folder = state.path("completed", stage)
    paths = sorted(folder.glob("task-*.json")) if folder.is_dir() else []
    expected_names = {f"task-{index:06d}.json" for index in range(expected_tasks)}
    actual_names = {path.name for path in paths}
    if actual_names != expected_names:
        raise RuntimeError(
            f"Filtering stage {stage} completion inventory is incomplete: "
            f"missing={sorted(expected_names - actual_names)[:8]}, "
            f"unexpected={sorted(actual_names - expected_names)[:8]}"
        )
    rows = []
    for path in paths:
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Filtering stage completion marker is unreadable: {path}") from exc
        if marker.get("execution_contract_sha256") != expected_execution_contract_sha256:
            raise RuntimeError(
                f"Filtering stage {stage} completion belongs to stale inputs or policy: {path.name}"
            )
        rows.append(
            {"task": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {
        "stage": stage,
        "tasks": expected_tasks,
        "execution_contract_sha256": expected_execution_contract_sha256,
        "marker_manifest_sha256": _json_sha256(rows),
    }


def _hash_files(files: list[Path], workers: int) -> dict[Path, str]:
    """SHA-256 many independent files, using the whole node.

    ``hashlib`` releases the GIL around each buffer, so this read-and-digest
    work threads well and is bounded by Lustre rather than by Python. The
    caller still folds the results in sorted order, so the receipt and its tree
    hash stay byte-identical to the sequential implementation.
    """

    if workers <= 1 or len(files) <= 1:
        return {path: sha256_file(path) for path in files}
    digests: dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(files))) as pool:
        futures = {pool.submit(sha256_file, path): path for path in files}
        for future in as_completed(futures):
            digests[futures[future]] = future.result()
    return digests


def _receipt_hash_workers(profile: dict[str, Any] | None) -> int:
    if not profile:
        return 1
    runtime = profile.get("runtime", {})
    return max(1, int(runtime.get("receipt_hash_workers", 1) or 1))


def _write_directory_content_receipt(
    source_root: Path,
    destination: Path,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Hash every immutable stage output once and publish an auditable manifest."""

    if not source_root.is_dir():
        raise RuntimeError(f"Filtering stage output is missing: {source_root}")
    files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and not path.name.endswith(".incomplete")
        and ".incomplete" not in path.parts
    )
    if not files:
        raise RuntimeError(f"Filtering stage output is empty: {source_root}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".incomplete")
    digests = _hash_files(files, workers)
    tree = hashlib.sha256()
    total_bytes = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for path in files:
            relative = str(path.relative_to(source_root))
            size = path.stat().st_size
            digest = digests[path]
            row = {"path": relative, "size": size, "sha256": digest}
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            tree.update(relative.encode("utf-8"))
            tree.update(b"\0")
            tree.update(str(size).encode("ascii"))
            tree.update(b"\0")
            tree.update(digest.encode("ascii"))
            tree.update(b"\n")
            total_bytes += size
    temporary.replace(destination)
    return {
        "root": str(source_root),
        "files": len(files),
        "bytes": total_bytes,
        "tree_sha256": tree.hexdigest(),
        "manifest": str(destination),
        "manifest_sha256": sha256_file(destination),
    }


def _validate_content_receipt(
    profile: dict[str, Any],
    content: dict[str, Any],
    *,
    require_live_content: bool,
) -> None:
    """Validate a stage-content manifest, even after its files were retired."""

    lustre_root = Path(profile["storage"]["lustre_root"]).resolve()
    source_root = Path(str(content.get("root", ""))).resolve()
    manifest_path = Path(str(content.get("manifest", ""))).resolve()
    for path in (source_root, manifest_path):
        try:
            path.relative_to(lustre_root)
        except ValueError as exc:
            raise RuntimeError(f"Verified stage artifact escapes Lustre root: {path}") from exc
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != content.get("manifest_sha256")
    ):
        raise RuntimeError(f"Verified stage manifest changed: {manifest_path}")
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tree = hashlib.sha256()
    total_bytes = 0
    for row in rows:
        relative = str(row.get("path", ""))
        size = int(row.get("size", -1))
        digest = str(row.get("sha256", ""))
        if not relative or size < 0 or len(digest) != 64:
            raise RuntimeError(f"Verified stage manifest has an invalid row: {row}")
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(size).encode("ascii"))
        tree.update(b"\0")
        tree.update(digest.encode("ascii"))
        tree.update(b"\n")
        total_bytes += size
    if (
        len(rows) != int(content.get("files", -1))
        or total_bytes != int(content.get("bytes", -1))
        or tree.hexdigest() != content.get("tree_sha256")
    ):
        raise RuntimeError(f"Verified stage aggregate changed for {source_root}")
    if not require_live_content:
        return
    current_files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and not path.name.endswith(".incomplete")
        and ".incomplete" not in path.parts
    )
    if [str(path.relative_to(source_root)) for path in current_files] != [
        str(row["path"]) for row in rows
    ]:
        raise RuntimeError(f"Verified stage file inventory changed for {source_root}")
    for row, path in zip(rows, current_files):
        if path.stat().st_size != int(row["size"]):
            raise RuntimeError(f"Verified stage artifact changed: {path}")
    digests = _hash_files(current_files, _receipt_hash_workers(profile))
    for row, path in zip(rows, current_files):
        if digests[path] != row["sha256"]:
            raise RuntimeError(f"Verified stage artifact changed: {path}")


def _verified_content_for_filter_chain(
    profile: dict[str, Any],
    state: StateStore,
    name: str,
    output: Path,
    destination: Path,
) -> dict[str, Any]:
    cleanup = state.read("cleanup", f"{name}.json")
    if not cleanup:
        content = _write_directory_content_receipt(
            output, destination, workers=_receipt_hash_workers(profile)
        )
        content["retained"] = True
        return content
    unsigned = {key: value for key, value in cleanup.items() if key != "cleanup_sha256"}
    if (
        cleanup.get("schema") != "metis.verified-cleanup/v1"
        or cleanup.get("name") != name
        or cleanup.get("cleanup_sha256") != _json_sha256(unsigned)
    ):
        raise RuntimeError(f"Cleanup receipt is corrupt for {name}")
    content = dict(cleanup.get("content") or {})
    _validate_content_receipt(profile, content, require_live_content=False)
    for deletion in cleanup.get("deletions", []):
        path = Path(str(deletion.get("path", ""))).resolve()
        try:
            path.relative_to(Path(profile["storage"]["lustre_root"]).resolve())
        except ValueError as exc:
            raise RuntimeError(f"Cleanup path escapes Lustre root: {path}") from exc
        if path.exists():
            raise RuntimeError(f"Cleanup receipt says retired path still exists: {path}")
    return content


def _contamination_input_receipt(contamination: Path) -> dict[str, Any]:
    index = contamination / "index.json"
    holdouts = contamination / "holdouts.jsonl"
    holdout_report = contamination / "HOLDOUTS.json"
    for path in (index, holdouts, holdout_report):
        if not path.is_file():
            raise RuntimeError(f"Required contamination artifact is missing: {path}")
    payload = json.loads(index.read_text(encoding="utf-8"))
    referenced = [index, holdouts, holdout_report]
    for filename in payload.get("arrays", {}).values():
        path = index.parent / str(filename)
        if not path.is_file():
            raise RuntimeError(f"Contamination index array is missing: {path}")
        referenced.append(path)
    rows = [
        {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in referenced
    ]
    return {"artifacts": rows, "manifest_sha256": _json_sha256(rows)}


def _write_filter_chain_receipt(
    profile: dict[str, Any],
    state: StateStore,
    destination: Path,
) -> dict[str, Any]:
    root = Path(profile["storage"]["lustre_root"])
    directories = profile["storage"]["directories"]
    eligible = root / directories["eligible"]
    total_tasks = build_input_count(state)
    exact_finders = int(profile["scheduler"]["exact_dedup"]["find_tasks"])
    span_finders = int(profile["scheduler"]["repeated_span"]["finder_tasks"])
    minhash_buckets = int(profile["scheduler"]["minhash"]["num_buckets"])
    minhash_priority_buckets = int(
        profile["scheduler"].get("minhash_priority", {}).get("bucket_count", 256)
    )
    code_finders = int(profile["scheduler"]["code_structural"]["finder_tasks"])
    final_finders = int(profile["scheduler"]["final_hash"]["finder_tasks"])
    specs = [
        ("normalize", root / directories["normalized"], [("normalize", total_tasks)]),
        (
            "exact_sha256",
            eligible / "exact",
            [("exact_signature", total_tasks), ("exact_find", exact_finders), ("exact_filter", total_tasks)],
        ),
        (
            "repeated_span",
            eligible / "repeated-span-deduped",
            [
                ("span_prefilter_signature", total_tasks),
                ("span_prefilter_find", span_finders),
                ("span_signature", total_tasks),
                ("span_find", span_finders),
                ("span_filter", total_tasks),
            ],
        ),
        (
            "minhash",
            eligible / "near-deduped",
            [
                ("minhash_signature", total_tasks),
                ("minhash_buckets", minhash_buckets),
                ("minhash_components", 1),
                ("minhash_priority_candidates", total_tasks),
                ("minhash_priority_resolve", minhash_priority_buckets),
                ("minhash_priority_finalize", total_tasks),
                ("minhash_priority_verify", 1),
                ("minhash_filter", total_tasks),
            ],
        ),
        (
            "code_structural",
            eligible / "code-structural-deduped",
            [("code_signature", total_tasks), ("code_find", code_finders), ("code_filter", total_tasks)],
        ),
        (
            "decontamination",
            eligible / "decontaminated",
            [("decontam_index", 1), ("decontam_filter", total_tasks)],
        ),
        (
            "final_sha256",
            eligible / "final",
            [
                ("final_hash_signature", total_tasks),
                ("final_hash_find", final_finders),
                ("final_hash_filter", total_tasks),
            ],
        ),
    ]
    content_root = destination.parent / "filter-content"
    stages: list[dict[str, Any]] = []
    for name, output, completions in specs:
        stages.append(
            {
                "name": name,
                "policy_sha256": _json_sha256(
                    {
                        "scheduler": profile.get("scheduler", {}),
                        "gates": profile.get("gates", {}),
                        "stage": name,
                    }
                ),
                "completions": [
                    _completion_inventory(
                        state,
                        stage,
                        count,
                        expected_execution_contract_sha256=_stage_execution_contract(
                            profile, state, stage
                        ),
                    )
                    for stage, count in completions
                ],
                "content": _verified_content_for_filter_chain(
                    profile,
                    state,
                    name,
                    output,
                    content_root / f"{name}.jsonl",
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema": "metis.filter-chain/v1",
        "created_at": utc_now(),
        "stages": stages,
        "contamination_inputs": _contamination_input_receipt(
            root / directories["contamination"]
        ),
    }
    payload["filter_chain_sha256"] = _json_sha256(payload)
    atomic_json(destination, payload)
    return payload


def _validate_filter_chain_artifacts(
    profile: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    unsigned = {
        key: value for key, value in payload.items() if key != "filter_chain_sha256"
    }
    if (
        payload.get("schema") != "metis.filter-chain/v1"
        or payload.get("filter_chain_sha256") != _json_sha256(unsigned)
    ):
        raise RuntimeError("Filtering/decontamination receipt failed its self-hash check")
    lustre_root = Path(profile["storage"]["lustre_root"]).resolve()
    for stage in payload.get("stages", []):
        content = stage.get("content", {})
        _validate_content_receipt(
            profile,
            content,
            require_live_content=bool(content.get("retained", True)),
        )
    for record in payload.get("contamination_inputs", {}).get("artifacts", []):
        path = Path(str(record.get("path", ""))).resolve()
        try:
            path.relative_to(lustre_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Contamination artifact escapes Lustre root: {path}"
            ) from exc
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"Contamination artifact changed: {path}")


def _safe_retire_path(root: Path, path: Path) -> dict[str, Any]:
    root = root.resolve()
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to retire path outside Lustre root: {path}") from exc
    if not relative.parts:
        raise RuntimeError("Refusing to retire the Lustre root")
    existed = path.exists()
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
    if path.exists():
        raise RuntimeError(f"Verified intermediate could not be retired: {path}")
    return {"path": str(path), "relative_path": str(relative), "existed": existed}


def _cleanup_filter_intermediate(
    profile: dict[str, Any],
    cleanup_stage: str,
) -> dict[str, Any]:
    """Hash a successor corpus, then retire only its verified predecessor."""

    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    eligible = root / directories["eligible"]
    dedup = root / directories["dedup"]
    total_tasks = build_input_count(state)
    exact_finders = int(profile["scheduler"]["exact_dedup"]["find_tasks"])
    span_finders = int(profile["scheduler"]["repeated_span"]["finder_tasks"])
    minhash_buckets = int(profile["scheduler"]["minhash"]["num_buckets"])
    minhash_priority_buckets = int(
        profile["scheduler"].get("minhash_priority", {}).get("bucket_count", 256)
    )
    code_finders = int(profile["scheduler"]["code_structural"]["finder_tasks"])
    final_finders = int(profile["scheduler"]["final_hash"]["finder_tasks"])
    cache = root / directories["cache"]
    specs: dict[str, dict[str, Any]] = {
        "cleanup_raw": {
            "name": "normalize",
            "output": root / directories["normalized"],
            "completions": [("normalize", total_tasks)],
            "delete": [
                root / directories["raw"],
                cache / "huggingface",
                cache / "common-crawl",
                cache / "tmp" / "materializers",
            ],
        },
        "cleanup_exact": {
            "name": "exact_sha256",
            "output": eligible / "exact",
            "completions": [
                ("exact_signature", total_tasks),
                ("exact_find", exact_finders),
                ("exact_filter", total_tasks),
            ],
            "delete": [root / directories["normalized"], dedup / "exact"],
        },
        "cleanup_span": {
            "name": "repeated_span",
            "output": eligible / "repeated-span-deduped",
            "completions": [
                ("span_prefilter_signature", total_tasks),
                ("span_prefilter_find", span_finders),
                ("span_signature", total_tasks),
                ("span_find", span_finders),
                ("span_filter", total_tasks),
            ],
            "delete": [eligible / "exact", dedup / "repeated-span"],
        },
        "cleanup_minhash": {
            "name": "minhash",
            "output": eligible / "near-deduped",
            "completions": [
                ("minhash_signature", total_tasks),
                ("minhash_buckets", minhash_buckets),
                ("minhash_components", 1),
                ("minhash_priority_candidates", total_tasks),
                ("minhash_priority_resolve", minhash_priority_buckets),
                ("minhash_priority_finalize", total_tasks),
                ("minhash_priority_verify", 1),
                ("minhash_filter", total_tasks),
            ],
            "delete": [eligible / "repeated-span-deduped", dedup / "minhash"],
        },
        "cleanup_code": {
            "name": "code_structural",
            "output": eligible / "code-structural-deduped",
            "completions": [
                ("code_signature", total_tasks),
                ("code_find", code_finders),
                ("code_filter", total_tasks),
            ],
            "delete": [eligible / "near-deduped", dedup / "code-structural"],
        },
        "cleanup_decontam": {
            "name": "decontamination",
            "output": eligible / "decontaminated",
            "completions": [("decontam_index", 1), ("decontam_filter", total_tasks)],
            "delete": [
                eligible / "code-structural-deduped",
                root / directories["contamination"] / "quarantine",
            ],
        },
        "cleanup_final_hash": {
            "name": "final_sha256",
            "output": eligible / "final",
            "completions": [
                ("final_hash_signature", total_tasks),
                ("final_hash_find", final_finders),
                ("final_hash_filter", total_tasks),
            ],
            "delete": [eligible / "decontaminated", dedup / "final-sha256"],
        },
    }
    try:
        spec = specs[cleanup_stage]
    except KeyError as exc:
        raise RuntimeError(f"Unknown verified cleanup stage: {cleanup_stage}") from exc
    name = str(spec["name"])
    final = state.read("cleanup", f"{name}.json")
    if final:
        unsigned = {key: value for key, value in final.items() if key != "cleanup_sha256"}
        if (
            final.get("schema") != "metis.verified-cleanup/v1"
            or final.get("cleanup_sha256") != _json_sha256(unsigned)
        ):
            raise RuntimeError(f"Existing cleanup receipt is corrupt for {name}")
        _validate_content_receipt(
            profile, dict(final["content"]), require_live_content=False
        )
        for deletion in final.get("deletions", []):
            if Path(str(deletion["path"])).exists():
                raise RuntimeError(
                    f"Retired intermediate unexpectedly reappeared: {deletion['path']}"
                )
        return final
    completion_receipts = [
        _completion_inventory(
            state,
            stage,
            count,
            expected_execution_contract_sha256=_stage_execution_contract(
                profile, state, stage
            ),
        )
        for stage, count in spec["completions"]
    ]
    pending = state.read("cleanup", "pending", f"{name}.json")
    if pending:
        unsigned_pending = {
            key: value for key, value in pending.items() if key != "pending_sha256"
        }
        if (
            pending.get("schema") != "metis.verified-cleanup-pending/v1"
            or pending.get("pending_sha256") != _json_sha256(unsigned_pending)
            or pending.get("completions") != completion_receipts
        ):
            raise RuntimeError(f"Pending cleanup receipt is corrupt for {name}")
        content = dict(pending["content"])
        _validate_content_receipt(profile, content, require_live_content=False)
    else:
        content = _write_directory_content_receipt(
            Path(spec["output"]),
            state.path("verified-content", f"{name}.jsonl"),
            workers=_receipt_hash_workers(profile),
        )
        pending = {
            "schema": "metis.verified-cleanup-pending/v1",
            "name": name,
            "cleanup_stage": cleanup_stage,
            "content": content,
            "completions": completion_receipts,
            "created_at": utc_now(),
        }
        pending["pending_sha256"] = _json_sha256(pending)
        state.write("cleanup", "pending", f"{name}.json", payload=pending)
        _validate_content_receipt(profile, content, require_live_content=True)
    # Retiring an intermediate is irreversible, and one of those intermediates
    # is the acquisition output that verify_acquisition_handoff re-checks on
    # every submit and every resume. Deleting it therefore does not just free
    # space, it ends the build's ability to be restarted at all: the graph can
    # only ever run forward from that point, and any interruption strands it
    # behind a re-download. Operators with room to spare keep the inputs.
    if profile.get("gates", {}).get("retain_stage_inputs"):
        deletions: list[dict[str, Any]] = []
    else:
        deletions = [_safe_retire_path(root, Path(path)) for path in spec["delete"]]
    content["retained"] = False
    payload: dict[str, Any] = {
        "schema": "metis.verified-cleanup/v1",
        "name": name,
        "cleanup_stage": cleanup_stage,
        "content": content,
        "completions": completion_receipts,
        "deletions": deletions,
        "completed_at": utc_now(),
    }
    payload["cleanup_sha256"] = _json_sha256(payload)
    state.write("cleanup", f"{name}.json", payload=payload)
    state.path("cleanup", "pending", f"{name}.json").unlink(missing_ok=True)
    state.complete(cleanup_stage, "task-000000", payload)
    return payload


def _cleanup_tokenizer_sample(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    existing = state.read("cleanup", "tokenizer_sample.json")
    if existing:
        unsigned = {key: value for key, value in existing.items() if key != "cleanup_sha256"}
        if existing.get("cleanup_sha256") != _json_sha256(unsigned):
            raise RuntimeError("Tokenizer-sample cleanup receipt is corrupt")
        for deletion in existing.get("deletions", []):
            if Path(str(deletion["path"])).exists():
                raise RuntimeError("Retired tokenizer sample unexpectedly reappeared")
        return existing
    contract = _production_tokenizer_contract(profile)
    pending = state.read("cleanup", "pending", "tokenizer_sample.json")
    if pending:
        unsigned_pending = {
            key: value for key, value in pending.items() if key != "pending_sha256"
        }
        if (
            pending.get("pending_sha256") != _json_sha256(unsigned_pending)
            or pending.get("tokenizer_contract") != contract
        ):
            raise RuntimeError("Pending tokenizer-sample cleanup receipt is corrupt")
    else:
        pending = {
            "schema": "metis.verified-cleanup-pending/v1",
            "name": "tokenizer_sample",
            "tokenizer_contract": contract,
            "created_at": utc_now(),
        }
        pending["pending_sha256"] = _json_sha256(pending)
        state.write(
            "cleanup", "pending", "tokenizer_sample.json", payload=pending
        )
    tokenizer_dir = root / profile["storage"]["directories"]["tokenizer"]
    deletions = [
        _safe_retire_path(root, tokenizer_dir / name)
        for name in ("sample.jsonl", "sample-parts", "scan", "SAMPLE_PLAN.json")
    ]
    payload: dict[str, Any] = {
        "schema": "metis.verified-cleanup/v1",
        "name": "tokenizer_sample",
        "tokenizer_contract": contract,
        "deletions": deletions,
        "completed_at": utc_now(),
    }
    payload["cleanup_sha256"] = _json_sha256(payload)
    state.write("cleanup", "tokenizer_sample.json", payload=payload)
    state.path("cleanup", "pending", "tokenizer_sample.json").unlink(missing_ok=True)
    state.complete("cleanup_tokenizer_sample", "task-000000", payload)
    return payload


def _cleanup_selection_inputs(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    selected = root / directories["selected"]
    selection_path = selected / "SELECTION.json"
    existing = state.read("cleanup", "selection_inputs.json")
    if existing:
        unsigned = {key: value for key, value in existing.items() if key != "cleanup_sha256"}
        if (
            existing.get("cleanup_sha256") != _json_sha256(unsigned)
            or not selection_path.is_file()
            or sha256_file(selection_path) != existing.get("selection_sha256")
        ):
            raise RuntimeError("Selection-input cleanup receipt is corrupt or stale")
        _validate_content_receipt(
            profile, dict(existing["content"]), require_live_content=True
        )
        for deletion in existing.get("deletions", []):
            if Path(str(deletion["path"])).exists():
                raise RuntimeError("Retired selection input unexpectedly reappeared")
        return existing
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    pending = state.read("cleanup", "pending", "selection_inputs.json")
    if pending:
        unsigned_pending = {
            key: value for key, value in pending.items() if key != "pending_sha256"
        }
        if (
            pending.get("pending_sha256") != _json_sha256(unsigned_pending)
            or pending.get("selection_sha256") != sha256_file(selection_path)
        ):
            raise RuntimeError("Pending selection-input cleanup receipt is corrupt")
        content = dict(pending["content"])
        _validate_content_receipt(profile, content, require_live_content=True)
    else:
        _validate_selection_artifacts(
            profile,
            state,
            selection,
            deep_token_count_validation=True,
        )
        content = _write_directory_content_receipt(
            selected,
            state.path("verified-content", "selection.jsonl"),
            workers=_receipt_hash_workers(profile),
        )
        _validate_content_receipt(profile, content, require_live_content=True)
        pending = {
            "schema": "metis.verified-cleanup-pending/v1",
            "name": "selection_inputs",
            "selection_sha256": sha256_file(selection_path),
            "content": content,
            "created_at": utc_now(),
        }
        pending["pending_sha256"] = _json_sha256(pending)
        state.write(
            "cleanup", "pending", "selection_inputs.json", payload=pending
        )
    deletions = [
        _safe_retire_path(root, root / directories["token_counts"]),
        _safe_retire_path(root, root / directories["eligible"] / "final"),
    ]
    payload: dict[str, Any] = {
        "schema": "metis.verified-cleanup/v1",
        "name": "selection_inputs",
        "selection_sha256": sha256_file(selection_path),
        "content": content,
        "deletions": deletions,
        "completed_at": utc_now(),
    }
    payload["cleanup_sha256"] = _json_sha256(payload)
    state.write("cleanup", "selection_inputs.json", payload=payload)
    state.path("cleanup", "pending", "selection_inputs.json").unlink(
        missing_ok=True
    )
    state.complete("cleanup_selection_inputs", "task-000000", payload)
    return payload


def _cleanup_pack_inputs(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    existing = state.read("cleanup", "pack_inputs.json")
    if existing:
        unsigned = {key: value for key, value in existing.items() if key != "cleanup_sha256"}
        if existing.get("cleanup_sha256") != _json_sha256(unsigned):
            raise RuntimeError("Pack-input cleanup receipt is corrupt")
        for deletion in existing.get("deletions", []):
            if Path(str(deletion["path"])).exists():
                raise RuntimeError("Retired pack input unexpectedly reappeared")
        return existing
    verification = state.read("VERIFICATION.json")
    if not verification:
        raise RuntimeError("Pack inputs cannot be retired before VERIFICATION.json exists")
    unsigned = {key: value for key, value in verification.items() if key != "verification_sha256"}
    if (
        verification.get("ok") is not True
        or verification.get("verification_sha256") != _json_sha256(unsigned)
    ):
        raise RuntimeError("Pack inputs cannot be retired against an invalid verification")
    pending = state.read("cleanup", "pending", "pack_inputs.json")
    if pending:
        unsigned_pending = {
            key: value for key, value in pending.items() if key != "pending_sha256"
        }
        if (
            pending.get("pending_sha256") != _json_sha256(unsigned_pending)
            or pending.get("verification_sha256")
            != verification["verification_sha256"]
        ):
            raise RuntimeError("Pending pack-input cleanup receipt is corrupt")
    else:
        pending = {
            "schema": "metis.verified-cleanup-pending/v1",
            "name": "pack_inputs",
            "verification_sha256": verification["verification_sha256"],
            "created_at": utc_now(),
        }
        pending["pending_sha256"] = _json_sha256(pending)
        state.write("cleanup", "pending", "pack_inputs.json", payload=pending)
    selected = root / profile["storage"]["directories"]["selected"]
    deletions = [
        _safe_retire_path(root, selected / "schedule"),
        _safe_retire_path(root, selected / "replay-pool"),
    ]
    payload: dict[str, Any] = {
        "schema": "metis.verified-cleanup/v1",
        "name": "pack_inputs",
        "verification_sha256": verification["verification_sha256"],
        "deletions": deletions,
        "completed_at": utc_now(),
    }
    payload["cleanup_sha256"] = _json_sha256(payload)
    state.write("cleanup", "pack_inputs.json", payload=payload)
    state.path("cleanup", "pending", "pack_inputs.json").unlink(missing_ok=True)
    state.complete("cleanup_pack_inputs", "task-000000", payload)
    return payload


def _cleanup_release_workspace(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    release_root = root / directories["release"]
    from .training_contract import validate_training_release

    validated = validate_training_release(
        release_root,
        repository_root() / "configs" / "metis16" / "pretraining.yaml",
    )
    existing = state.read("cleanup", "release_workspace.json")
    if existing:
        unsigned = {key: value for key, value in existing.items() if key != "cleanup_sha256"}
        if (
            existing.get("cleanup_sha256") != _json_sha256(unsigned)
            or existing.get("release") != validated
        ):
            raise RuntimeError("Release-workspace cleanup receipt is corrupt or stale")
        for deletion in existing.get("deletions", []):
            if Path(str(deletion["path"])).exists():
                raise RuntimeError("Retired release workspace unexpectedly reappeared")
        return existing
    pending = state.read("cleanup", "pending", "release_workspace.json")
    if pending:
        unsigned_pending = {
            key: value for key, value in pending.items() if key != "pending_sha256"
        }
        if (
            pending.get("pending_sha256") != _json_sha256(unsigned_pending)
            or pending.get("release") != validated
        ):
            raise RuntimeError("Pending release-workspace cleanup receipt is corrupt")
    else:
        pending = {
            "schema": "metis.verified-cleanup-pending/v1",
            "name": "release_workspace",
            "release": validated,
            "created_at": utc_now(),
        }
        pending["pending_sha256"] = _json_sha256(pending)
        state.write(
            "cleanup", "pending", "release_workspace.json", payload=pending
        )
    candidates = [
        root / directories[name]
        for name in ("raw", "normalized", "eligible", "dedup", "token_counts", "selected")
    ] + [
        root / directories["tokenizer"],
        root / directories["cache"],
    ]
    deletions = [
        _safe_retire_path(root, path)
        for path in candidates
        if path.resolve() != release_root.resolve()
    ]
    payload: dict[str, Any] = {
        "schema": "metis.verified-cleanup/v1",
        "name": "release_workspace",
        "release": validated,
        "deletions": deletions,
        "completed_at": utc_now(),
    }
    payload["cleanup_sha256"] = _json_sha256(payload)
    state.write("cleanup", "release_workspace.json", payload=payload)
    state.path("cleanup", "pending", "release_workspace.json").unlink(
        missing_ok=True
    )
    state.complete("cleanup_release_workspace", "task-000000", payload)
    return payload


def _normalize_task(profile: dict[str, Any], task_index: int) -> dict[str, Any]:
    import zstandard as zstd

    root, state = _paths(profile)
    execution_contract = _stage_execution_contract(profile, state, "normalize")
    build_inputs = state.read("build.inputs.json")
    if not build_inputs:
        raise RuntimeError("build.inputs.json is missing; prepare the immutable Rhea build inputs first")
    try:
        file_record = build_inputs["inputs"][task_index]
    except IndexError as exc:
        raise ValueError(f"Unknown normalization input {task_index}") from exc
    input_integrity: dict[str, Any] | None = None
    if profile.get("gates", {}).get("require_deep_handoff_verification", False):
        from .handoff_verification import require_verified_build_input

        input_integrity = require_verified_build_input(profile, state, file_record)
    manifest = _manifest(profile)
    sources = {source["id"]: source for source in manifest["sources"]}
    source_id = str(file_record.get("source_id") or "")
    source = sources.get(source_id)
    if source is None and file_record.get("kind") != "remote_source_plan":
        raise RuntimeError(f"Build input references unknown manifest source: {source_id!r}")
    source_driver = (
        str(source.get("acquisition", {}).get("driver") or "")
        if source is not None
        else ""
    )
    input_driver = str(file_record.get("driver") or "")
    # The final opt-out re-check exists to honour publishers who withdrew after
    # the corpus was acquired, so what matters is whether the text came out of
    # Common Crawl -- not which driver fetched it. A packaged extraction such as
    # FineWeb is Common Crawl text that a third party filtered at build time,
    # which makes it *more* exposed to later withdrawals than a fresh crawl, not
    # less. Gating on the driver alone silently exempted it.
    common_crawl_derived = bool(
        source is not None
        and source.get("provenance", {}).get("common_crawl_derived")
    )
    final_opt_out_policy = None
    if source_driver == "common_crawl_ranges":
        if input_driver != "common_crawl_ranges":
            raise RuntimeError(
                f"Common Crawl build input {task_index} lost its acquisition driver identity"
            )
        final_opt_out_policy = load_frozen_common_crawl_opt_out(root, state)
    elif common_crawl_derived:
        final_opt_out_policy = load_frozen_common_crawl_opt_out(root, state)
    output_dir = root / profile["storage"]["directories"]["normalized"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"task-{task_index:06d}.jsonl.zst"
    temporary_output = output.with_suffix(output.suffix + ".incomplete")
    report = output_dir / f"task-{task_index:06d}.report.json"
    if report.exists() and output.exists():
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("execution_contract_sha256") != execution_contract:
            report.unlink()
            output.unlink()
        else:
            if (
                output.stat().st_size != int(payload.get("output_size", -1))
                or sha256_file(output) != payload.get("output_sha256")
            ):
                raise RuntimeError(f"Normalized output changed after completion: {output}")
            if input_integrity is not None:
                from .handoff_verification import verify_build_input_after_read

                verify_build_input_after_read(input_integrity)
            return payload
    if report.exists() != output.exists():
        # Both files are derived from an immutable, deep-verified input. A
        # crash between their atomic publications is safely restartable.
        report.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
    counts = {"input": 0, "accepted": 0, "rejected": 0, "no_text": 0, "remote_plans": 0}
    rejection_reasons: dict[str, int] = {}
    temporary_output.unlink(missing_ok=True)
    try:
        with temporary_output.open("wb") as raw:
            with zstd.ZstdCompressor(level=6).stream_writer(raw) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                    if file_record.get("kind") == "remote_source_plan":
                        counts["remote_plans"] += 1
                    else:
                        assert source is not None
                        profile_name = source["processing"]["quality_profile"]
                        source_priority = int(source["processing"].get("priority", 1))
                        for row_index, row in enumerate(_iter_rows(Path(file_record["local_path"]))):
                            counts["input"] += 1
                            text = _text_from_row(row)
                            if not text:
                                counts["no_text"] += 1
                                continue
                            if final_opt_out_policy is not None:
                                opt_out_reason, _matched_url = final_common_crawl_opt_out_reason(
                                    row, final_opt_out_policy
                                )
                                if opt_out_reason:
                                    counts["rejected"] += 1
                                    rejection_reasons[opt_out_reason] = (
                                        rejection_reasons.get(opt_out_reason, 0) + 1
                                    )
                                    continue
                            metadata = _normalization_evidence(row, source, file_record, text)
                            if final_opt_out_policy is not None:
                                metadata.update(
                                    {
                                        "final_common_crawl_opt_out_reapplied": True,
                                        "final_common_crawl_opt_out_snapshot_sha256": (
                                            final_opt_out_policy.snapshot_sha256
                                        ),
                                    }
                                )
                            metadata.update(
                                {
                                    "source_id": source_id,
                                    "category": source["category"],
                                    "source_revision": file_record.get("revision"),
                                    "source_file": file_record.get("repo_path"),
                                    "license_status": source["license"]["status"],
                                    "generated": bool(source["provenance"].get("generated")),
                                    "transformed": bool(source["provenance"].get("transformed")),
                                    "human_original": not bool(source["provenance"].get("generated"))
                                    and not bool(source["provenance"].get("transformed")),
                                    "fresh": bool(source["provenance"].get("fresh")),
                                }
                            )
                            hygiene_reason = code_hygiene_reason(text, metadata)
                            if hygiene_reason:
                                counts["rejected"] += 1
                                rejection_reasons[hygiene_reason] = rejection_reasons.get(hygiene_reason, 0) + 1
                                continue
                            if source["license"]["status"] in {"per_record_required", "inherited", "requires_review"} and not metadata.get("license"):
                                counts["rejected"] += 1
                                rejection_reasons["missing_license"] = rejection_reasons.get("missing_license", 0) + 1
                                continue
                            decision = evaluate_quality(
                                text,
                                profile_name=profile_name,
                                metadata=metadata,
                                fail_closed=bool(profile.get("gates", {}).get("fail_closed", True)),
                            )
                            if not decision.keep:
                                counts["rejected"] += 1
                                rejection_reasons[decision.reason] = rejection_reasons.get(decision.reason, 0) + 1
                                continue
                            doc_id = row.get("id") or row.get("uuid") or f"{source_id}:{task_index}:{row_index}"
                            payload = {
                                "id": str(doc_id),
                                "text": text,
                                "metadata": {
                                    **metadata,
                                    "priority": priority_score(source_priority, metadata),
                                    "quality_features": decision.features,
                                },
                            }
                            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                            counts["accepted"] += 1
        if counts["remote_plans"]:
            raise RuntimeError(
                f"Normalization task {task_index} contains {counts['remote_plans']} unresolved remote acquisition plan(s). "
                "A selection plan is not training data; materialize it before submitting the build graph."
            )
        # Zero accepted from zero inputs is an empty file, not a failed gate.
        # Proof-Pile-2 ships genuinely empty shards -- github-MATLAB-train-0001
        # through -0003 and github-coq-train-0001 are 13-byte zstd frames
        # carrying no rows -- and failing on them stops the whole afterok graph
        # over a file the publisher released empty. There is nothing to reject
        # and nothing to accept, so the honest output is an empty one. The gate
        # still fires whenever records were read and every one was dropped,
        # which is the case it was written for.
        # A task that accepts nothing is recorded, not fatal. This guard was
        # written to catch a profile demanding evidence no publisher ships, and
        # it has never once caught that here -- `preflight-profiles` finds those
        # in a minute by sampling every source, and `select` enforces the
        # corpus-wide floor at the end. What the guard actually caught three
        # times was a property of one file: a 13-byte empty shard, an OpenStax
        # book whose task holds exactly one document, and a single unusually
        # repetitive AlgebraicStack file. Each time it converted "this file
        # yields nothing" into "the build stops", because a failed task fails
        # its array element and afterok stops all 49 downstream jobs.
        #
        # The zero-yield fact is kept in the task report and in the stage
        # summary, so a source that genuinely normalizes to nothing is visible
        # in the receipts rather than silent -- and it is visible corpus-wide at
        # the `minimum_unique_tokens` gate, which is where a real systematic
        # failure belongs.
        if counts["accepted"] == 0:
            print(
                json.dumps(
                    {
                        "stage": "normalize",
                        "task_index": task_index,
                        "zero_yield": True,
                        "inputs": counts["input"],
                        "rejection_reasons": rejection_reasons,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if input_integrity is not None:
            from .handoff_verification import verify_build_input_after_read

            verify_build_input_after_read(input_integrity)
        temporary_output.replace(output)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise
    payload = {
        "stage": "normalize",
        "task_index": task_index,
        "execution_contract_sha256": execution_contract,
        "output": str(output),
        "output_size": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "input_integrity": (
            {
                "artifact_id": input_integrity["artifact_id"],
                "marker_sha256": input_integrity["marker_sha256"],
                "sha256": input_integrity["sha256"],
            }
            if input_integrity is not None
            else None
        ),
        "counts": counts,
        "rejection_reasons": rejection_reasons,
        # Explicit so a corpus-wide sweep for "what normalized to nothing" is a
        # grep over the receipts rather than an inference from two counters.
        "zero_yield": counts["accepted"] == 0,
        "common_crawl_opt_out": (
            {
                "reapplied": True,
                "snapshot_sha256": final_opt_out_policy.snapshot_sha256,
            }
            if final_opt_out_policy is not None
            else None
        ),
        "completed_at": utc_now(),
    }
    atomic_json(report, payload)
    state.complete("normalize", f"task-{task_index:06d}", payload)
    return payload


def _content(doc: Any) -> str:
    return str(doc.text)


def _priority(doc: Any) -> int:
    return int(doc.metadata.get("priority", 1))


def _local_executor(profile: dict[str, Any], stage: str, task_index: int, tasks: int, pipeline: list[Any]) -> None:
    from datatrove.executor.local import LocalPipelineExecutor

    root, state = _paths(profile)
    execution_contract = _stage_execution_contract(profile, state, stage)
    # DataTrove's own completion logs are resumable, but only within the exact
    # manifest/input/policy contract that produced them.
    logs = (
        root
        / profile["storage"]["directories"]["logs"]
        / stage
        / execution_contract[:24]
    )
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=str(logs),
        tasks=tasks,
        workers=1,
        local_tasks=1,
        local_rank_offset=task_index,
        skip_completed=True,
    )
    executor.run()


def _datatrove_stage(profile: dict[str, Any], stage: str, task_index: int) -> dict[str, Any]:
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets, MinhashDedupFilter
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers.jsonl import JsonlWriter
    from datatrove.utils.hashing import HashConfig
    from .datatrove_blocks import build_regex_word_tokenizer

    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    normalized = root / directories["normalized"]
    eligible = root / directories["eligible"]
    dedup = root / directories["dedup"]
    contamination = root / directories["contamination"]
    total_tasks = build_input_count(state)
    finder_tasks = int(profile["scheduler"]["exact_dedup"]["find_tasks"])
    mh_profile = profile["scheduler"]["minhash"]
    mh_config = MinhashConfig(
        n_grams=int(mh_profile["n_grams"]),
        num_buckets=int(mh_profile["num_buckets"]),
        hashes_per_bucket=int(mh_profile["hashes_per_bucket"]),
        seed=16062026,
        hash_config=HashConfig(precision=64),
    )
    reader = JsonlReader(str(normalized), glob_pattern="task-*.jsonl.zst", compression="zstd", shuffle_files=False)
    exact_sig = dedup / "exact" / "signatures"
    exact_dups = dedup / "exact" / "duplicates"
    exact_output = eligible / "exact"
    span_profile = profile["scheduler"]["repeated_span"]
    span_finders = int(span_profile["finder_tasks"])
    span_prefilter = dedup / "repeated-span" / "prefilter"
    span_candidates = dedup / "repeated-span" / "candidates"
    span_sig = dedup / "repeated-span" / "signatures"
    span_remove = dedup / "repeated-span" / "remove_ids"
    span_output = eligible / "repeated-span-deduped"
    mh_sig = dedup / "minhash" / "signatures"
    mh_buckets = dedup / "minhash" / "buckets"
    mh_priority_work = dedup / "minhash" / "priority"
    mh_remove = dedup / "minhash" / "remove_ids"
    mh_output = eligible / "near-deduped"
    code_profile = profile["scheduler"]["code_structural"]
    code_finders = int(code_profile["finder_tasks"])
    code_sig = dedup / "code-structural" / "signatures"
    code_remove = dedup / "code-structural" / "remove_ids"
    code_output = eligible / "code-structural-deduped"
    final_profile = profile["scheduler"]["final_hash"]
    final_finders = int(final_profile["finder_tasks"])
    final_sig = dedup / "final-sha256" / "signatures"
    final_remove = dedup / "final-sha256" / "remove_ids"
    final_output = eligible / "final"

    if stage == "exact_signature":
        from .final_dedup import write_final_signatures

        report = write_final_signatures(
            reader.run(rank=task_index, world_size=total_tasks),
            exact_sig,
            rank=task_index,
            finder_workers=finder_tasks,
        )
        state.write("exact-sha256", "signature-reports", f"task-{task_index:06d}.json", payload=report)
    elif stage == "exact_find":
        from .final_dedup import find_final_duplicates

        temp_root = Path(profile.get("runtime", {}).get("temp_dir", "cache/tmp"))
        if not temp_root.is_absolute():
            temp_root = root / temp_root
        report = find_final_duplicates(
            exact_sig,
            exact_dups,
            bucket=task_index,
            finder_workers=finder_tasks,
            expected_ranks=total_tasks,
            temporary_directory=temp_root / "exact-sha256-finders",
        )
        state.write("exact-sha256", "finder-reports", f"bucket-{task_index:04d}.json", payload=report)
    elif stage == "exact_filter":
        from .final_dedup import build_sha256_filter

        quarantine = JsonlWriter(str(dedup / "exact" / "quarantine"))
        _local_executor(
            profile,
            stage,
            task_index,
            total_tasks,
            [
                reader,
                build_sha256_filter(
                    exact_dups,
                    finder_workers=finder_tasks,
                    reason="exact_sha256_duplicate",
                    annotate_hash=False,
                    exclusion_writer=quarantine,
                ),
                JsonlWriter(str(exact_output)),
            ],
        )
    elif stage == "span_prefilter_signature":
        from .span_dedup import write_span_prefilter_signatures

        exact_reader = JsonlReader(str(exact_output), shuffle_files=False)
        report = write_span_prefilter_signatures(
            exact_reader.run(rank=task_index, world_size=total_tasks),
            span_prefilter,
            rank=task_index,
            finder_workers=span_finders,
            sentence_count=int(span_profile.get("sentence_count", 3)),
            minimum_span_words=int(span_profile.get("minimum_span_words", 24)),
            maximum_open_files=int(span_profile.get("maximum_open_files", 32)),
        )
        state.write(
            "repeated-span",
            "prefilter-signature-reports",
            f"task-{task_index:06d}.json",
            payload=report,
        )
    elif stage == "span_prefilter_find":
        from .span_dedup import find_repeated_span_candidates

        temp_root = Path(profile.get("runtime", {}).get("temp_dir", "cache/tmp"))
        if not temp_root.is_absolute():
            temp_root = root / temp_root
        report = find_repeated_span_candidates(
            span_prefilter,
            span_candidates,
            bucket=task_index,
            finder_workers=span_finders,
            total_ranks=total_tasks,
            sentence_count=int(span_profile.get("sentence_count", 3)),
            minimum_span_words=int(span_profile.get("minimum_span_words", 24)),
            chunk_records=int(span_profile.get("external_sort_chunk_records", 250_000)),
            maximum_open_runs=int(span_profile.get("external_sort_max_open_runs", 64)),
            temporary_directory=temp_root / "repeated-span-prefilter-finders",
        )
        state.write(
            "repeated-span",
            "prefilter-finder-reports",
            f"bucket-{task_index:04d}.json",
            payload=report,
        )
    elif stage == "span_signature":
        from .span_dedup import write_span_signatures

        exact_reader = JsonlReader(str(exact_output), shuffle_files=False)
        report = write_span_signatures(
            exact_reader.run(rank=task_index, world_size=total_tasks),
            span_sig,
            rank=task_index,
            finder_workers=span_finders,
            sentence_count=int(span_profile.get("sentence_count", 3)),
            minimum_span_words=int(span_profile.get("minimum_span_words", 24)),
            maximum_open_files=int(span_profile.get("maximum_open_files", 32)),
            candidate_root=span_candidates,
            total_ranks=total_tasks,
        )
        state.write("repeated-span", "signature-reports", f"task-{task_index:06d}.json", payload=report)
    elif stage == "span_find":
        from .span_dedup import find_span_duplicates

        temp_root = Path(profile.get("runtime", {}).get("temp_dir", "cache/tmp"))
        if not temp_root.is_absolute():
            temp_root = root / temp_root
        report = find_span_duplicates(
            span_sig,
            span_remove,
            bucket=task_index,
            finder_workers=span_finders,
            total_ranks=total_tasks,
            sentence_count=int(span_profile.get("sentence_count", 3)),
            minimum_span_words=int(span_profile.get("minimum_span_words", 24)),
            chunk_records=int(span_profile.get("external_sort_chunk_records", 250_000)),
            maximum_open_runs=int(span_profile.get("external_sort_max_open_runs", 64)),
            temporary_directory=temp_root / "repeated-span-finders",
        )
        state.write("repeated-span", "finder-reports", f"bucket-{task_index:04d}.json", payload=report)
    elif stage == "span_filter":
        from .span_dedup import build_span_dedup_filter

        exact_reader = JsonlReader(str(exact_output), shuffle_files=False)
        quarantine = JsonlWriter(str(dedup / "repeated-span" / "quarantine"))
        _local_executor(
            profile,
            stage,
            task_index,
            total_tasks,
            [
                exact_reader,
                build_span_dedup_filter(
                    span_remove,
                    finder_workers=span_finders,
                    sentence_count=int(span_profile.get("sentence_count", 3)),
                    minimum_span_words=int(span_profile.get("minimum_span_words", 24)),
                    minimum_remaining_words=int(span_profile.get("minimum_remaining_words", 50)),
                    minimum_remaining_sentences=int(
                        span_profile.get("minimum_remaining_sentences", 3)
                    ),
                    quarantine_writer=quarantine,
                ),
                JsonlWriter(str(span_output)),
            ],
        )
    elif stage == "minhash_signature":
        span_reader = JsonlReader(str(span_output), shuffle_files=False)
        _local_executor(
            profile,
            stage,
            task_index,
            total_tasks,
            [span_reader, MinhashDedupSignature(str(mh_sig), config=mh_config, language=build_regex_word_tokenizer())],
        )
    elif stage == "minhash_buckets":
        _local_executor(profile, stage, task_index, mh_config.num_buckets, [MinhashDedupBuckets(str(mh_sig), str(mh_buckets), config=mh_config)])
        from .datatrove_blocks import write_minhash_bucket_output_manifest

        report = write_minhash_bucket_output_manifest(
            mh_buckets,
            mh_priority_work / "bucket-inventory",
            bucket=task_index,
            expected_buckets=mh_config.num_buckets,
        )
        state.write(
            "minhash-priority",
            "bucket-reports",
            f"bucket-{task_index:06d}.json",
            payload=report,
        )
    elif stage == "minhash_components":
        from .datatrove_blocks import cluster_priority_minhash_pairs

        priority_profile = profile["scheduler"].get("minhash_priority", {})
        cluster_report = cluster_priority_minhash_pairs(
            mh_buckets,
            mh_priority_work,
            total_tasks=total_tasks,
            bucket_count=int(priority_profile.get("bucket_count", 256)),
            sqlite_cache_mb=int(priority_profile.get("sqlite_cache_mb", 256)),
            transaction_rows=int(priority_profile.get("transaction_rows", 100_000)),
            bucket_inventory_folder=mh_priority_work / "bucket-inventory",
            expected_duplicate_buckets=mh_config.num_buckets,
            temporary_directory=_stage_temporary_directory(
                profile, root, "minhash-components"
            ),
        )
        state.write("minhash-components-report.json", payload=cluster_report)
    elif stage == "minhash_priority_candidates":
        from .datatrove_blocks import write_priority_minhash_rank_candidates

        priority_profile = profile["scheduler"].get("minhash_priority", {})
        report = write_priority_minhash_rank_candidates(
            span_output,
            mh_priority_work,
            rank=task_index,
            total_tasks=total_tasks,
            max_open_files=int(priority_profile.get("maximum_open_files", 32)),
        )
        state.write(
            "minhash-priority",
            "candidate-reports",
            f"rank-{task_index:06d}.json",
            payload=report,
        )
    elif stage == "minhash_priority_resolve":
        from .datatrove_blocks import resolve_priority_minhash_bucket

        priority_profile = profile["scheduler"].get("minhash_priority", {})
        report = resolve_priority_minhash_bucket(
            mh_priority_work,
            bucket=task_index,
            total_tasks=total_tasks,
            sqlite_cache_mb=int(priority_profile.get("sqlite_cache_mb", 256)),
            transaction_rows=int(priority_profile.get("transaction_rows", 100_000)),
            temporary_directory=_stage_temporary_directory(
                profile, root, "minhash-priority-resolve"
            ),
        )
        state.write(
            "minhash-priority",
            "resolver-reports",
            f"bucket-{task_index:06d}.json",
            payload=report,
        )
    elif stage == "minhash_priority_finalize":
        from .datatrove_blocks import finalize_priority_minhash_rank_removals

        priority_profile = profile["scheduler"].get("minhash_priority", {})
        report = finalize_priority_minhash_rank_removals(
            mh_priority_work,
            mh_remove,
            rank=task_index,
            total_tasks=total_tasks,
            sqlite_cache_mb=int(priority_profile.get("sqlite_cache_mb", 256)),
            transaction_rows=int(priority_profile.get("transaction_rows", 100_000)),
            temporary_directory=_stage_temporary_directory(
                profile, root, "minhash-priority-finalize"
            ),
        )
        state.write(
            "minhash-priority",
            "finalizer-reports",
            f"rank-{task_index:06d}.json",
            payload=report,
        )
    elif stage == "minhash_priority_verify":
        from .datatrove_blocks import verify_priority_minhash_completion

        report = verify_priority_minhash_completion(
            mh_priority_work,
            mh_remove,
            total_tasks=total_tasks,
        )
        state.write("minhash-priority", "COMPLETE.json", payload=report)
    elif stage == "minhash_filter":
        from .datatrove_blocks import require_verified_priority_minhash_rank

        require_verified_priority_minhash_rank(
            mh_priority_work,
            mh_remove,
            rank=task_index,
            total_tasks=total_tasks,
        )
        span_reader = JsonlReader(str(span_output), shuffle_files=False)
        _local_executor(profile, stage, task_index, total_tasks, [span_reader, MinhashDedupFilter(str(mh_remove)), JsonlWriter(str(mh_output))])
    elif stage == "code_signature":
        from .code_dedup import write_code_signatures

        mh_reader = JsonlReader(str(mh_output), shuffle_files=False)
        report = write_code_signatures(
            mh_reader.run(rank=task_index, world_size=total_tasks),
            code_sig,
            rank=task_index,
            finder_workers=code_finders,
            block_tokens=int(code_profile.get("block_tokens", 96)),
        )
        state.write("code-structural", "signature-reports", f"task-{task_index:06d}.json", payload=report)
    elif stage == "code_find":
        from .code_dedup import find_code_duplicates

        temp_root = Path(profile.get("runtime", {}).get("temp_dir", "cache/tmp"))
        if not temp_root.is_absolute():
            temp_root = root / temp_root
        report = find_code_duplicates(
            code_sig,
            code_remove,
            bucket=task_index,
            finder_workers=code_finders,
            expected_ranks=total_tasks,
            temporary_directory=temp_root / "code-structural-finders",
        )
        state.write("code-structural", "finder-reports", f"bucket-{task_index:04d}.json", payload=report)
    elif stage == "code_filter":
        from .code_dedup import build_code_structural_filter

        mh_reader = JsonlReader(str(mh_output), shuffle_files=False)
        quarantine = JsonlWriter(str(dedup / "code-structural" / "quarantine"))
        _local_executor(
            profile,
            stage,
            task_index,
            total_tasks,
            [
                mh_reader,
                build_code_structural_filter(
                    code_remove,
                    finder_workers=code_finders,
                    duplicate_fraction=float(code_profile.get("duplicate_fraction", 0.80)),
                    block_tokens=int(code_profile.get("block_tokens", 96)),
                    exclusion_writer=quarantine,
                ),
                JsonlWriter(str(code_output)),
            ],
        )
    elif stage == "decontam_index":
        holdouts = contamination / "holdouts.jsonl"
        if not holdouts.exists():
            raise RuntimeError(f"Fail-closed: benchmark holdout bundle is missing at {holdouts}")
        from .datatrove_blocks import save_contamination_index
        from .decontaminate import ContaminationIndex

        policy = load_yaml(repository_root() / "manifests" / "contamination" / "eval-holdouts.yaml")["policy"]
        index = ContaminationIndex.build(
            _iter_rows(holdouts),
            ngram_size=int(policy["ngram_size"]),
            minimum_matching_ngrams=int(policy["minimum_matching_ngrams"]),
            short_ngram_size=int(policy["short_ngram_size"]),
            minimum_short_matching_ngrams=int(policy["minimum_short_matching_ngrams"]),
            code_ngram_size=int(policy["code_ngram_size"]),
            minimum_code_matching_ngrams=int(policy["minimum_code_matching_ngrams"]),
            code_skeleton_ngram_size=int(policy["code_skeleton_ngram_size"]),
            minimum_code_skeleton_matching_ngrams=int(
                policy["minimum_code_skeleton_matching_ngrams"]
            ),
            maximum_shingle_rows=int(policy["maximum_shingle_rows"]),
        )
        save_contamination_index(index, contamination / "index.json")
    elif stage == "decontam_filter":
        from .datatrove_blocks import build_datatrove_decontamination_filter

        index_path = contamination / "index.json"
        if not index_path.exists():
            raise RuntimeError(f"Fail-closed: decontamination index is missing at {index_path}")
        code_reader = JsonlReader(str(code_output), shuffle_files=False)
        quarantine = JsonlWriter(str(contamination / "quarantine"))
        benchmark_registry = load_yaml(
            repository_root() / "manifests" / "contamination" / "eval-holdouts.yaml"
        )
        _local_executor(
            profile,
            stage,
            task_index,
            total_tasks,
            [
                code_reader,
                build_datatrove_decontamination_filter(
                    index_path,
                    exclusion_writer=quarantine,
                    benchmark_registry=benchmark_registry,
                ),
                JsonlWriter(str(eligible / "decontaminated")),
            ],
        )
    elif stage == "final_hash_signature":
        from .final_dedup import write_final_signatures

        decontaminated = JsonlReader(str(eligible / "decontaminated"), shuffle_files=False)
        report = write_final_signatures(
            decontaminated.run(rank=task_index, world_size=total_tasks),
            final_sig,
            rank=task_index,
            finder_workers=final_finders,
        )
        state.write("final-sha256", "signature-reports", f"task-{task_index:06d}.json", payload=report)
    elif stage == "final_hash_find":
        from .final_dedup import find_final_duplicates

        temp_root = Path(profile.get("runtime", {}).get("temp_dir", "cache/tmp"))
        if not temp_root.is_absolute():
            temp_root = root / temp_root
        report = find_final_duplicates(
            final_sig,
            final_remove,
            bucket=task_index,
            finder_workers=final_finders,
            expected_ranks=total_tasks,
            temporary_directory=temp_root / "final-sha256-finders",
        )
        state.write("final-sha256", "finder-reports", f"bucket-{task_index:04d}.json", payload=report)
    elif stage == "final_hash_filter":
        from .final_dedup import build_final_hash_filter

        decontaminated = JsonlReader(str(eligible / "decontaminated"), shuffle_files=False)
        quarantine = JsonlWriter(str(dedup / "final-sha256" / "quarantine"))
        _local_executor(
            profile,
            stage,
            task_index,
            total_tasks,
            [
                decontaminated,
                build_final_hash_filter(
                    final_remove,
                    finder_workers=final_finders,
                    exclusion_writer=quarantine,
                ),
                JsonlWriter(str(final_output)),
            ],
        )
    else:
        raise ValueError(f"Unsupported DataTrove stage {stage}")
    payload = {
        "stage": stage,
        "task_index": task_index,
        "execution_contract_sha256": _stage_execution_contract(
            profile, state, stage
        ),
        "completed_at": utc_now(),
    }
    state.complete(stage, f"task-{task_index:06d}", payload)
    return payload


def _iter_jsonl_folder(folder: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(folder.glob("**/*.jsonl*")):
        yield from _iter_rows(path)


def _tokenizer_sample_targets(manifest: dict[str, Any]) -> tuple[dict[str, int], dict[str, int], int]:
    """Stratified per-category and per-source byte targets for the sample.

    This is the immutable sampling contract: identical whether the sample is
    produced by one process or by thousands of parallel shards.
    """

    target = int(manifest["tokenizer"]["sample_target_bytes"])
    minimum_category = int(manifest["tokenizer"]["min_sample_bytes_per_category"])
    category_weights = {
        category["id"]: sum(int(value) for value in category["phase_tokens"].values())
        for category in manifest["categories"]
    }
    base = minimum_category * len(category_weights)
    if base > target:
        raise RuntimeError("Tokenizer category floors exceed the total sample target")
    category_extra = hamilton_apportion(target - base, category_weights)
    category_targets = {
        category: minimum_category + category_extra[category] for category in category_weights
    }
    source_targets: dict[str, int] = {}
    for category, category_target in category_targets.items():
        sources = [source for source in manifest["sources"] if source["category"] == category]
        weights = {
            source["id"]: sum(int(value) for value in source["phase_tokens"].values())
            for source in sources
        }
        source_targets.update(hamilton_apportion(category_target, weights))
    return category_targets, source_targets, target


def _tokenizer_sample_paths(profile: dict[str, Any], task_index: int, total_tasks: int) -> tuple[Path, list[Path]]:
    eligible = (
        Path(profile["storage"]["lustre_root"])
        / profile["storage"]["directories"]["eligible"]
        / "final"
    )
    paths = sorted(
        path
        for path in eligible.glob("**/*.jsonl*")
        if path.is_file()
        and not path.name.endswith(".incomplete")
        and ".incomplete" not in path.parts
    )
    return eligible, paths[task_index::total_tasks]


def _tokenizer_sample_scan(profile: dict[str, Any], task_index: int) -> dict[str, Any]:
    """Count available sample bytes per source over this shard's files.

    A counting pass with no output lets the planner apportion exact per-shard
    quotas, so the parallel sample reproduces the serial per-source totals
    instead of relying on an oversampling heuristic.
    """

    root, state = _paths(profile)
    manifest = _manifest(profile)
    _category_targets, source_targets, _target = _tokenizer_sample_targets(manifest)
    total_tasks = build_input_count(state)
    eligible, assigned = _tokenizer_sample_paths(profile, task_index, total_tasks)
    output_dir = root / profile["storage"]["directories"]["tokenizer"] / "scan"
    output_dir.mkdir(parents=True, exist_ok=True)
    assigned_inputs = [_file_inventory(path, relative_to=eligible) for path in assigned]
    assigned_inventory_sha = _json_sha256(assigned_inputs)
    available: dict[str, int] = {}
    documents: dict[str, int] = {}
    for path in assigned:
        for row in _iter_rows(path):
            metadata = row.get("metadata", {})
            source_id = metadata.get("source_id") or row.get("source_id")
            if source_id not in source_targets:
                continue
            text = str(row.get("text", ""))
            if not text:
                continue
            source_id = str(source_id)
            available[source_id] = available.get(source_id, 0) + len(text.encode("utf-8"))
            documents[source_id] = documents.get(source_id, 0) + 1
    payload = {
        "schema": "metis.tokenizer-sample-scan/v1",
        "stage": "tokenizer_sample_scan",
        "task_index": task_index,
        "world_size": total_tasks,
        "assigned_input_inventory_sha256": assigned_inventory_sha,
        "available_bytes": available,
        "available_documents": documents,
    }
    atomic_json(output_dir / f"task-{task_index:06d}.json", payload)
    state.complete("tokenizer_sample_scan", f"task-{task_index:06d}", payload)
    return payload


def _tokenizer_sample_plan(profile: dict[str, Any]) -> dict[str, Any]:
    """Apportion each source's byte target across the shards that hold it."""

    root, state = _paths(profile)
    manifest = _manifest(profile)
    category_targets, source_targets, target = _tokenizer_sample_targets(manifest)
    total_tasks = build_input_count(state)
    scan_dir = root / profile["storage"]["directories"]["tokenizer"] / "scan"
    availability: dict[str, dict[str, int]] = {}
    for task_index in range(total_tasks):
        report = state.read("completed", "tokenizer_sample_scan", f"task-{task_index:06d}.json")
        if not report:
            raise RuntimeError(f"Tokenizer sample scan is incomplete: task-{task_index:06d}")
        if int(report.get("world_size", -1)) != total_tasks:
            raise RuntimeError(
                f"Tokenizer sample scan task-{task_index:06d} was produced against a different partitioning"
            )
        for source_id, byte_count in report.get("available_bytes", {}).items():
            availability.setdefault(str(source_id), {})[str(task_index)] = int(byte_count)

    quotas: dict[str, dict[str, int]] = {}
    short: dict[str, int] = {}
    for source_id in sorted(source_targets):
        wanted = int(source_targets[source_id])
        holders = availability.get(source_id, {})
        total_available = sum(holders.values())
        if total_available < wanted:
            short[source_id] = wanted - total_available
            continue
        # Fill the largest holders first rather than spreading the target
        # proportionally. A shard emits whole documents, so any shard given a
        # partial quota overshoots it by up to one document. Consuming holders
        # completely means only the final shard of each source is partial, which
        # reproduces the serial sampler's single bounded overshoot per source
        # instead of one per shard. Ties break on task index, so the plan is
        # deterministic for a given scan.
        remaining = wanted
        ordered = sorted(
            (task for task in holders if holders[task] > 0),
            key=lambda task: (-holders[task], int(task)),
        )
        for task in ordered:
            if remaining <= 0:
                break
            take = min(holders[task], remaining)
            quotas.setdefault(str(task), {})[source_id] = int(take)
            remaining -= take
        if remaining:
            raise RuntimeError(
                f"Tokenizer sample apportionment could not place {remaining:,} bytes for {source_id}"
            )
    if short:
        raise RuntimeError(
            f"Tokenizer sample exhausted before its stratified source targets: {short}"
        )

    plan = {
        "schema": "metis.tokenizer-sample-plan/v1",
        "created_at": utc_now(),
        "world_size": total_tasks,
        "target_bytes": target,
        "category_targets": category_targets,
        "source_targets": source_targets,
        "task_quotas": quotas,
    }
    plan["plan_sha256"] = _json_sha256({k: v for k, v in plan.items() if k != "created_at"})
    atomic_json(scan_dir.parent / "SAMPLE_PLAN.json", plan)
    payload = {
        "stage": "tokenizer_sample_plan",
        "world_size": total_tasks,
        "target_bytes": target,
        "shards_with_quota": len(quotas),
        "plan_sha256": plan["plan_sha256"],
    }
    state.complete("tokenizer_sample_plan", "task-000000", payload)
    return payload


def _tokenizer_sample(profile: dict[str, Any], task_index: int) -> dict[str, Any]:
    """Write this shard's exact planned slice of the tokenizer sample."""

    import zstandard as zstd

    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    tokenizer_dir = root / directories["tokenizer"]
    plan_path = tokenizer_dir / "SAMPLE_PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in plan.items() if key not in {"created_at", "plan_sha256"}}
    if plan.get("plan_sha256") != _json_sha256(unsigned):
        raise RuntimeError("SAMPLE_PLAN.json failed its self-hash check")
    total_tasks = build_input_count(state)
    if int(plan.get("world_size", -1)) != total_tasks:
        raise RuntimeError("SAMPLE_PLAN.json was produced against a different partitioning")
    quota = {str(k): int(v) for k, v in plan.get("task_quotas", {}).get(str(task_index), {}).items()}

    parts_dir = tokenizer_dir / "sample-parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    output = parts_dir / f"task-{task_index:06d}.jsonl.zst"
    temporary_output = output.with_suffix(output.suffix + ".incomplete")
    if not quota:
        # A shard with no planned bytes still publishes an empty part so the
        # trainer's input set is a complete, gap-free sequence.
        temporary_output.unlink(missing_ok=True)
        with temporary_output.open("wb") as raw:
            with zstd.ZstdCompressor(level=6).stream_writer(raw):
                pass
        os.replace(temporary_output, output)
        payload = {
            "stage": "tokenizer_sample",
            "task_index": task_index,
            "bytes": 0,
            "source_bytes": {},
            "output": str(output),
        }
        state.complete("tokenizer_sample", f"task-{task_index:06d}", payload)
        return payload

    _eligible, assigned = _tokenizer_sample_paths(profile, task_index, total_tasks)
    written: dict[str, int] = {source_id: 0 for source_id in quota}
    total = 0
    temporary_output.unlink(missing_ok=True)
    with temporary_output.open("wb") as raw:
        with zstd.ZstdCompressor(level=6).stream_writer(raw) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for path in assigned:
                    if all(written[source] >= quota[source] for source in quota):
                        break
                    for row in _iter_rows(path):
                        metadata = row.get("metadata", {})
                        source_id = metadata.get("source_id") or row.get("source_id")
                        source_id = str(source_id) if source_id is not None else ""
                        if source_id not in quota or written[source_id] >= quota[source_id]:
                            continue
                        text = str(row.get("text", ""))
                        if not text:
                            continue
                        handle.write(
                            json.dumps(
                                {
                                    "source_id": source_id,
                                    "category": metadata.get("category"),
                                    "text": text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        size = len(text.encode("utf-8"))
                        written[source_id] += size
                        total += size
    short = {
        source_id: quota[source_id] - written[source_id]
        for source_id in quota
        if written[source_id] < quota[source_id]
    }
    if short:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Tokenizer sample shard {task_index} fell short of its planned quota: {short}"
        )
    os.replace(temporary_output, output)
    payload = {
        "stage": "tokenizer_sample",
        "task_index": task_index,
        "bytes": total,
        "source_bytes": written,
        "plan_sha256": plan["plan_sha256"],
        "output": str(output),
    }
    state.complete("tokenizer_sample", f"task-{task_index:06d}", payload)
    return payload


def _tokenizer_sample_parts(profile: dict[str, Any], state: StateStore) -> list[Path]:
    """Every planned sample shard, in deterministic order, with no gaps.

    The trainer consumes the shards directly, so the parallel sample never has
    to be concatenated back into one multi-hundred-GB file.
    """

    root = Path(profile["storage"]["lustre_root"])
    parts_dir = root / profile["storage"]["directories"]["tokenizer"] / "sample-parts"
    total_tasks = build_input_count(state)
    parts: list[Path] = []
    for task_index in range(total_tasks):
        part = parts_dir / f"task-{task_index:06d}.jsonl.zst"
        if not part.is_file():
            raise RuntimeError(f"Tokenizer sample shard is missing: {part}")
        parts.append(part)
    return parts


def _tokenizer_train(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    manifest = _manifest(profile)
    output_dir = root / profile["storage"]["directories"]["tokenizer"]
    parts = _tokenizer_sample_parts(profile, state)

    def texts() -> Iterator[str]:
        for part in parts:
            for row in _iter_rows(part):
                yield str(row["text"])

    payload = train_tokenizer(
        texts(),
        output_dir=output_dir,
        vocabulary_size=int(manifest["tokenizer"]["vocabulary_size_including_special_tokens"]),
        special_tokens=list(manifest["tokenizer"]["special_tokens"]),
    )
    audit_limits: dict[str, int] = {}

    def audit_samples() -> Iterator[dict[str, Any]]:
        for part in parts:
            for row in _iter_rows(part):
                category = str(row.get("category", "unknown"))
                if audit_limits.get(category, 0) >= 500:
                    continue
                audit_limits[category] = audit_limits.get(category, 0) + 1
                yield row

    validation = validate_tokenizer(output_dir / "tokenizer.json", audit_samples())
    validation["sampled_documents_by_category"] = audit_limits
    validation_path = output_dir / "TOKENIZER_VALIDATION.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not validation["ok"]:
        raise RuntimeError(
            f"Tokenizer round-trip validation failed for {validation['roundtrip_failure_count']} sampled documents"
        )
    payload["validation"] = validation
    payload["validation_path"] = str(validation_path)
    state.complete("tokenizer_train", "task-000000", payload)
    return payload


def _token_count(profile: dict[str, Any], task_index: int) -> dict[str, Any]:
    import zstandard as zstd

    root, state = _paths(profile)
    manifest = _manifest(profile)
    directories = profile["storage"]["directories"]
    source_dir = root / directories["eligible"] / "final"
    output_dir = root / directories["token_counts"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"task-{task_index:06d}.jsonl.zst"
    temporary_output = output.with_suffix(output.suffix + ".incomplete")
    report_path = output_dir / f"task-{task_index:06d}.report.json"
    total_tasks = build_input_count(state)
    paths = sorted(
        path
        for path in source_dir.glob("**/*.jsonl*")
        if path.is_file()
        and not path.name.endswith(".incomplete")
        and ".incomplete" not in path.parts
    )
    assigned = paths[task_index::total_tasks]
    assigned_inputs = [
        _file_inventory(path, relative_to=source_dir)
        for path in assigned
    ]
    assigned_inventory_sha = _json_sha256(assigned_inputs)
    tokenizer_contract = _production_tokenizer_contract(profile, manifest)
    if report_path.exists() and output.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != "metis.token-count-task/v2"
            or int(payload.get("task_index", -1)) != task_index
            or int(payload.get("world_size", -1)) != total_tasks
            or payload.get("assigned_inputs") != assigned_inputs
            or payload.get("assigned_input_inventory_sha256") != assigned_inventory_sha
            or payload.get("tokenizer_contract") != tokenizer_contract
        ):
            raise RuntimeError(
                f"Token-count task {task_index} no longer matches its immutable inputs/tokenizer"
            )
        _require_inventory_file(
            output_dir,
            dict(payload.get("output_artifact") or {}),
            f"token-count output {task_index}",
        )
        return payload
    if report_path.exists() != output.exists():
        # Both are deterministic derivatives of the immutable final-corpus
        # inventory. A crash between their atomic publications is therefore
        # safely recoverable by discarding the orphan and rebuilding.
        report_path.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
    tokenizer_path = root / directories["tokenizer"] / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    source_tokens: dict[str, int] = {}
    documents = 0
    temporary_output.unlink(missing_ok=True)
    try:
        with temporary_output.open("wb") as raw:
            with zstd.ZstdCompressor(level=6).stream_writer(raw) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                    for path in assigned:
                        for row in _iter_rows(path):
                            metadata = row.get("metadata", {})
                            text = str(row.get("text", ""))
                            source_id = str(metadata.get("source_id", row.get("source_id", "")))
                            if not source_id or not text:
                                continue
                            token_count = len(
                                tokenizer.encode(text, add_special_tokens=False).ids
                            ) + 1
                            doc_id = str(
                                row.get(
                                    "id",
                                    metadata.get("doc_id", f"{task_index}:{documents}"),
                                )
                            )
                            content_sha = str(metadata.get("final_content_sha256", ""))
                            recomputed_sha = content_sha256(text).hex()
                            if len(content_sha) != 64 or content_sha != recomputed_sha:
                                raise RuntimeError(
                                    "Final SHA-256 audit evidence does not match text for "
                                    f"{source_id}:{doc_id}"
                                )
                            record = {
                                "source_id": source_id,
                                "category": metadata.get("category"),
                                "doc_id": doc_id,
                                "text": text,
                                "token_count": token_count,
                                "content_sha256": content_sha,
                                "text_sha256": hashlib.sha256(
                                    text.encode("utf-8")
                                ).hexdigest(),
                                "generated": bool(metadata.get("generated", False)),
                                "transformed": bool(metadata.get("transformed", False)),
                                "priority": int(metadata.get("priority", 1)),
                                "license": metadata.get("license"),
                                "license_status": metadata.get("license_status"),
                                "context_group_id": context_group_id(
                                    source_id, doc_id, metadata
                                ),
                                "context_structure": structural_evidence(
                                    text, metadata
                                ),
                            }
                            handle.write(
                                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                            )
                            source_tokens[source_id] = (
                                source_tokens.get(source_id, 0) + token_count
                            )
                            documents += 1
        temporary_output.replace(output)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise
    payload = {
        "schema": "metis.token-count-task/v2",
        "stage": "token_count",
        "task_index": task_index,
        "world_size": total_tasks,
        "documents": documents,
        "tokens": sum(source_tokens.values()),
        "source_tokens": source_tokens,
        "assigned_inputs": assigned_inputs,
        "assigned_input_inventory_sha256": assigned_inventory_sha,
        "tokenizer_contract": tokenizer_contract,
        "output_artifact": _file_inventory(output, relative_to=output_dir),
        "completed_at": utc_now(),
    }
    atomic_json(report_path, payload)
    state.complete("token_count", f"task-{task_index:06d}", payload)
    return payload


def _token_count_contract(
    profile: dict[str, Any],
    state: StateStore,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(profile["storage"]["lustre_root"])
    directories = profile["storage"]["directories"]
    final_root = root / directories["eligible"] / "final"
    token_root = root / directories["token_counts"]
    expected_tasks = build_input_count(state)
    expected_names = {
        f"task-{index:06d}.report.json" for index in range(expected_tasks)
    }
    report_paths = sorted(token_root.glob("task-*.report.json"))
    actual_names = {path.name for path in report_paths}
    if actual_names != expected_names:
        raise RuntimeError(
            "Token-count report inventory is incomplete: "
            f"missing={sorted(expected_names - actual_names)[:8]}, "
            f"unexpected={sorted(actual_names - expected_names)[:8]}"
        )
    final_paths = sorted(
        path
        for path in final_root.glob("**/*.jsonl*")
        if path.is_file()
        and not path.name.endswith(".incomplete")
        and ".incomplete" not in path.parts
    )
    expected_inventory = [
        _file_inventory(path, relative_to=final_root) for path in final_paths
    ]
    tokenizer_contract = _production_tokenizer_contract(profile)
    task_contracts: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    observed_inputs: list[dict[str, Any]] = []
    for task_index, report_path in enumerate(report_paths):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_inputs = expected_inventory[task_index::expected_tasks]
        expected_input_sha = _json_sha256(expected_inputs)
        if (
            report.get("schema") != "metis.token-count-task/v2"
            or int(report.get("task_index", -1)) != task_index
            or int(report.get("world_size", -1)) != expected_tasks
            or report.get("assigned_inputs") != expected_inputs
            or report.get("assigned_input_inventory_sha256") != expected_input_sha
            or report.get("tokenizer_contract") != tokenizer_contract
        ):
            raise RuntimeError(
                f"Token-count report {task_index} is stale or not bound to the final corpus"
            )
        output_record = dict(report.get("output_artifact") or {})
        output_path = _require_inventory_file(
            token_root,
            output_record,
            f"token-count output {task_index}",
        )
        expected_output = token_root / f"task-{task_index:06d}.jsonl.zst"
        if output_path != expected_output.resolve():
            raise RuntimeError(
                f"Token-count task {task_index} points at an unexpected output: {output_path}"
            )
        reports.append(report)
        observed_inputs.extend(expected_inputs)
        task_contracts.append(
            {
                "task_index": task_index,
                "report": _file_inventory(report_path, relative_to=token_root),
                "output": output_record,
                "assigned_input_inventory_sha256": expected_input_sha,
                "documents": int(report.get("documents", -1)),
                "tokens": int(report.get("tokens", -1)),
                "source_tokens": {
                    str(key): int(value)
                    for key, value in report.get("source_tokens", {}).items()
                },
            }
        )
    if sorted(observed_inputs, key=lambda item: item["path"]) != expected_inventory:
        raise RuntimeError("Token-count tasks do not cover the exact final-corpus inventory")
    payload: dict[str, Any] = {
        "schema": "metis.token-count-set/v1",
        "created_at": utc_now(),
        "tasks": task_contracts,
        "task_count": expected_tasks,
        "final_input_inventory": expected_inventory,
        "final_input_inventory_sha256": _json_sha256(expected_inventory),
        "tokenizer_contract": tokenizer_contract,
        "documents": sum(int(report["documents"]) for report in reports),
        "tokens": sum(int(report["tokens"]) for report in reports),
    }
    payload["contract_sha256"] = _json_sha256(payload)
    return payload, reports


class _RestartableTokenCountRows:
    def __init__(self, token_root: Path, tasks: list[dict[str, Any]]) -> None:
        self.token_root = token_root
        self.tasks = tasks

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for task in self.tasks:
            yield from _iter_rows(self.token_root / task["output"]["path"])


def _context_output_root(profile: dict[str, Any]) -> Path:
    root = Path(profile["storage"]["lustre_root"])
    directories = profile["storage"]["directories"]
    relative = str(
        directories.get(
            "context_release",
            "releases/metis-1.6-context-extension-r1",
        )
    )
    output_root = (root / relative).resolve()
    try:
        output_root.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("context-extension release escapes the Lustre root") from exc
    return output_root


def _context_plan(profile: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    manifest = _manifest(profile)
    plan = manifest.get("context_extension_plan")
    if not isinstance(plan, dict):
        raise RuntimeError("manifest has no validated context-extension plan")
    path = Path(str(plan.get("_path") or "")).resolve()
    if not path.is_file():
        raise RuntimeError("context-extension plan file is missing")
    return plan, path


def _load_or_create_context_pack_plan(
    output_root: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    path = output_root / "PACK_PLAN.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        unsigned = {
            key: value for key, value in payload.items() if key != "plan_sha256"
        }
        if (
            payload.get("schema") != CONTEXT_PACK_PLAN_SCHEMA
            or payload.get("release") != plan["release"]
            or payload.get("plan_sha256") != _json_sha256(unsigned)
            or int(payload.get("pack_tasks", -1)) != 96
        ):
            raise RuntimeError("persisted context PACK_PLAN.json is stale or corrupt")
        return payload
    output_root.mkdir(parents=True, exist_ok=True)
    payload = build_context_pack_plan(plan)
    atomic_json(path, payload)
    return payload


def _context_token_count_contract(
    profile: dict[str, Any],
    state: StateStore,
    output_root: Path,
) -> tuple[dict[str, Any], str]:
    path = output_root / "TOKEN_COUNT_CONTRACT.json"
    rebuilt, _reports = _token_count_contract(profile, state)
    if path.exists():
        persisted = json.loads(path.read_text(encoding="utf-8"))
        unsigned = {
            key: value for key, value in persisted.items() if key != "contract_sha256"
        }
        if (
            persisted.get("schema") != "metis.token-count-set/v1"
            or persisted.get("contract_sha256") != _json_sha256(unsigned)
        ):
            raise RuntimeError("context token-count contract is corrupt")
        rebuilt["created_at"] = persisted.get("created_at")
        rebuilt["contract_sha256"] = _json_sha256(
            {
                key: value
                for key, value in rebuilt.items()
                if key != "contract_sha256"
            }
        )
        if rebuilt != persisted:
            raise RuntimeError(
                "context token-count inputs changed after selection began"
            )
        contract = persisted
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(path, rebuilt)
        contract = rebuilt
    return contract, sha256_file(path)


def _context_select(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    output_root = _context_output_root(profile)
    plan, _plan_path = _context_plan(profile)
    pack_plan = _load_or_create_context_pack_plan(output_root, plan)
    token_contract, token_contract_file_sha = _context_token_count_contract(
        profile, state, output_root
    )
    tokenizer_contract = dict(token_contract["tokenizer_contract"])
    selection_path = output_root / "SELECTION.json"
    if selection_path.exists():
        payload = validate_context_selection(
            output_root,
            plan=plan,
            pack_plan=pack_plan,
            token_count_contract_sha256=token_contract_file_sha,
            tokenizer_contract=tokenizer_contract,
        )
        state.complete("context_select", "task-000000", payload)
        return payload
    token_root = root / directories["token_counts"]
    records = _RestartableTokenCountRows(
        token_root, [dict(row) for row in token_contract["tasks"]]
    )
    payload = build_context_selection(
        records,
        plan=plan,
        pack_plan=pack_plan,
        output_root=output_root,
        token_count_contract_sha256=token_contract_file_sha,
        tokenizer_contract=tokenizer_contract,
    )
    state.complete("context_select", "task-000000", payload)
    return payload


def _context_prepare(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    output_root = _context_output_root(profile)
    plan, _plan_path = _context_plan(profile)
    pack_plan = _load_or_create_context_pack_plan(output_root, plan)
    token_contract, token_contract_file_sha = _context_token_count_contract(
        profile, state, output_root
    )
    validate_context_selection(
        output_root,
        plan=plan,
        pack_plan=pack_plan,
        token_count_contract_sha256=token_contract_file_sha,
        tokenizer_contract=token_contract["tokenizer_contract"],
    )
    payload = initialize_context_arrays(
        output_root,
        pack_plan=pack_plan,
        plan=plan,
    )
    tokenizer_contract = dict(token_contract["tokenizer_contract"])
    evaluation = pack_context_evaluation(
        output_root,
        plan=plan,
        tokenizer_path=(
            root / directories["tokenizer"] / "tokenizer.json"
        ),
        tokenizer_sha256=str(tokenizer_contract["tokenizer_sha256"]),
        eos_id=int(tokenizer_contract["eos_token_id"]),
    )
    payload["gate_evaluation_receipt_sha256"] = evaluation[
        "receipt_sha256"
    ]
    payload["initialization_sha256"] = _json_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "initialization_sha256"
        }
    )
    atomic_json(output_root / "ARRAYS_INITIALIZED.json", payload)
    state.complete("context_prepare", "task-000000", payload)
    return payload


def _context_pack(profile: dict[str, Any], task_index: int) -> dict[str, Any]:
    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    output_root = _context_output_root(profile)
    plan, _plan_path = _context_plan(profile)
    pack_plan = _load_or_create_context_pack_plan(output_root, plan)
    token_contract, token_contract_file_sha = _context_token_count_contract(
        profile, state, output_root
    )
    tokenizer_contract = dict(token_contract["tokenizer_contract"])
    validate_context_selection(
        output_root,
        plan=plan,
        pack_plan=pack_plan,
        token_count_contract_sha256=token_contract_file_sha,
        tokenizer_contract=tokenizer_contract,
    )
    initialization_path = output_root / "ARRAYS_INITIALIZED.json"
    if not initialization_path.is_file():
        raise RuntimeError("context arrays were not initialized before packing")
    task_id = f"task-{task_index:06d}"
    receipt_path = output_root / "pack-receipts" / f"{task_id}.json"
    if receipt_path.exists():
        payload = validate_context_pack_receipt(
            output_root,
            pack_plan=pack_plan,
            tokenizer_sha256=str(tokenizer_contract["tokenizer_sha256"]),
            task_index=task_index,
            deep_array_validation=True,
        )
        state.complete("context_pack", task_id, payload)
        return payload
    tokenizer_path = root / directories["tokenizer"] / "tokenizer.json"
    eos_id = int(tokenizer_contract["eos_token_id"])
    payload = pack_context_task(
        output_root,
        pack_plan=pack_plan,
        plan=plan,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=str(tokenizer_contract["tokenizer_sha256"]),
        eos_id=eos_id,
        # The compact causal loader masks padding, so EOS is the safest
        # tokenizer-defined padding value when the tokenizer has no pad token.
        pad_id=eos_id,
        task_index=task_index,
    )
    state.complete("context_pack", task_id, payload)
    return payload


def _context_verify(profile: dict[str, Any]) -> dict[str, Any]:
    _root, state = _paths(profile)
    output_root = _context_output_root(profile)
    plan, plan_path = _context_plan(profile)
    pack_plan = _load_or_create_context_pack_plan(output_root, plan)
    token_contract, token_contract_file_sha = _context_token_count_contract(
        profile, state, output_root
    )
    selection = validate_context_selection(
        output_root,
        plan=plan,
        pack_plan=pack_plan,
        token_count_contract_sha256=token_contract_file_sha,
        tokenizer_contract=token_contract["tokenizer_contract"],
    )
    payload = verify_and_seal_context_release(
        output_root,
        plan=plan,
        pack_plan=pack_plan,
        selection=selection,
        tokenizer_contract=token_contract["tokenizer_contract"],
        context_plan_path=plan_path,
    )
    state.complete("context_verify", "task-000000", payload)
    return payload


def _validate_selection_artifacts(
    profile: dict[str, Any],
    state: StateStore,
    selection: dict[str, Any],
    *,
    deep_token_count_validation: bool,
    validate_all_schedule: bool = True,
) -> dict[str, Any]:
    if selection.get("schema") != "metis.selection-release/v2":
        raise RuntimeError("Selection uses an obsolete or unknown schema")
    root = Path(profile["storage"]["lustre_root"])
    directories = profile["storage"]["directories"]
    selected_root = root / directories["selected"]
    token_contract_path = selected_root / "TOKEN_COUNT_CONTRACT.json"
    if not token_contract_path.is_file():
        raise RuntimeError("TOKEN_COUNT_CONTRACT.json is missing from selection")
    token_contract = json.loads(token_contract_path.read_text(encoding="utf-8"))
    unsigned_contract = {
        key: value for key, value in token_contract.items() if key != "contract_sha256"
    }
    if (
        token_contract.get("schema") != "metis.token-count-set/v1"
        or token_contract.get("contract_sha256") != _json_sha256(unsigned_contract)
        or selection.get("token_count_contract_sha256")
        != sha256_file(token_contract_path)
    ):
        raise RuntimeError("Selection token-count contract is corrupt or mismatched")
    tokenizer_contract = _production_tokenizer_contract(profile)
    if (
        selection.get("tokenizer_contract") != tokenizer_contract
        or token_contract.get("tokenizer_contract") != tokenizer_contract
    ):
        raise RuntimeError("Selection is not bound to the current production tokenizer")
    if deep_token_count_validation:
        rebuilt, _ = _token_count_contract(profile, state)
        if rebuilt != token_contract:
            # created_at is deterministic only within the persisted contract,
            # so compare the immutable fields after retaining its timestamp.
            rebuilt["created_at"] = token_contract.get("created_at")
            rebuilt["contract_sha256"] = _json_sha256(
                {
                    key: value
                    for key, value in rebuilt.items()
                    if key != "contract_sha256"
                }
            )
            if rebuilt != token_contract:
                raise RuntimeError(
                    "Token-count inputs/outputs changed after selection"
                )
    schedule_contract = []
    seen_global: set[int] = set()
    for shard in selection.get("shards", []):
        global_index = int(shard.get("global_index", -1))
        if global_index in seen_global:
            raise RuntimeError(f"Duplicate selection global shard {global_index}")
        seen_global.add(global_index)
        path = Path(str(shard.get("path", ""))).resolve()
        try:
            path.relative_to((selected_root / "schedule").resolve())
        except ValueError as exc:
            raise RuntimeError(f"Selection schedule path escapes its root: {path}") from exc
        if validate_all_schedule:
            if (
                not path.is_file()
                or path.stat().st_size != int(shard.get("size", -1))
                or sha256_file(path) != shard.get("sha256")
            ):
                raise RuntimeError(f"Selection schedule shard changed: {path}")
        schedule_contract.append(
            {
                key: shard[key]
                for key in (
                    "phase",
                    "phase_index",
                    "global_index",
                    "target_tokens",
                    "size",
                    "sha256",
                )
            }
        )
    if seen_global != set(range(len(seen_global))):
        raise RuntimeError("Selection global shard indices are not contiguous")
    if selection.get("schedule_manifest_sha256") != _json_sha256(schedule_contract):
        raise RuntimeError("Selection schedule manifest hash is invalid")
    return {
        "tokenizer_contract": tokenizer_contract,
        "token_count_contract_path": token_contract_path,
        "token_count_contract": token_contract,
    }


def _select(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    output_root = root / directories["selected"]
    selection_path = output_root / "SELECTION.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        _validate_selection_artifacts(
            profile,
            state,
            selection,
            deep_token_count_validation=True,
        )
        return selection
    manifest = _manifest(profile)
    _require_metis16_schedule_contract(manifest)
    token_contract, report_payloads = _token_count_contract(profile, state)
    output_root.mkdir(parents=True, exist_ok=True)
    token_contract_path = output_root / "TOKEN_COUNT_CONTRACT.json"
    atomic_json(token_contract_path, token_contract)
    eligible_tokens: dict[str, int] = {}
    for report in report_payloads:
        for source_id, tokens in report["source_tokens"].items():
            eligible_tokens[source_id] = eligible_tokens.get(source_id, 0) + int(tokens)

    def records() -> Iterator[dict[str, Any]]:
        token_root = root / directories["token_counts"]
        for task in token_contract["tasks"]:
            path = token_root / task["output"]["path"]
            yield from _iter_rows(path)

    payload = build_selection(
        records(),
        manifest=manifest,
        eligible_tokens=eligible_tokens,
        output_root=output_root,
        shard_tokens=int(profile["storage"]["final_shard_tokens"]),
        token_count_contract_sha256=sha256_file(token_contract_path),
        tokenizer_contract=token_contract["tokenizer_contract"],
    )
    state.complete("select", "task-000000", payload)
    return payload


def _pack(profile: dict[str, Any], task_index: int) -> dict[str, Any]:
    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    selected = root / directories["selected"]
    selection_path = selected / "SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_sha = sha256_file(selection_path)
    selection_artifacts = _validate_selection_artifacts(
        profile,
        state,
        selection,
        deep_token_count_validation=False,
        validate_all_schedule=False,
    )
    matching = [shard for shard in selection["shards"] if int(shard["global_index"]) == task_index]
    if len(matching) != 1:
        raise ValueError(f"Selection does not contain exactly one global shard {task_index}")
    shard = matching[0]
    schedule_path = Path(str(shard["path"])).resolve()
    try:
        schedule_path.relative_to((selected / "schedule").resolve())
    except ValueError as exc:
        raise RuntimeError(f"Pack schedule path escapes selection root: {schedule_path}") from exc
    if (
        not schedule_path.is_file()
        or schedule_path.stat().st_size != int(shard.get("size", -1))
        or sha256_file(schedule_path) != shard.get("sha256")
    ):
        raise RuntimeError(f"Pack schedule shard is missing or changed: {schedule_path}")
    task_id = f"task-{task_index:06d}"
    if state.is_complete("pack", task_id):
        completed = state.read("completed", "pack", f"{task_id}.json")
        if (
            completed.get("schema") != "metis.pack-task/v2"
            or completed.get("selection_sha256") != selection_sha
            or completed.get("tokenizer_contract")
            != selection_artifacts["tokenizer_contract"]
            or completed.get("token_count_contract_sha256")
            != selection.get("token_count_contract_sha256")
            or completed.get("schedule_sha256") != shard.get("sha256")
            or completed.get("phase") != shard.get("phase")
            or int(completed.get("phase_index", -1))
            != int(shard.get("phase_index", -2))
            or int(completed.get("tokens", -1))
            != int(shard.get("target_tokens", -2))
        ):
            raise RuntimeError(
                f"Completed pack task {task_index} is stale relative to selection/tokenizer"
            )
        binary = Path(str(completed.get("binary", "")))
        index = Path(str(completed.get("index", "")))
        if (
            not binary.is_file()
            or binary.stat().st_size != int(completed["tokens"]) * 2
            or sha256_file(binary) != completed.get("binary_sha256")
            or not index.is_file()
            or sha256_file(index) != completed.get("index_sha256")
        ):
            raise RuntimeError(f"Completed pack task {task_index} artifacts changed")
        return completed
    tokenizer = Tokenizer.from_file(
        str(root / directories["tokenizer"] / "tokenizer.json")
    )
    eos_id = int(selection_artifacts["tokenizer_contract"]["eos_token_id"])
    release_root = root / directories["release"]
    phase_dir = release_root / shard["phase"].replace("_", "-")
    phase_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard-{int(shard['phase_index']):05d}"
    binary = phase_dir / f"{stem}.bin"
    index = phase_dir / f"{stem}.index.jsonl"
    temporary_binary = binary.with_suffix(".bin.incomplete")
    temporary_index = index.with_suffix(".jsonl.incomplete")
    written = 0
    documents = 0
    source_tokens: dict[str, int] = {}
    quota_source_tokens: dict[str, int] = {}
    replacement_tokens: dict[str, dict[str, int]] = {}
    license_tokens: dict[str, dict[str, int]] = {}
    generated_tokens = 0
    transformed_tokens = 0
    generated_or_transformed_tokens = 0
    unique_tokens = 0
    replay_tokens = 0
    missing_license_tokens = 0
    with temporary_binary.open("wb") as binary_handle, temporary_index.open("w", encoding="utf-8") as index_handle:
        for record in _iter_rows(schedule_path):
            recomputed_content_sha = content_sha256(str(record["text"])).hex()
            recomputed_text_sha = hashlib.sha256(
                str(record["text"]).encode("utf-8")
            ).hexdigest()
            if (
                recomputed_content_sha != record.get("content_sha256")
                or recomputed_text_sha != record.get("text_sha256")
            ):
                raise RuntimeError(
                    "Selected text no longer matches its exact/final-dedup hashes for "
                    f"{record['source_id']}:{record['doc_id']}"
                )
            ids = tokenizer.encode(str(record["text"]), add_special_tokens=False).ids + [eos_id]
            start = int(record["token_start"])
            count = int(record["token_count"])
            selected_ids = ids[start : start + count]
            if len(selected_ids) != count:
                raise RuntimeError(f"Token slice is out of bounds for {record['source_id']}:{record['doc_id']}")
            array = np.asarray(selected_ids, dtype=np.dtype("<u2"))
            binary_handle.write(array.tobytes())
            index_handle.write(
                json.dumps(
                    {
                        "start": written,
                        "end": written + count,
                        "source_id": record["source_id"],
                        "quota_source_id": record.get(
                            "quota_source_id", record["source_id"]
                        ),
                        "replacement_for_source_id": record.get(
                            "replacement_for_source_id"
                        ),
                        "replacement": bool(record.get("replacement", False)),
                        "doc_id": record["doc_id"],
                        "replay": bool(record["replay"]),
                        "exposure": int(record.get("exposure", 0)),
                        "token_start": start,
                        "content_sha256": record.get("content_sha256"),
                        "text_sha256": record.get("text_sha256"),
                        "license": record.get("license"),
                        "license_status": record.get("license_status"),
                        "generated": bool(record.get("generated", False)),
                        "transformed": bool(record.get("transformed", False)),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            written += count
            if record.get("replay"):
                replay_tokens += count
            else:
                unique_tokens += count
            documents += 1
            source_tokens[record["source_id"]] = source_tokens.get(record["source_id"], 0) + count
            quota_source_id = str(
                record.get("quota_source_id", record["source_id"])
            )
            quota_source_tokens[quota_source_id] = (
                quota_source_tokens.get(quota_source_id, 0) + count
            )
            if bool(record.get("replacement", False)):
                by_target = replacement_tokens.setdefault(quota_source_id, {})
                actual_source_id = str(record["source_id"])
                by_target[actual_source_id] = (
                    by_target.get(actual_source_id, 0) + count
                )
            license_expression = str(record.get("license") or "")
            if not license_expression:
                missing_license_tokens += count
            else:
                by_license = license_tokens.setdefault(record["source_id"], {})
                by_license[license_expression] = by_license.get(license_expression, 0) + count
            if record.get("generated"):
                generated_tokens += count
            if record.get("transformed"):
                transformed_tokens += count
            if record.get("generated") or record.get("transformed"):
                generated_or_transformed_tokens += count
    if written != int(shard["target_tokens"]):
        raise RuntimeError(f"Pack task {task_index} wrote {written:,}, expected {int(shard['target_tokens']):,}")
    if shard["phase"] == "phase_c" and (generated_tokens or transformed_tokens):
        raise RuntimeError(
            "Phase C shard contains generated/transformed tokens: "
            f"{generated_tokens:,}/{transformed_tokens:,}"
        )
    temporary_binary.replace(binary)
    temporary_index.replace(index)
    payload = {
        "schema": "metis.pack-task/v2",
        "stage": "pack",
        "task_index": task_index,
        "phase": shard["phase"],
        "phase_index": shard["phase_index"],
        "tokens": written,
        "documents": documents,
        "source_tokens": source_tokens,
        "quota_source_tokens": quota_source_tokens,
        "replacement_tokens": replacement_tokens,
        "license_tokens": license_tokens,
        "missing_license_tokens": missing_license_tokens,
        "generated_tokens": generated_tokens,
        "transformed_tokens": transformed_tokens,
        "generated_or_transformed_tokens": generated_or_transformed_tokens,
        "unique_tokens": unique_tokens,
        "replay_tokens": replay_tokens,
        "binary": str(binary),
        "binary_bytes": binary.stat().st_size,
        "binary_sha256": sha256_file(binary),
        "index": str(index),
        "index_sha256": sha256_file(index),
        "selection_sha256": selection_sha,
        "token_count_contract_sha256": selection["token_count_contract_sha256"],
        "tokenizer_contract": selection_artifacts["tokenizer_contract"],
        "schedule_sha256": shard["sha256"],
        "completed_at": utc_now(),
    }
    state.complete("pack", task_id, payload)
    return payload


def _integer_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _integer_tree(item) for key, item in value.items()}
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return value


def _require_metis16_schedule_contract(manifest: dict[str, Any]) -> None:
    schedule = manifest.get("schedule", {})
    phases = schedule.get("phases", {})
    expected_starts = {
        "phase_a": 0,
        "phase_b": 700_000_000_000,
        "phase_c": 950_000_000_000,
    }
    expected_targets = {
        "phase_a": 700_000_000_000,
        "phase_b": 250_000_000_000,
        "phase_c": 50_000_000_000,
    }
    if int(schedule.get("target_tokens", -1)) != 1_000_000_000_000:
        raise RuntimeError("Metis-1.6 schedule is not exactly 1T")
    if (
        int(schedule.get("unique_target_tokens", -1)) != 950_000_000_000
        or int(schedule.get("replay_target_tokens", -1)) != 50_000_000_000
    ):
        raise RuntimeError("Metis-1.6 schedule is not exactly 950B unique/50B replay")
    unique_sum = 0
    replay_sum = 0
    for phase, target in expected_targets.items():
        payload = phases.get(phase, {})
        unique = int(payload.get("unique_tokens", -1))
        replay = int(payload.get("replay_tokens", -1))
        if (
            int(payload.get("start_token", -1)) != expected_starts[phase]
            or int(payload.get("target_tokens", -1)) != target
            or unique < 0
            or replay < 0
            or unique + replay != target
        ):
            raise RuntimeError(f"Metis-1.6 {phase} schedule contract is invalid")
        unique_sum += unique
        replay_sum += replay
    if unique_sum != 950_000_000_000 or replay_sum != 50_000_000_000:
        raise RuntimeError("Per-phase unique/replay sums do not equal 950B/50B")
    for source in manifest.get("sources", []):
        if any(
            int(source.get("phase_tokens", {}).get(phase, 0)) < 0
            for phase in expected_targets
        ):
            raise RuntimeError(f"Source {source.get('id')} has a negative phase quota")


def _validate_selection_contract(
    profile: dict[str, Any], manifest: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    _require_metis16_schedule_contract(manifest)
    expected_replay = replay_quotas(manifest)
    expected_unique = unique_quotas(manifest, expected_replay)
    comparisons = {
        "replay_quotas": expected_replay,
        "unique_quotas": expected_unique,
        "replay_written": expected_replay,
        "unique_written": expected_unique,
    }
    for field, expected in comparisons.items():
        if _integer_tree(selection.get(field)) != _integer_tree(expected):
            raise RuntimeError(f"Selection {field} does not match the immutable manifest quota")
    replacement_allocation = selection.get("replacement_allocation")
    if not isinstance(replacement_allocation, dict):
        raise RuntimeError("Selection replacement allocation is missing")
    rebuilt_replacement = allocate_replacements(
        manifest,
        requirements=expected_unique,
        available_tokens=replacement_allocation.get("available_tokens", {}),
    )
    if replacement_allocation != rebuilt_replacement:
        raise RuntimeError(
            "Selection replacement allocation does not match the immutable policy"
        )
    if int(selection.get("replacement_tokens", -1)) != int(
        rebuilt_replacement["replacement_tokens"]
    ):
        raise RuntimeError("Selection replacement-token total is inconsistent")
    unique_tokens = sum(sum(phases.values()) for phases in expected_unique.values())
    replay_tokens = sum(sum(phases.values()) for phases in expected_replay.values())
    target_tokens = int(manifest["schedule"]["target_tokens"])
    declared_unique = int(manifest["schedule"].get("unique_target_tokens", -1))
    declared_replay = int(manifest["schedule"].get("replay_target_tokens", -1))
    if (
        unique_tokens != declared_unique
        or replay_tokens != declared_replay
        or declared_unique != 950_000_000_000
        or declared_replay != 50_000_000_000
    ):
        raise RuntimeError(
            "Manifest/derived unique-replay contract is not exactly 950B/50B"
        )
    if unique_tokens + replay_tokens != target_tokens:
        raise RuntimeError("Unique plus replay selection does not equal the 1T exposure schedule")
    if (
        int(selection.get("unique_tokens", -1)) != unique_tokens
        or int(selection.get("replay_tokens", -1)) != replay_tokens
    ):
        raise RuntimeError("Selection unique/replay headline totals are inconsistent")
    expected_phase = {
        phase: int(manifest["schedule"]["phases"][phase]["target_tokens"])
        for phase in ("phase_a", "phase_b", "phase_c")
    }
    if _integer_tree(selection.get("phase_tokens")) != expected_phase:
        raise RuntimeError("Selection phase totals do not match the immutable manifest")
    shard_phase = {phase: 0 for phase in expected_phase}
    for shard in selection.get("shards", []):
        phase = str(shard.get("phase", ""))
        if phase not in shard_phase:
            raise RuntimeError(f"Selection contains unknown phase {phase!r}")
        shard_phase[phase] += int(shard.get("target_tokens", 0))
    if shard_phase != expected_phase:
        raise RuntimeError("Selection schedule shards do not exactly cover all phases")
    minimum_unique = int(profile.get("gates", {}).get("minimum_unique_tokens", unique_tokens))
    if unique_tokens < minimum_unique:
        raise RuntimeError(
            f"Selection contains {unique_tokens:,} unique tokens, below the {minimum_unique:,} gate"
        )
    maximum_exposures = int(manifest["selection"]["replay"]["maximum_document_exposures"])
    if int(selection.get("maximum_document_exposures", -1)) != maximum_exposures:
        raise RuntimeError("Selection lost the maximum-document-exposures contract")
    if int(selection.get("selection_seed", -1)) != int(manifest["selection"]["seed"]):
        raise RuntimeError("Selection seed does not match the manifest")
    return {
        "unique_tokens": unique_tokens,
        "replay_tokens": replay_tokens,
        "maximum_document_exposures": maximum_exposures,
        "selection_seed": int(manifest["selection"]["seed"]),
    }


def _audit_packed_index(
    index_path: Path,
    *,
    expected_tokens: int,
    maximum_exposures: int,
) -> dict[str, Any]:
    cursor = 0
    documents = 0
    source_tokens: dict[str, int] = {}
    quota_source_tokens: dict[str, int] = {}
    replacement_tokens: dict[str, dict[str, int]] = {}
    license_tokens: dict[str, dict[str, int]] = {}
    missing_license_tokens = 0
    unique_tokens = 0
    replay_tokens = 0
    generated_tokens = 0
    transformed_tokens = 0
    generated_or_transformed_tokens = 0
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            start = int(row.get("start", -1))
            end = int(row.get("end", -1))
            if start != cursor or end <= start or end > expected_tokens:
                raise RuntimeError(
                    f"Packed index offsets are non-contiguous at {index_path}:{documents + 1}"
                )
            count = end - start
            replay = bool(row.get("replay", False))
            exposure = int(row.get("exposure", -1))
            if (replay and not 1 <= exposure < maximum_exposures) or (
                not replay and exposure != 0
            ):
                raise RuntimeError(
                    f"Packed index exposure contract is invalid at {index_path}:{documents + 1}"
                )
            content_hash = str(row.get("content_sha256", ""))
            text_hash = str(row.get("text_sha256", ""))
            if len(content_hash) != 64 or len(text_hash) != 64:
                raise RuntimeError(
                    f"Packed index content hash is invalid at {index_path}:{documents + 1}"
                )
            source_id = str(row.get("source_id", ""))
            if not source_id:
                raise RuntimeError(
                    f"Packed index source is missing at {index_path}:{documents + 1}"
                )
            source_tokens[source_id] = source_tokens.get(source_id, 0) + count
            quota_source_id = str(row.get("quota_source_id") or source_id)
            if not quota_source_id:
                raise RuntimeError(
                    f"Packed index quota source is missing at {index_path}:{documents + 1}"
                )
            replacement = bool(row.get("replacement", False))
            replacement_for = row.get("replacement_for_source_id")
            if replacement != (quota_source_id != source_id):
                raise RuntimeError(
                    f"Packed replacement flag is inconsistent at {index_path}:{documents + 1}"
                )
            if replacement and str(replacement_for or "") != quota_source_id:
                raise RuntimeError(
                    f"Packed replacement target is inconsistent at {index_path}:{documents + 1}"
                )
            if not replacement and replacement_for not in (None, ""):
                raise RuntimeError(
                    f"Non-replacement row carries a replacement target at {index_path}:{documents + 1}"
                )
            quota_source_tokens[quota_source_id] = (
                quota_source_tokens.get(quota_source_id, 0) + count
            )
            if replacement:
                by_target = replacement_tokens.setdefault(quota_source_id, {})
                by_target[source_id] = by_target.get(source_id, 0) + count
            if replay:
                replay_tokens += count
            else:
                unique_tokens += count
            generated = bool(row.get("generated", False))
            transformed = bool(row.get("transformed", False))
            if generated:
                generated_tokens += count
            if transformed:
                transformed_tokens += count
            if generated or transformed:
                generated_or_transformed_tokens += count
            license_expression = str(row.get("license") or "")
            if not license_expression:
                missing_license_tokens += count
            else:
                by_license = license_tokens.setdefault(source_id, {})
                by_license[license_expression] = (
                    by_license.get(license_expression, 0) + count
                )
            cursor = end
            documents += 1
    if cursor != expected_tokens:
        raise RuntimeError(
            f"Packed index covers {cursor:,} tokens, expected {expected_tokens:,}: {index_path}"
        )
    return {
        "documents": documents,
        "source_tokens": source_tokens,
        "quota_source_tokens": quota_source_tokens,
        "replacement_tokens": replacement_tokens,
        "license_tokens": license_tokens,
        "missing_license_tokens": missing_license_tokens,
        "unique_tokens": unique_tokens,
        "replay_tokens": replay_tokens,
        "generated_tokens": generated_tokens,
        "transformed_tokens": transformed_tokens,
        "generated_or_transformed_tokens": generated_or_transformed_tokens,
    }


VERIFY_AUDIT_FIELDS = (
    "documents",
    "source_tokens",
    "quota_source_tokens",
    "replacement_tokens",
    "license_tokens",
    "missing_license_tokens",
    "unique_tokens",
    "replay_tokens",
    "generated_tokens",
    "transformed_tokens",
    "generated_or_transformed_tokens",
)


def _verify_shard_binding(
    *,
    selection: dict[str, Any],
    selection_sha: str,
    tokenizer_contract: dict[str, Any],
    maximum_exposures: int,
) -> str:
    """Hash the policy a per-shard verification receipt is valid under."""

    return _json_sha256(
        {
            "schema": "metis.verify-shard-binding/v1",
            "selection_sha256": selection_sha,
            "token_count_contract_sha256": selection.get("token_count_contract_sha256"),
            "tokenizer_contract": tokenizer_contract,
            "maximum_document_exposures": int(maximum_exposures),
        }
    )


def _verify_shard_payload(
    profile: dict[str, Any],
    state: StateStore,
    *,
    shard: dict[str, Any],
    selection: dict[str, Any],
    selection_sha: str,
    tokenizer_contract: dict[str, Any],
    maximum_exposures: int,
) -> dict[str, Any]:
    """Re-hash and re-audit one packed shard against its pack completion.

    This is the whole per-shard integrity gate: the same checks the serial
    verifier ran, moved into an independently schedulable task so 1.82TiB is
    not read through a single process.
    """

    root = Path(profile["storage"]["lustre_root"])
    directories = profile["storage"]["directories"]
    global_index = int(shard["global_index"])
    task_id = f"task-{global_index:06d}"
    report = state.read("completed", "pack", f"{task_id}.json")
    if not report:
        raise RuntimeError(f"Pack completion is missing: {task_id}")
    if (
        report.get("schema") != "metis.pack-task/v2"
        or int(report.get("task_index", -1)) != global_index
        or report.get("phase") != shard["phase"]
        or int(report.get("phase_index", -1)) != int(shard["phase_index"])
        or int(report.get("tokens", -1)) != int(shard["target_tokens"])
        or report.get("selection_sha256") != selection_sha
        or report.get("token_count_contract_sha256")
        != selection.get("token_count_contract_sha256")
        or report.get("tokenizer_contract") != tokenizer_contract
        or report.get("schedule_sha256") != shard.get("sha256")
    ):
        raise RuntimeError(f"Pack completion is stale or mismatched: {task_id}")
    binary = Path(report["binary"])
    expected_phase_root = (
        root / directories["release"] / str(shard["phase"]).replace("_", "-")
    ).resolve()
    try:
        binary.resolve().relative_to(expected_phase_root)
    except ValueError as exc:
        raise RuntimeError(f"Packed binary escapes its phase directory: {binary}") from exc
    if binary.stat().st_size != int(report["tokens"]) * 2:
        raise RuntimeError(f"uint16 byte size mismatch: {binary}")
    if sha256_file(binary) != report["binary_sha256"]:
        raise RuntimeError(f"Binary checksum mismatch: {binary}")
    index_path = Path(report["index"])
    try:
        index_path.resolve().relative_to(expected_phase_root)
    except ValueError as exc:
        raise RuntimeError(f"Packed index escapes its phase directory: {index_path}") from exc
    if sha256_file(index_path) != report["index_sha256"]:
        raise RuntimeError(f"Index checksum mismatch: {index_path}")
    audited = _audit_packed_index(
        index_path,
        expected_tokens=int(report["tokens"]),
        maximum_exposures=maximum_exposures,
    )
    for field in VERIFY_AUDIT_FIELDS:
        if _integer_tree(report.get(field)) != _integer_tree(audited[field]):
            raise RuntimeError(
                f"Pack report {task_id} {field} does not match its hashed index"
            )
    if report["phase"] == "phase_c" and int(report["generated_or_transformed_tokens"]):
        raise RuntimeError(f"Generated/transformed data found in phase C: {binary}")
    if int(audited["missing_license_tokens"]):
        raise RuntimeError(f"Shard contains records without license evidence: {binary}")
    return report


def _verify_selection_context(
    profile: dict[str, Any], state: StateStore
) -> dict[str, Any]:
    root = Path(profile["storage"]["lustre_root"])
    manifest = _manifest(profile)
    selection_path = root / profile["storage"]["directories"]["selected"] / "SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_artifacts = _validate_selection_artifacts(
        profile,
        state,
        selection,
        deep_token_count_validation=not bool(state.read("cleanup", "selection_inputs.json")),
    )
    selection_contract = _validate_selection_contract(profile, manifest, selection)
    tokenizer_contract = selection_artifacts["tokenizer_contract"]
    maximum_exposures = int(selection_contract["maximum_document_exposures"])
    selection_sha = sha256_file(selection_path)
    return {
        "manifest": manifest,
        "selection": selection,
        "selection_path": selection_path,
        "selection_sha": selection_sha,
        "selection_artifacts": selection_artifacts,
        "selection_contract": selection_contract,
        "tokenizer_contract": tokenizer_contract,
        "maximum_exposures": maximum_exposures,
        "binding": _verify_shard_binding(
            selection=selection,
            selection_sha=selection_sha,
            tokenizer_contract=tokenizer_contract,
            maximum_exposures=maximum_exposures,
        ),
    }


def _verify_shard(profile: dict[str, Any], task_index: int) -> dict[str, Any]:
    _root, state = _paths(profile)
    context = _verify_selection_context(profile, state)
    shards = context["selection"]["shards"]
    if task_index >= len(shards):
        # pack_tasks is derived from the manifest schedule and may round above
        # the actual shard count; the surplus tasks are complete by definition.
        payload = {
            "schema": "metis.verify-shard/v1",
            "task_index": task_index,
            "shard_present": False,
            "binding_sha256": context["binding"],
        }
        state.complete("verify_shard", f"task-{task_index:06d}", payload)
        return payload
    shard = shards[task_index]
    if int(shard["global_index"]) != task_index:
        raise RuntimeError(
            f"SELECTION.json shard {task_index} declares global_index {shard['global_index']}"
        )
    report = _verify_shard_payload(
        profile,
        state,
        shard=shard,
        selection=context["selection"],
        selection_sha=context["selection_sha"],
        tokenizer_contract=context["tokenizer_contract"],
        maximum_exposures=context["maximum_exposures"],
    )
    payload = {
        "schema": "metis.verify-shard/v1",
        "task_index": task_index,
        "shard_present": True,
        "binding_sha256": context["binding"],
        "pack_report": report,
    }
    state.complete("verify_shard", f"task-{task_index:06d}", payload)
    return payload


def _verify(profile: dict[str, Any]) -> dict[str, Any]:
    # First, before any filesystem work. This is the gate that actually holds
    # an unreviewed corpus back -- the compute preflight only warns, because the
    # ledger being reviewed is produced by the build itself. A policy refusal
    # should not depend on the state store being readable to fire.
    if profile.get("gates", {}).get("require_license_ledger") and not profile.get("gates", {}).get("license_review_complete", False):
        raise RuntimeError("Fail-closed: the source/license review has not been marked complete in the CPU build profile")
    root, state = _paths(profile)
    manifest = _manifest(profile)
    directories = profile["storage"]["directories"]
    context = _verify_selection_context(profile, state)
    selection = context["selection"]
    selection_path = context["selection_path"]
    selection_artifacts = context["selection_artifacts"]
    selection_contract = context["selection_contract"]
    selection_sha = context["selection_sha"]
    tokenizer_contract = context["tokenizer_contract"]
    maximum_exposures = context["maximum_exposures"]
    binding = context["binding"]
    pack_reports = []
    for shard in selection["shards"]:
        global_index = int(shard["global_index"])
        task_id = f"task-{global_index:06d}"
        # Prefer the parallel per-shard receipt, but only when it was produced
        # under this exact selection/tokenizer/exposure policy. Anything stale
        # or missing is re-verified here rather than trusted.
        receipt = state.read("completed", "verify_shard", f"{task_id}.json")
        if (
            receipt
            and receipt.get("schema") == "metis.verify-shard/v1"
            and receipt.get("binding_sha256") == binding
            and receipt.get("shard_present") is True
            and int(receipt.get("task_index", -1)) == global_index
        ):
            pack_reports.append(receipt["pack_report"])
            continue
        pack_reports.append(
            _verify_shard_payload(
                profile,
                state,
                shard=shard,
                selection=selection,
                selection_sha=selection_sha,
                tokenizer_contract=tokenizer_contract,
                maximum_exposures=maximum_exposures,
            )
        )
    packed_unique_tokens = sum(int(report.get("unique_tokens", 0)) for report in pack_reports)
    packed_replay_tokens = sum(int(report.get("replay_tokens", 0)) for report in pack_reports)
    if (
        packed_unique_tokens != int(selection_contract["unique_tokens"])
        or packed_replay_tokens != int(selection_contract["replay_tokens"])
    ):
        raise RuntimeError(
            "Packed unique/replay totals do not match SELECTION.json: "
            f"{packed_unique_tokens:,}/{packed_replay_tokens:,}"
        )
    generated_or_transformed_tokens = sum(
        int(report.get("generated_or_transformed_tokens", 0))
        for report in pack_reports
    )
    actual_generated_tokens = sum(
        int(report.get("generated_tokens", 0)) for report in pack_reports
    )
    actual_transformed_tokens = sum(
        int(report.get("transformed_tokens", 0)) for report in pack_reports
    )
    target_tokens = int(manifest["schedule"]["target_tokens"])
    generated_share = (
        generated_or_transformed_tokens / target_tokens if target_tokens else 0.0
    )
    maximum_generated_share = float(profile.get("gates", {}).get("maximum_generated_share", 1.0))
    if generated_share > maximum_generated_share:
        raise RuntimeError(
            f"Generated-token share {generated_share:.6f} exceeds gate {maximum_generated_share:.6f}"
        )
    actual: dict[str, dict[str, int]] = {
        source["id"]: {phase: 0 for phase in ("phase_a", "phase_b", "phase_c")}
        for source in manifest["sources"]
    }
    quota_actual: dict[str, dict[str, int]] = {
        source["id"]: {phase: 0 for phase in ("phase_a", "phase_b", "phase_c")}
        for source in manifest["sources"]
    }
    phase_tokens = {phase: 0 for phase in ("phase_a", "phase_b", "phase_c")}
    for report in pack_reports:
        phase = report["phase"]
        phase_tokens[phase] += int(report["tokens"])
        for source_id, tokens in report["source_tokens"].items():
            actual[source_id][phase] += int(tokens)
        for source_id, tokens in report["quota_source_tokens"].items():
            quota_actual[source_id][phase] += int(tokens)
    expected_phase = {phase: int(manifest["schedule"]["phases"][phase]["target_tokens"]) for phase in phase_tokens}
    if phase_tokens != expected_phase:
        raise RuntimeError(f"Phase totals mismatch: {phase_tokens} != {expected_phase}")
    mismatches = {}
    for source in manifest["sources"]:
        expected = {phase: int(source["phase_tokens"].get(phase, 0)) for phase in phase_tokens}
        if quota_actual[source["id"]] != expected:
            mismatches[source["id"]] = {
                "quota_actual": quota_actual[source["id"]],
                "expected": expected,
            }
    if mismatches:
        raise RuntimeError(f"Source/phase token mismatches: {mismatches}")
    expected_generated = sum(
        sum(int(value) for value in source["phase_tokens"].values())
        for source in manifest["sources"]
        if source["provenance"].get("generated")
    )
    if actual_generated_tokens > expected_generated:
        raise RuntimeError(
            "Replacement increased generated provenance above the manifest ceiling: "
            f"{actual_generated_tokens:,} > {expected_generated:,}"
        )
    source_metadata = {source["id"]: source for source in manifest["sources"]}
    actual_category_phase = {
        category["id"]: {phase: 0 for phase in phase_tokens}
        for category in manifest["categories"]
    }
    actual_fresh_buckets: dict[str, int] = {}
    for source_id, phases in actual.items():
        source = source_metadata[source_id]
        category = str(source["category"])
        for phase, tokens in phases.items():
            actual_category_phase[category][phase] += int(tokens)
        provenance = source.get("provenance", {})
        if provenance.get("fresh"):
            bucket = str(provenance.get("freshness_bucket") or "")
            actual_fresh_buckets[bucket] = actual_fresh_buckets.get(bucket, 0) + sum(
                int(value) for value in phases.values()
            )
    expected_category_phase = {
        category["id"]: {
            phase: int(category["phase_tokens"].get(phase, 0))
            for phase in phase_tokens
        }
        for category in manifest["categories"]
    }
    if actual_category_phase != expected_category_phase:
        raise RuntimeError(
            "Replacement changed category/phase totals: "
            f"{actual_category_phase} != {expected_category_phase}"
        )
    expected_fresh_buckets = {
        str(bucket): int(tokens)
        for bucket, tokens in manifest["freshness_layer"]["buckets"].items()
    }
    if actual_fresh_buckets != expected_fresh_buckets:
        raise RuntimeError(
            "Replacement changed freshness-bucket totals: "
            f"{actual_fresh_buckets} != {expected_fresh_buckets}"
        )
    release_root = root / directories["release"]
    provenance_root = release_root / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)
    filter_chain_path = provenance_root / "FILTER_CHAIN.json"
    filter_chain = _write_filter_chain_receipt(profile, state, filter_chain_path)
    ledger_path = provenance_root / "LICENSE_LEDGER.jsonl"
    with ledger_path.open("w", encoding="utf-8") as ledger:
        for source in manifest["sources"]:
            license_payload = source["license"]
            observed: dict[str, int] = {}
            for report in pack_reports:
                for expression, tokens in report.get("license_tokens", {}).get(source["id"], {}).items():
                    observed[expression] = observed.get(expression, 0) + int(tokens)
            ledger.write(
                json.dumps(
                    {
                        "source_id": source["id"],
                        "license_status": license_payload["status"],
                        "license_expression": license_payload["expression"],
                        "observed_license_tokens": observed,
                        "training_recipe_disposition": "verified_for_training",
                        "data_publication_requires_separate_review": True,
                        "access": source["access"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    shard_manifest_path = provenance_root / "SHARDS.jsonl"
    with shard_manifest_path.open("w", encoding="utf-8") as shard_manifest:
        for report in sorted(pack_reports, key=lambda item: int(item["task_index"])):
            shard_manifest.write(
                json.dumps(
                    {
                        "task_index": int(report["task_index"]),
                        "phase": report["phase"],
                        "phase_index": int(report["phase_index"]),
                        "tokens": int(report["tokens"]),
                        "binary": str(Path(report["binary"]).relative_to(release_root)),
                        "binary_sha256": report["binary_sha256"],
                        "index": str(Path(report["index"]).relative_to(release_root)),
                        "index_sha256": report["index_sha256"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    payload = {
        "schema": "metis.verification/v2",
        "ok": True,
        "verified_at": utc_now(),
        "target_tokens": sum(phase_tokens.values()),
        "phase_tokens": phase_tokens,
        "source_phase_tokens": quota_actual,
        "actual_source_phase_tokens": actual,
        "replacement_tokens": sum(
            sum(int(tokens) for tokens in donors.values())
            for report in pack_reports
            for donors in report.get("replacement_tokens", {}).values()
        ),
        "replacement_allocation": selection.get("replacement_allocation"),
        "manifest_sha256": sha256_file(Path(manifest["_path"])),
        "manifest_contract_sha256": _manifest_contract_sha256(manifest),
        "source_lock_sha256": sha256_file(state.path("sources.lock.json")),
        "build_inputs_sha256": sha256_file(state.path("build.inputs.json")),
        "selection_sha256": selection_sha,
        "token_count_contract_sha256": sha256_file(
            selection_artifacts["token_count_contract_path"]
        ),
        "tokenizer_contract": tokenizer_contract,
        "selection_contract": selection_contract,
        "packed_unique_tokens": packed_unique_tokens,
        "packed_replay_tokens": packed_replay_tokens,
        "generated_tokens": actual_generated_tokens,
        "transformed_tokens": actual_transformed_tokens,
        "generated_or_transformed_tokens": generated_or_transformed_tokens,
        "generated_share": generated_share,
        "maximum_generated_share": maximum_generated_share,
        "filter_chain": str(filter_chain_path),
        "filter_chain_sha256": sha256_file(filter_chain_path),
        "filter_chain_contract_sha256": filter_chain["filter_chain_sha256"],
        "shards": len(pack_reports),
        "license_ledger": str(ledger_path),
        "license_ledger_sha256": sha256_file(ledger_path),
        "shard_manifest": str(shard_manifest_path),
        "shard_manifest_sha256": sha256_file(shard_manifest_path),
    }
    payload["verification_sha256"] = _json_sha256(payload)
    state.write("VERIFICATION.json", payload=payload)
    state.complete("verify", "task-000000", payload)
    return payload


def _release(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    manifest = _manifest(profile)
    directories = profile["storage"]["directories"]
    verification = state.read("VERIFICATION.json")
    if not verification or not verification.get("ok"):
        raise RuntimeError("Verified release gate has not passed")
    if verification.get("schema") != "metis.verification/v2":
        raise RuntimeError("Verified release gate uses an obsolete schema")
    unsigned_verification = {
        key: value for key, value in verification.items() if key != "verification_sha256"
    }
    if verification.get("verification_sha256") != _json_sha256(unsigned_verification):
        raise RuntimeError("VERIFICATION.json failed its self-hash check")
    release_root = root / directories["release"]
    tokenizer = root / directories["tokenizer"] / "tokenizer.json"
    selection = root / directories["selected"] / "SELECTION.json"
    source_lock = state.path("sources.lock.json")
    build_inputs = state.path("build.inputs.json")
    token_count_contract = (
        root / directories["selected"] / "TOKEN_COUNT_CONTRACT.json"
    )
    current_manifest_sha = sha256_file(Path(manifest["_path"]))
    current_manifest_contract_sha = _manifest_contract_sha256(manifest)
    current_tokenizer_contract = _production_tokenizer_contract(profile, manifest)
    selection_payload = json.loads(selection.read_text(encoding="utf-8"))
    selection_artifacts = _validate_selection_artifacts(
        profile,
        state,
        selection_payload,
        deep_token_count_validation=False,
        validate_all_schedule=not bool(state.read("cleanup", "pack_inputs.json")),
    )
    current_selection_contract = _validate_selection_contract(
        profile,
        manifest,
        selection_payload,
    )
    if (
        verification.get("manifest_sha256") != current_manifest_sha
        or verification.get("manifest_contract_sha256")
        != current_manifest_contract_sha
        or verification.get("source_lock_sha256") != sha256_file(source_lock)
        or verification.get("build_inputs_sha256") != sha256_file(build_inputs)
        or verification.get("tokenizer_contract") != current_tokenizer_contract
        or verification.get("selection_contract") != current_selection_contract
        or verification.get("token_count_contract_sha256")
        != sha256_file(token_count_contract)
        or verification.get("target_tokens")
        != int(manifest["schedule"]["target_tokens"])
        or _integer_tree(verification.get("phase_tokens"))
        != {
            phase: int(manifest["schedule"]["phases"][phase]["target_tokens"])
            for phase in ("phase_a", "phase_b", "phase_c")
        }
    ):
        raise RuntimeError(
            "Verification is stale relative to the manifest, tokenizer, or frozen build inputs"
        )
    if (
        selection_artifacts["token_count_contract_path"].resolve()
        != token_count_contract.resolve()
    ):
        raise RuntimeError("Selection resolved an unexpected token-count contract")
    ledger = release_root / "provenance" / "LICENSE_LEDGER.jsonl"
    if profile.get("gates", {}).get("require_license_ledger") and not ledger.is_file():
        raise RuntimeError(f"Fail-closed: license ledger is missing at {ledger}")
    if (
        not ledger.is_file()
        or sha256_file(ledger) != verification.get("license_ledger_sha256")
    ):
        raise RuntimeError("License ledger changed after verification")
    release_root.mkdir(parents=True, exist_ok=True)
    if sha256_file(selection) != verification.get("selection_sha256"):
        raise RuntimeError("SELECTION.json changed after verification")
    filter_chain = Path(str(verification.get("filter_chain") or ""))
    if not filter_chain.is_file() or sha256_file(filter_chain) != verification.get("filter_chain_sha256"):
        raise RuntimeError("Filtering/decontamination receipt changed after verification")
    filter_payload = json.loads(filter_chain.read_text(encoding="utf-8"))
    filter_unsigned = {
        key: value for key, value in filter_payload.items() if key != "filter_chain_sha256"
    }
    if filter_payload.get("filter_chain_sha256") != _json_sha256(filter_unsigned):
        raise RuntimeError("Filtering/decontamination receipt failed its self-hash check")
    _validate_filter_chain_artifacts(profile, filter_payload)
    shard_manifest = release_root / "provenance" / "SHARDS.jsonl"
    if (
        not shard_manifest.is_file()
        or sha256_file(shard_manifest)
        != verification.get("shard_manifest_sha256")
    ):
        raise RuntimeError("Shard manifest changed after verification")
    release_tokenizer = release_root / "tokenizer"
    release_manifests = release_root / "manifests"
    release_reports = release_root / "reports"
    for directory in (release_tokenizer, release_manifests, release_reports):
        if directory.exists():
            shutil.rmtree(directory)
    release_tokenizer.mkdir(parents=True, exist_ok=True)
    release_manifests.mkdir(parents=True, exist_ok=True)
    release_reports.mkdir(parents=True, exist_ok=True)
    tokenizer_root = tokenizer.parent
    for name in (
        "tokenizer.json",
        "vocab.json",
        "TOKENIZER_RELEASE.json",
        "TOKENIZER_VALIDATION.json",
        CANONICAL_IDS_MANIFEST,
        CANONICAL_IDS_BINARY,
    ):
        source = tokenizer_root / name
        if not source.exists():
            raise RuntimeError(f"Tokenizer release artifact is missing: {source}")
        shutil.copy2(source, release_tokenizer / name)
    shutil.copy2(Path(manifest["_path"]), release_manifests / "metis-1.6.yaml")
    manifest_repository = repository_root() / "manifests"
    replacement_policy_file = manifest.get("replacement_policy_file")
    if replacement_policy_file or manifest.get("replacement_policy"):
        replacement_policy_path = manifest_repository / str(
            replacement_policy_file or "replacements.yaml"
        )
        if not replacement_policy_path.is_file():
            raise RuntimeError(
                f"Replacement policy is missing from the repository: {replacement_policy_path}"
            )
        shutil.copy2(replacement_policy_path, release_manifests / "replacements.yaml")
    for subdirectory in ("sources", "contamination", "registries", "licenses"):
        source_directory = manifest_repository / subdirectory
        if source_directory.exists():
            shutil.copytree(source_directory, release_manifests / subdirectory, dirs_exist_ok=True)
    shutil.copy2(source_lock, release_manifests / "sources.lock.json")
    if not build_inputs.is_file():
        raise RuntimeError("Frozen build.inputs.json is missing from the Rhea build state")
    shutil.copy2(build_inputs, release_manifests / "build.inputs.json")
    shutil.copy2(selection, release_manifests / "SELECTION.json")
    shutil.copy2(
        token_count_contract,
        release_manifests / "TOKEN_COUNT_CONTRACT.json",
    )
    copied_hashes = (
        (
            release_manifests / "metis-1.6.yaml",
            current_manifest_sha,
            "data manifest",
        ),
        (
            release_manifests / "sources.lock.json",
            verification["source_lock_sha256"],
            "source lock",
        ),
        (
            release_manifests / "build.inputs.json",
            verification["build_inputs_sha256"],
            "build inputs",
        ),
        (
            release_manifests / "SELECTION.json",
            verification["selection_sha256"],
            "selection",
        ),
        (
            release_manifests / "TOKEN_COUNT_CONTRACT.json",
            verification["token_count_contract_sha256"],
            "token-count contract",
        ),
        (
            release_tokenizer / "tokenizer.json",
            current_tokenizer_contract["tokenizer_sha256"],
            "tokenizer",
        ),
        (
            release_tokenizer / "vocab.json",
            current_tokenizer_contract["vocab_sha256"],
            "tokenizer vocabulary",
        ),
        (
            release_tokenizer / "TOKENIZER_RELEASE.json",
            current_tokenizer_contract["tokenizer_release_sha256"],
            "tokenizer release report",
        ),
        (
            release_tokenizer / "TOKENIZER_VALIDATION.json",
            current_tokenizer_contract["tokenizer_validation_sha256"],
            "tokenizer validation report",
        ),
        (
            release_tokenizer / CANONICAL_IDS_MANIFEST,
            current_tokenizer_contract["ngram_canonical_map_manifest_sha256"],
            "N-gram canonical-ID manifest",
        ),
        (
            release_tokenizer / CANONICAL_IDS_BINARY,
            current_tokenizer_contract["ngram_canonical_ids_sha256"],
            "N-gram canonical-ID binary",
        ),
    )
    for copied, expected_sha, label in copied_hashes:
        if sha256_file(copied) != expected_sha:
            raise RuntimeError(f"{label.title()} changed while release artifacts were copied")
    verification_path = release_reports / "VERIFICATION.json"
    atomic_json(verification_path, verification)
    payload = {
        "schema": "metis.data-release/v2",
        "release": manifest["release"],
        "released_at": utc_now(),
        "target_tokens": verification["target_tokens"],
        "phase_tokens": verification["phase_tokens"],
        "token_dtype": profile["storage"]["final_token_dtype"],
        "token_endianness": "little",
        "tokenizer_sha256": current_tokenizer_contract["tokenizer_sha256"],
        "ngram_canonical_map_manifest_sha256": current_tokenizer_contract[
            "ngram_canonical_map_manifest_sha256"
        ],
        "ngram_canonical_map_self_sha256": current_tokenizer_contract[
            "ngram_canonical_map_self_sha256"
        ],
        "ngram_canonical_ids_sha256": current_tokenizer_contract[
            "ngram_canonical_ids_sha256"
        ],
        "tokenizer_contract": current_tokenizer_contract,
        "selection_sha256": verification["selection_sha256"],
        "token_count_contract_sha256": verification[
            "token_count_contract_sha256"
        ],
        "filter_chain_sha256": sha256_file(filter_chain),
        "license_ledger_sha256": sha256_file(ledger),
        "shard_manifest_sha256": sha256_file(shard_manifest),
        "verification_file_sha256": sha256_file(verification_path),
        "source_lock_sha256": sha256_file(source_lock),
        "build_inputs_sha256": sha256_file(build_inputs),
        "manifest_sha256": current_manifest_sha,
        "manifest_contract_sha256": current_manifest_contract_sha,
        "manifest_bundle_sha256": _tree_sha256(release_manifests),
        "verification": verification,
        "artifacts": {
            "tokenizer": "tokenizer/tokenizer.json",
            "tokenizer_vocab": "tokenizer/vocab.json",
            "tokenizer_release": "tokenizer/TOKENIZER_RELEASE.json",
            "tokenizer_validation": "tokenizer/TOKENIZER_VALIDATION.json",
            "ngram_canonical_map": f"tokenizer/{CANONICAL_IDS_MANIFEST}",
            "ngram_canonical_ids": f"tokenizer/{CANONICAL_IDS_BINARY}",
            "manifest": "manifests/metis-1.6.yaml",
            "manifest_bundle": "manifests",
            "source_lock": "manifests/sources.lock.json",
            "build_inputs": "manifests/build.inputs.json",
            "selection": "manifests/SELECTION.json",
            "token_count_contract": "manifests/TOKEN_COUNT_CONTRACT.json",
            "verification": "reports/VERIFICATION.json",
            "filter_chain": "provenance/FILTER_CHAIN.json",
            "license_ledger": "provenance/LICENSE_LEDGER.jsonl",
            "shard_manifest": "provenance/SHARDS.jsonl",
        },
    }
    payload["release_sha256"] = _json_sha256(payload)
    release_descriptor = release_root / "RELEASE.json"
    atomic_json(release_descriptor, payload)
    try:
        from .training_contract import validate_training_release

        validate_training_release(
            release_root,
            repository_root() / "configs" / "metis16" / "pretraining.yaml",
        )
    except BaseException:
        # Never leave a descriptor that looks releasable when the independent
        # Portage-side reader rejects its provenance or shard inventory.
        release_descriptor.unlink(missing_ok=True)
        raise
    state.complete("release", "task-000000", payload)
    return payload


def run_stage(profile: dict[str, Any], stage: str, task_index: int) -> dict[str, Any]:
    _require_safety_space(profile, stage)
    if stage == "download":
        return run_download_task(profile, task_index)
    if stage == "handoff_signature":
        from .handoff_verification import verify_handoff_artifact

        _, state = _paths(profile)
        return verify_handoff_artifact(profile, state, task_index)
    if stage == "handoff_verify":
        from .handoff_verification import reduce_handoff_verification

        _, state = _paths(profile)
        return reduce_handoff_verification(profile, state)
    if stage == "normalize":
        return _normalize_task(profile, task_index)
    if stage in {
        "cleanup_raw",
        "cleanup_exact",
        "cleanup_span",
        "cleanup_minhash",
        "cleanup_code",
        "cleanup_decontam",
        "cleanup_final_hash",
    }:
        return _cleanup_filter_intermediate(profile, stage)
    if stage in {
        "exact_signature", "exact_find", "exact_filter", "span_prefilter_signature",
        "span_prefilter_find", "span_signature", "span_find", "span_filter",
        "minhash_signature", "minhash_buckets", "minhash_components",
        "minhash_priority_candidates", "minhash_priority_resolve",
        "minhash_priority_finalize", "minhash_priority_verify", "minhash_filter",
        "code_signature", "code_find",
        "code_filter",
        "decontam_index", "decontam_filter", "final_hash_signature", "final_hash_find",
        "final_hash_filter",
    }:
        return _datatrove_stage(profile, stage, task_index)
    if stage == "tokenizer_sample_scan":
        return _tokenizer_sample_scan(profile, task_index)
    if stage == "tokenizer_sample_plan":
        return _tokenizer_sample_plan(profile)
    if stage == "tokenizer_sample":
        return _tokenizer_sample(profile, task_index)
    if stage == "tokenizer_train":
        return _tokenizer_train(profile)
    if stage == "cleanup_tokenizer_sample":
        return _cleanup_tokenizer_sample(profile)
    if stage == "token_count":
        return _token_count(profile, task_index)
    if stage == "context_select":
        return _context_select(profile)
    if stage == "context_prepare":
        return _context_prepare(profile)
    if stage == "context_pack":
        return _context_pack(profile, task_index)
    if stage == "context_verify":
        return _context_verify(profile)
    if stage == "select":
        return _select(profile)
    if stage == "cleanup_selection_inputs":
        return _cleanup_selection_inputs(profile)
    if stage == "pack":
        return _pack(profile, task_index)
    if stage == "verify_shard":
        return _verify_shard(profile, task_index)
    if stage == "verify":
        return _verify(profile)
    if stage == "cleanup_pack_inputs":
        return _cleanup_pack_inputs(profile)
    if stage == "release":
        return _release(profile)
    if stage == "cleanup_release_workspace":
        return _cleanup_release_workspace(profile)
    raise RuntimeError(f"Unknown stage {stage!r}")


def _run_task_worker(payload: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    """Run one global task index in a dedicated process.

    Each stage task already owns a disjoint output path and completion marker,
    so several of them may run concurrently inside one node allocation without
    any additional coordination.
    """

    profile, stage, task_index = payload
    started = time.monotonic()
    try:
        result = run_stage(profile, stage, task_index)
    except Exception as exc:  # noqa: BLE001 - reported per task, not raised
        return {
            "task_index": task_index,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    return {
        "task_index": task_index,
        "ok": True,
        "payload": result,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _run_task_group(
    profile: dict[str, Any],
    stage: str,
    *,
    first_index: int,
    task_count: int,
    task_limit: int,
    workers: int,
) -> int:
    indices = list(range(first_index, first_index + task_count))
    if task_limit > 0:
        # The final group of a chunk is normally partial, and must never reach
        # into the index range owned by another submission.
        indices = [index for index in indices if index < task_limit]
    _, state = _paths(profile)
    pending = [
        index for index in indices if not state.is_complete(stage, f"task-{index:06d}")
    ]
    skipped = len(indices) - len(pending)
    if not pending:
        print(
            json.dumps(
                {"stage": stage, "requested": len(indices), "skipped_complete": skipped, "ran": 0},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    parallelism = max(1, min(workers or len(pending), len(pending)))
    results: list[dict[str, Any]] = []
    if parallelism == 1:
        for index in pending:
            results.append(_run_task_worker((profile, stage, index)))
    else:
        # Fork keeps the per-task import cost off the critical path. The parent
        # has only parsed arguments and stat'ed completion markers by now, so
        # no native thread pool is live at the fork point; the batch wrapper
        # additionally pins every worker to a single thread. Arrow's pools are
        # created at import rather than by environment variable, so they are
        # quiesced here before the fork instead.
        try:
            import pyarrow

            pyarrow.set_io_thread_count(1)
            pyarrow.set_cpu_count(1)
        except Exception:  # noqa: BLE001 - thread pinning is best effort
            pass
        try:
            context = multiprocessing.get_context("fork")
        except ValueError:  # pragma: no cover - non-POSIX fallback
            context = multiprocessing.get_context()
        with ProcessPoolExecutor(max_workers=parallelism, mp_context=context) as pool:
            futures = {
                pool.submit(_run_task_worker, (profile, stage, index)): index
                for index in pending
            }
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda row: int(row["task_index"]))
    failures = [row for row in results if not row["ok"]]
    summary = {
        "stage": stage,
        "first_index": first_index,
        "requested": len(indices),
        "skipped_complete": skipped,
        "ran": len(results),
        "parallelism": parallelism,
        "failed": len(failures),
        "slowest_seconds": max((row["elapsed_seconds"] for row in results), default=0.0),
        "tasks": [
            {key: row[key] for key in ("task_index", "ok", "elapsed_seconds") if key in row}
            for row in results
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    for row in failures:
        print(f"FAIL task-{row['task_index']:06d} {row['error']}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument(
        "--task-count",
        type=int,
        default=1,
        help="Consecutive global task indices to run inside this allocation",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=0,
        help="Exclusive upper bound on global task index for this submission",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Concurrent worker processes; defaults to the pending task count",
    )
    args = parser.parse_args(argv)
    _, profile = load_profile(args.profile)
    if args.task_count <= 1:
        try:
            payload = run_stage(profile, args.stage, args.task_index)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        except Exception as exc:
            print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    try:
        return _run_task_group(
            profile,
            args.stage,
            first_index=args.task_index,
            task_count=args.task_count,
            task_limit=args.task_limit,
            workers=args.workers,
        )
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
