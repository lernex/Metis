"""Bind each stage to the code that actually produces its output.

The 1.6 contract bound every stage to the repository commit, by way of the
``repository_commit`` field inside ``sources.lock.json``. Two consequences
followed, and both of them cost real time on the 1.6 build:

* Re-resolving the lock rewrote ``resolved_at`` and therefore changed the
  contract, so a lock refresh invalidated finished stage work even when the
  resolved sources were byte-identical.
* Any commit invalidated every stage. Fixing a writer in ``span_dedup`` threw
  away a verified normalize pass over 1,862 shards, and committing a paragraph
  to a lessons document would have done the same.

The commit is the wrong identity. What actually determines a stage's output is
the code that stage runs, so that is what this module fingerprints: the shared
machinery every stage depends on, plus the modules specific to the stage. A fix
to ``span_dedup.py`` now invalidates the span stages and leaves normalize alone.

The default is deliberately conservative. A stage with no entry in
``STAGE_MODULES`` binds to every module in the package, which reproduces the old
whole-repository behaviour rather than silently accepting stale work. Adding a
stage without thinking about it therefore fails safe.

``stage_runner.py`` is in the shared set because it holds the implementation of
every stage in one file; editing it invalidates everything, exactly as before.
Splitting it per stage is the follow-on that makes this map pay off in full.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent

# Shared machinery. A change here can alter any stage's output, so every stage
# includes it and every stage re-runs when it moves.
COMMON_MODULES: tuple[str, ...] = (
    "config.py",
    "state.py",
    "manifest.py",
    "runtime_lock.py",
    "external_sort.py",
    "slurm.py",
    "stage_code.py",
    "stage_runner.py",
)

# Modules whose behaviour is confined to particular stages.
STAGE_MODULES: dict[str, tuple[str, ...]] = {
    "download": (
        "download.py",
        "local_download.py",
        "materializers.py",
        "source_builders.py",
        "source_lock.py",
        "freshweb.py",
        "acquisition",
    ),
    "holdouts": ("holdouts.py",),
    "handoff_signature": ("handoff.py", "handoff_verification.py"),
    "handoff_verify": ("handoff.py", "handoff_verification.py"),
    "normalize": (
        "quality.py",
        "normalization_evidence.py",
        "source_builders.py",
        "materializers.py",
        "datatrove_blocks.py",
        "freshweb.py",
        "replacement.py",
    ),
    "exact_signature": ("dedup.py",),
    "exact_find": ("dedup.py",),
    "exact_filter": ("dedup.py",),
    "span_prefilter_signature": ("span_dedup.py", "dedup.py"),
    "span_prefilter_find": ("span_dedup.py", "dedup.py"),
    "span_signature": ("span_dedup.py", "dedup.py"),
    "span_find": ("span_dedup.py", "dedup.py"),
    "span_filter": ("span_dedup.py", "dedup.py"),
    "minhash_signature": ("dedup.py",),
    "minhash_buckets": ("dedup.py",),
    "minhash_components": ("dedup.py",),
    "minhash_priority_candidates": ("dedup.py",),
    "minhash_priority_resolve": ("dedup.py",),
    "minhash_priority_finalize": ("dedup.py",),
    "minhash_priority_verify": ("dedup.py",),
    "minhash_filter": ("dedup.py",),
    "code_signature": ("code_dedup.py", "dedup.py"),
    "code_find": ("code_dedup.py", "dedup.py"),
    "code_filter": ("code_dedup.py", "dedup.py"),
    "decontam_index": ("decontaminate.py", "ngram_canonical.py"),
    "decontam_filter": ("decontaminate.py", "ngram_canonical.py"),
    "final_hash_signature": ("final_dedup.py", "dedup.py"),
    "final_hash_find": ("final_dedup.py", "dedup.py"),
    "final_hash_filter": ("final_dedup.py", "dedup.py"),
    "cleanup_raw": (),
    "cleanup_exact": (),
    "cleanup_span": (),
    "cleanup_minhash": (),
    "cleanup_code": (),
    "cleanup_decontam": (),
    "cleanup_final_hash": (),
    "cleanup_tokenizer_sample": ("tokenizer.py",),
    "tokenizer_sample": ("tokenizer.py",),
    "tokenizer_train": ("tokenizer.py", "ngram_canonical.py"),
    "token_count": ("tokenizer.py",),
    "context": ("context_extension.py", "context_manifest.py"),
    "select": ("selection.py", "replacement.py"),
    "pack": ("packing.py", "tokenizer.py"),
    "verify": ("reporting.py", "repository_license.py", "training_contract.py"),
    "release": ("reporting.py", "repository_license.py", "training_contract.py"),
}


def _iter_module_files(name: str) -> list[Path]:
    """Resolve one map entry to the files it names, package directories included."""

    target = _PACKAGE_ROOT / name
    if target.is_dir():
        return sorted(path for path in target.rglob("*.py") if path.is_file())
    if target.is_file():
        return [target]
    raise RuntimeError(f"Stage code map names a module that does not exist: {name}")


def _all_module_names() -> tuple[str, ...]:
    names = {path.name for path in _PACKAGE_ROOT.glob("*.py")}
    names.update(
        path.name for path in _PACKAGE_ROOT.iterdir() if path.is_dir() and not path.name.startswith("__")
    )
    return tuple(sorted(names))


@lru_cache(maxsize=256)
def stage_code_sha256(stage: str) -> str:
    """Fingerprint the code a stage runs.

    Called once per task across thousands of array elements, so the result is
    memoised; the underlying source cannot change inside a running job.
    """

    if stage in STAGE_MODULES:
        names = tuple(COMMON_MODULES) + tuple(STAGE_MODULES[stage])
    else:
        # Unmapped stage: bind to everything rather than guess.
        names = _all_module_names()
    digest = hashlib.sha256()
    seen: set[Path] = set()
    for name in sorted(set(names)):
        for path in _iter_module_files(name):
            if path in seen:
                continue
            seen.add(path)
            digest.update(str(path.relative_to(_PACKAGE_ROOT)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def acquisition_code_sha256() -> str:
    """Fingerprint the code that produced the downloads.

    ``require_repository_commit_match`` used to compare raw commit SHAs, which
    made the build refuse to start whenever anything at all had been committed
    since acquisition. The property worth protecting is narrower: that the
    acquisition artifacts were produced by the acquisition code now on disk.
    """

    return stage_code_sha256("download")
