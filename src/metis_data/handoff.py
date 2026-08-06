from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator

from .config import repository_root
from .freshweb import OptOutPolicy, parse_opt_out_registry, snapshot_common_crawl_opt_out
from .runtime_lock import runtime_contract, runtime_identity
from .source_lock import validate_source_lock_integrity
from .state import StateStore, utc_now
from .replacement import allocate_replacements
from .selection import replay_quotas, unique_quotas


HANDOFF_SCHEMA = "metis.acquisition-handoff/v4"
INDEX_PAYLOAD_ROLES = {"source_index", "metadata_index", "retrieval_index"}


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_digest(manifest: dict[str, Any]) -> str:
    def without_local_paths(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: without_local_paths(item)
                for key, item in value.items()
                if key not in {"_path"}
            }
        if isinstance(value, list):
            return [without_local_paths(item) for item in value]
        return value

    return _json_digest(without_local_paths(manifest))


def _repository_state() -> dict[str, Any]:
    root = repository_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}
    from .stage_code import acquisition_code_sha256

    # The commit stays for provenance and for reading a build's history. The
    # fingerprint is what verification compares, so an unrelated commit no
    # longer looks like the acquisition code changed underneath the downloads.
    return {
        "commit": commit,
        "dirty": dirty,
        "acquisition_code_sha256": acquisition_code_sha256(),
    }


def _iter_output_records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        local_path = value.get("local_path")
        if local_path and value.get("materialized") is False:
            raise RuntimeError("Acquisition output is an unresolved remote plan")
        if local_path and value.get("ready_for_training_build") is False:
            raise RuntimeError("Acquisition output is not ready for the CPU build")
        is_materialized_dataset = value.get("kind") == "materialized_dataset" or (
            local_path and Path(str(local_path)).is_dir()
        )
        if is_materialized_dataset:
            # A dataset directory is only a container. Bind the immutable
            # receipt and every shard named by that receipt, never the
            # directory inode itself.
            receipt = value.get("receipt")
            shards = value.get("shards")
            if not receipt or not isinstance(shards, list) or not shards:
                raise RuntimeError("materialized_dataset output requires a receipt and at least one shard")
            yield {
                "kind": "materialization_receipt",
                "source_id": value.get("source_id"),
                "local_path": str(receipt),
            }
            for shard in shards:
                if not isinstance(shard, dict) or not shard.get("path"):
                    raise RuntimeError("materialized_dataset receipt contains an invalid shard entry")
                yield {
                    "kind": "materialized_shard",
                    "source_id": value.get("source_id"),
                    "local_path": str(shard["path"]),
                    "size": shard.get("size"),
                    "sha256": shard.get("sha256"),
                }
            return
        if local_path:
            yield value
        for key in ("files", "outputs", "artifacts"):
            nested = value.get(key)
            if isinstance(nested, list):
                for item in nested:
                    yield from _iter_output_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_output_records(item)


def _path_under_root(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Acquisition output is outside the configured Lustre root: {resolved}") from exc


def _rebase_output_path(path_value: Any, root: Path, recorded_root: Path | None) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        return (root / path).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
        return resolved
    except ValueError:
        pass
    if recorded_root is not None:
        try:
            relative = resolved.relative_to(recorded_root)
            return (root / relative).resolve()
        except ValueError:
            pass
    return resolved


def _artifact_record(
    record: dict[str, Any],
    root: Path,
    *,
    verify_hash: bool,
    require_hash: bool = False,
    recorded_root: Path | None = None,
) -> dict[str, Any]:
    path = _rebase_output_path(record["local_path"], root, recorded_root)
    relative = _path_under_root(path, root)
    if not path.is_file():
        raise RuntimeError(f"Acquisition output is missing or is not a file: {path}")
    actual_size = path.stat().st_size
    recorded_size = record.get("size")
    if recorded_size is not None and int(recorded_size) != actual_size:
        raise RuntimeError(f"Acquisition output size changed: {path}")
    recorded_hash = str(record.get("sha256") or "")
    actual_hash = (
        _sha256_file(path)
        if verify_hash or (require_hash and not recorded_hash)
        else recorded_hash
    )
    if recorded_hash and verify_hash and recorded_hash != actual_hash:
        raise RuntimeError(f"Acquisition output hash changed: {path}")
    return {
        "path": str(relative),
        "size": actual_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": actual_hash,
        "kind": str(record.get("kind", "file")),
        "source_id": record.get("source_id"),
    }


def _validate_outputs(
    profile: dict[str, Any],
    state: StateStore,
    *,
    verify_hashes: bool,
    require_hashes: bool = False,
    recorded_root: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    root = Path(profile["storage"]["lustre_root"]).resolve()
    lock_path = state.path("sources.lock.json")
    if not lock_path.is_file():
        raise RuntimeError("sources.lock.json is missing")
    lock = state.read("sources.lock.json")
    completion_digests: list[dict[str, str]] = []
    artifacts: list[dict[str, Any]] = []
    for task in lock.get("download_tasks", []):
        task_id = str(task["task_id"])
        completion_path = state.path("completed", "download", f"{task_id}.json")
        if not completion_path.is_file():
            raise RuntimeError(f"Acquisition task is incomplete: {task_id}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if task.get("task_sha256") and completion.get("task_sha256") != task.get("task_sha256"):
            raise RuntimeError(f"Acquisition completion does not match its immutable task: {task_id}")
        completion_digests.append({"task_id": task_id, "sha256": _sha256_file(completion_path)})
        outputs = list(_iter_output_records(completion.get("files", [])))
        if not outputs:
            raise RuntimeError(f"Acquisition task has no materialized outputs: {task_id}")
        for output in outputs:
            if output.get("kind") == "remote_source_plan" or output.get("materialized") is False:
                raise RuntimeError(f"Acquisition task still contains an unresolved remote plan: {task_id}")
            if output.get("ready_for_training_build") is False:
                raise RuntimeError(f"Acquisition output is not ready for the CPU build: {task_id}")
            artifacts.append(
                _artifact_record(
                    output,
                    root,
                    verify_hash=verify_hashes,
                    require_hash=require_hashes,
                    recorded_root=recorded_root,
                )
            )
    return artifacts, _json_digest(completion_digests)


def _iter_candidate_estimates(value: Any) -> Iterator[tuple[str, int, str]]:
    """Yield one conservative candidate-token estimate per physical output.

    Retrieval indexes are deliberately excluded: repository metadata is not
    source code and can never satisfy the corresponding source target.
    """

    if isinstance(value, list):
        for item in value:
            yield from _iter_candidate_estimates(item)
        return
    if not isinstance(value, dict):
        return
    if value.get("payload_role") in INDEX_PAYLOAD_ROLES:
        return
    source_id = str(value.get("source_id") or "")
    estimate = value.get("candidate_token_estimate")
    estimator = str(value.get("candidate_estimator") or "")
    if estimate is None and value.get("estimated_tokens") is not None:
        estimate = value.get("estimated_tokens")
        estimator = estimator or "materializer_estimated_tokens"
    if estimate is None and value.get("text_bytes") is not None:
        estimate = int(value.get("text_bytes", 0)) // 4
        estimator = estimator or "accepted_utf8_text_bytes_divided_by_4"
    if source_id and estimate is not None:
        yield source_id, max(0, int(estimate)), estimator or "unspecified_conservative_estimator"
        # Dataset wrappers already account for their shards.
        if value.get("kind") == "materialized_dataset":
            return
    for key in ("files", "outputs", "artifacts"):
        yield from _iter_candidate_estimates(value.get(key))


def _iter_materialized_datasets(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_materialized_datasets(item)
        return
    if not isinstance(value, dict):
        return
    if value.get("kind") == "materialized_dataset":
        yield value
        return
    for key in ("files", "outputs", "artifacts"):
        yield from _iter_materialized_datasets(value.get(key))


def _final_opt_out_candidate_estimates(
    profile: dict[str, Any],
    state: StateStore,
    policy: OptOutPolicy,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Recount FreshWeb candidates against the handoff-frozen opt-out policy."""

    import zstandard as zstd

    root = Path(profile["storage"]["lustre_root"]).resolve()
    estimates: dict[str, int] = {}
    removals: dict[str, dict[str, int]] = {}
    lock = state.read("sources.lock.json", default={})
    for task in lock.get("download_tasks", []):
        completion = state.read(
            "completed", "download", f"{task['task_id']}.json", default={}
        )
        for dataset in _iter_materialized_datasets(completion.get("files", [])):
            if dataset.get("driver") != "common_crawl_ranges":
                continue
            source_id = str(dataset.get("source_id") or "")
            if not source_id:
                raise RuntimeError("Common Crawl materialized dataset has no source_id")
            kept_tokens = 0
            removed_tokens = 0
            removed_records = 0
            for shard in dataset.get("shards", []):
                shard_path = Path(str(shard.get("path") or "")).resolve()
                _path_under_root(shard_path, root)
                if not shard_path.is_file():
                    raise RuntimeError(
                        f"Common Crawl shard is missing during final opt-out recount: {shard_path}"
                    )
                with shard_path.open("rb") as raw:
                    with zstd.ZstdDecompressor().stream_reader(raw) as stream:
                        with io.TextIOWrapper(stream, encoding="utf-8") as handle:
                            for line in handle:
                                if not line.strip():
                                    continue
                                row = json.loads(line)
                                text = str(row.get("text") or "")
                                metadata = row.get("metadata") or {}
                                url = str(
                                    metadata.get("url")
                                    or metadata.get("original_url")
                                    or metadata.get("canonical_url")
                                    or ""
                                )
                                tokens = max(1, len(text.encode("utf-8")) // 4)
                                if url and policy.reason(url):
                                    removed_tokens += tokens
                                    removed_records += 1
                                else:
                                    kept_tokens += tokens
            estimates[source_id] = estimates.get(source_id, 0) + kept_tokens
            report = removals.setdefault(
                source_id, {"removed_tokens": 0, "removed_records": 0}
            )
            report["removed_tokens"] += removed_tokens
            report["removed_records"] += removed_records
    return estimates, removals


def _validate_materialized_token_targets(
    lock: dict[str, Any],
    state: StateStore,
    manifest: dict[str, Any],
    *,
    final_opt_out_policy: OptOutPolicy | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require conservative candidate headroom for every source and category."""

    expected = {
        str(source["id"]): int(source.get("candidate_tokens", 0))
        for source in lock.get("sources", [])
    }
    drivers = {
        str(source["id"]): str(source.get("driver") or "")
        for source in lock.get("sources", [])
    }
    actual: dict[str, int] = {}
    estimators: dict[str, set[str]] = {}
    for task in lock.get("download_tasks", []):
        completion = state.read("completed", "download", f"{task['task_id']}.json", default={})
        for source_id, tokens, estimator in _iter_candidate_estimates(completion.get("files", [])):
            actual[source_id] = actual.get(source_id, 0) + tokens
            estimators.setdefault(source_id, set()).add(estimator)
    opt_out_removals: dict[str, dict[str, int]] = {}
    if final_opt_out_policy is not None:
        if profile is None:
            raise ValueError("profile is required for a final Common Crawl opt-out recount")
        adjusted, opt_out_removals = _final_opt_out_candidate_estimates(
            profile, state, final_opt_out_policy
        )
        for source_id, tokens in adjusted.items():
            actual[source_id] = tokens
            estimators[source_id] = {
                "final_opt_out_eligible_utf8_text_bytes_divided_by_4"
            }
    replacement_allocation: dict[str, Any] | None = None
    if manifest.get("replacement_policy"):
        replay = replay_quotas(manifest)
        unique = unique_quotas(manifest, replay)
        unique_requirements = {
            source_id: sum(int(tokens) for tokens in phases.values())
            for source_id, phases in unique.items()
        }
        replacement_allocation = allocate_replacements(
            manifest,
            requirements=unique,
            available_tokens=actual,
        )
        received: dict[str, int] = {}
        donated: dict[str, int] = {}
        for transfer in replacement_allocation["transfers"]:
            target_id = str(transfer["target_source_id"])
            donor_id = str(transfer["actual_source_id"])
            tokens = int(transfer["tokens"])
            received[target_id] = received.get(target_id, 0) + tokens
            donated[donor_id] = donated.get(donor_id, 0) + tokens
        report = {
            source_id: {
                "driver": drivers.get(source_id, ""),
                "candidate_token_target": expected.get(source_id, 0),
                "final_unique_requirement": unique_requirements[source_id],
                "estimated_materialized_tokens": actual.get(source_id, 0),
                "estimators": sorted(estimators.get(source_id, set())),
                "final_opt_out_removals": opt_out_removals.get(
                    source_id, {"removed_tokens": 0, "removed_records": 0}
                ),
                "own_unique_requirement_met": (
                    actual.get(source_id, 0) >= unique_requirements[source_id]
                ),
                "replacement_tokens_received": received.get(source_id, 0),
                "reserve_tokens_donated": donated.get(source_id, 0),
                "target_met": not bool(
                    replacement_allocation.get("unresolved", {}).get(source_id)
                ),
            }
            for source_id in sorted(unique_requirements)
        }
        category_targets = unique_requirements
    else:
        report = {
            source_id: {
                "driver": drivers.get(source_id, ""),
                "candidate_token_target": target,
                "estimated_materialized_tokens": actual.get(source_id, 0),
                "estimators": sorted(estimators.get(source_id, set())),
                "final_opt_out_removals": opt_out_removals.get(
                    source_id, {"removed_tokens": 0, "removed_records": 0}
                ),
                "target_met": actual.get(source_id, 0) >= target,
            }
            for source_id, target in sorted(expected.items())
        }
        short = [
            source_id for source_id, row in report.items() if not row["target_met"]
        ]
        if short:
            detail = ", ".join(
                f"{source_id}={report[source_id]['estimated_materialized_tokens']:,}/"
                f"{report[source_id]['candidate_token_target']:,}"
                for source_id in short
            )
            raise RuntimeError(f"Acquisition candidate-token target shortfall: {detail}")
        category_targets = expected

    source_categories = {
        str(source["id"]): str(source["category"])
        for source in manifest.get("sources", [])
    }
    category_report: dict[str, dict[str, Any]] = {}
    for source_id, target in category_targets.items():
        category = source_categories.get(source_id)
        if not category:
            raise RuntimeError(f"Source lock contains an unknown manifest source: {source_id}")
        row = category_report.setdefault(
            category,
            {"candidate_token_target": 0, "estimated_materialized_tokens": 0},
        )
        row["candidate_token_target"] += target
        row["estimated_materialized_tokens"] += actual.get(source_id, 0)
    for row in category_report.values():
        row["target_met"] = (
            int(row["estimated_materialized_tokens"])
            >= int(row["candidate_token_target"])
        )
    category_short = [name for name, row in category_report.items() if not row["target_met"]]
    if category_short:
        raise RuntimeError(
            "Acquisition category candidate-token shortfall: " + ", ".join(category_short)
        )
    return {
        "sources": report,
        "categories": dict(sorted(category_report.items())),
        "replacement_allocation": replacement_allocation,
    }


def _uses_common_crawl(manifest: dict[str, Any]) -> bool:
    """Whether any source needs the frozen publisher opt-out snapshot.

    This has to match what normalization actually demands, and for a while it
    did not. stage_runner asks for the snapshot whenever a source declares
    `provenance.common_crawl_derived`, deliberately -- a packaged extraction
    such as FreshWeb is Common Crawl text a third party filtered at build time,
    which makes it more exposed to a later withdrawal than a fresh crawl, not
    less. This function still only looked at the driver, so it wrote the
    snapshot for `common_crawl_ranges` sources alone.

    The disagreement stayed invisible while some source used that driver. When
    the common_crawl_ranges sources were withdrawn it became load-bearing: no
    source had the driver, so no snapshot was written, while
    metis_freshweb_2025 still declared common_crawl_derived and its 34
    normalization tasks failed on the missing snapshot. Every archived handoff
    on disk lacks the block for this reason.
    """

    return any(
        source.get("acquisition", {}).get("driver") == "common_crawl_ranges"
        or bool((source.get("provenance") or {}).get("common_crawl_derived"))
        for source in manifest.get("sources", [])
    )


def _snapshot_final_common_crawl_policy(
    profile: dict[str, Any], manifest: dict[str, Any], root: Path
) -> dict[str, Any] | None:
    """Freeze the latest publisher opt-outs immediately before handoff.

    FreshWeb applies a live snapshot during each WARC run. This second snapshot
    closes the acquisition-time race: publishers who opted out while the long
    download was running are removed when Rhea normalizes the materialized
    records.
    """

    if not _uses_common_crawl(manifest):
        return None
    destination = (
        root
        / profile["storage"]["directories"]["contamination"]
        / "common-crawl-opt-out"
    )
    pointer = snapshot_common_crawl_opt_out(destination)
    latest_path = destination / "LATEST_COMMON_CRAWL_OPT_OUT.json"
    records: dict[str, dict[str, Any]] = {}
    for name, raw_path, expected_sha in (
        ("snapshot", pointer["path"], pointer["sha256"]),
        ("rules", pointer["rules_path"], pointer["rules_sha256"]),
        ("metadata", pointer["metadata_path"], None),
        ("latest", latest_path, None),
    ):
        path = Path(str(raw_path)).resolve()
        relative = _path_under_root(path, root)
        if not path.is_file():
            raise RuntimeError(f"Common Crawl opt-out {name} artifact is missing: {path}")
        digest = _sha256_file(path)
        if expected_sha and digest != expected_sha:
            raise RuntimeError(f"Common Crawl opt-out {name} checksum changed: {path}")
        records[name] = {
            "path": str(relative),
            "size": path.stat().st_size,
            "sha256": digest,
        }
    return {
        "parser_version": pointer["parser_version"],
        "source": pointer["source"],
        "last_updated": pointer.get("last_updated"),
        "domains": int(pointer["domains"]),
        "url_paths": int(pointer["url_paths"]),
        "url_rules": int(pointer["url_rules"]),
        "input_entries": int(pointer["input_entries"]),
        "unparsed_entries": int(pointer["unparsed_entries"]),
        "artifacts": records,
        "normalization_reapplication_required": True,
    }


def _verify_final_common_crawl_policy(
    handoff: dict[str, Any],
    manifest: dict[str, Any],
    root: Path,
    *,
    verify_hashes: bool,
) -> None:
    required = _uses_common_crawl(manifest)
    policy = handoff.get("common_crawl_opt_out")
    if required and not isinstance(policy, dict):
        raise RuntimeError("Final Common Crawl opt-out snapshot is missing from the acquisition handoff")
    if not policy:
        return
    if policy.get("normalization_reapplication_required") is not True:
        raise RuntimeError("Common Crawl opt-out reapplication is not required by the handoff")
    artifacts = policy.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"snapshot", "rules", "metadata", "latest"}:
        raise RuntimeError("Common Crawl opt-out handoff artifacts are incomplete")
    for name, record in artifacts.items():
        path = (root / str(record.get("path", ""))).resolve()
        _path_under_root(path, root)
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size", -1))
            or (verify_hashes and _sha256_file(path) != record.get("sha256"))
        ):
            raise RuntimeError(f"Common Crawl opt-out {name} artifact changed after handoff")


def write_acquisition_handoff(
    profile: dict[str, Any], manifest: dict[str, Any], state: StateStore
) -> dict[str, Any]:
    existing = state.read("ACQUISITION_READY.json")
    if existing:
        verify_acquisition_handoff(profile, manifest, state)
        return existing
    root = Path(profile["storage"]["lustre_root"]).resolve()
    holdouts = root / profile["storage"]["directories"]["contamination"] / "holdouts.jsonl"
    holdout_report = root / profile["storage"]["directories"]["contamination"] / "HOLDOUTS.json"
    if not holdouts.is_file():
        raise RuntimeError("Evaluation holdouts are missing; acquisition cannot be handed to the CPU build")
    if not holdout_report.is_file():
        raise RuntimeError("Evaluation holdout provenance report is missing")
    repository = _repository_state()
    if profile.get("gates", {}).get("require_clean_repository") and repository["dirty"]:
        raise RuntimeError("Refusing acquisition handoff from a dirty repository checkout")
    # Download workers already verified and recorded upstream checksums. Reuse
    # those hashes here; compute a hash only for a materializer that omitted one.
    # This avoids rereading terabytes immediately after acquisition.
    artifacts, completion_digest = _validate_outputs(
        profile,
        state,
        verify_hashes=False,
        require_hashes=True,
    )
    lock_path = state.path("sources.lock.json")
    lock = validate_source_lock_integrity(state.read("sources.lock.json"))
    manifest_source_ids = {str(source["id"]) for source in manifest.get("sources", [])}
    lock_source_ids = {str(source.get("id") or "") for source in lock.get("sources", [])}
    if lock_source_ids != manifest_source_ids:
        raise RuntimeError(
            "The source lock source inventory differs from the active manifest: "
            f"missing={sorted(manifest_source_ids - lock_source_ids)}, "
            f"unexpected={sorted(lock_source_ids - manifest_source_ids)}"
        )
    active_runtime_contract = runtime_contract()
    if lock.get("runtime_contract") != active_runtime_contract:
        raise RuntimeError(
            "The source lock is not bound to the active hash-locked Python runtime"
        )
    if lock.get("manifest_sha256") and lock.get("manifest_sha256") != _manifest_digest(manifest):
        raise RuntimeError("The source lock is bound to a different data manifest")
    # The lock's commit is provenance. Its binding to the manifest and the
    # pinned runtime, checked directly above, is what decides which bytes the
    # downloads contain; the commit alone does not. Enforcing it here meant the
    # handoff could not be rebuilt after any commit, which is how a build gets
    # stranded between a lock it cannot reuse and inputs it cannot re-verify.
    common_crawl_opt_out = _snapshot_final_common_crawl_policy(profile, manifest, root)
    final_opt_out_policy: OptOutPolicy | None = None
    if common_crawl_opt_out is not None:
        snapshot_record = common_crawl_opt_out["artifacts"]["snapshot"]
        snapshot_path = (root / str(snapshot_record["path"])).resolve()
        final_opt_out_policy = parse_opt_out_registry(snapshot_path.read_bytes())
        if final_opt_out_policy.snapshot_sha256 != snapshot_record["sha256"]:
            raise RuntimeError("Final Common Crawl opt-out snapshot changed before recount")
    materialized_token_targets = _validate_materialized_token_targets(
        lock,
        state,
        manifest,
        final_opt_out_policy=final_opt_out_policy,
        profile=profile,
    )
    payload = {
        "schema": HANDOFF_SCHEMA,
        "created_at": utc_now(),
        "release": manifest["release"],
        "lustre_root": str(root),
        "manifest_sha256": _manifest_digest(manifest),
        "source_lock_sha256": _sha256_file(lock_path),
        "runtime_contract": active_runtime_contract,
        "acquisition_runtime": lock.get("resolver_runtime", runtime_identity()),
        "completion_markers_sha256": completion_digest,
        "holdouts": {
            "path": str(holdouts.relative_to(root)),
            "size": holdouts.stat().st_size,
            "sha256": _sha256_file(holdouts),
            "report_path": str(holdout_report.relative_to(root)),
            "report_size": holdout_report.stat().st_size,
            "report_sha256": _sha256_file(holdout_report),
        },
        "repository": repository,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(int(item["size"]) for item in artifacts),
        "materialized_token_targets": materialized_token_targets,
        "common_crawl_opt_out": common_crawl_opt_out,
    }
    payload["handoff_sha256"] = _json_digest(payload)
    state.write("ACQUISITION_READY.json", payload=payload)
    return payload


def verify_acquisition_handoff(
    profile: dict[str, Any],
    manifest: dict[str, Any],
    state: StateStore,
    *,
    verify_artifact_hashes: bool = False,
) -> dict[str, Any]:
    handoff = state.read("ACQUISITION_READY.json")
    if not handoff:
        raise RuntimeError("ACQUISITION_READY.json is missing; finish acquisition on login2 first")
    if handoff.get("schema") != HANDOFF_SCHEMA:
        raise RuntimeError(f"Unsupported acquisition handoff schema: {handoff.get('schema')}")
    expected_handoff_hash = handoff.get("handoff_sha256")
    unsigned = {key: value for key, value in handoff.items() if key != "handoff_sha256"}
    if expected_handoff_hash != _json_digest(unsigned):
        raise RuntimeError("ACQUISITION_READY.json failed its self-hash check")
    root = Path(profile["storage"]["lustre_root"]).resolve()
    recorded_root = Path(str(handoff.get("lustre_root", ""))).expanduser().resolve()
    relocated = recorded_root != root
    if relocated and not profile.get("gates", {}).get("allow_relocated_lustre_root", False):
        raise RuntimeError(
            f"Rhea sees a different Lustre path ({root}) than login2 recorded ({handoff.get('lustre_root')})"
        )
    if handoff.get("release") != manifest.get("release"):
        raise RuntimeError("Acquisition release does not match this build profile")
    if handoff.get("manifest_sha256") != _manifest_digest(manifest):
        raise RuntimeError("The data manifest changed after acquisition")
    active_runtime_contract = runtime_contract()
    if handoff.get("runtime_contract") != active_runtime_contract:
        raise RuntimeError("The hash-locked Python runtime changed after acquisition")
    _verify_final_common_crawl_policy(
        handoff,
        manifest,
        root,
        verify_hashes=verify_artifact_hashes,
    )
    lock_path = state.path("sources.lock.json")
    if not lock_path.is_file() or handoff.get("source_lock_sha256") != _sha256_file(lock_path):
        raise RuntimeError("The immutable source lock changed after acquisition")
    lock = state.read("sources.lock.json")
    if lock.get("runtime_contract") != active_runtime_contract:
        raise RuntimeError("The immutable source lock has a different Python runtime contract")
    artifacts, completion_digest = _validate_outputs(
        profile,
        state,
        verify_hashes=verify_artifact_hashes,
        require_hashes=False,
        recorded_root=recorded_root,
    )
    if handoff.get("completion_markers_sha256") != completion_digest:
        raise RuntimeError("Download completion markers changed after acquisition")
    def artifact_identity(item: dict[str, Any]) -> tuple[Any, ...]:
        base: tuple[Any, ...] = (
            str(item["path"]),
            int(item["size"]),
            int(item.get("mtime_ns", 0)),
        )
        return base + ((str(item["sha256"]),) if verify_artifact_hashes else ())

    expected_artifacts = {artifact_identity(item) for item in handoff.get("artifacts", [])}
    actual_artifacts = {artifact_identity(item) for item in artifacts}
    if expected_artifacts != actual_artifacts:
        raise RuntimeError("Materialized acquisition artifacts changed after handoff")
    if not verify_artifact_hashes:
        expected_hashes = {
            (str(item["path"]), str(item["sha256"]))
            for item in handoff.get("artifacts", [])
            if item.get("sha256")
        }
        recorded_hashes = {
            (str(item["path"]), str(item["sha256"]))
            for item in artifacts
            if item.get("sha256")
        }
        if not recorded_hashes <= expected_hashes:
            raise RuntimeError("Materialized acquisition checksums changed after handoff")
    holdout = handoff["holdouts"]
    holdout_path = root / holdout["path"]
    if (
        not holdout_path.is_file()
        or holdout_path.stat().st_size != int(holdout["size"])
        or (
            verify_artifact_hashes
            and _sha256_file(holdout_path) != holdout["sha256"]
        )
    ):
        raise RuntimeError("Evaluation holdouts changed after acquisition")
    holdout_report_path = root / str(holdout.get("report_path", ""))
    if (
        not holdout_report_path.is_file()
        or holdout_report_path.stat().st_size != int(holdout.get("report_size", -1))
        or (
            verify_artifact_hashes
            and _sha256_file(holdout_report_path) != holdout.get("report_sha256")
        )
    ):
        raise RuntimeError("Evaluation holdout provenance report changed after acquisition")
    if profile.get("gates", {}).get("require_repository_commit_match"):
        from .stage_code import acquisition_code_sha256

        current = _repository_state()
        expected = handoff.get("repository", {})
        if current["dirty"]:
            raise RuntimeError("Rhea checkout is dirty; refusing to build against uncommitted code")
        # The property worth protecting is that the acquisition artifacts were
        # produced by the acquisition code now on disk -- not that nothing at
        # all has been committed since. Comparing bare commits made every
        # unrelated fix look like tampering. Handoffs written before this
        # fingerprint existed fall back to the commit comparison.
        recorded_fingerprint = expected.get("acquisition_code_sha256")
        if recorded_fingerprint is None:
            if current["commit"] != expected.get("commit"):
                raise RuntimeError(
                    "Rhea checkout does not match the clean acquisition repository commit"
                )
        elif recorded_fingerprint != acquisition_code_sha256():
            raise RuntimeError(
                "The acquisition code changed after the downloads were produced; "
                "re-run acquisition so the handoff matches the code that made it"
            )
    return {
        "ok": True,
        "schema": HANDOFF_SCHEMA,
        "handoff_sha256": expected_handoff_hash,
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(int(item["size"]) for item in artifacts),
        "verified_artifact_hashes": verify_artifact_hashes,
        "login2_lustre_root": str(recorded_root),
        "current_lustre_root": str(root),
        "relocated_mount": relocated,
        "runtime_contract": active_runtime_contract,
        "verification_runtime": runtime_identity(),
    }
