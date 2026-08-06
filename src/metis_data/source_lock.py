from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import EntryNotFoundError

from .acquisition.github import month_windows
from .manifest import candidate_plan, matches_any, total_phase_tokens
from .replacement import allocate_replacements, replacement_chains
from .selection import replay_quotas, unique_quotas
from .runtime_lock import runtime_contract, runtime_identity
from .state import StateStore, utc_now


SOURCE_LOCK_SCHEMA = "metis.source-lock/v4"
SOURCE_LOCK_RESOLVER_VERSION = "metis-source-resolver-2026-07-24-v5"


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_sha256(manifest: dict[str, Any]) -> str:
    def portable(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items() if key != "_path"}
        if isinstance(value, list):
            return [portable(item) for item in value]
        return value

    return _json_sha256(portable(manifest))


def source_lock_sha256(lock: dict[str, Any]) -> str:
    """Return the canonical self-hash for a source lock.

    The self-hash protects the complete lock payload, while each task hash keeps
    task identity independently auditable and content-addressed.
    """

    unsigned = {key: value for key, value in lock.items() if key != "lock_sha256"}
    return _json_sha256(unsigned)


def validate_source_lock_integrity(lock: dict[str, Any]) -> dict[str, Any]:
    if lock.get("schema") != SOURCE_LOCK_SCHEMA:
        raise RuntimeError(
            "An older or unknown sources.lock.json already exists. Never overwrite acquisition state in place; "
            "use a new data release/root or explicitly archive the old state after review."
        )
    recorded_lock_hash = str(lock.get("lock_sha256") or "")
    if recorded_lock_hash != source_lock_sha256(lock):
        raise RuntimeError("The immutable source lock failed its whole-lock self-hash check")
    tasks = lock.get("download_tasks")
    if not isinstance(tasks, list):
        raise RuntimeError("The immutable source lock has no valid download task list")
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise RuntimeError(f"The immutable source lock contains an invalid task at index {index}")
        items = task.get("items")
        planned_bytes = task.get("planned_bytes")
        if not isinstance(items, list) or (
            not isinstance(planned_bytes, int)
            or isinstance(planned_bytes, bool)
            or planned_bytes < 0
        ):
            raise RuntimeError(
                f"The immutable source lock contains an invalid task payload at index {index}"
            )
        expected_task_hash = _json_sha256(
            {
                "items": items,
                "planned_bytes": planned_bytes,
            }
        )
        expected_task_id = f"download-{index:06d}-{expected_task_hash[:16]}"
        if (
            type(task.get("task_index")) is not int
            or task.get("task_index") != index
            or task.get("task_sha256") != expected_task_hash
            or task.get("task_id") != expected_task_id
        ):
            raise RuntimeError(
                f"The immutable source lock contains a task identity mismatch at index {index}"
            )
    return lock


def _repository_commit() -> tuple[str, bool]:
    try:
        root = Path(__file__).resolve().parents[2]
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
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _validate_existing_lock(
    existing: dict[str, Any], manifest: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    validate_source_lock_integrity(existing)
    if existing.get("release") != manifest.get("release"):
        raise RuntimeError("The existing source lock belongs to a different data release")
    expected_manifest = _manifest_sha256(manifest)
    if existing.get("manifest_sha256") != expected_manifest:
        raise RuntimeError("The data manifest changed after the immutable source lock was created")
    expected_runtime = runtime_contract()
    if existing.get("runtime_contract") != expected_runtime:
        raise RuntimeError(
            "The hash-locked Python runtime changed after the immutable source lock was created"
        )
    commit, dirty = _repository_commit()
    if profile.get("gates", {}).get("require_clean_repository") and dirty:
        raise RuntimeError("Refusing to reuse the source lock from a dirty repository checkout")
    # The commit is recorded as provenance, not enforced as identity. What the
    # lock must still agree with is the manifest and the pinned runtime, both
    # checked above; those are what decide which bytes get downloaded. Refusing
    # on the commit alone meant no code fix could ever be deployed into a
    # running build, because re-resolving at the new commit rewrote the lock and
    # invalidated every completed stage. Stage-level code identity is enforced
    # where it belongs, per stage, by stage_code_sha256.
    return existing


def _stable_key(seed: int, repo_id: str, path: str) -> str:
    return hashlib.sha256(f"{seed}\0{repo_id}\0{path}".encode()).hexdigest()


def _iter_repo_files(
    api: HfApi,
    repo_id: str,
    revision: str,
    patterns: Iterable[str],
) -> list[dict[str, Any]]:
    pattern_list = tuple(str(pattern) for pattern in patterns)
    files_by_path: dict[str, dict[str, Any]] = {}
    for tree_root in _repo_tree_roots(pattern_list):
        try:
            tree = api.list_repo_tree(
                repo_id,
                path_in_repo=tree_root,
                repo_type="dataset",
                revision=revision,
                recursive=True,
                # expand=True is deliberately not requested. It adds only
                # lastCommit and securityFileStatus, neither of which this
                # resolver reads, and the Hub caps an expanded tree page at 50
                # entries instead of 1000 -- twenty times the HTTP round trips
                # for the same listing. On a login node whose egress is
                # inspected by an endpoint-security agent, per-request latency
                # dominates, and that multiplier turned this preflight into a
                # multi-hour metadata transfer. Every field the lock records
                # (path, size, oid, and the LFS sha256) is returned either way,
                # so the resolved lock is byte-identical.
                expand=False,
            )
            for item in tree:
                path = getattr(item, "path", "")
                if not path or not matches_any(path, pattern_list):
                    continue
                size = int(getattr(item, "size", 0) or 0)
                if size <= 0:
                    continue
                lfs = getattr(item, "lfs", None)
                lfs_sha256 = (
                    lfs.get("sha256")
                    if isinstance(lfs, dict)
                    else getattr(lfs, "sha256", None)
                )
                files_by_path[path] = {
                    "path": path,
                    "size": size,
                    "blob_id": getattr(item, "blob_id", None),
                    "lfs_sha256": lfs_sha256,
                }
        except EntryNotFoundError:
            # A stale literal prefix should produce the resolver's normal
            # fail-closed "no files matching" error, not an opaque Hub 404.
            continue
    return [files_by_path[path] for path in sorted(files_by_path)]


def _repo_tree_roots(patterns: Iterable[str]) -> tuple[str | None, ...]:
    """Return the smallest safe Hub subtrees covering the allow-patterns.

    Hugging Face's recursive root listing can contain tens of thousands of
    entries. A source such as TxT360 already declares a literal partition
    prefix, so walking the entire repository is both unnecessary and can turn
    a preflight into a multi-hour metadata transfer.
    """

    roots: set[str | None] = set()
    for raw_pattern in patterns:
        pattern = str(raw_pattern).strip("/")
        segments = pattern.split("/") if pattern else []
        literal_segments: list[str] = []
        saw_glob = False
        for segment in segments:
            if any(character in segment for character in "*?["):
                saw_glob = True
                break
            literal_segments.append(segment)
        if not literal_segments:
            roots.add(None)
            continue
        if not saw_glob:
            # An exact pattern may name a file; listing its parent is safe.
            literal_segments = literal_segments[:-1]
        roots.add("/".join(literal_segments) or None)

    if not roots or None in roots:
        return (None,)
    ordered = sorted(roots, key=lambda value: (len(str(value).split("/")), str(value)))
    minimal: list[str] = []
    for root in ordered:
        assert root is not None
        if any(root == parent or root.startswith(parent + "/") for parent in minimal):
            continue
        minimal.append(root)
    return tuple(minimal)


def _select_files(
    files: list[dict[str, Any]],
    *,
    seed: int,
    repo_id: str,
    target_bytes: int,
    take_all: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    ordered = sorted(files, key=lambda item: _stable_key(seed, repo_id, item["path"]))
    if take_all:
        return ordered, sum(int(item["size"]) for item in ordered) < target_bytes
    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    for item in ordered:
        selected.append(item)
        selected_bytes += item["size"]
        if selected_bytes >= target_bytes:
            break
    return selected, selected_bytes < target_bytes


def _hf_accesses(source: dict[str, Any]) -> list[dict[str, Any]]:
    access = source["access"]
    if access.get("type") == "huggingface":
        return [access]
    if access.get("type") == "repository_index":
        return [
            {
                "type": "huggingface",
                "repo_id": component["repo_id"],
                "revision": component["revision"],
                "gated": component.get("gated", False),
                "allow_patterns": component.get(
                    "allow_patterns", ["**/*.parquet", "**/*.jsonl", "**/*.jsonl.zst"]
                ),
                "take_all": True,
            }
            for component in access.get("components", [])
        ]
    return []


GITHUB_DRIVERS = {"github_repositories", "github_discussions"}


def _apportion_integer(total: int, weights: list[int]) -> list[int]:
    if total < 0 or not weights or any(weight <= 0 for weight in weights):
        raise ValueError("Apportionment requires a non-negative total and positive weights")
    denominator = sum(weights)
    values = [(total * weight) // denominator for weight in weights]
    remainder_order = sorted(
        range(len(weights)),
        key=lambda index: (-(total * weights[index] % denominator), index),
    )
    for index in remainder_order[: total - sum(values)]:
        values[index] += 1
    return values


def _github_partition_items(source: dict[str, Any], source_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Create one credential-free, deterministic builder item per calendar month."""

    driver = str(source["acquisition"]["driver"])
    if driver not in GITHUB_DRIVERS:
        raise ValueError(f"Not a GitHub acquisition driver: {driver}")
    access = source["access"]
    # Whitelist the only fields needed by the materializer. A token/header
    # accidentally added to a manifest or caller can never enter the source
    # lock, task report, or log through this structure.
    public_access = {
        "type": str(access.get("type", "github_public")),
        "cutoff_start": str(access["cutoff_start"]),
        "cutoff_end": str(access["cutoff_end"]),
    }
    windows = month_windows(public_access["cutoff_start"], public_access["cutoff_end"])
    days = [(end - start).days + 1 for start, end in windows]
    token_targets = _apportion_integer(int(source_plan["candidate_tokens"]), days)
    byte_targets = _apportion_integer(int(source_plan["planned_download_bytes"]), days)
    if any(target <= 0 for target in token_targets):
        raise RuntimeError(f"{source['id']} has too few candidate tokens for monthly GitHub partitions")
    items: list[dict[str, Any]] = []
    for index, ((start, end), candidate_tokens, planned_bytes) in enumerate(
        zip(windows, token_targets, byte_targets, strict=True)
    ):
        partition = {
            "id": f"{start:%Y-%m}",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "ordinal": index,
            "total_partitions": len(windows),
            "days": days[index],
        }
        items.append(
            {
                "kind": "builder",
                "source_id": source["id"],
                "driver": driver,
                "access": public_access,
                "partition": partition,
                "planned_download_bytes": planned_bytes,
                "candidate_tokens": candidate_tokens,
            }
        )
    return items


def resolve_sources(manifest: dict[str, Any], profile: dict[str, Any], state: StateStore) -> dict[str, Any]:
    existing = state.read("sources.lock.json")
    if existing is not None:
        return _validate_existing_lock(existing, manifest, profile)
    repository_commit, repository_dirty = _repository_commit()
    resolved_runtime_contract = runtime_contract()
    resolver_runtime = runtime_identity()
    if profile.get("gates", {}).get("require_clean_repository") and repository_dirty:
        raise RuntimeError("Refusing to create the immutable source lock from a dirty repository checkout")
    api = HfApi(token=get_token())
    seed = int(manifest.get("selection", {}).get("seed", 0))
    plan_by_source = {row["id"]: row for row in candidate_plan(manifest)["sources"]}
    resolved_sources: list[dict[str, Any]] = []
    all_download_items: list[dict[str, Any]] = []

    for source in manifest["sources"]:
        source_id = source["id"]
        source_plan = plan_by_source[source_id]
        hf_accesses = _hf_accesses(source)
        resolved: dict[str, Any] = {
            "id": source_id,
            "driver": source["acquisition"]["driver"],
            "final_exposure_tokens": total_phase_tokens(source),
            "candidate_tokens": source_plan["candidate_tokens"],
            "planned_download_bytes": source_plan["planned_download_bytes"],
            "compressed_bytes_per_token": float(
                source["acquisition"].get("compressed_bytes_per_token", 0.8)
            ),
            "repositories": [],
        }
        if source["acquisition"]["driver"] in GITHUB_DRIVERS:
            github_items = _github_partition_items(source, source_plan)
            all_download_items.extend(github_items)
            resolved["partitions"] = [
                {
                    **item["partition"],
                    "candidate_tokens": item["candidate_tokens"],
                    "planned_download_bytes": item["planned_download_bytes"],
                }
                for item in github_items
            ]
        elif hf_accesses:
            per_repo_target = max(1, source_plan["planned_download_bytes"] // len(hf_accesses))
            for access in hf_accesses:
                repo_id = access["repo_id"]
                revision = access["revision"]
                info = api.dataset_info(repo_id, revision=revision, timeout=60)
                if info.sha != revision:
                    raise RuntimeError(f"Pinned revision drift for {repo_id}: expected {revision}, resolved {info.sha}")
                files = _iter_repo_files(
                    api,
                    repo_id,
                    revision,
                    access.get("allow_patterns", ["**/*.parquet", "**/*.jsonl*"]),
                )
                selected, short = _select_files(
                    files,
                    seed=seed,
                    repo_id=repo_id,
                    target_bytes=per_repo_target,
                    take_all=bool(access.get("take_all")),
                )
                if not selected:
                    raise RuntimeError(
                        f"Pinned Hugging Face source {repo_id}@{revision} has no files matching "
                        f"{access.get('allow_patterns', ['**/*.parquet', '**/*.jsonl*'])}"
                    )
                if short:
                    chains, _ = replacement_chains(manifest)
                    if not chains.get(source_id):
                        raise RuntimeError(
                            f"Pinned Hugging Face source {repo_id}@{revision} exposes only "
                            f"{sum(item['size'] for item in selected):,} matching bytes, below the "
                            f"{per_repo_target:,}-byte candidate target and has no compatible donor; "
                            "revise the manifest rather than underfill. "
                            f"take_all={bool(access.get('take_all'))!r} means select every matching file, not "
                            "permit an unresolved acquisition shortfall"
                        )
                resolved_repo = {
                    "repo_id": repo_id,
                    "revision": revision,
                    "gated": access.get("gated", False),
                    "selected_bytes": sum(item["size"] for item in selected),
                    "available_bytes": sum(item["size"] for item in files),
                    "candidate_shortfall_at_byte_estimate": short,
                    "files": selected,
                }
                resolved["repositories"].append(resolved_repo)
                for item in selected:
                    compressed_bytes_per_token = float(
                        source["acquisition"].get("compressed_bytes_per_token", 0.8)
                    )
                    all_download_items.append(
                        {
                            "kind": "hf_file",
                            "source_id": source_id,
                            "repo_id": repo_id,
                            "revision": revision,
                            "payload_role": (
                                "source_index"
                                if source["acquisition"]["driver"] == "repository_index"
                                else "training_records"
                            ),
                            "candidate_token_estimate": int(
                                int(item["size"]) / compressed_bytes_per_token
                            ),
                            "candidate_estimator": "compressed_bytes_divided_by_manifest_ratio",
                            **item,
                        }
                    )
            selected_source_bytes = sum(
                int(repository["selected_bytes"])
                for repository in resolved["repositories"]
            )
            compressed_bytes_per_token = float(
                source["acquisition"].get("compressed_bytes_per_token", 0.8)
            )
            resolved["estimated_candidate_tokens_at_source_lock"] = int(
                selected_source_bytes / compressed_bytes_per_token
            )
            resolved["candidate_target_met_at_source_lock"] = (
                resolved["estimated_candidate_tokens_at_source_lock"]
                >= int(source_plan["candidate_tokens"])
            )
            if source["acquisition"]["driver"] == "repository_index":
                # The Hugging Face payload is an immutable retrieval index, not
                # repository source code.  Keep a separate unresolved item in
                # the lock so neither status nor normalization can count index
                # rows as materialized training documents.
                all_download_items.append(
                    {
                        "kind": "builder",
                        "source_id": source_id,
                        "driver": "repository_index",
                        "access": source["access"],
                        "license": source["license"],
                        "planned_download_bytes": source_plan["planned_download_bytes"],
                        "candidate_tokens": source_plan["candidate_tokens"],
                    }
                )
        elif source["access"].get("type") == "derived":
            resolved["derived_from"] = source["access"].get("parents", [])
            # A parent reference describes a derivation recipe; it is not a
            # payload.  Record it as an unresolved materialization task until a
            # tested builder writes actual canonical records.
            all_download_items.append(
                {
                    "kind": "builder",
                    "source_id": source_id,
                    "driver": "derived_after_download",
                    "access": source["access"],
                    "license": source["license"],
                    "planned_download_bytes": source_plan["planned_download_bytes"],
                    "candidate_tokens": source_plan["candidate_tokens"],
                }
            )
        else:
            all_download_items.append(
                {
                    "kind": "builder",
                    "source_id": source_id,
                    "driver": source["acquisition"]["driver"],
                    "access": source["access"],
                    "license": source["license"],
                    "planned_download_bytes": source_plan["planned_download_bytes"],
                    "candidate_tokens": source_plan["candidate_tokens"],
                }
            )
        resolved_sources.append(resolved)

    replacement_feasibility: dict[str, Any] | None = None
    if manifest.get("replacement_policy") and manifest.get("schedule"):
        replay = replay_quotas(manifest)
        unique = unique_quotas(manifest, replay)
        source_lock_available: dict[str, int] = {}
        for source in resolved_sources:
            source_id = str(source["id"])
            if source.get("driver") == "hf_snapshot":
                source_lock_available[source_id] = int(
                    source.get("estimated_candidate_tokens_at_source_lock", 0)
                )
            else:
                # Builders such as repository hydration, canonical Git,
                # GitHub, and Common Crawl enforce materialized estimates at
                # handoff. Their target is the honest lock-time estimate.
                source_lock_available[source_id] = int(
                    source.get("candidate_tokens", 0)
                )
        replacement_feasibility = allocate_replacements(
            manifest,
            requirements=unique,
            available_tokens=source_lock_available,
        )

    target_task_bytes = int(profile.get("scheduler", {}).get("download", {}).get("target_bytes_per_task", 20_000_000_000))
    tasks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for item in all_download_items:
        item_bytes = int(item.get("size", item.get("planned_download_bytes", 0)))
        if current and current_bytes + item_bytes > target_task_bytes:
            tasks.append({"items": current, "planned_bytes": current_bytes})
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item_bytes
        if item.get("kind") == "builder" or current_bytes >= target_task_bytes:
            tasks.append({"items": current, "planned_bytes": current_bytes})
            current = []
            current_bytes = 0
    if current:
        tasks.append({"items": current, "planned_bytes": current_bytes})
    for index, task in enumerate(tasks):
        task["task_index"] = index
        task["task_sha256"] = _json_sha256(
            {"items": task["items"], "planned_bytes": task["planned_bytes"]}
        )
        task["task_id"] = f"download-{index:06d}-{task['task_sha256'][:16]}"

    lock = {
        "schema": SOURCE_LOCK_SCHEMA,
        "resolver_version": SOURCE_LOCK_RESOLVER_VERSION,
        "release": manifest["release"],
        "resolved_at": utc_now(),
        "source_manifest": manifest["_path"],
        "manifest_sha256": _manifest_sha256(manifest),
        "repository_commit": repository_commit,
        "runtime_contract": resolved_runtime_contract,
        "resolver_runtime": resolver_runtime,
        "sources": resolved_sources,
        "replacement_feasibility": replacement_feasibility,
        "download_tasks": tasks,
    }
    lock["lock_sha256"] = source_lock_sha256(lock)
    state.write("sources.lock.json", payload=lock)
    return lock
