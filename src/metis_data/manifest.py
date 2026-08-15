from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from .config import load_yaml, repository_root
from .normalization_evidence import validated_attestations


PHASES = ("phase_a", "phase_b", "phase_c")

# The freshness layer is checked against a constant, not merely against its own
# declared target, so it cannot shrink by arithmetic -- dropping a fresh source
# and rebalancing the remainder would otherwise still reconcile. Changing this
# number is the deliberate act of reducing how current the corpus is.
#
# 1.6 was planned at 90B across four buckets. Three were withdrawn: every fresh
# source used the common_crawl_ranges driver, which cannot run on this
# generation's acquisition host -- a login node whose ledger sits on Lustre with
# no node-local scratch, where eleven days yielded 184KiB of documents against
# the 90B target. General web survives only because FineWeb ships an already
# built Common Crawl extraction; no equivalent packaged corpus exists for recent
# software docs, science, or specifications. See manifests/sources/web.yaml.
# The freshness layer's size follows its sources; what is fixed is that it
# stays a real share of the corpus rather than shrinking to nothing.
MINIMUM_FRESHNESS_SHARE = 0.03
HEX40 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ValidationResult:
    manifest: dict[str, Any]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_valid(self) -> dict[str, Any]:
        if self.errors:
            joined = "\n - ".join(self.errors)
            raise ValueError(f"Metis data manifest validation failed:\n - {joined}")
        return self.manifest


def default_manifest_path() -> Path:
    return repository_root() / "manifests" / "metis-1.6.yaml"


def total_phase_tokens(item: dict[str, Any]) -> int:
    return sum(int(item.get("phase_tokens", {}).get(phase, 0)) for phase in PHASES)


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path or default_manifest_path()).expanduser().resolve()
    manifest = load_yaml(manifest_path)
    sources: list[dict[str, Any]] = []
    for relative in manifest.get("source_files", []):
        source_path = (manifest_path.parent / relative).resolve()
        source_payload = load_yaml(source_path)
        for source in source_payload.get("sources", []):
            source = dict(source)
            source["_manifest_file"] = str(source_path)
            sources.append(source)
    manifest["sources"] = sources
    replacement_path = manifest.get("replacement_policy_file")
    if replacement_path:
        replacement_payload = load_yaml((manifest_path.parent / replacement_path).resolve())
        manifest["replacement_policy"] = replacement_payload
    context_path = manifest.get("context_extension_plan_file")
    if context_path:
        from .context_manifest import load_context_plan

        manifest["context_extension_plan"] = load_context_plan(
            (manifest_path.parent / context_path).resolve(),
            base_manifest={**manifest, "sources": sources},
        )
    manifest["_path"] = str(manifest_path)
    return manifest


def _sum_by_phase(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    return {
        phase: sum(int(item.get("phase_tokens", {}).get(phase, 0)) for item in items)
        for phase in PHASES
    }


def validate_manifest(path: str | Path | None = None) -> ValidationResult:
    manifest = load_manifest(path)
    errors: list[str] = []
    warnings: list[str] = []
    schedule = manifest.get("schedule", {})
    target = int(schedule.get("target_tokens", 0))
    if manifest.get("schema") != "metis.data-manifest/v1":
        errors.append("schema must be metis.data-manifest/v1")
    if target <= 0:
        errors.append(f"schedule.target_tokens must be positive, got {target:,}")

    phases = schedule.get("phases", {})
    scheduled_by_phase = {phase: int(phases.get(phase, {}).get("target_tokens", 0)) for phase in PHASES}
    # The schedule was pinned to a literal 1T/700B/250B/50B, which asserted a
    # number nobody could know until token_count -- stage 27 -- and made the
    # corpus wrong rather than the plan when the two disagreed. What actually
    # has to hold is that the schedule is internally consistent and that the
    # corpus can supply it; the floor for the latter is
    # gates.minimum_unique_tokens, enforced at select against measured counts.
    if sum(scheduled_by_phase.values()) != target:
        errors.append(
            f"phase targets {scheduled_by_phase} must sum to target_tokens {target:,}"
        )
    if any(value < 0 for value in scheduled_by_phase.values()):
        errors.append(f"phase targets must be non-negative, got {scheduled_by_phase}")
    unique = int(schedule.get("unique_target_tokens", 0))
    replay = int(schedule.get("replay_target_tokens", 0))
    if unique <= 0 or replay < 0:
        errors.append(
            f"schedule unique/replay targets must be positive/non-negative, got {unique:,}/{replay:,}"
        )
    if unique + replay != target:
        errors.append("unique_target_tokens + replay_target_tokens must equal target_tokens")
    expected_start = 0
    phase_unique_total = 0
    phase_replay_total = 0
    for phase in PHASES:
        phase_payload = phases.get(phase, {})
        phase_target = scheduled_by_phase[phase]
        phase_unique = int(phase_payload.get("unique_tokens", 0))
        phase_replay = int(phase_payload.get("replay_tokens", 0))
        phase_start = int(phase_payload.get("start_token", -1))
        if min(phase_target, phase_unique, phase_replay, phase_start) < 0:
            errors.append(f"{phase} schedule values must be nonnegative")
        if phase_start != expected_start:
            errors.append(
                f"{phase}.start_token must be the canonical contiguous cursor "
                f"{expected_start:,}, got {phase_start:,}"
            )
        if phase_unique + phase_replay != phase_target:
            errors.append(f"{phase} unique_tokens + replay_tokens does not match phase target")
        expected_start += phase_target
        phase_unique_total += phase_unique
        phase_replay_total += phase_replay
    if expected_start != target:
        errors.append(f"phase cursors end at {expected_start:,}, not target {target:,}")
    if (phase_unique_total, phase_replay_total) != (unique, replay):
        errors.append(
            "phase unique/replay quotas do not reconcile with schedule totals: "
            f"{phase_unique_total:,}/{phase_replay_total:,} versus {unique:,}/{replay:,}"
        )

    categories = manifest.get("categories", [])
    category_ids = {item.get("id") for item in categories}
    for category in categories:
        if any(int(category.get("phase_tokens", {}).get(phase, 0)) < 0 for phase in PHASES):
            errors.append(f"category {category.get('id')}: phase token targets must be nonnegative")
    category_by_phase = _sum_by_phase(categories)
    if category_by_phase != scheduled_by_phase:
        errors.append(f"category phase totals {category_by_phase} do not match schedule {scheduled_by_phase}")

    sources = manifest.get("sources", [])
    source_ids = [str(source.get("id", "")) for source in sources]
    if len(source_ids) != len(set(source_ids)):
        errors.append("source ids must be unique")
    source_by_phase = _sum_by_phase(sources)
    if source_by_phase != scheduled_by_phase:
        errors.append(f"source phase totals {source_by_phase} do not match schedule {scheduled_by_phase}")

    by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in category_ids}
    for source in sources:
        source_id = str(source.get("id", "<missing>"))
        category = source.get("category")
        if category not in category_ids:
            errors.append(f"{source_id}: unknown category {category!r}")
            continue
        by_category[category].append(source)
        phase_values = [int(source.get("phase_tokens", {}).get(phase, 0)) for phase in PHASES]
        if any(value < 0 for value in phase_values):
            errors.append(f"{source_id}: phase token targets must be nonnegative")
        if total_phase_tokens(source) <= 0:
            errors.append(f"{source_id}: source target must be positive")
        for key in ("provenance", "access", "license", "acquisition", "processing"):
            if not isinstance(source.get(key), dict):
                errors.append(f"{source_id}: missing {key} mapping")
        access = source.get("access", {})
        if access.get("type") == "huggingface":
            if not access.get("repo_id"):
                errors.append(f"{source_id}: Hugging Face source missing repo_id")
            if not HEX40.match(str(access.get("revision", ""))):
                errors.append(f"{source_id}: Hugging Face revision must be a pinned 40-character commit")
        for component in access.get("components", []):
            if not HEX40.match(str(component.get("revision", ""))):
                errors.append(f"{source_id}: component revision must be a pinned 40-character commit")
        if int(source.get("phase_tokens", {}).get("phase_c", 0)) and source.get("provenance", {}).get("generated"):
            errors.append(f"{source_id}: generated data is forbidden in phase_c")
        license_status = source.get("license", {}).get("status")
        if not license_status or license_status == "unresolved":
            errors.append(f"{source_id}: unresolved license status")
        try:
            validated_attestations(source)
        except ValueError as error:
            errors.append(str(error))

    for category in categories:
        category_id = category["id"]
        expected = {phase: int(category.get("phase_tokens", {}).get(phase, 0)) for phase in PHASES}
        actual = _sum_by_phase(by_category.get(category_id, []))
        if actual != expected:
            errors.append(f"category {category_id}: source totals {actual} do not match {expected}")

    freshness = manifest.get("freshness_layer", {})
    fresh_sources = [source for source in sources if source.get("provenance", {}).get("fresh")]
    fresh_total = sum(total_phase_tokens(source) for source in fresh_sources)
    # Declared rather than constant, for the same reason as the schedule: the
    # layer is as large as the fresh sources actually supply. What must hold is
    # that the manifest's declaration matches them and that the layer is a
    # meaningful share of the corpus rather than a rounding error.
    if fresh_total != int(freshness.get("target_tokens", 0)):
        errors.append(
            "freshness_layer.target_tokens must equal its sources' embedded tokens, "
            f"declared {int(freshness.get('target_tokens', 0)):,} against {fresh_total:,}"
        )
    if target and fresh_total < MINIMUM_FRESHNESS_SHARE * target:
        errors.append(
            f"freshness layer is {fresh_total:,} tokens, below "
            f"{MINIMUM_FRESHNESS_SHARE:.1%} of the {target:,}-token schedule"
        )
    expected_fresh_buckets = {key: int(value) for key, value in freshness.get("buckets", {}).items()}
    actual_fresh_buckets: dict[str, int] = {}
    for source in fresh_sources:
        bucket = str(source.get("provenance", {}).get("freshness_bucket", ""))
        actual_fresh_buckets[bucket] = actual_fresh_buckets.get(bucket, 0) + total_phase_tokens(source)
    if actual_fresh_buckets != expected_fresh_buckets:
        errors.append(f"fresh bucket totals {actual_fresh_buckets} do not match {expected_fresh_buckets}")

    vocab = int(manifest.get("tokenizer", {}).get("vocabulary_size_including_special_tokens", 0))
    if vocab != 65_536:
        errors.append(f"tokenizer vocabulary including special tokens must be 65,536, got {vocab:,}")

    excluded = {str(item.get("id", "")).lower() for item in manifest.get("hard_exclusions", [])}
    for source in sources:
        blobs = [source.get("id", ""), source.get("access", {}).get("repo_id", "")]
        if any(str(blob).lower() in excluded for blob in blobs):
            errors.append(f"excluded benchmark/content appears as a training source: {source.get('id')}")

    generated_tokens = sum(
        total_phase_tokens(source)
        for source in sources
        if source.get("provenance", {}).get("generated") or source.get("provenance", {}).get("transformed")
    )
    if target and generated_tokens / target > 0.15:
        errors.append(f"explicitly generated/transformed share {generated_tokens / target:.1%} exceeds 15% cap")
    if target and generated_tokens / target > 0.12:
        warnings.append(f"explicitly generated/transformed share is {generated_tokens / target:.1%}; review before release")

    from .replacement import (
        replacement_resilience_report,
        validate_replacement_policy,
    )
    from .selection import replay_quotas, unique_quotas

    errors.extend(validate_replacement_policy(manifest))
    if not errors:
        plan_rows = []
        for source in sources:
            final_tokens = total_phase_tokens(source)
            multiplier = Decimal(
                str(source["acquisition"].get("candidate_multiplier", 1.0))
            )
            plan_rows.append(
                (source["id"], int(Decimal(final_tokens) * multiplier))
            )
        resilience = replacement_resilience_report(
            manifest,
            requirements=unique_quotas(manifest, replay_quotas(manifest)),
            candidate_tokens=dict(plan_rows),
        )
        if not resilience["all_sources_have_automatic_shortfall_path"]:
            missing_paths = sorted(
                source_id
                for source_id, row in resilience["sources"].items()
                if not row["automatic_shortfall_path_available"]
            )
            errors.append(
                "replacement policy lacks an automatic donor or cold-reserve path for "
                f"{missing_paths}"
            )

    return ValidationResult(manifest=manifest, errors=tuple(errors), warnings=tuple(warnings))


def candidate_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    context_reserve: dict[str, int] = {}
    context_targets: dict[str, int] = {}
    context_plan = manifest.get("context_extension_plan")
    if context_plan:
        from .context_manifest import (
            context_candidate_targets,
            context_retrieval_reserve_tokens,
        )

        context_reserve = context_retrieval_reserve_tokens(context_plan)
        context_targets = context_candidate_targets(context_plan)
    rows = []
    for source in manifest["sources"]:
        final_tokens = total_phase_tokens(source)
        acquisition = source["acquisition"]
        multiplier = Decimal(str(acquisition.get("candidate_multiplier", 1.0)))
        base_candidate_tokens = int(Decimal(final_tokens) * multiplier)
        reserve_tokens = int(context_reserve.get(source["id"], 0))
        candidate_tokens = base_candidate_tokens + reserve_tokens
        bytes_per_token = Decimal(str(acquisition.get("compressed_bytes_per_token", 0.8)))
        rows.append(
            {
                "id": source["id"],
                "category": source["category"],
                "driver": acquisition["driver"],
                "final_exposure_tokens": final_tokens,
                "base_candidate_tokens": base_candidate_tokens,
                "context_extension_retrieval_reserve_tokens": reserve_tokens,
                "context_extension_candidate_target_tokens": int(
                    context_targets.get(source["id"], 0)
                ),
                "candidate_tokens": candidate_tokens,
                "planned_download_bytes": int(Decimal(candidate_tokens) * bytes_per_token),
                "phase_tokens": source["phase_tokens"],
                "fresh": bool(source["provenance"].get("fresh")),
            }
        )
    from .replacement import replacement_resilience_report
    from .selection import replay_quotas, unique_quotas

    candidate_tokens = {row["id"]: int(row["candidate_tokens"]) for row in rows}
    unique = unique_quotas(manifest, replay_quotas(manifest))
    return {
        "release": manifest["release"],
        "target_tokens": manifest["schedule"]["target_tokens"],
        "planned_candidate_tokens": sum(row["candidate_tokens"] for row in rows),
        "planned_download_bytes": sum(row["planned_download_bytes"] for row in rows),
        "context_extension": (
            {
                "release": context_plan["release"],
                "token_budget": int(context_plan["token_budget"]),
                "checkpoint_gates": list(context_plan["checkpoint_gates"]),
                "candidate_targets": context_targets,
                "additional_retrieval_reserve_tokens": sum(
                    context_reserve.values()
                ),
            }
            if context_plan
            else None
        ),
        "replacement_resilience": replacement_resilience_report(
            manifest,
            requirements=unique,
            candidate_tokens=candidate_tokens,
        ),
        "sources": rows,
    }


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    normalized = path.lstrip("/")
    for pattern in patterns:
        normalized_pattern = str(pattern).lstrip("/")
        if fnmatch.fnmatch(normalized, normalized_pattern):
            return True
        # Python's fnmatch treats the slash in ``**/`` literally, whereas
        # Hugging Face allow-patterns commonly use it to mean "at any depth",
        # including the repository root.  Keep our resolver consistent so a
        # root-level ``data.parquet`` is not silently omitted by
        # ``**/*.parquet``.
        if normalized_pattern.startswith("**/") and fnmatch.fnmatch(normalized, normalized_pattern[3:]):
            return True
    return False


def dump_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
