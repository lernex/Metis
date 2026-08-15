from __future__ import annotations

import gzip
import hashlib
import io
import json

try:  # matches stage_runner: orjson decodes these shards about 2.5x faster
    import orjson as _orjson

    def _loads(value: Any) -> Any:
        return _orjson.loads(value)

except ImportError:  # pragma: no cover
    def _loads(value: Any) -> Any:
        return json.loads(value)

import math
import os
import re
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import zstandard as zstd
from tokenizers import Tokenizer

from .context_manifest import (
    CONTEXT_GATES,
    CONTEXT_SEQUENCE_MIX,
    CONTEXT_TOKEN_BUDGET,
    context_quota_rows,
    validate_context_plan,
)
from .state import atomic_json, utc_now


CONTEXT_SELECTION_SCHEMA = "metis.context-selection/v1"
CONTEXT_PACK_PLAN_SCHEMA = "metis.context-pack-plan/v1"
CONTEXT_PACK_TASK_SCHEMA = "metis.context-pack-task/v1"
CONTEXT_VERIFICATION_SCHEMA = "metis.context-verification/v1"
CONTEXT_DATA_SCHEMA = "metis.context-extension-data/v1"
SEALED_ARTIFACT_SCHEMA = "metis.sealed-artifact/v1"
MMAP_BUNDLE_SCHEMA = "metis.posttraining-mmap/v1"
COMPACT_LAYOUT = "metis.compact-causal/v1"
LANES = ("natural_long", "dependency_constructed", "short_replay")
CONTEXT_EVALUATION_ARRAYS = (
    "context_evaluation_input_ids",
    "context_evaluation_probe_target_ids",
    "context_evaluation_probe_positions",
    "context_evaluation_split_fingerprint",
)
DEFAULT_PACK_TASKS = 96
_HEADING = re.compile(
    r"(?im)^(?:#{1,6}\s+|(?:chapter|section|part|appendix)\s+[A-Z0-9IVXLC.-]+\b)"
)
_LONG_REFERENCE = re.compile(
    r"(?i)\b(?:see|refer to|as (?:shown|described|defined|proved)|"
    r"above|below|earlier|later)\s+(?:in\s+)?(?:chapter|section|part|"
    r"appendix|figure|table|equation|theorem|lemma|definition|listing|file|"
    r"module|class|function)?"
)
_CODE_DEPENDENCY = re.compile(
    r"(?m)^(?:\s*(?:from|import|include|require|use|mod|package)\b|"
    r"\s*#include\s*[<\"]|\s*(?:class|interface|trait|impl|def|fn)\s+)"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _json_sha256(value: Any, *, omit: Sequence[str] = ()) -> str:
    ignored = set(omit)
    if isinstance(value, Mapping):
        value = {key: item for key, item in value.items() if key not in ignored}
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve()).as_posix()
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def structural_evidence(
    text: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Cheap, deterministic prefilter for documents with long-range structure.

    This is deliberately only a prefilter. The Portage context gate calibrates
    it against full-context versus 4K-conditioned NLL on a sealed sample after
    the base checkpoint exists.
    """

    metadata = metadata or {}
    headings = len(_HEADING.findall(text))
    references = len(_LONG_REFERENCE.findall(text))
    code_dependencies = len(_CODE_DEPENDENCY.findall(text))
    paragraph_count = sum(1 for part in re.split(r"\n\s*\n", text) if part.strip())
    repository = bool(
        metadata.get("repository")
        or metadata.get("repo_name")
        or metadata.get("repository_url")
    )
    scholarly = bool(
        metadata.get("doi")
        or metadata.get("paper_id")
        or metadata.get("arxiv_id")
        or metadata.get("pmcid")
    )
    longform = bool(
        metadata.get("book_id")
        or metadata.get("volume_id")
        or metadata.get("chapter")
    )
    score = 0
    score += int(len(text) >= 32_768)
    score += int(len(text) >= 131_072)
    score += int(headings >= 4 or paragraph_count >= 24)
    score += int(references >= 4 or code_dependencies >= 8)
    score += int(repository or scholarly or longform)
    return {
        "schema": "metis.long-range-structural-evidence/v1",
        "score": score,
        "characters": len(text),
        "headings": headings,
        "cross_references": references,
        "code_dependencies": code_dependencies,
        "paragraphs": paragraph_count,
        "repository_document": repository,
        "scholarly_document": scholarly,
        "longform_document": longform,
    }


def context_group_id(
    source_id: str,
    doc_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    metadata = metadata or {}
    candidate = next(
        (
            str(metadata[key])
            for key in (
                "repository",
                "repo_name",
                "repository_url",
                "book_id",
                "volume_id",
                "paper_id",
                "doi",
                "arxiv_id",
                "pmcid",
            )
            if metadata.get(key)
        ),
        doc_id,
    )
    return hashlib.sha256(
        f"{source_id}\0{candidate}".encode("utf-8")
    ).hexdigest()


def _hamilton(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    positive = {str(key): int(value) for key, value in weights.items() if int(value) > 0}
    denominator = sum(positive.values())
    if total < 0 or (total and denominator <= 0):
        raise ValueError("invalid Hamilton apportionment")
    floors = {
        key: total * weight // denominator for key, weight in positive.items()
    }
    remainders = sorted(
        (
            (total * weight % denominator, key)
            for key, weight in positive.items()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for _remainder, key in remainders[: total - sum(floors.values())]:
        floors[key] += 1
    return {str(key): floors.get(str(key), 0) for key in weights}


def context_lane_quota_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_context_plan(plan)
    source_rows = context_quota_rows(plan)
    result: list[dict[str, Any]] = []
    for gate_index in range(len(CONTEXT_GATES)):
        gate_rows = [
            row for row in source_rows if int(row["gate_index"]) == gate_index
        ]
        gate_total = sum(int(row["tokens"]) for row in gate_rows)
        lane_totals = _hamilton(
            gate_total,
            {
                lane: int(round(CONTEXT_SEQUENCE_MIX[lane] * 100))
                for lane in LANES
            },
        )
        for lane in LANES:
            source_allocations = _hamilton(
                lane_totals[lane],
                {
                    str(row["source_id"]): int(row["tokens"])
                    for row in gate_rows
                },
            )
            domains = {
                str(row["source_id"]): str(row["domain"]) for row in gate_rows
            }
            for source_id, tokens in source_allocations.items():
                result.append(
                    {
                        "gate_index": gate_index,
                        "gate_target_tokens": CONTEXT_GATES[gate_index],
                        "lane": lane,
                        "source_id": source_id,
                        "domain": domains[source_id],
                        "tokens": tokens,
                    }
                )
    for gate_index in range(len(CONTEXT_GATES)):
        observed = sum(
            int(row["tokens"])
            for row in result
            if int(row["gate_index"]) == gate_index
        )
        if observed != CONTEXT_GATES[gate_index] - (
            CONTEXT_GATES[gate_index - 1] if gate_index else 0
        ):
            raise AssertionError("context lane quota gate is not exact")
    return result


def context_evaluation_domain_targets(
    plan: Mapping[str, Any],
) -> dict[str, int]:
    validate_context_plan(plan)
    return _hamilton(
        int(plan["selection"]["gate_evaluation_records"]),
        {
            str(group["id"]): sum(
                int(row["tokens"])
                for row in plan["sources"]
                if row["domain"] == group["id"]
            )
            for group in plan["fallbacks"]["groups"]
        },
    )


def _fallback_chains(plan: Mapping[str, Any]) -> dict[str, list[str]]:
    chains: dict[str, list[str]] = {}
    for group in plan["fallbacks"]["groups"]:
        donors = [str(value) for value in group["donor_order"]]
        for member in group["members"]:
            source_id = str(member)
            chains[source_id] = [
                donor for donor in donors if donor != source_id
            ]
    return chains


def allocate_context_replacements(
    plan: Mapping[str, Any],
    *,
    requirements: Sequence[Mapping[str, Any]],
    available_tokens: Mapping[str, int],
) -> dict[str, Any]:
    """Fill exact gate/lane/source quotas from ordered same-domain surplus."""

    validate_context_plan(plan)
    source_ids = {str(row["id"]) for row in plan["sources"]}
    chains = _fallback_chains(plan)
    rows = [
        {
            **dict(raw),
            "tokens": int(raw["tokens"]),
            "row_id": (
                f"{int(raw['gate_index'])}:{raw['lane']}:{raw['source_id']}"
            ),
        }
        for raw in requirements
        if int(raw["tokens"]) > 0
    ]
    if {str(row["source_id"]) for row in rows} - source_ids:
        raise ValueError("context requirements reference unknown sources")

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(row)
    own: dict[str, int] = {}
    surplus: dict[str, int] = {}
    assignments: list[dict[str, Any]] = []
    deficits: dict[str, int] = {}
    for source_id in sorted(source_ids):
        source_rows = by_source.get(source_id, [])
        required = sum(int(row["tokens"]) for row in source_rows)
        available = max(0, int(available_tokens.get(source_id, 0)))
        own_total = min(required, available)
        own[source_id] = own_total
        surplus[source_id] = max(0, available - own_total)
        apportioned = _hamilton(
            own_total,
            {row["row_id"]: int(row["tokens"]) for row in source_rows},
        )
        for row in source_rows:
            amount = int(apportioned[row["row_id"]])
            deficits[row["row_id"]] = int(row["tokens"]) - amount
            if amount:
                assignments.append(
                    {
                        **{key: row[key] for key in (
                            "gate_index",
                            "gate_target_tokens",
                            "lane",
                            "domain",
                        )},
                        "target_source_id": source_id,
                        "actual_source_id": source_id,
                        "tokens": amount,
                        "replacement": False,
                    }
                )

    row_map = {str(row["row_id"]): row for row in rows}
    for row_id in sorted(
        deficits,
        key=lambda value: (
            int(row_map[value]["gate_index"]),
            LANES.index(str(row_map[value]["lane"])),
            str(row_map[value]["source_id"]),
        ),
    ):
        needed = deficits[row_id]
        row = row_map[row_id]
        target = str(row["source_id"])
        for donor in chains[target]:
            take = min(needed, surplus.get(donor, 0))
            if take <= 0:
                continue
            assignments.append(
                {
                    **{key: row[key] for key in (
                        "gate_index",
                        "gate_target_tokens",
                        "lane",
                        "domain",
                    )},
                    "target_source_id": target,
                    "actual_source_id": donor,
                    "tokens": take,
                    "replacement": True,
                }
            )
            surplus[donor] -= take
            needed -= take
            if not needed:
                break
        deficits[row_id] = needed

    unresolved = {
        row_id: tokens for row_id, tokens in deficits.items() if int(tokens) > 0
    }
    if unresolved:
        raise RuntimeError(
            "context source exhaustion after ordered same-domain fallbacks: "
            + json.dumps(unresolved, sort_keys=True)
        )
    by_actual: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        by_actual[str(assignment["actual_source_id"])].append(assignment)
    for values in by_actual.values():
        values.sort(
            key=lambda row: (
                int(row["gate_index"]),
                LANES.index(str(row["lane"])),
                str(row["target_source_id"]),
            )
        )
    return {
        "schema": "metis.context-replacement-allocation/v1",
        "assignments": assignments,
        "assignments_by_actual_source": dict(by_actual),
        "available_tokens": {
            source_id: int(available_tokens.get(source_id, 0))
            for source_id in sorted(source_ids)
        },
        "unresolved": {},
        "replacement_tokens": sum(
            int(row["tokens"]) for row in assignments if row["replacement"]
        ),
    }


def _balanced_parts(total: int, parts: int) -> list[int]:
    if total < 0 or parts <= 0:
        raise ValueError("balanced partition requires non-negative total and positive parts")
    quotient, remainder = divmod(total, parts)
    return [quotient + int(index < remainder) for index in range(parts)]


def build_context_pack_plan(
    plan: Mapping[str, Any],
    *,
    pack_tasks: int = DEFAULT_PACK_TASKS,
) -> dict[str, Any]:
    validate_context_plan(plan)
    if pack_tasks != 96:
        raise ValueError("production context packing currently requires exactly 96 tasks")
    multiple = int(plan["packing_multiple_records"])
    records = math.ceil(CONTEXT_TOKEN_BUDGET / int(plan["train_context"]))
    records = math.ceil(records / multiple) * multiple
    records_per_gate = records // len(CONTEXT_GATES)
    if records % len(CONTEXT_GATES) or records_per_gate % multiple:
        raise AssertionError("context records do not divide exactly over gates")

    task_rows: list[dict[str, Any]] = []
    global_record = 0
    task_index = 0
    quota_rows = context_lane_quota_rows(plan)
    domain_order = [
        str(group["id"]) for group in plan["fallbacks"]["groups"]
    ]
    if len(domain_order) != len(set(domain_order)):
        raise AssertionError("context fallback domains must be unique")
    # The dependency-constructed lane packs one task per domain, so the domain
    # count is part of the per-gate task budget rather than a free parameter.
    # This was pinned at seven, which silently made the packer depend on the
    # number of fallback groups in the plan: dropping a domain for lack of
    # supply left 93 tasks against a required 96 and failed here, two stages
    # after the decision. Derive the split instead. At seven domains this
    # yields the historical 22/3/7.
    tasks_per_gate, remainder = divmod(pack_tasks, len(CONTEXT_GATES))
    if remainder:
        raise AssertionError("context pack tasks do not divide over gates")
    replay_tasks = 3
    natural_tasks = tasks_per_gate - replay_tasks - len(domain_order)
    if natural_tasks < replay_tasks:
        raise AssertionError(
            f"{len(domain_order)} context domains leave only {natural_tasks} "
            f"tasks per gate for the natural-long lane, which carries "
            f"{CONTEXT_SEQUENCE_MIX['natural_long']:.0%} of the mix"
        )
    lane_task_counts = {
        "natural_long": natural_tasks,
        "short_replay": replay_tasks,
    }
    for gate_index, gate_target in enumerate(CONTEXT_GATES):
        gate_tokens = gate_target - (
            CONTEXT_GATES[gate_index - 1] if gate_index else 0
        )
        lane_tokens = _hamilton(
            gate_tokens,
            {
                lane: int(round(CONTEXT_SEQUENCE_MIX[lane] * 100))
                for lane in LANES
            },
        )
        lane_records = _hamilton(
            records_per_gate,
            {
                lane: int(round(CONTEXT_SEQUENCE_MIX[lane] * 100))
                for lane in LANES
            },
        )
        for lane in LANES:
            if lane == "dependency_constructed":
                tokens_by_domain = {
                    domain: sum(
                        int(row["tokens"])
                        for row in quota_rows
                        if (
                            int(row["gate_index"]) == gate_index
                            and row["lane"] == lane
                            and row["domain"] == domain
                        )
                    )
                    for domain in domain_order
                }
                if (
                    any(tokens <= 0 for tokens in tokens_by_domain.values())
                    or sum(tokens_by_domain.values()) != lane_tokens[lane]
                ):
                    raise AssertionError(
                        "dependency construction domain quotas do not reconcile"
                    )
                records_by_domain = _hamilton(
                    lane_records[lane],
                    tokens_by_domain,
                )
                task_specs = [
                    (
                        local_index,
                        int(records_by_domain[domain]),
                        int(tokens_by_domain[domain]),
                        domain,
                    )
                    for local_index, domain in enumerate(domain_order)
                ]
            else:
                task_count = lane_task_counts[lane]
                records_by_task = _balanced_parts(
                    lane_records[lane], task_count
                )
                tokens_by_task = _balanced_parts(
                    lane_tokens[lane], task_count
                )
                task_specs = [
                    (local_index, task_records, task_tokens, None)
                    for local_index, (task_records, task_tokens) in enumerate(
                        zip(records_by_task, tokens_by_task, strict=True)
                    )
                ]
            for local_index, task_records, task_tokens, domain in task_specs:
                if task_records <= 0 or not (
                    task_records <= task_tokens <= task_records * int(plan["train_context"])
                ):
                    raise AssertionError("context pack task record/token geometry is invalid")
                task_row = {
                    "task_index": task_index,
                    "gate_index": gate_index,
                    "gate_target_tokens": gate_target,
                    "lane": lane,
                    "lane_task_index": local_index,
                    "record_start": global_record,
                    "record_end": global_record + task_records,
                    "records": task_records,
                    "active_tokens": task_tokens,
                    "selection_path": (
                        f"selection/task-{task_index:06d}.jsonl.zst"
                    ),
                }
                if domain is not None:
                    task_row["domain"] = domain
                task_rows.append(task_row)
                global_record += task_records
                task_index += 1
    if (
        task_index != pack_tasks
        or global_record != records
        or sum(int(row["active_tokens"]) for row in task_rows)
        != CONTEXT_TOKEN_BUDGET
    ):
        raise AssertionError("context pack plan does not reconcile")
    payload: dict[str, Any] = {
        "schema": CONTEXT_PACK_PLAN_SCHEMA,
        "release": plan["release"],
        "created_at": utc_now(),
        "records": records,
        "sequence_length": int(plan["train_context"]),
        "active_tokens": CONTEXT_TOKEN_BUDGET,
        "checkpoint_gates": list(CONTEXT_GATES),
        "packing_multiple_records": multiple,
        "pack_tasks": pack_tasks,
        "tasks": task_rows,
    }
    payload["plan_sha256"] = _json_sha256(payload)
    return payload


class _ZstdWriters:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.handles: dict[int, io.TextIOWrapper] = {}

    def write(self, task_index: int, payload: Mapping[str, Any]) -> None:
        handle = self.handles.get(task_index)
        if handle is None:
            path = self.root / f"task-{task_index:06d}.jsonl.zst"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)
            raw = path.open("wb")
            handle = io.TextIOWrapper(
                zstd.ZstdCompressor(level=6).stream_writer(raw, closefd=True),
                encoding="utf-8",
            )
            self.handles[task_index] = handle
        handle.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False) + "\n")

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


class _AtomicZstdRows:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_name(path.name + ".incomplete")
        self.handle: io.TextIOWrapper | None = None

    def __enter__(self) -> "_AtomicZstdRows":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self.temporary.unlink(missing_ok=True)
        raw = self.temporary.open("wb")
        self.handle = io.TextIOWrapper(
            zstd.ZstdCompressor(level=6).stream_writer(raw, closefd=True),
            encoding="utf-8",
        )
        return self

    def write(self, payload: Mapping[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("atomic Zstandard writer is not open")
        self.handle.write(
            json.dumps(dict(payload), sort_keys=True, ensure_ascii=False)
            + "\n"
        )

    def __exit__(self, exception_type: Any, *_args: Any) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if exception_type is None:
            os.replace(self.temporary, self.path)
        else:
            self.temporary.unlink(missing_ok=True)


def _stable_fraction(*values: Any) -> float:
    digest = hashlib.sha256(
        "\0".join(str(value) for value in values).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _assignment_queues(
    allocation: Mapping[str, Any],
) -> dict[str, deque[dict[str, Any]]]:
    result: dict[str, deque[dict[str, Any]]] = {}
    for source_id, raw_rows in allocation["assignments_by_actual_source"].items():
        rows = deque()
        for raw in raw_rows:
            row = dict(raw)
            row["remaining"] = int(row["tokens"])
            rows.append(row)
        result[str(source_id)] = rows
    return result


def _consume_assignment(
    queue: deque[dict[str, Any]],
    available: int,
) -> Iterator[tuple[dict[str, Any], int]]:
    remaining = available
    while queue and remaining:
        assignment = queue[0]
        take = min(remaining, int(assignment["remaining"]))
        if take:
            yield assignment, take
            remaining -= take
            assignment["remaining"] = int(assignment["remaining"]) - take
        if int(assignment["remaining"]) == 0:
            queue.popleft()


def _task_routers(
    pack_plan: Mapping[str, Any],
) -> dict[tuple[int, str, str], deque[dict[str, Any]]]:
    result: dict[tuple[int, str, str], deque[dict[str, Any]]] = defaultdict(
        deque
    )
    for raw in pack_plan["tasks"]:
        row = dict(raw)
        row["remaining"] = int(row["active_tokens"])
        lane = str(row["lane"])
        domain = (
            str(row.get("domain") or "")
            if lane == "dependency_constructed"
            else ""
        )
        if lane == "dependency_constructed" and not domain:
            raise RuntimeError(
                "dependency construction task is not domain-bound"
            )
        result[(int(row["gate_index"]), lane, domain)].append(row)
    return dict(result)


def _route_fragment(
    routers: dict[tuple[int, str, str], deque[dict[str, Any]]],
    writers: _ZstdWriters,
    *,
    assignment: Mapping[str, Any],
    record: Mapping[str, Any],
    token_start: int,
    token_count: int,
) -> None:
    lane = str(assignment["lane"])
    domain = (
        str(assignment["domain"])
        if lane == "dependency_constructed"
        else ""
    )
    queue = routers[(int(assignment["gate_index"]), lane, domain)]
    remaining = token_count
    offset = token_start
    while remaining:
        if not queue:
            raise RuntimeError("context selection exceeded its pack-task routing quota")
        task = queue[0]
        take = min(remaining, int(task["remaining"]))
        payload = {
            **dict(record),
            "gate_index": int(assignment["gate_index"]),
            "gate_target_tokens": int(assignment["gate_target_tokens"]),
            "lane": lane,
            "context_domain": str(assignment["domain"]),
            "quota_source_id": str(assignment["target_source_id"]),
            "replacement": bool(assignment["replacement"]),
            "replacement_for_source_id": (
                str(assignment["target_source_id"])
                if assignment["replacement"]
                else None
            ),
            "token_start": offset,
            "token_count": take,
        }
        writers.write(int(task["task_index"]), payload)
        task["remaining"] = int(task["remaining"]) - take
        if int(task["remaining"]) == 0:
            queue.popleft()
        remaining -= take
        offset += take


def build_context_selection(
    records: Iterable[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    pack_plan: Mapping[str, Any],
    output_root: Path,
    token_count_contract_sha256: str,
    tokenizer_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Select exact 6B tranches with domain-preserving measured fallbacks.

    ``records`` must be restartable because selection intentionally performs a
    measurement pass followed by one deterministic materialization pass.
    Pass a callable-backed iterable or a concrete sequence.
    """

    validate_context_plan(plan)
    if pack_plan.get("schema") != CONTEXT_PACK_PLAN_SCHEMA:
        raise ValueError("context selection requires a valid pack plan")
    if iter(records) is records:
        raise ValueError("context selection records must be restartable")
    source_domains = {
        str(row["id"]): str(row["domain"]) for row in plan["sources"]
    }
    source_ids = set(source_domains)
    minimum_long = int(plan["selection"]["minimum_long_document_tokens"])
    minimum_score = int(plan["selection"]["minimum_structural_score"])
    total_available: dict[str, int] = defaultdict(int)
    long_available: dict[str, int] = defaultdict(int)
    documents = 0
    for raw in records:
        source_id = str(raw.get("source_id") or "")
        if source_id not in source_ids:
            continue
        tokens = int(raw.get("token_count", 0))
        evidence = raw.get("context_structure")
        score = (
            int(evidence.get("score", -1))
            if isinstance(evidence, Mapping)
            else -1
        )
        if tokens <= 0:
            continue
        total_available[source_id] += tokens
        if tokens >= minimum_long and score >= minimum_score:
            long_available[source_id] += tokens
        documents += 1

    quota_rows = context_lane_quota_rows(plan)
    long_requirements = [
        row for row in quota_rows if row["lane"] != "short_replay"
    ]
    short_requirements = [
        row for row in quota_rows if row["lane"] == "short_replay"
    ]
    long_allocation = allocate_context_replacements(
        plan,
        requirements=long_requirements,
        available_tokens=long_available,
    )
    long_used: dict[str, int] = defaultdict(int)
    for assignment in long_allocation["assignments"]:
        long_used[str(assignment["actual_source_id"])] += int(assignment["tokens"])
    remaining_available = {
        source_id: max(
            0,
            int(total_available.get(source_id, 0))
            - int(long_used.get(source_id, 0)),
        )
        for source_id in source_ids
    }
    short_allocation = allocate_context_replacements(
        plan,
        requirements=short_requirements,
        available_tokens=remaining_available,
    )

    long_queues = _assignment_queues(long_allocation)
    short_queues = _assignment_queues(short_allocation)
    routers = _task_routers(pack_plan)
    selection_root = output_root / "selection"
    selection_root.mkdir(parents=True, exist_ok=True)
    for path in selection_root.glob("task-*.jsonl.zst"):
        path.unlink()
    writers = _ZstdWriters(selection_root)
    selected_tokens = 0
    selected_documents: set[tuple[str, str]] = set()
    license_tokens: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    evaluation_records = int(plan["selection"]["gate_evaluation_records"])
    evaluation_context = int(plan["selection"]["gate_evaluation_context"])
    evaluation_domain_targets = context_evaluation_domain_targets(plan)
    evaluation_path = output_root / "evaluation" / "records.jsonl.zst"
    evaluation_selected = 0
    evaluation_tokens = 0
    evaluation_sources: dict[str, int] = defaultdict(int)
    evaluation_domains: dict[str, int] = defaultdict(int)
    try:
        with _AtomicZstdRows(evaluation_path) as evaluation_writer:
            for raw in records:
                source_id = str(raw.get("source_id") or "")
                if source_id not in source_ids:
                    continue
                record = dict(raw)
                tokens = int(record.get("token_count", 0))
                if tokens <= 0:
                    continue
                domain = source_domains[source_id]
                evidence = record.get("context_structure")
                long_eligible = bool(
                    tokens >= minimum_long
                    and isinstance(evidence, Mapping)
                    and int(evidence.get("score", -1)) >= minimum_score
                )
                cursor = 0
                if long_eligible:
                    queue = long_queues.get(source_id, deque())
                    for assignment, take in _consume_assignment(queue, tokens):
                        _route_fragment(
                            routers,
                            writers,
                            assignment=assignment,
                            record=record,
                            token_start=cursor,
                            token_count=take,
                        )
                        cursor += take
                        selected_tokens += take
                        selected_documents.add(
                            (source_id, str(record.get("doc_id")))
                        )
                        expression = str(record.get("license") or "")
                        if not expression:
                            raise RuntimeError(
                                f"context selection has no license for {source_id}"
                            )
                        license_tokens[source_id][expression] += take
                        if cursor == tokens:
                            break
                if cursor < tokens:
                    queue = short_queues.get(source_id, deque())
                    for assignment, take in _consume_assignment(
                        queue, tokens - cursor
                    ):
                        _route_fragment(
                            routers,
                            writers,
                            assignment=assignment,
                            record=record,
                            token_start=cursor,
                            token_count=take,
                        )
                        cursor += take
                        selected_tokens += take
                        selected_documents.add(
                            (source_id, str(record.get("doc_id")))
                        )
                        expression = str(record.get("license") or "")
                        if not expression:
                            raise RuntimeError(
                                f"context selection has no license for {source_id}"
                            )
                        license_tokens[source_id][expression] += take
                        if cursor == tokens:
                            break
                # Evaluation slices never overlap a training slice. Requiring
                # one contiguous deploy-length span per record makes the
                # full-vs-4K NLL comparison a real long-range measurement.
                if (
                    evaluation_selected < evaluation_records
                    and long_eligible
                    and tokens - cursor >= evaluation_context
                    and evaluation_domains[domain]
                    < int(evaluation_domain_targets[domain])
                ):
                    evaluation_writer.write(
                        {
                            **record,
                            "context_domain": domain,
                            "token_start": cursor,
                            "token_count": evaluation_context,
                            "evaluation_only": True,
                        }
                    )
                    evaluation_selected += 1
                    evaluation_tokens += evaluation_context
                    evaluation_sources[source_id] += 1
                    evaluation_domains[domain] += 1
    finally:
        writers.close()

    remaining_assignments = {
        "long": {
            source: sum(int(row["remaining"]) for row in queue)
            for source, queue in long_queues.items()
            if any(int(row["remaining"]) for row in queue)
        },
        "short": {
            source: sum(int(row["remaining"]) for row in queue)
            for source, queue in short_queues.items()
            if any(int(row["remaining"]) for row in queue)
        },
    }
    remaining_routes = {
        f"{gate}:{lane}:{domain or 'all'}": sum(
            int(row["remaining"]) for row in queue
        )
        for (gate, lane, domain), queue in routers.items()
        if any(int(row["remaining"]) for row in queue)
    }
    if (
        selected_tokens != CONTEXT_TOKEN_BUDGET
        or evaluation_selected != evaluation_records
        or evaluation_tokens != evaluation_records * evaluation_context
        or dict(evaluation_domains) != evaluation_domain_targets
        or remaining_assignments["long"]
        or remaining_assignments["short"]
        or remaining_routes
    ):
        raise RuntimeError(
            "context selection did not fill its exact release: "
            + json.dumps(
                {
                    "selected_tokens": selected_tokens,
                    "evaluation_records": evaluation_selected,
                    "evaluation_tokens": evaluation_tokens,
                    "evaluation_domains": dict(evaluation_domains),
                    "evaluation_domain_targets": evaluation_domain_targets,
                    "remaining_assignments": remaining_assignments,
                    "remaining_routes": remaining_routes,
                },
                sort_keys=True,
            )
        )

    files = []
    for task in pack_plan["tasks"]:
        path = output_root / str(task["selection_path"])
        if not path.is_file():
            raise RuntimeError(f"context selection task is missing: {path}")
        files.append(_file_record(path, root=output_root))
    files.append(_file_record(evaluation_path, root=output_root))
    payload: dict[str, Any] = {
        "schema": CONTEXT_SELECTION_SCHEMA,
        "release": plan["release"],
        "created_at": utc_now(),
        "documents_scanned": documents,
        "documents_selected": len(selected_documents),
        "gate_evaluation_records": evaluation_selected,
        "gate_evaluation_tokens": evaluation_tokens,
        "gate_evaluation_context": evaluation_context,
        "gate_evaluation_sources": dict(evaluation_sources),
        "gate_evaluation_domains": dict(evaluation_domains),
        "gate_evaluation_domain_targets": evaluation_domain_targets,
        "gate_evaluation_path": evaluation_path.relative_to(
            output_root
        ).as_posix(),
        "active_tokens": selected_tokens,
        "long_available_tokens": dict(long_available),
        "total_available_tokens": dict(total_available),
        "long_allocation": long_allocation,
        "short_allocation": short_allocation,
        "pack_plan_sha256": pack_plan["plan_sha256"],
        "token_count_contract_sha256": token_count_contract_sha256,
        "tokenizer_contract": dict(tokenizer_contract),
        "files": files,
        "license_tokens": {
            source: dict(expressions)
            for source, expressions in license_tokens.items()
        },
    }
    payload["selection_sha256"] = _json_sha256(payload)
    atomic_json(output_root / "SELECTION.json", payload)
    return payload


def validate_context_selection(
    output_root: Path,
    *,
    plan: Mapping[str, Any],
    pack_plan: Mapping[str, Any],
    token_count_contract_sha256: str,
    tokenizer_contract: Mapping[str, Any],
) -> dict[str, Any]:
    path = output_root / "SELECTION.json"
    if not path.is_file():
        raise RuntimeError("context SELECTION.json is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_evaluation_domains = context_evaluation_domain_targets(plan)
    if (
        payload.get("schema") != CONTEXT_SELECTION_SCHEMA
        or payload.get("release") != plan["release"]
        or payload.get("selection_sha256")
        != _json_sha256(payload, omit=("selection_sha256",))
        or int(payload.get("active_tokens", -1)) != CONTEXT_TOKEN_BUDGET
        or payload.get("pack_plan_sha256") != pack_plan["plan_sha256"]
        or payload.get("token_count_contract_sha256")
        != token_count_contract_sha256
        or payload.get("tokenizer_contract") != dict(tokenizer_contract)
        or int(payload.get("gate_evaluation_records", -1))
        != int(plan["selection"]["gate_evaluation_records"])
        or int(payload.get("gate_evaluation_tokens", -1))
        != int(plan["selection"]["gate_evaluation_records"])
        * int(plan["selection"]["gate_evaluation_context"])
        or int(payload.get("gate_evaluation_context", -1))
        != int(plan["selection"]["gate_evaluation_context"])
        or payload.get("gate_evaluation_domains")
        != expected_evaluation_domains
        or payload.get("gate_evaluation_domain_targets")
        != expected_evaluation_domains
    ):
        raise RuntimeError("context selection contract is stale or corrupt")
    expected_paths = {
        str(task["selection_path"]) for task in pack_plan["tasks"]
    }
    expected_paths.add(str(payload.get("gate_evaluation_path", "")))
    observed_paths: set[str] = set()
    for raw in payload.get("files", []):
        record = dict(raw)
        relative = str(record.get("path") or "")
        target = (output_root / relative).resolve()
        try:
            target.relative_to(output_root.resolve())
        except ValueError as exc:
            raise RuntimeError("context selection inventory escapes its root") from exc
        if (
            relative in observed_paths
            or not target.is_file()
            or target.stat().st_size != int(record.get("bytes", -1))
            or _file_sha256(target) != record.get("sha256")
        ):
            raise RuntimeError(f"context selection inventory changed: {relative}")
        observed_paths.add(relative)
    if observed_paths != expected_paths:
        raise RuntimeError("context selection inventory does not match the pack plan")
    return payload


def initialize_context_arrays(
    output_root: Path,
    *,
    pack_plan: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    validate_context_plan(plan)
    if (
        pack_plan.get("schema") != CONTEXT_PACK_PLAN_SCHEMA
        or pack_plan.get("plan_sha256")
        != _json_sha256(pack_plan, omit=("plan_sha256",))
    ):
        raise ValueError("cannot initialize arrays from an invalid context pack plan")
    arrays_root = output_root / "arrays"
    arrays_root.mkdir(parents=True, exist_ok=True)
    specifications = {
        "input_ids": (
            np.dtype("<u2"),
            (int(pack_plan["records"]), int(pack_plan["sequence_length"])),
        ),
        "document_start": (
            np.dtype("u1"),
            (int(pack_plan["records"]), int(pack_plan["sequence_length"])),
        ),
        "sequence_lengths": (
            np.dtype("<u4"),
            (int(pack_plan["records"]),),
        ),
        "gate_ids": (
            np.dtype("u1"),
            (int(pack_plan["records"]),),
        ),
        "context_evaluation_input_ids": (
            np.dtype("<u2"),
            (
                int(plan["selection"]["gate_evaluation_records"]),
                int(plan["selection"]["gate_evaluation_context"]),
            ),
        ),
        "context_evaluation_probe_target_ids": (
            np.dtype("<u2"),
            (int(plan["selection"]["gate_evaluation_records"]),),
        ),
        "context_evaluation_probe_positions": (
            np.dtype("<u4"),
            (int(plan["selection"]["gate_evaluation_records"]),),
        ),
        "context_evaluation_split_fingerprint": (
            np.dtype("<u8"),
            (int(plan["selection"]["gate_evaluation_records"]), 4),
        ),
    }
    initialized = []
    for name, (dtype, shape) in specifications.items():
        path = arrays_root / f"{name}.npy"
        if path.exists():
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != dtype:
                raise RuntimeError(f"existing context array {name} has wrong shape/dtype")
        else:
            array = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=dtype,
                shape=shape,
            )
            array.flush()
            del array
        initialized.append(
            {
                "name": name,
                "path": path.relative_to(output_root).as_posix(),
                "dtype": dtype.name,
                "shape": list(shape),
            }
        )
    payload: dict[str, Any] = {
        "schema": "metis.context-array-initialization/v1",
        "created_at": utc_now(),
        "pack_plan_sha256": pack_plan["plan_sha256"],
        "arrays": initialized,
    }
    payload["initialization_sha256"] = _json_sha256(payload)
    atomic_json(output_root / "ARRAYS_INITIALIZED.json", payload)
    return payload


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.name.endswith(".jsonl.zst"):
        raw = path.open("rb")
        handle: Any = io.BufferedReader(
            zstd.ZstdDecompressor().stream_reader(raw), buffer_size=1 << 22
        )
    elif path.name.endswith(".jsonl.gz"):
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        for line in handle:
            if line.strip():
                payload = _loads(line)
                if not isinstance(payload, dict):
                    raise RuntimeError(f"invalid JSON row in {path}")
                yield payload


@dataclass
class _TokenChunk:
    ids: np.ndarray
    starts: np.ndarray


def _encoded_slice(
    tokenizer: Tokenizer,
    row: Mapping[str, Any],
    *,
    eos_id: int,
) -> np.ndarray:
    text = str(row.get("text") or "")
    content_sha = str(row.get("content_sha256") or "")
    if not text or not _SHA256.fullmatch(content_sha):
        raise RuntimeError("context selection row has no text/content hash")
    from .final_dedup import content_sha256

    if content_sha256(text).hex() != content_sha:
        raise RuntimeError("context selection text changed after final dedup")
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    ids.append(eos_id)
    start = int(row["token_start"])
    count = int(row["token_count"])
    selected = ids[start : start + count]
    if len(selected) != count:
        raise RuntimeError(
            f"context token slice is out of bounds for {row.get('source_id')}:{row.get('doc_id')}"
        )
    return np.asarray(selected, dtype=np.dtype("<u2"))


def pack_context_evaluation(
    output_root: Path,
    *,
    plan: Mapping[str, Any],
    tokenizer_path: Path,
    tokenizer_sha256: str,
    eos_id: int,
) -> dict[str, Any]:
    """Pack a disjoint natural-long calibration set and associative probe."""

    validate_context_plan(plan)
    selection_path = output_root / "evaluation" / "records.jsonl.zst"
    if not selection_path.is_file():
        raise RuntimeError("context gate-evaluation selection is missing")
    receipt_path = output_root / "evaluation" / "PACK_RECEIPT.json"
    records = int(plan["selection"]["gate_evaluation_records"])
    context = int(plan["selection"]["gate_evaluation_context"])
    prefix_tokens = int(
        plan["selection"]["gate_evaluation_probe_prefix_tokens"]
    )
    if receipt_path.is_file():
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != "metis.context-gate-evaluation-pack/v1"
            or payload.get("receipt_sha256")
            != _json_sha256(payload, omit=("receipt_sha256",))
            or payload.get("selection_file_sha256")
            != _file_sha256(selection_path)
            or payload.get("tokenizer_sha256") != tokenizer_sha256
            or int(payload.get("records", -1)) != records
            or int(payload.get("context", -1)) != context
        ):
            raise RuntimeError(
                "existing context gate-evaluation receipt is stale"
            )
        return payload
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if (
        _file_sha256(tokenizer_path) != tokenizer_sha256
        or tokenizer.get_vocab_size(with_added_tokens=True) != 65_536
        or not 0 <= eos_id < 65_536
    ):
        raise RuntimeError("context gate evaluation tokenizer changed")
    arrays_root = output_root / "arrays"
    input_ids = np.load(
        arrays_root / "context_evaluation_input_ids.npy",
        mmap_mode="r+",
        allow_pickle=False,
    )
    target_ids = np.load(
        arrays_root / "context_evaluation_probe_target_ids.npy",
        mmap_mode="r+",
        allow_pickle=False,
    )
    positions = np.load(
        arrays_root / "context_evaluation_probe_positions.npy",
        mmap_mode="r+",
        allow_pickle=False,
    )
    fingerprints = np.load(
        arrays_root / "context_evaluation_split_fingerprint.npy",
        mmap_mode="r+",
        allow_pickle=False,
    )
    if (
        input_ids.shape != (records, context)
        or target_ids.shape != (records,)
        or positions.shape != (records,)
        or fingerprints.shape != (records, 4)
    ):
        raise RuntimeError("context gate-evaluation arrays have wrong shapes")
    seen: set[bytes] = set()
    rows = 0
    minimum_distance = 65_536
    for row in _iter_rows(selection_path):
        if rows >= records:
            raise RuntimeError("context gate evaluation has excess rows")
        encoded = _encoded_slice(tokenizer, row, eos_id=eos_id)
        if encoded.shape != (context,):
            raise RuntimeError(
                "context gate evaluation row is not exactly deploy length"
            )
        identity = (
            f"{row.get('content_sha256')}\0{int(row['token_start'])}"
        ).encode("utf-8")
        digest = hashlib.sha256(identity).digest()
        if digest in seen:
            raise RuntimeError("context gate evaluation duplicated a split")
        seen.add(digest)
        latest_anchor = context - minimum_distance - prefix_tokens - 1
        if latest_anchor <= 1_024:
            raise RuntimeError("context associative probe geometry is invalid")
        anchor = 1_024 + (
            int.from_bytes(digest[:8], "big") % (latest_anchor - 1_024)
        )
        target = int(encoded[anchor + prefix_tokens])
        query_start = context - prefix_tokens
        encoded[query_start:context] = encoded[
            anchor : anchor + prefix_tokens
        ]
        input_ids[rows] = encoded
        target_ids[rows] = np.uint16(target)
        positions[rows] = np.uint32(context - 1)
        fingerprints[rows] = np.frombuffer(
            digest, dtype=np.dtype("<u8")
        )
        rows += 1
    if rows != records:
        raise RuntimeError(
            f"context gate evaluation has {rows} rows, expected {records}"
        )
    input_ids.flush()
    target_ids.flush()
    positions.flush()
    fingerprints.flush()
    array_hashes = {
        "context_evaluation_input_ids": _file_sha256(
            arrays_root / "context_evaluation_input_ids.npy"
        ),
        "context_evaluation_probe_target_ids": _file_sha256(
            arrays_root / "context_evaluation_probe_target_ids.npy"
        ),
        "context_evaluation_probe_positions": _file_sha256(
            arrays_root / "context_evaluation_probe_positions.npy"
        ),
        "context_evaluation_split_fingerprint": _file_sha256(
            arrays_root / "context_evaluation_split_fingerprint.npy"
        ),
    }
    del input_ids, target_ids, positions, fingerprints
    payload: dict[str, Any] = {
        "schema": "metis.context-gate-evaluation-pack/v1",
        "records": records,
        "context": context,
        "tail_tokens": int(
            plan["selection"]["gate_evaluation_tail_tokens"]
        ),
        "probe_prefix_tokens": prefix_tokens,
        "minimum_probe_distance": minimum_distance,
        "selection_file_sha256": _file_sha256(selection_path),
        "tokenizer_sha256": tokenizer_sha256,
        "array_sha256": array_hashes,
        "receipt_sha256": "",
    }
    payload["receipt_sha256"] = _json_sha256(
        payload, omit=("receipt_sha256",)
    )
    atomic_json(receipt_path, payload)
    return payload


def _natural_chunks(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Tokenizer,
    eos_id: int,
) -> Iterator[_TokenChunk]:
    previous: tuple[str, str, int] | None = None
    for row in rows:
        ids = _encoded_slice(tokenizer, row, eos_id=eos_id)
        starts = np.zeros(ids.shape, dtype=np.uint8)
        identity = (
            str(row.get("context_group_id") or row.get("doc_id")),
            str(row.get("doc_id")),
            int(row.get("token_start", 0)),
        )
        contiguous = bool(
            previous
            and previous[0] == identity[0]
            and previous[1] == identity[1]
            and previous[2] == identity[2]
        )
        if not contiguous and starts.size:
            starts[0] = 1
        previous = (
            identity[0],
            identity[1],
            identity[2] + int(row["token_count"]),
        )
        yield _TokenChunk(ids, starts)


def _short_chunks(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Tokenizer,
    eos_id: int,
    short_tokens: int,
) -> Iterator[_TokenChunk]:
    for row in rows:
        ids = _encoded_slice(tokenizer, row, eos_id=eos_id)
        starts = np.zeros(ids.shape, dtype=np.uint8)
        starts[::short_tokens] = 1
        yield _TokenChunk(ids, starts)


def _constructed_chunks(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Tokenizer,
    eos_id: int,
    chunk_tokens: int,
    expected_domain: str,
) -> Iterator[_TokenChunk]:
    def documents() -> Iterator[np.ndarray]:
        identity: tuple[str, str, str] | None = None
        fragments: list[np.ndarray] = []
        for row in rows:
            if str(row.get("context_domain") or "") != expected_domain:
                raise RuntimeError(
                    "dependency construction crossed its sealed domain"
                )
            next_identity = (
                str(row.get("source_id") or ""),
                str(row.get("doc_id") or ""),
                str(row.get("content_sha256") or ""),
            )
            if not all(next_identity):
                raise RuntimeError(
                    "dependency construction row has no document identity"
                )
            if identity is not None and next_identity != identity:
                yield np.concatenate(fragments)
                fragments = []
            identity = next_identity
            fragments.append(
                _encoded_slice(tokenizer, row, eos_id=eos_id)
            )
        if fragments:
            yield np.concatenate(fragments)

    iterator = iter(documents())
    while True:
        try:
            first = next(iterator)
        except StopIteration:
            return
        try:
            second = next(iterator)
        except StopIteration:
            # Exact token accounting can leave one document in a domain task.
            # Split that original document into two preserved-text streams so
            # the construction remains domain-local without duplicating or
            # dropping a token.
            if first.size <= 1:
                encoded = [first]
            else:
                split = first.size // 2
                encoded = [first[:split], first[split:]]
        else:
            encoded = [first, second]
        positions = [0 for _ in encoded]
        turn = 0
        emitted_any = False
        while any(
            position < values.size
            for position, values in zip(positions, encoded, strict=True)
        ):
            lane = turn % len(encoded)
            turn += 1
            if positions[lane] >= encoded[lane].size:
                continue
            start = positions[lane]
            end = min(encoded[lane].size, start + chunk_tokens)
            ids = encoded[lane][start:end]
            starts = np.zeros(ids.shape, dtype=np.uint8)
            if not emitted_any and starts.size:
                starts[0] = 1
            emitted_any = True
            positions[lane] = end
            yield _TokenChunk(ids, starts)


def _sequence_lengths(tokens: int, records: int, maximum: int) -> list[int]:
    values = _balanced_parts(tokens, records)
    if min(values) <= 1 or max(values) > maximum:
        raise RuntimeError("context task sequence lengths are invalid")
    return values


def pack_context_task(
    output_root: Path,
    *,
    pack_plan: Mapping[str, Any],
    plan: Mapping[str, Any],
    tokenizer_path: Path,
    tokenizer_sha256: str,
    eos_id: int,
    pad_id: int,
    task_index: int,
) -> dict[str, Any]:
    validate_context_plan(plan)
    tasks = [
        dict(row)
        for row in pack_plan["tasks"]
        if int(row["task_index"]) == task_index
    ]
    if len(tasks) != 1:
        raise ValueError(f"unknown context pack task {task_index}")
    task = tasks[0]
    selection_path = output_root / str(task["selection_path"])
    if not selection_path.is_file():
        raise RuntimeError(f"context selection file is missing: {selection_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if _file_sha256(tokenizer_path) != tokenizer_sha256:
        raise RuntimeError("context pack tokenizer hash changed")
    if tokenizer.get_vocab_size(with_added_tokens=True) != 65_536:
        raise RuntimeError("context pack requires the final 65,536-token tokenizer")
    selected_tokens = sum(
        int(row["token_count"]) for row in _iter_rows(selection_path)
    )
    if selected_tokens != int(task["active_tokens"]):
        raise RuntimeError(
            f"context selection task {task_index} has {selected_tokens:,} tokens, "
            f"expected {int(task['active_tokens']):,}"
        )
    lane = str(task["lane"])
    if lane == "natural_long":
        chunks = _natural_chunks(
            _iter_rows(selection_path), tokenizer=tokenizer, eos_id=eos_id
        )
    elif lane == "short_replay":
        chunks = _short_chunks(
            _iter_rows(selection_path),
            tokenizer=tokenizer,
            eos_id=eos_id,
            short_tokens=int(plan["selection"]["short_replay_document_tokens"]),
        )
    elif lane == "dependency_constructed":
        expected_domain = str(task.get("domain") or "")
        if not expected_domain:
            raise RuntimeError(
                "dependency construction task has no sealed domain"
            )
        chunks = _constructed_chunks(
            _iter_rows(selection_path),
            tokenizer=tokenizer,
            eos_id=eos_id,
            chunk_tokens=int(plan["selection"]["dependency_chunk_tokens"]),
            expected_domain=expected_domain,
        )
    else:
        raise RuntimeError(f"unknown context lane {lane}")

    arrays_root = output_root / "arrays"
    input_ids = np.load(arrays_root / "input_ids.npy", mmap_mode="r+", allow_pickle=False)
    document_start = np.load(
        arrays_root / "document_start.npy", mmap_mode="r+", allow_pickle=False
    )
    sequence_lengths = np.load(
        arrays_root / "sequence_lengths.npy", mmap_mode="r+", allow_pickle=False
    )
    gate_ids = np.load(arrays_root / "gate_ids.npy", mmap_mode="r+", allow_pickle=False)
    opening = int(task["record_start"])
    closing = int(task["record_end"])
    lengths = _sequence_lengths(
        int(task["active_tokens"]),
        int(task["records"]),
        int(plan["train_context"]),
    )
    input_ids[opening:closing] = np.uint16(pad_id)
    document_start[opening:closing] = np.uint8(0)
    sequence_lengths[opening:closing] = np.asarray(lengths, dtype=np.dtype("<u4"))
    gate_ids[opening:closing] = np.uint8(task["gate_index"])

    chunk_iterator = iter(chunks)
    current: _TokenChunk | None = None
    current_offset = 0
    written = 0
    for local_record, active_length in enumerate(lengths):
        record_index = opening + local_record
        position = 0
        first = True
        while position < active_length:
            if current is None or current_offset == current.ids.size:
                try:
                    current = next(chunk_iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"context pack task {task_index} exhausted its token stream"
                    ) from exc
                current_offset = 0
            take = min(active_length - position, current.ids.size - current_offset)
            input_ids[record_index, position : position + take] = current.ids[
                current_offset : current_offset + take
            ]
            document_start[record_index, position : position + take] = current.starts[
                current_offset : current_offset + take
            ]
            if first:
                document_start[record_index, 0] = 1
                first = False
            position += take
            current_offset += take
            written += take
    if current is not None and current_offset != current.ids.size:
        raise RuntimeError(f"context pack task {task_index} left a partial token chunk")
    try:
        next(chunk_iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError(f"context pack task {task_index} has unused selected tokens")
    if written != int(task["active_tokens"]):
        raise RuntimeError(f"context pack task {task_index} wrote the wrong token count")
    input_ids.flush()
    document_start.flush()
    sequence_lengths.flush()
    gate_ids.flush()
    slice_hashes = {
        "input_ids": _array_slice_sha256(input_ids, opening, closing),
        "document_start": _array_slice_sha256(
            document_start, opening, closing
        ),
        "sequence_lengths": _array_slice_sha256(
            sequence_lengths, opening, closing
        ),
        "gate_ids": _array_slice_sha256(gate_ids, opening, closing),
    }
    del input_ids, document_start, sequence_lengths, gate_ids

    receipt: dict[str, Any] = {
        "schema": CONTEXT_PACK_TASK_SCHEMA,
        "release": plan["release"],
        "task_index": task_index,
        "pack_plan_sha256": pack_plan["plan_sha256"],
        "selection_file_sha256": _file_sha256(selection_path),
        "selection_file_bytes": selection_path.stat().st_size,
        "tokenizer_sha256": tokenizer_sha256,
        "gate_index": int(task["gate_index"]),
        "lane": lane,
        "domain": (
            str(task["domain"])
            if lane == "dependency_constructed"
            else None
        ),
        "record_start": opening,
        "record_end": closing,
        "records": int(task["records"]),
        "active_tokens": written,
        "minimum_sequence_tokens": min(lengths),
        "maximum_sequence_tokens": max(lengths),
        "array_slice_sha256": slice_hashes,
        "completed_at": utc_now(),
    }
    receipt["receipt_sha256"] = _json_sha256(receipt)
    receipt_path = output_root / "pack-receipts" / f"task-{task_index:06d}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_path, receipt)
    return receipt


def _array_slice_sha256(
    array: np.ndarray,
    opening: int,
    closing: int,
    *,
    rows_per_chunk: int = 32,
) -> str:
    digest = hashlib.sha256()
    for start in range(opening, closing, rows_per_chunk):
        end = min(closing, start + rows_per_chunk)
        digest.update(np.ascontiguousarray(array[start:end]).tobytes())
    return digest.hexdigest()


def validate_context_pack_receipt(
    output_root: Path,
    *,
    pack_plan: Mapping[str, Any],
    tokenizer_sha256: str,
    task_index: int,
    deep_array_validation: bool = True,
) -> dict[str, Any]:
    matching = [
        dict(row)
        for row in pack_plan["tasks"]
        if int(row["task_index"]) == task_index
    ]
    if len(matching) != 1:
        raise ValueError(f"unknown context pack task {task_index}")
    task = matching[0]
    path = output_root / "pack-receipts" / f"task-{task_index:06d}.json"
    if not path.is_file():
        raise RuntimeError(f"context pack receipt is missing: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    selection_path = output_root / str(task["selection_path"])
    if (
        receipt.get("schema") != CONTEXT_PACK_TASK_SCHEMA
        or receipt.get("receipt_sha256")
        != _json_sha256(receipt, omit=("receipt_sha256",))
        or receipt.get("pack_plan_sha256") != pack_plan["plan_sha256"]
        or receipt.get("tokenizer_sha256") != tokenizer_sha256
        or int(receipt.get("task_index", -1)) != task_index
        or receipt.get("lane") != task["lane"]
        or receipt.get("domain")
        != (
            task.get("domain")
            if task["lane"] == "dependency_constructed"
            else None
        )
        or int(receipt.get("record_start", -1)) != int(task["record_start"])
        or int(receipt.get("record_end", -1)) != int(task["record_end"])
        or int(receipt.get("active_tokens", -1)) != int(task["active_tokens"])
        or not selection_path.is_file()
        or selection_path.stat().st_size
        != int(receipt.get("selection_file_bytes", -1))
        or _file_sha256(selection_path) != receipt.get("selection_file_sha256")
    ):
        raise RuntimeError(f"context pack receipt is stale or corrupt: {path}")
    if deep_array_validation:
        arrays_root = output_root / "arrays"
        opening = int(task["record_start"])
        closing = int(task["record_end"])
        observed: dict[str, str] = {}
        for name in (
            "input_ids",
            "document_start",
            "sequence_lengths",
            "gate_ids",
        ):
            array = np.load(
                arrays_root / f"{name}.npy", mmap_mode="r", allow_pickle=False
            )
            observed[name] = _array_slice_sha256(array, opening, closing)
            del array
        if observed != receipt.get("array_slice_sha256"):
            raise RuntimeError(
                f"context pack task {task_index} array slice changed after completion"
            )
    return receipt


def verify_and_seal_context_release(
    output_root: Path,
    *,
    plan: Mapping[str, Any],
    pack_plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    tokenizer_contract: Mapping[str, Any],
    context_plan_path: Path,
) -> dict[str, Any]:
    validate_context_plan(plan)
    if selection.get("selection_sha256") != _json_sha256(
        selection, omit=("selection_sha256",)
    ):
        raise RuntimeError("context selection failed its self-hash")
    receipts = []
    for task in pack_plan["tasks"]:
        path = output_root / "pack-receipts" / (
            f"task-{int(task['task_index']):06d}.json"
        )
        if not path.is_file():
            raise RuntimeError(f"context pack receipt is missing: {path}")
        receipt = validate_context_pack_receipt(
            output_root,
            pack_plan=pack_plan,
            tokenizer_sha256=str(tokenizer_contract["tokenizer_sha256"]),
            task_index=int(task["task_index"]),
            deep_array_validation=True,
        )
        receipts.append(receipt)
    if sum(int(row["active_tokens"]) for row in receipts) != CONTEXT_TOKEN_BUDGET:
        raise RuntimeError("context pack receipts do not sum to 18B")

    arrays_root = output_root / "arrays"
    arrays = {
        "input_ids": np.load(
            arrays_root / "input_ids.npy", mmap_mode="r", allow_pickle=False
        ),
        "document_start": np.load(
            arrays_root / "document_start.npy", mmap_mode="r", allow_pickle=False
        ),
        "sequence_lengths": np.load(
            arrays_root / "sequence_lengths.npy", mmap_mode="r", allow_pickle=False
        ),
        "gate_ids": np.load(
            arrays_root / "gate_ids.npy", mmap_mode="r", allow_pickle=False
        ),
    }
    expected_records = int(pack_plan["records"])
    sequence_length = int(pack_plan["sequence_length"])
    if (
        arrays["input_ids"].shape != (expected_records, sequence_length)
        or arrays["input_ids"].dtype != np.dtype("<u2")
        or arrays["document_start"].shape != (expected_records, sequence_length)
        or arrays["document_start"].dtype != np.uint8
        or arrays["sequence_lengths"].shape != (expected_records,)
        or arrays["sequence_lengths"].dtype != np.dtype("<u4")
        or arrays["gate_ids"].shape != (expected_records,)
        or arrays["gate_ids"].dtype != np.uint8
    ):
        raise RuntimeError("context compact arrays have an invalid shape or dtype")
    lengths = np.asarray(arrays["sequence_lengths"], dtype=np.int64)
    if (
        int(lengths.sum()) != CONTEXT_TOKEN_BUDGET
        or np.any(lengths <= 1)
        or np.any(lengths > sequence_length)
        or not np.all(
            arrays["document_start"][
                np.arange(expected_records), np.zeros(expected_records, dtype=np.int64)
            ]
            == 1
        )
        or np.any(arrays["gate_ids"] > 2)
    ):
        raise RuntimeError("context compact array content contract failed")
    gate_tokens = [
        int(lengths[np.asarray(arrays["gate_ids"]) == gate].sum())
        for gate in range(3)
    ]
    if gate_tokens != [6_000_000_000, 6_000_000_000, 6_000_000_000]:
        raise RuntimeError(f"context compact gate tokens are not exact: {gate_tokens}")
    arrays.clear()

    tokenizer_sha = str(tokenizer_contract["tokenizer_sha256"])
    canonical_map_sha = str(
        tokenizer_contract["ngram_canonical_map_self_sha256"]
    )
    canonical_ids_sha = str(tokenizer_contract["ngram_canonical_ids_sha256"])
    evaluation_receipt_path = (
        output_root / "evaluation" / "PACK_RECEIPT.json"
    )
    if not evaluation_receipt_path.is_file():
        raise RuntimeError("context gate-evaluation pack receipt is missing")
    evaluation_receipt = json.loads(
        evaluation_receipt_path.read_text(encoding="utf-8")
    )
    evaluation_records = int(
        plan["selection"]["gate_evaluation_records"]
    )
    evaluation_context = int(
        plan["selection"]["gate_evaluation_context"]
    )
    evaluation_hashes = evaluation_receipt.get("array_sha256")
    if (
        evaluation_receipt.get("schema")
        != "metis.context-gate-evaluation-pack/v1"
        or evaluation_receipt.get("receipt_sha256")
        != _json_sha256(
            evaluation_receipt, omit=("receipt_sha256",)
        )
        or evaluation_receipt.get("tokenizer_sha256") != tokenizer_sha
        or int(evaluation_receipt.get("records", -1))
        != evaluation_records
        or int(evaluation_receipt.get("context", -1))
        != evaluation_context
        or not isinstance(evaluation_hashes, Mapping)
        or set(evaluation_hashes) != set(CONTEXT_EVALUATION_ARRAYS)
    ):
        raise RuntimeError("context gate-evaluation receipt is invalid")
    evaluation_arrays = {
        name: np.load(
            arrays_root / f"{name}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for name in CONTEXT_EVALUATION_ARRAYS
    }
    if (
        evaluation_arrays["context_evaluation_input_ids"].shape
        != (evaluation_records, evaluation_context)
        or evaluation_arrays["context_evaluation_input_ids"].dtype
        != np.dtype("<u2")
        or evaluation_arrays[
            "context_evaluation_probe_target_ids"
        ].shape
        != (evaluation_records,)
        or evaluation_arrays[
            "context_evaluation_probe_target_ids"
        ].dtype
        != np.dtype("<u2")
        or evaluation_arrays[
            "context_evaluation_probe_positions"
        ].shape
        != (evaluation_records,)
        or evaluation_arrays[
            "context_evaluation_probe_positions"
        ].dtype
        != np.dtype("<u4")
        or evaluation_arrays[
            "context_evaluation_split_fingerprint"
        ].shape
        != (evaluation_records, 4)
        or evaluation_arrays[
            "context_evaluation_split_fingerprint"
        ].dtype
        != np.dtype("<u8")
        or np.any(
            evaluation_arrays["context_evaluation_probe_positions"]
            != evaluation_context - 1
        )
        or len(
            {
                tuple(int(value) for value in row)
                for row in evaluation_arrays[
                    "context_evaluation_split_fingerprint"
                ]
            }
        )
        != evaluation_records
    ):
        raise RuntimeError("context gate-evaluation arrays are invalid")
    for name, array in evaluation_arrays.items():
        path = arrays_root / f"{name}.npy"
        if _file_sha256(path) != evaluation_hashes.get(name):
            raise RuntimeError(
                f"context gate-evaluation array changed: {name}"
            )
        del array
    evaluation_arrays.clear()
    array_specs: dict[str, Any] = {}
    payload_paths: list[Path] = []
    for name in (
        "input_ids",
        "document_start",
        "sequence_lengths",
        "gate_ids",
        *CONTEXT_EVALUATION_ARRAYS,
    ):
        path = arrays_root / f"{name}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        array_specs[name] = {
            "path": path.relative_to(output_root).as_posix(),
            "dtype": array.dtype.name,
            "shape": list(array.shape),
        }
        del array
        payload_paths.append(path)
    bundle: dict[str, Any] = {
        "schema": MMAP_BUNDLE_SCHEMA,
        "stage": "context_extension",
        "family": "shared",
        "parent_checkpoint_sha256": "unbound",
        "tokenizer_sha256": tokenizer_sha,
        "vocabulary_size": 65_536,
        "records": expected_records,
        "sequence_length": sequence_length,
        "compact_layout": COMPACT_LAYOUT,
        "unique_active_tokens": CONTEXT_TOKEN_BUDGET,
        "training_tokens": CONTEXT_TOKEN_BUDGET,
        "checkpoint_gates": list(CONTEXT_GATES),
        "gate_policy": dict(plan["gate_policy"]),
        "ngram_canonical_map_self_sha256": canonical_map_sha,
        "ngram_canonical_ids_sha256": canonical_ids_sha,
        "training": {
            "epochs": 1,
            "micro_batch_size": 1,
            "gradient_accumulation": 1,
            "shuffle_seed": int(plan["selection"]["seed"]),
            "checkpoint_interval_steps": 0,
            "learning_rate": 1.0e-4,
            "minimum_learning_rate_ratio": 0.10,
            "warmup_steps": 32,
            "gradient_clip": 1.0,
        },
        "arrays": array_specs,
        "pack_plan_sha256": pack_plan["plan_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "long_range_calibration": {
            "required": True,
            "implementation": "metis.long-range-information/v1",
            "sample_records": evaluation_records,
            "context": evaluation_context,
            "tail_tokens": int(
                plan["selection"]["gate_evaluation_tail_tokens"]
            ),
            "probe_prefix_tokens": int(
                plan["selection"][
                    "gate_evaluation_probe_prefix_tokens"
                ]
            ),
            "score": "full_context_nll_gain_over_4096",
            "training_disjoint": True,
            "domain_records": dict(
                selection["gate_evaluation_domains"]
            ),
            "evaluation_pack_receipt_sha256": evaluation_receipt[
                "receipt_sha256"
            ],
        },
        "bundle_sha256": "",
    }
    bundle["bundle_sha256"] = _json_sha256(bundle, omit=("bundle_sha256",))
    bundle_path = output_root / "BUNDLE.json"
    atomic_json(bundle_path, bundle)
    payload_paths.append(bundle_path)

    copied_plan = output_root / "CONTEXT_PLAN.yaml"
    if context_plan_path.resolve() != copied_plan.resolve():
        copied_plan.write_bytes(context_plan_path.read_bytes())
    payload_paths.append(copied_plan)
    selection_path = output_root / "SELECTION.json"
    pack_plan_path = output_root / "PACK_PLAN.json"
    payload_paths.extend([selection_path, pack_plan_path])
    payload_paths.append(evaluation_receipt_path)
    license_path = output_root / "LICENSE_LEDGER.jsonl"
    with license_path.open("w", encoding="utf-8") as handle:
        for source_id in sorted(selection["license_tokens"]):
            handle.write(
                json.dumps(
                    {
                        "source_id": source_id,
                        "observed_license_tokens": selection["license_tokens"][
                            source_id
                        ],
                        "training_recipe_disposition": "verified_for_training",
                        "data_publication_requires_separate_review": True,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    payload_paths.append(license_path)
    pack_receipt_index = output_root / "PACK_RECEIPTS.json"
    atomic_json(
        pack_receipt_index,
        {
            "schema": "metis.context-pack-receipts/v1",
            "receipts": [
                {
                    "task_index": int(receipt["task_index"]),
                    "receipt_sha256": receipt["receipt_sha256"],
                }
                for receipt in receipts
            ],
        },
    )
    payload_paths.append(pack_receipt_index)
    file_records = [
        _file_record(path, root=output_root)
        for path in sorted(payload_paths)
    ]
    envelope: dict[str, Any] = {
        "envelope_schema": SEALED_ARTIFACT_SCHEMA,
        "schema": CONTEXT_DATA_SCHEMA,
        "complete": True,
        "files": file_records,
        "metadata": {
            "backend_contract": MMAP_BUNDLE_SCHEMA,
            "bundle_manifest": bundle_path.relative_to(output_root).as_posix(),
            "records": expected_records,
            "tokens": CONTEXT_TOKEN_BUDGET,
            "base_context": 4_096,
            "train_context": sequence_length,
            "deploy_context": int(plan["deploy_context"]),
            "single_jump": True,
            "long_sequence_fraction": 0.90,
            "base_sequence_fraction": 0.10,
            "pretrain_style_fraction": 0.80,
            "synthetic_long_context_fraction": 0.20,
            "document_length_histogram_verified": True,
            "checkpoint_gates": list(CONTEXT_GATES),
            "compact_layout": COMPACT_LAYOUT,
            "gate_evaluation_records": evaluation_records,
            "gate_evaluation_context": evaluation_context,
            "gate_evaluation_domains": dict(
                selection["gate_evaluation_domains"]
            ),
            "gate_evaluation_pack_receipt_sha256": evaluation_receipt[
                "receipt_sha256"
            ],
            "tokenizer_sha256": tokenizer_sha,
            "context_plan_sha256": _file_sha256(copied_plan),
            "selection_sha256": selection["selection_sha256"],
            "pack_plan_sha256": pack_plan["plan_sha256"],
            "license_ledger": license_path.relative_to(output_root).as_posix(),
        },
        "manifest_sha256": "",
    }
    envelope["manifest_sha256"] = _json_sha256(
        envelope, omit=("manifest_sha256",)
    )
    manifest_path = output_root / "MANIFEST.json"
    atomic_json(manifest_path, envelope)
    verification: dict[str, Any] = {
        "schema": CONTEXT_VERIFICATION_SCHEMA,
        "release": plan["release"],
        "verified_at": utc_now(),
        "manifest": str(manifest_path),
        "manifest_file_sha256": _file_sha256(manifest_path),
        "manifest_sha256": envelope["manifest_sha256"],
        "bundle_sha256": bundle["bundle_sha256"],
        "active_tokens": CONTEXT_TOKEN_BUDGET,
        "records": expected_records,
        "checkpoint_gates": list(CONTEXT_GATES),
        "gate_tranche_tokens": gate_tokens,
        "compact_bytes": sum(
            (output_root / spec["path"]).stat().st_size
            for spec in array_specs.values()
        ),
        "selection_sha256": selection["selection_sha256"],
        "pack_plan_sha256": pack_plan["plan_sha256"],
        "ok": True,
    }
    verification["verification_sha256"] = _json_sha256(verification)
    atomic_json(output_root / "VERIFICATION.json", verification)
    return verification
