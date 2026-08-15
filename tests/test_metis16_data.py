from __future__ import annotations

import copy
import concurrent.futures
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
import struct
import threading
import time
from pathlib import Path
from unittest import mock

from metis_data.decontaminate import ContaminationIndex, benchmark_genealogy_match
from metis_data.code_dedup import (
    code_hygiene_reason,
    find_code_duplicates,
    load_code_removals,
    write_code_signatures,
)
from metis_data.dedup import deduplicate_records
from metis_data.final_dedup import find_final_duplicates, load_final_removals, write_final_signatures
from metis_data.holdouts import _benchmark_fragments, _benchmark_jobs
from metis_data.config import load_profile, load_yaml, validate_storage_root
from metis_data.context_extension import (
    allocate_context_replacements,
    build_context_pack_plan,
    context_evaluation_domain_targets,
    context_lane_quota_rows,
    structural_evidence,
)
from metis_data.context_manifest import (
    CONTEXT_GATES,
    context_candidate_targets,
    context_retrieval_reserve_tokens,
    validate_context_plan,
)
from metis_data.build_inputs import prepare_build_inputs
from metis_data.handoff import write_acquisition_handoff, verify_acquisition_handoff
from metis_data.local_download import (
    _lane_configuration,
    _pending_task_waves,
    _run_task_in_lanes,
    _supervisor_lock,
)
from metis_data.manifest import (
    MINIMUM_FRESHNESS_SHARE,
    load_manifest,
    matches_any,
    total_phase_tokens,
    validate_manifest,
)
from metis_data.packing import pack_release
from metis_data.quality import evaluate_quality
from metis_data.runtime_lock import runtime_contract
from metis_data.source_lock import source_lock_sha256
from metis_data.selection import build_selection, hamilton_apportion, replay_quotas, unique_quotas
from metis_data.tokenizer import train_tokenizer, validate_tokenizer
from metis_data.training_contract import phase_for_token
from metis_data.datatrove_blocks import (
    build_priority_minhash_removals,
    load_contamination_index,
    save_contamination_index,
)
from metis_data.slurm import BUILD_GRAPH, _indices_expression, _submit_array_chunks
from metis_data import stage_runner
from metis_data.state import StateStore

from tests.contamination_fixtures import write_contamination_inputs


class ManifestTests(unittest.TestCase):
    def test_production_manifest_is_exact(self) -> None:
        result = validate_manifest()
        self.assertTrue(result.ok, result.errors)
        manifest = result.manifest
        # Refitted from measured supply. The 1T/950B/50B plan was aspirational:
        # token_count measured 849.5B available, so the schedule was rebuilt at
        # 90% of what each category actually has. These are pinned so the mix
        # cannot drift by accident, but they are a release-specific fact, not a
        # law -- refitting the mix is expected to move them deliberately.
        schedule = manifest["schedule"]
        self.assertEqual(schedule["target_tokens"], 804_755_808_835)
        self.assertEqual(schedule["unique_target_tokens"], 764_518_018_395)
        self.assertEqual(schedule["replay_target_tokens"], 40_237_790_440)
        # The relationships hold through any refit and are what actually matter.
        self.assertEqual(
            schedule["unique_target_tokens"] + schedule["replay_target_tokens"],
            schedule["target_tokens"],
        )
        self.assertEqual(
            sum(
                int(phase["unique_tokens"])
                for phase in schedule["phases"].values()
            ),
            schedule["unique_target_tokens"],
        )
        self.assertEqual(
            sum(
                int(phase["replay_tokens"])
                for phase in schedule["phases"].values()
            ),
            schedule["replay_target_tokens"],
        )
        self.assertEqual(
            schedule["phases"]["phase_b"]["unique_tokens"],
            209_668_641_449,
        )
        self.assertEqual(
            schedule["phases"]["phase_b"]["replay_tokens"],
            0,
        )
        # Was 90B across four buckets, then pinned to a constant. Both are the
        # wrong shape: the layer is as large as the fresh sources actually
        # supply, so a constant only forces the declaration and the sources to
        # be edited in lockstep. Pin the two invariants the validator enforces
        # instead -- the declaration matches its sources, and the layer is a
        # meaningful share rather than a rounding error.
        fresh_tokens = sum(
            total_phase_tokens(source)
            for source in manifest["sources"]
            if source.get("provenance", {}).get("fresh")
        )
        self.assertEqual(
            manifest["freshness_layer"]["target_tokens"], fresh_tokens
        )
        self.assertGreaterEqual(
            fresh_tokens,
            MINIMUM_FRESHNESS_SHARE * manifest["schedule"]["total_tokens"],
        )
        self.assertEqual(manifest["tokenizer"]["vocabulary_size_including_special_tokens"], 65_536)
        source_ids = [source["id"] for source in manifest["sources"]]
        self.assertGreaterEqual(len(source_ids), 50)
        self.assertEqual(len(source_ids), len(set(source_ids)))
        generated_or_transformed = sum(
            sum(source["phase_tokens"].values())
            for source in manifest["sources"]
            if source["provenance"].get("generated") or source["provenance"].get("transformed")
        )
        self.assertEqual(generated_or_transformed, 93_000_000_000)

    def test_phase_c_contains_no_generated_sources(self) -> None:
        manifest = load_manifest()
        offenders = [
            source["id"]
            for source in manifest["sources"]
            if source["provenance"].get("generated") and source["phase_tokens"].get("phase_c", 0)
        ]
        self.assertEqual(offenders, [])

    def test_pretraining_phase_boundaries_are_token_based(self) -> None:
        contract = Path(__file__).resolve().parents[1] / "configs" / "metis16" / "pretraining.yaml"
        self.assertEqual(phase_for_token(contract, 0), "phase_a")
        self.assertEqual(phase_for_token(contract, 699_999_999_999), "phase_a")
        self.assertEqual(phase_for_token(contract, 700_000_000_000), "phase_b")
        self.assertEqual(phase_for_token(contract, 950_000_000_000), "phase_c")

    def test_hugging_face_double_star_patterns_include_repository_root(self) -> None:
        self.assertTrue(matches_any("data.parquet", ["**/*.parquet"]))
        self.assertTrue(matches_any("nested/data.parquet", ["**/*.parquet"]))
        self.assertFalse(matches_any("data.jsonl", ["**/*.parquet"]))

    def test_context_extension_plan_is_exact_and_autonomously_gated(self) -> None:
        manifest = load_manifest()
        plan = manifest["context_extension_plan"]
        validate_context_plan(plan, base_manifest=manifest)
        self.assertEqual(
            sum(int(row["tokens"]) for row in plan["sources"]),
            18_000_000_000,
        )
        self.assertEqual(tuple(plan["checkpoint_gates"]), CONTEXT_GATES)
        self.assertEqual(
            plan["selection"]["gate_evaluation_records"], 384
        )
        self.assertTrue(
            plan["selection"]["long_range_filter"][
                "model_calibration_at_context_gates"
            ]
        )
        candidate = context_candidate_targets(plan)
        reserve = context_retrieval_reserve_tokens(plan)
        evaluation_domains = context_evaluation_domain_targets(plan)
        self.assertEqual(sum(evaluation_domains.values()), 384)
        self.assertEqual(
            set(evaluation_domains),
            {
                str(group["id"])
                for group in plan["fallbacks"]["groups"]
            },
        )
        for row in plan["sources"]:
            source_id = row["id"]
            self.assertEqual(
                candidate[source_id],
                int(row["tokens"]) + reserve[source_id],
            )

    def test_context_pack_plan_and_lane_quotas_hit_every_gate_exactly(self) -> None:
        plan = load_manifest()["context_extension_plan"]
        rows = context_lane_quota_rows(plan)
        for gate_index, gate in enumerate(CONTEXT_GATES):
            previous = CONTEXT_GATES[gate_index - 1] if gate_index else 0
            self.assertEqual(
                sum(
                    int(row["tokens"])
                    for row in rows
                    if int(row["gate_index"]) == gate_index
                ),
                gate - previous,
            )
        pack_plan = build_context_pack_plan(plan)
        self.assertEqual(pack_plan["pack_tasks"], 96)
        self.assertEqual(pack_plan["active_tokens"], 18_000_000_000)
        self.assertEqual(
            [
                sum(
                    int(task["active_tokens"])
                    for task in pack_plan["tasks"]
                    if int(task["gate_index"]) == gate
                )
                for gate in range(3)
            ],
            [6_000_000_000] * 3,
        )
        expected_domains = {
            str(group["id"]) for group in plan["fallbacks"]["groups"]
        }
        for gate_index in range(3):
            constructed = [
                task
                for task in pack_plan["tasks"]
                if (
                    int(task["gate_index"]) == gate_index
                    and task["lane"] == "dependency_constructed"
                )
            ]
            self.assertEqual(
                {str(task["domain"]) for task in constructed},
                expected_domains,
            )
            self.assertEqual(len(constructed), len(expected_domains))
            for task in constructed:
                expected_tokens = sum(
                    int(row["tokens"])
                    for row in rows
                    if (
                        int(row["gate_index"]) == gate_index
                        and row["lane"] == "dependency_constructed"
                        and row["domain"] == task["domain"]
                    )
                )
                self.assertEqual(
                    int(task["active_tokens"]),
                    expected_tokens,
                )

    def test_context_fallbacks_never_cross_domains(self) -> None:
        plan = load_manifest()["context_extension_plan"]
        requirements = [
            {
                "gate_index": 0,
                "gate_target_tokens": CONTEXT_GATES[0],
                "lane": "natural_long",
                "source_id": "metis_freshweb_2025",
                "domain": "general_reference",
                "tokens": 100,
            }
        ]
        available = {
            row["id"]: (
                100 if row["id"] == "wikimedia_reference" else 0
            )
            for row in plan["sources"]
        }
        allocation = allocate_context_replacements(
            plan,
            requirements=requirements,
            available_tokens=available,
        )
        self.assertEqual(len(allocation["assignments"]), 1)
        replacement = allocation["assignments"][0]
        self.assertEqual(
            replacement["actual_source_id"], "wikimedia_reference"
        )
        self.assertEqual(replacement["domain"], "general_reference")
        self.assertTrue(replacement["replacement"])

    def test_structural_long_range_prefilter_detects_real_dependencies(self) -> None:
        text = (
            "# Chapter I\nSee theorem above and refer to section below.\n\n"
            "import package\nclass Solver:\n    pass\n\n"
        ) * 3_000
        evidence = structural_evidence(
            text,
            {"repository": "metis/example"},
        )
        self.assertGreaterEqual(evidence["score"], 3)
        self.assertGreater(evidence["code_dependencies"], 8)


class QualityAndDedupTests(unittest.TestCase):
    def test_quality_gate_is_fail_closed_and_rejects_secrets(self) -> None:
        missing = evaluate_quality(
            "A sufficiently long educational explanation " * 30,
            profile_name="web_edu_v1",
            metadata={},
        )
        self.assertFalse(missing.keep)
        self.assertEqual(missing.reason, "missing_educational_score")
        secret = evaluate_quality(
            ("A normal technical explanation with credential AKIAABCDEFGHIJKLMNOP inside. " * 20),
            profile_name="web_general_v1",
            metadata={"quality_score": 0.99, "language_probability": 0.99},
        )
        self.assertFalse(secret.keep)
        self.assertEqual(secret.reason, "secret")

    def test_exact_dedup_keeps_higher_priority(self) -> None:
        records = [
            {"doc_id": "low", "text": "The same useful explanation.", "priority": 10},
            {"doc_id": "high", "text": "  THE same useful explanation.  ", "priority": 20},
            {"doc_id": "unique", "text": "A completely different document.", "priority": 5},
        ]
        result = deduplicate_records(records)
        self.assertEqual({row["doc_id"] for row in result.kept}, {"high", "unique"})
        self.assertEqual(result.removed[0]["doc_id"], "low")

    def test_decontamination_removes_exact_and_ngram_matches(self) -> None:
        holdout = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
        index = ContaminationIndex.build([holdout], ngram_size=5, minimum_matching_ngrams=2)
        self.assertEqual(index.reason(holdout), "benchmark_exact")
        self.assertEqual(index.reason("prefix one two three four five six seven suffix"), "benchmark_ngram")
        self.assertIsNone(index.reason("independent prose with no benchmark overlap at all"))

    def test_decontamination_catches_short_and_code_fragments(self) -> None:
        index = ContaminationIndex.build(
            [
                "Which planet is known as the red planet because of iron oxide?",
                "def benchmark_secret(value):\n    return value * value + 17\n",
            ],
            ngram_size=13,
            minimum_matching_ngrams=2,
            short_ngram_size=5,
            minimum_short_matching_ngrams=2,
            code_ngram_size=6,
            minimum_code_matching_ngrams=2,
        )
        self.assertEqual(
            index.reason("A copied prompt asks which planet is known as the red planet because of iron oxide today"),
            "benchmark_short_ngram",
        )
        self.assertEqual(
            index.reason("# copied benchmark\ndef benchmark_secret(value):\n    return value * value + 17\nprint('x')"),
            "benchmark_code_ngram",
        )

    def test_decontamination_catches_code_with_renamed_identifiers_and_literals(self) -> None:
        index = ContaminationIndex.build(
            [
                "def benchmark_solution(values):\n"
                "    total = 0\n"
                "    for value in values:\n"
                "        total = total + value * 17\n"
                "    return total\n"
            ],
            code_ngram_size=20,
            minimum_code_matching_ngrams=2,
            code_skeleton_ngram_size=8,
            minimum_code_skeleton_matching_ngrams=2,
        )
        renamed = (
            "def copied_answer(items):\n"
            "    accumulator = 91\n"
            "    for element in items:\n"
            "        accumulator = accumulator + element * 23\n"
            "    return accumulator\n"
        )
        self.assertEqual(index.reason(renamed), "benchmark_code_skeleton_ngram")

    def test_code_hygiene_rejects_repository_noise(self) -> None:
        self.assertEqual(
            code_hygiene_reason("const x = 1;", {"category": "code", "repo_path": "node_modules/a.js"}),
            "vendored_or_build_tree",
        )
        self.assertEqual(
            code_hygiene_reason("const x = 1;", {"category": "code", "repo_path": "package-lock.json"}),
            "lockfile",
        )
        self.assertEqual(
            code_hygiene_reason("def useful():\n    return 1\n", {"category": "code", "repo_path": "src/useful.py"}),
            None,
        )

    def test_code_and_final_exact_dedup_keep_higher_priority(self) -> None:
        try:
            from datatrove.data import Document
        except ImportError:
            self.skipTest("DataTrove is installed by the Metis-1.6 data runtime")
        low = Document(
            text="def copied_function(value):\n    result = value + 1\n    return result\n",
            id="low",
            metadata={"category": "code", "repo_path": "a.py", "priority": 10},
        )
        high = Document(
            text="def copied_function(value):\n    result = value + 1\n    return result\n",
            id="high",
            metadata={"category": "code", "repo_path": "b.py", "priority": 20},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code_signatures = root / "code-signatures"
            code_removals = root / "code-removals"
            write_code_signatures([low], code_signatures, rank=0, finder_workers=2, block_tokens=16)
            write_code_signatures([high], code_signatures, rank=1, finder_workers=2, block_tokens=16)
            for bucket in range(2):
                find_code_duplicates(code_signatures, code_removals, bucket=bucket)
            low_files, _ = load_code_removals(code_removals, rank=0, finder_workers=2)
            high_files, _ = load_code_removals(code_removals, rank=1, finder_workers=2)
            self.assertEqual(low_files, {0})
            self.assertEqual(high_files, set())

            final_signatures = root / "final-signatures"
            final_removals = root / "final-removals"
            write_final_signatures([low], final_signatures, rank=0, finder_workers=2)
            write_final_signatures([high], final_signatures, rank=1, finder_workers=2)
            for bucket in range(2):
                find_final_duplicates(final_signatures, final_removals, bucket=bucket)
            self.assertEqual(load_final_removals(final_removals, rank=0, finder_workers=2), {0})
            self.assertEqual(load_final_removals(final_removals, rank=1, finder_workers=2), set())

    def test_holdout_fragment_extraction_includes_context_answers_and_tests(self) -> None:
        row = {
            "question": "What does the function return?",
            "context": "The function increments its input.",
            "choices": ["zero", "the input plus one"],
            "solution": "The input plus one.",
            "test_list": ["assert f(1) == 2"],
        }
        fragments = list(_benchmark_fragments(row))
        kinds = {kind for kind, _ in fragments}
        self.assertTrue({"query", "context", "choices", "answer", "code"} <= kinds)

    def test_holdout_registry_is_broad_pinned_and_nonsemantic(self) -> None:
        registry = load_yaml(Path(__file__).resolve().parents[1] / "manifests" / "contamination" / "eval-holdouts.yaml")
        self.assertEqual(len(registry["benchmarks"]), 63)
        jobs = sum(
            len(entry.get("files", [])) if entry.get("files") else len(list(_benchmark_jobs(entry)))
            for entry in registry["benchmarks"]
        )
        self.assertEqual(jobs, 203)
        self.assertEqual(len({entry["family"] for entry in registry["benchmarks"]}), 36)
        self.assertEqual(registry["policy"]["expected_benchmark_registries"], 63)
        self.assertEqual(registry["policy"]["expected_family_labels"], 36)
        self.assertEqual(registry["policy"]["expected_jobs"], 203)
        self.assertFalse(registry["policy"]["semantic_dedup"])
        self.assertEqual(registry["policy"]["maximum_shingle_rows"], 32)
        self.assertTrue(registry["policy"]["explicit_genealogy_match"])
        self.assertTrue(registry["policy"]["quarantine_outputs"])
        ids = [entry["id"] for entry in registry["benchmarks"]]
        self.assertEqual(len(ids), len(set(ids)))
        for entry in registry["benchmarks"]:
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")

    def test_benchmark_genealogy_rejects_explicit_seed_lineage_without_domain_false_positives(self) -> None:
        registry = load_yaml(
            Path(__file__).resolve().parents[1]
            / "manifests"
            / "contamination"
            / "eval-holdouts.yaml"
        )
        self.assertEqual(
            benchmark_genealogy_match(
                {"upstream_metadata": {"seed_dataset": "openai/gsm8k"}}, registry
            ),
            "gsm8k",
        )
        self.assertEqual(
            benchmark_genealogy_match({"benchmark_name": "gpqa_diamond"}, registry),
            "gpqa",
        )
        self.assertIsNone(
            benchmark_genealogy_match(
                {"dataset_name": "a broad mathematics and applications corpus"}, registry
            )
        )
        self.assertEqual(
            benchmark_genealogy_match({"evaluation_benchmark": "MATH"}, registry),
            "math",
        )

    def test_contamination_index_roundtrips_as_memory_mapped_binary(self) -> None:
        source = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
        index = ContaminationIndex.build([source], short_ngram_size=5, code_ngram_size=6)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.json"
            write_contamination_inputs(Path(temporary), index, [source])
            save_contamination_index(index, path)
            loaded = load_contamination_index(path)
            self.assertEqual(loaded.reason(source), "benchmark_exact")
            self.assertEqual(
                loaded.reason("prefix one two three four five six seven suffix"),
                "benchmark_short_ngram",
            )
            self.assertTrue((Path(temporary) / "index.exact.npy").exists())

    def test_near_dedup_keeps_higher_priority(self) -> None:
        try:
            import datatrove  # noqa: F401
        except ImportError:
            self.skipTest("DataTrove is installed by the Metis-1.6 data runtime")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = root / "documents"
            duplicates = root / "duplicates"
            removals = root / "removals"
            documents.mkdir()
            duplicates.mkdir()
            (documents / "00000.jsonl").write_text(
                json.dumps({"id": "low", "text": "similar document", "metadata": {"priority": 10}}) + "\n",
                encoding="utf-8",
            )
            (documents / "00001.jsonl").write_text(
                json.dumps({"id": "high", "text": "similar document!", "metadata": {"priority": 20}}) + "\n",
                encoding="utf-8",
            )
            (duplicates / "pairs.dups").write_bytes(struct.pack("<4I", 0, 0, 1, 0))
            report = build_priority_minhash_removals(duplicates, removals, documents, total_tasks=2)
            self.assertEqual(report["removed"], 1)
            self.assertEqual(struct.unpack("<I", (removals / "000000.remove").read_bytes())[0], 0)
            self.assertFalse((removals / "000001.remove").exists())


class AcquisitionTruthTests(unittest.TestCase):
    def test_slurm_range_is_compact_and_rejects_oversized_arrays(self) -> None:
        self.assertEqual(_indices_expression(range(1000), 200, 1000), "0-999%200")
        with self.assertRaises(ValueError):
            _indices_expression(range(1_000_000_000), 200, 1000)

    def test_production_manifest_has_no_unmaterialized_derived_sources(self) -> None:
        manifest = load_manifest()
        derived = [
            source["id"]
            for source in manifest["sources"]
            if source["access"].get("type") == "derived"
            or source["acquisition"].get("driver") == "derived_after_download"
        ]
        self.assertEqual(derived, [])

    def test_build_inputs_freeze_payloads_expand_shards_and_exclude_source_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state = StateStore(root / "state")
            hf_payload = root / "raw" / "hf-source" / "part.jsonl"
            source_index = root / "raw" / "repo-source" / "metadata.parquet"
            materialized = root / "raw" / "repo-source" / "materialized"
            hf_payload.parent.mkdir(parents=True)
            source_index.parent.mkdir(parents=True)
            materialized.mkdir(parents=True)
            hf_payload.write_text('{"text":"training payload"}\n', encoding="utf-8")
            source_index.write_bytes(b"retrieval metadata only")
            shards = []
            for index, content in enumerate((b'{"text":"one"}\n', b'{"text":"two"}\n')):
                path = materialized / f"part-{index:05d}.jsonl.zst"
                path.write_bytes(content)
                shards.append(
                    {
                        "path": str(path),
                        "size": path.stat().st_size,
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
            lock = {
                "release": "test-release",
                "sources": [{"id": "hf-source"}, {"id": "repo-source"}],
                "download_tasks": [
                    {"task_id": "download-000000"},
                    {"task_id": "download-000001"},
                ],
            }
            state.write("sources.lock.json", payload=lock)
            hf_record = {
                "kind": "hf_file",
                "source_id": "hf-source",
                "local_path": str(hf_payload),
                "size": hf_payload.stat().st_size,
                "sha256": hashlib.sha256(hf_payload.read_bytes()).hexdigest(),
                "payload_role": "training_records",
            }
            state.complete("download", "download-000000", {"files": [hf_record]})
            state.complete(
                "download",
                "download-000001",
                {
                    "files": [
                        {
                            "kind": "hf_file",
                            "source_id": "repo-source",
                            "local_path": str(source_index),
                            "size": source_index.stat().st_size,
                            "sha256": hashlib.sha256(source_index.read_bytes()).hexdigest(),
                            "payload_role": "source_index",
                        },
                        {
                            "kind": "materialized_dataset",
                            "source_id": "repo-source",
                            "local_path": str(materialized),
                            "shards": shards,
                            "materialized": True,
                        },
                    ]
                },
            )
            profile = {"storage": {"lustre_root": str(root)}}
            frozen = prepare_build_inputs(profile, state)
            self.assertEqual(frozen["input_count"], 3)
            self.assertEqual(frozen["files_by_source"], {"hf-source": 1, "repo-source": 2})
            self.assertNotIn(str(source_index), {item["local_path"] for item in frozen["inputs"]})
            self.assertEqual(prepare_build_inputs(profile, state), frozen)

            extra = root / "raw" / "hf-source" / "late.jsonl"
            extra.write_text('{"text":"late payload"}\n', encoding="utf-8")
            extra_record = {
                "kind": "hf_file",
                "source_id": "hf-source",
                "local_path": str(extra),
                "size": extra.stat().st_size,
                "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
            }
            state.complete("download", "download-000000", {"files": [hf_record, extra_record]})
            with self.assertRaisesRegex(RuntimeError, "Frozen build.inputs.json differs"):
                prepare_build_inputs(profile, state)

    def test_build_inputs_fail_closed_when_a_manifest_source_has_no_training_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state = StateStore(root / "state")
            payload = root / "raw" / "present" / "part.jsonl"
            payload.parent.mkdir(parents=True)
            payload.write_text('{"text":"present"}\n', encoding="utf-8")
            state.write(
                "sources.lock.json",
                payload={
                    "release": "test-release",
                    "sources": [{"id": "present"}, {"id": "missing"}],
                    "download_tasks": [{"task_id": "download-000000"}],
                },
            )
            state.complete(
                "download",
                "download-000000",
                {
                    "files": [
                        {
                            "kind": "hf_file",
                            "source_id": "present",
                            "local_path": str(payload),
                            "size": payload.stat().st_size,
                            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                        }
                    ]
                },
            )
            with self.assertRaisesRegex(RuntimeError, "missing"):
                prepare_build_inputs({"storage": {"lustre_root": str(root)}}, state)

    def test_build_inputs_fail_closed_on_duplicate_training_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state = StateStore(root / "state")
            payload = root / "raw" / "source" / "part.jsonl"
            payload.parent.mkdir(parents=True)
            payload.write_text('{"text":"one physical payload"}\n', encoding="utf-8")
            record = {
                "kind": "hf_file",
                "source_id": "source",
                "local_path": str(payload),
                "size": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
            state.write(
                "sources.lock.json",
                payload={
                    "release": "test-release",
                    "sources": [{"id": "source"}],
                    "download_tasks": [
                        {"task_id": "download-000000"},
                        {"task_id": "download-000001"},
                    ],
                },
            )
            state.complete("download", "download-000000", {"files": [record]})
            state.complete("download", "download-000001", {"files": [{**record, "repo_path": "alias.jsonl"}]})
            with self.assertRaisesRegex(RuntimeError, "duplicate training-record path"):
                prepare_build_inputs({"storage": {"lustre_root": str(root)}}, state)

    def test_slurm_arrays_are_chunked_with_global_task_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            profile = {
                "storage": {"lustre_root": str(root), "directories": {"logs": "logs"}},
                "scheduler": {
                    "max_array_size": 3,
                    "normalize": {"max_concurrent": 2},
                },
            }
            jobs = _submit_array_chunks(
                stage="normalize",
                global_indices=range(8),
                profile_path=root / "rhea.yaml",
                profile=profile,
                dependency="previous-job",
                dry_run=True,
            )
            self.assertEqual([job.array for job in jobs], ["0-2%2", "0-2%2", "0-1%2"])
            self.assertEqual([job.task_offset for job in jobs], [0, 3, 6])
            self.assertEqual([job.dependency for job in jobs], ["previous-job"] * 3)
            self.assertEqual(len({job.job_id for job in jobs}), 3)
            for job in jobs:
                exports = [argument for argument in job.command if argument.startswith("--export=")]
                self.assertEqual(len(exports), 1)
                self.assertIn(f"METIS_TASK_OFFSET={job.task_offset}", exports[0])

    def test_repeated_span_dedup_precedes_minhash_in_build_graph(self) -> None:
        stages = [stage for stage, _ in BUILD_GRAPH]
        self.assertLess(stages.index("exact_filter"), stages.index("span_prefilter_signature"))
        self.assertLess(
            stages.index("span_prefilter_signature"),
            stages.index("span_prefilter_find"),
        )
        self.assertLess(stages.index("span_prefilter_find"), stages.index("span_signature"))
        self.assertLess(stages.index("span_signature"), stages.index("span_find"))
        self.assertLess(stages.index("span_find"), stages.index("span_filter"))
        self.assertLess(stages.index("span_filter"), stages.index("minhash_signature"))
        self.assertLess(stages.index("minhash_buckets"), stages.index("minhash_components"))
        self.assertLess(
            stages.index("minhash_components"),
            stages.index("minhash_priority_candidates"),
        )
        self.assertLess(
            stages.index("minhash_priority_candidates"),
            stages.index("minhash_priority_resolve"),
        )
        self.assertLess(
            stages.index("minhash_priority_resolve"),
            stages.index("minhash_priority_finalize"),
        )
        self.assertLess(
            stages.index("minhash_priority_finalize"),
            stages.index("minhash_priority_verify"),
        )
        self.assertLess(
            stages.index("minhash_priority_verify"),
            stages.index("minhash_filter"),
        )
        self.assertEqual(stages[stages.index("normalize") + 1], "cleanup_raw")
        self.assertEqual(stages[stages.index("exact_filter") + 1], "cleanup_exact")
        self.assertEqual(stages[stages.index("span_filter") + 1], "cleanup_span")
        self.assertEqual(stages[stages.index("minhash_filter") + 1], "cleanup_minhash")
        self.assertEqual(stages[stages.index("code_filter") + 1], "cleanup_code")
        self.assertEqual(stages[stages.index("decontam_filter") + 1], "cleanup_decontam")
        self.assertEqual(
            stages[stages.index("final_hash_filter") + 1],
            "cleanup_final_hash",
        )

    def test_verified_cleanup_hashes_successor_before_retiring_predecessor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state = StateStore(root / "state")
            state.write("build.inputs.json", payload={"input_count": 1})
            state.complete(
                "normalize",
                "task-000000",
                {"execution_contract_sha256": "contract"},
            )
            normalized = root / "normalized"
            normalized.mkdir()
            (normalized / "task-000000.jsonl").write_text(
                '{"text":"verified successor"}\n',
                encoding="utf-8",
            )
            raw = root / "raw"
            raw.mkdir()
            (raw / "candidate.jsonl").write_text(
                '{"text":"raw predecessor"}\n',
                encoding="utf-8",
            )
            cache = root / "cache" / "huggingface"
            cache.mkdir(parents=True)
            (cache / "blob").write_bytes(b"cached")
            profile = {
                "storage": {
                    "lustre_root": str(root),
                    "directories": {
                        "raw": "raw",
                        "normalized": "normalized",
                        "eligible": "eligible",
                        "dedup": "dedup",
                        "contamination": "contamination",
                        "cache": "cache",
                        "state": "state",
                    },
                },
                "scheduler": {
                    "exact_dedup": {"find_tasks": 1},
                    "repeated_span": {"finder_tasks": 1},
                    "minhash": {"num_buckets": 1},
                    "minhash_priority": {"bucket_count": 1},
                    "code_structural": {"finder_tasks": 1},
                    "final_hash": {"finder_tasks": 1},
                },
            }
            with mock.patch(
                "metis_data.stage_runner._stage_execution_contract",
                return_value="contract",
            ):
                receipt = stage_runner._cleanup_filter_intermediate(
                    profile,
                    "cleanup_raw",
                )
            self.assertFalse(raw.exists())
            self.assertFalse(cache.exists())
            self.assertTrue(normalized.exists())
            self.assertEqual(receipt["content"]["files"], 1)
            self.assertFalse(receipt["content"]["retained"])
            stage_runner._validate_content_receipt(
                profile,
                receipt["content"],
                require_live_content=True,
            )

    def test_slurm_wrapper_adds_array_id_to_exported_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_python = root / "python"
            fake_python.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
            fake_python.chmod(0o755)
            environment = {
                **os.environ,
                "METIS_PYTHON": str(fake_python),
                # Slurm runs a staged copy of this script, so it cannot find the
                # checkout on its own. See tests/test_slurm_script_staging.py.
                "METIS_ROOT": str(Path(__file__).resolve().parents[1]),
                "METIS_PROFILE": str(root / "rhea.yaml"),
                "METIS_STAGE": "normalize",
                "METIS_TASK_OFFSET": "3000",
                "SLURM_ARRAY_TASK_ID": "17",
            }
            result = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "slurm" / "metis16" / "stage.sbatch")],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            arguments = result.stdout.splitlines()
            self.assertEqual(arguments[arguments.index("--task-index") + 1], "3017")

    def test_production_profile_requires_explicit_safe_lustre_root(self) -> None:
        profile = {
            "storage": {
                "require_explicit_root": True,
                "lustre_root": "auto",
                "forbidden_roots": ["/", "/lus", "/lus/lustre1"],
            }
        }
        with self.assertRaises(RuntimeError):
            validate_storage_root(profile, Path("/lus/lustre1"))
        with tempfile.TemporaryDirectory() as temporary:
            profile_path = Path(temporary) / "production.yaml"
            profile_path.write_text(
                "name: production\n"
                "storage:\n"
                "  lustre_root: ${METIS_LUSTRE_ROOT:-auto}\n"
                "  require_explicit_root: true\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("METIS_LUSTRE_ROOT", None)
                with self.assertRaises(RuntimeError):
                    load_profile(profile_path)

    def test_login2_and_rhea_profiles_have_separate_fail_closed_roles(self) -> None:
        root = Path(__file__).resolve().parents[1] / "configs" / "metis16"
        login2 = load_yaml(root / "login2.yaml")
        rhea = load_yaml(root / "rhea.yaml")
        portage = load_yaml(root / "portage.yaml")
        self.assertEqual(login2["operator"]["roles"], ["acquisition"])
        self.assertEqual(login2["acquisition"]["mode"], "screen_foreground")
        self.assertTrue(login2["storage"]["require_explicit_root"])
        self.assertEqual(rhea["operator"]["roles"], ["compute"])
        self.assertEqual(rhea["acquisition"]["mode"], "external_complete")
        self.assertFalse(rhea["scheduler"]["site_values_confirmed"])
        self.assertEqual(rhea["scheduler"]["account"], "${METIS_SLURM_ACCOUNT:-auto}")
        self.assertFalse(rhea["scheduler"]["repeated_span"]["semantic_dedup"])
        self.assertEqual(portage["operator"]["roles"], ["legacy_disabled"])
        self.assertTrue(portage["storage"]["require_explicit_root"])

    def test_acquisition_supervisor_lock_is_singleton(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = StateStore(Path(temporary) / "state")
            with _supervisor_lock(state):
                with self.assertRaises(RuntimeError):
                    with _supervisor_lock(state):
                        self.fail("second supervisor unexpectedly acquired the lock")

    def test_task_lock_reclaims_a_proven_dead_process_on_the_same_host(self) -> None:
        import socket

        with tempfile.TemporaryDirectory() as temporary:
            state = StateStore(Path(temporary) / "state")
            lock = state.path("locks", "download", "task.lock")
            lock.mkdir(parents=True)
            (lock / "OWNER.json").write_text(
                json.dumps({"pid": 2_000_000_000, "hostname": socket.gethostname()}),
                encoding="utf-8",
            )
            with state.task_lock("download", "task") as reclaimed:
                self.assertEqual(reclaimed, lock)
                owner = json.loads((lock / "OWNER.json").read_text(encoding="utf-8"))
                self.assertEqual(owner["pid"], os.getpid())
            self.assertFalse(lock.exists())

    def test_acquisition_handoff_detects_mutated_source_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state = StateStore(root / "state")
            artifact = root / "raw" / "source" / "part.bin"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"immutable candidate data")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            task_payload = {"items": [], "planned_bytes": 0}
            task_sha256 = hashlib.sha256(
                json.dumps(
                    task_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            task_id = f"download-000000-{task_sha256[:16]}"
            source_lock = {
                "schema": "metis.source-lock/v4",
                "release": "test-release",
                "sources": [],
                "runtime_contract": runtime_contract(),
                "download_tasks": [
                    {
                        **task_payload,
                        "task_index": 0,
                        "task_sha256": task_sha256,
                        "task_id": task_id,
                    }
                ],
            }
            source_lock["lock_sha256"] = source_lock_sha256(source_lock)
            state.write("sources.lock.json", payload=source_lock)
            state.complete(
                "download",
                task_id,
                {
                    "task_id": task_id,
                    "task_sha256": task_sha256,
                    "files": [
                        {
                            "kind": "hf_file",
                            "source_id": "source",
                            "local_path": str(artifact),
                            "size": artifact.stat().st_size,
                            "sha256": digest,
                        }
                    ],
                },
            )
            contamination = root / "contamination"
            contamination.mkdir()
            (contamination / "holdouts.jsonl").write_text('{"text":"holdout"}\n', encoding="utf-8")
            (contamination / "HOLDOUTS.json").write_text(
                '{"schema":"metis.holdout-bundle/test"}\n', encoding="utf-8"
            )
            profile = {
                "storage": {
                    "lustre_root": str(root),
                    "directories": {"state": "state", "contamination": "contamination"},
                },
                "gates": {"require_clean_repository": False, "require_repository_commit_match": False},
            }
            manifest = {"release": "test-release", "sources": []}
            handoff = write_acquisition_handoff(profile, manifest, state)
            self.assertEqual(handoff["artifact_count"], 1)
            self.assertTrue(verify_acquisition_handoff(profile, manifest, state)["ok"])
            original = artifact.read_bytes()
            artifact.write_bytes(b"x" * len(original))
            with self.assertRaisesRegex(RuntimeError, "hash changed"):
                verify_acquisition_handoff(profile, manifest, state, verify_artifact_hashes=True)
            artifact.write_bytes(original)
            state.write("sources.lock.json", payload={**source_lock, "tampered": True})
            with self.assertRaisesRegex(RuntimeError, "source lock changed"):
                verify_acquisition_handoff(profile, manifest, state)

    def test_handoff_expands_materialized_dataset_receipt_and_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state = StateStore(root / "state")
            dataset = root / "raw" / "dynamic-source"
            dataset.mkdir(parents=True)
            shards = []
            for index, payload in enumerate((b"first shard", b"second shard")):
                path = dataset / f"part-{index:05d}.jsonl.zst"
                path.write_bytes(payload)
                shards.append(
                    {
                        "path": str(path),
                        "size": path.stat().st_size,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            receipt = dataset / "ACQUISITION_RECEIPT.json"
            receipt.write_text(json.dumps({"shards": shards}, sort_keys=True), encoding="utf-8")
            task_payload = {"items": [], "planned_bytes": 0}
            task_sha256 = hashlib.sha256(
                json.dumps(
                    task_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            task_id = f"download-000000-{task_sha256[:16]}"
            source_lock = {
                "schema": "metis.source-lock/v4",
                "release": "test",
                "sources": [],
                "runtime_contract": runtime_contract(),
                "download_tasks": [
                    {
                        **task_payload,
                        "task_index": 0,
                        "task_sha256": task_sha256,
                        "task_id": task_id,
                    }
                ],
            }
            source_lock["lock_sha256"] = source_lock_sha256(source_lock)
            state.write(
                "sources.lock.json",
                payload=source_lock,
            )
            state.complete(
                "download",
                task_id,
                {
                    "task_sha256": task_sha256,
                    "files": [
                        {
                            "kind": "materialized_dataset",
                            "source_id": "dynamic-source",
                            "local_path": str(dataset),
                            "receipt": str(receipt),
                            "shards": shards,
                            "materialized": True,
                            "ready_for_training_build": True,
                        }
                    ]
                },
            )
            contamination = root / "contamination"
            contamination.mkdir()
            (contamination / "holdouts.jsonl").write_text('{"text":"holdout"}\n', encoding="utf-8")
            (contamination / "HOLDOUTS.json").write_text(
                '{"schema":"metis.holdout-bundle/test"}\n', encoding="utf-8"
            )
            profile = {
                "storage": {
                    "lustre_root": str(root),
                    "directories": {"state": "state", "contamination": "contamination"},
                },
                "gates": {"require_clean_repository": False, "require_repository_commit_match": False},
            }
            handoff = write_acquisition_handoff(profile, {"release": "test", "sources": []}, state)
            self.assertEqual(handoff["artifact_count"], 3)
            paths = {item["path"] for item in handoff["artifacts"]}
            self.assertIn(str(receipt.relative_to(root)), paths)
            self.assertTrue(all(str(dataset.relative_to(root)) != path for path in paths))
            self.assertTrue(verify_acquisition_handoff(profile, {"release": "test", "sources": []}, state)["ok"])

    def test_release_bundles_the_frozen_build_input_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repository = root / "repository"
            manifest_path = repository / "manifests" / "metis-1.6.yaml"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("release: test-release\n", encoding="utf-8")
            directories = {
                "state": "state",
                "release": "release",
                "tokenizer": "tokenizer",
                "selected": "selected",
            }
            profile = {
                "manifest": str(manifest_path),
                "storage": {
                    "lustre_root": str(root),
                    "directories": directories,
                    "final_token_dtype": "uint16",
                },
                "gates": {"require_license_ledger": False},
            }
            tokenizer_root = root / directories["tokenizer"]
            tokenizer_root.mkdir()
            for name in (
                "tokenizer.json",
                "vocab.json",
                "TOKENIZER_RELEASE.json",
                "TOKENIZER_VALIDATION.json",
                "NGRAM_CANONICAL_IDS.json",
                "NGRAM_CANONICAL_IDS.uint16",
            ):
                (tokenizer_root / name).write_text(f"{name}\n", encoding="utf-8")
            selection_path = root / directories["selected"] / "SELECTION.json"
            selection_path.parent.mkdir()
            selection_path.write_text('{"schema":"selection-test"}\n', encoding="utf-8")
            token_count_contract = selection_path.parent / "TOKEN_COUNT_CONTRACT.json"
            token_count_contract.write_text('{"schema":"token-count-test"}\n', encoding="utf-8")
            state = StateStore(root / directories["state"])
            state.write("sources.lock.json", payload={"release": "test-release", "sources": []})
            build_inputs = {
                "schema": "metis.build-inputs/v1",
                "release": "test-release",
                "input_count": 1,
                "inputs": [{"input_id": "abc", "source_id": "source"}],
            }
            state.write("build.inputs.json", payload=build_inputs)
            provenance = root / directories["release"] / "provenance"
            provenance.mkdir(parents=True)
            filter_chain_path = provenance / "FILTER_CHAIN.json"
            filter_chain = {"schema": "metis.filter-chain/v1", "stages": []}
            filter_chain["filter_chain_sha256"] = stage_runner._json_sha256(filter_chain)
            filter_chain_path.write_text(
                json.dumps(filter_chain, sort_keys=True) + "\n", encoding="utf-8"
            )
            ledger_path = provenance / "LICENSE_LEDGER.jsonl"
            ledger_path.write_text('{"source_id":"source"}\n', encoding="utf-8")
            shard_manifest_path = provenance / "SHARDS.jsonl"
            shard_manifest_path.write_text("", encoding="utf-8")
            tokenizer_contract = {
                "schema": "test-tokenizer-contract",
                "tokenizer_sha256": hashlib.sha256(
                    (tokenizer_root / "tokenizer.json").read_bytes()
                ).hexdigest(),
                "vocab_sha256": hashlib.sha256(
                    (tokenizer_root / "vocab.json").read_bytes()
                ).hexdigest(),
                "tokenizer_release_sha256": hashlib.sha256(
                    (tokenizer_root / "TOKENIZER_RELEASE.json").read_bytes()
                ).hexdigest(),
                "tokenizer_validation_sha256": hashlib.sha256(
                    (tokenizer_root / "TOKENIZER_VALIDATION.json").read_bytes()
                ).hexdigest(),
                "ngram_canonical_map_manifest_sha256": hashlib.sha256(
                    (tokenizer_root / "NGRAM_CANONICAL_IDS.json").read_bytes()
                ).hexdigest(),
                "ngram_canonical_map_self_sha256": "c" * 64,
                "ngram_canonical_ids_sha256": hashlib.sha256(
                    (tokenizer_root / "NGRAM_CANONICAL_IDS.uint16").read_bytes()
                ).hexdigest(),
            }
            selection_contract = {"schema": "test-selection-contract"}
            verification = {
                "schema": "metis.verification/v2",
                "ok": True,
                "target_tokens": 10,
                "phase_tokens": {"phase_a": 7, "phase_b": 2, "phase_c": 1},
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "source_lock_sha256": hashlib.sha256(
                    state.path("sources.lock.json").read_bytes()
                ).hexdigest(),
                "build_inputs_sha256": hashlib.sha256(
                    state.path("build.inputs.json").read_bytes()
                ).hexdigest(),
                "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                "selection_contract": selection_contract,
                "token_count_contract_sha256": hashlib.sha256(
                    token_count_contract.read_bytes()
                ).hexdigest(),
                "tokenizer_contract": tokenizer_contract,
                "filter_chain": str(filter_chain_path),
                "filter_chain_sha256": hashlib.sha256(filter_chain_path.read_bytes()).hexdigest(),
                "license_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                "shard_manifest_sha256": hashlib.sha256(
                    shard_manifest_path.read_bytes()
                ).hexdigest(),
            }
            manifest = {
                "_path": str(manifest_path),
                "release": "test-release",
                "schedule": {
                    "target_tokens": 10,
                    "phases": {
                        "phase_a": {"target_tokens": 7},
                        "phase_b": {"target_tokens": 2},
                        "phase_c": {"target_tokens": 1},
                    },
                },
            }
            verification["manifest_contract_sha256"] = (
                stage_runner._manifest_contract_sha256(manifest)
            )
            verification["verification_sha256"] = stage_runner._json_sha256(
                {
                    key: value
                    for key, value in verification.items()
                    if key != "verification_sha256"
                }
            )
            state.write("VERIFICATION.json", payload=verification)
            with (
                mock.patch("metis_data.stage_runner._manifest", return_value=manifest),
                mock.patch("metis_data.stage_runner.repository_root", return_value=repository),
                mock.patch(
                    "metis_data.stage_runner._production_tokenizer_contract",
                    return_value=tokenizer_contract,
                ),
                mock.patch(
                    "metis_data.stage_runner._validate_selection_artifacts",
                    return_value={
                        "token_count_contract_path": token_count_contract,
                        "tokenizer_contract": tokenizer_contract,
                    },
                ),
                mock.patch(
                    "metis_data.stage_runner._validate_selection_contract",
                    return_value=selection_contract,
                ),
                mock.patch("metis_data.stage_runner._validate_filter_chain_artifacts"),
                mock.patch(
                    "metis_data.training_contract.validate_training_release",
                    return_value={"ok": True},
                ),
            ):
                released = stage_runner._release(profile)
            bundled = root / directories["release"] / "manifests" / "build.inputs.json"
            self.assertEqual(json.loads(bundled.read_text(encoding="utf-8")), build_inputs)
            self.assertEqual(released["artifacts"]["build_inputs"], "manifests/build.inputs.json")
            self.assertEqual(
                released["build_inputs_sha256"],
                hashlib.sha256(bundled.read_bytes()).hexdigest(),
            )

    def test_acquisition_waves_put_downloaded_payloads_before_materializers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = StateStore(Path(temporary) / "state")
            lock = {
                "download_tasks": [
                    {"task_id": "builder", "items": [{"kind": "builder", "driver": "repository_index"}]},
                    {"task_id": "payload", "items": [{"kind": "hf_file"}]},
                    {"task_id": "fresh", "items": [{"kind": "builder", "driver": "common_crawl_ranges"}]},
                    {"task_id": "done", "items": [{"kind": "hf_file"}]},
                    {"task_id": "github", "items": [{"kind": "builder", "driver": "github_discussions"}]},
                ]
            }
            state.complete("download", "done", {"files": [{"kind": "hf_file"}]})
            self.assertEqual(_pending_task_waves(lock, state), [(0, [1]), (1, [0, 2]), (2, [4])])

    def test_github_builders_share_one_concurrency_lane(self) -> None:
        profile = {
            "acquisition": {
                "driver_lanes": {
                    "repository_index": "github",
                    "github_discussions": "github",
                },
                "lane_max_workers": {"github": 1},
            }
        }
        lock = {
            "download_tasks": [
                {"items": [{"kind": "builder", "driver": "repository_index"}]},
                {"items": [{"kind": "builder", "driver": "github_discussions"}]},
            ]
        }
        driver_lanes, semaphores = _lane_configuration(profile)
        active = 0
        maximum_active = 0
        counter_lock = threading.Lock()

        def fake_download(_profile: dict, task_index: int) -> dict:
            nonlocal active, maximum_active
            with counter_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with counter_lock:
                active -= 1
            return {"task_index": task_index}

        with mock.patch("metis_data.local_download.run_download_task", side_effect=fake_download):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(_run_task_in_lanes, profile, lock, index, driver_lanes, semaphores)
                    for index in range(2)
                ]
                for future in futures:
                    future.result()
        self.assertEqual(maximum_active, 1)

    def test_screen_launcher_never_puts_tokens_on_screen_command_line(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "ops" / "start-acquisition.sh").read_text(
            encoding="utf-8"
        )
        screen_lines = [line for line in script.splitlines() if line.strip().startswith("screen -DmS")]
        self.assertEqual(len(screen_lines), 1)
        self.assertNotIn("HF_TOKEN", screen_lines[0])
        self.assertNotIn("GITHUB_TOKEN", screen_lines[0])


class SelectionAndTokenizerTests(unittest.TestCase):
    def test_hamilton_apportion_is_exact(self) -> None:
        apportioned = hamilton_apportion(11, {"a": 5, "b": 3, "c": 2})
        self.assertEqual(sum(apportioned.values()), 11)
        self.assertGreater(apportioned["a"], apportioned["c"])

    def test_tiny_selection_hits_unique_replay_and_shard_contracts(self) -> None:
        manifest = {
            "selection": {"seed": 1, "replay": {"maximum_document_exposures": 4}},
            "schedule": {
                "target_tokens": 100,
                "phases": {
                    "phase_a": {"target_tokens": 60, "replay_tokens": 0},
                    "phase_b": {"target_tokens": 30, "replay_tokens": 10},
                    "phase_c": {"target_tokens": 10, "replay_tokens": 10},
                },
            },
            "sources": [
                {"id": "a", "phase_tokens": {"phase_a": 40, "phase_b": 20, "phase_c": 6}},
                {"id": "b", "phase_tokens": {"phase_a": 20, "phase_b": 10, "phase_c": 4}},
            ],
        }
        records = [
            {"source_id": "a", "doc_id": f"a{i}", "text": "alpha", "token_count": 10, "generated": False}
            for i in range(10)
        ] + [
            {"source_id": "b", "doc_id": f"b{i}", "text": "beta", "token_count": 10, "generated": False}
            for i in range(6)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            result = build_selection(
                records,
                manifest=manifest,
                eligible_tokens={"a": 100, "b": 60},
                output_root=Path(temporary),
                shard_tokens=10,
            )
            self.assertEqual(sum(item["target_tokens"] for item in result["shards"]), 100)
            self.assertEqual(len(result["shards"]), 10)
            self.assertEqual(sum(sum(value.values()) for value in result["replay_written"].values()), 20)

    def test_tokenizer_roundtrip_and_uint16_packing(self) -> None:
        corpus = [
            "ordinary English prose with punctuation.",
            "def hello(name: str) -> str:\n    return f'hello {name}'",
            r"For $x^2 + y^2 = z^2$, preserve \LaTeX{}.",
            "Unicode names: José, 李, Δx and emoji 🧪.",
        ] * 50
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = train_tokenizer(
                iter(corpus),
                output_dir=root / "tokenizer",
                vocabulary_size=512,
                special_tokens=["<|endoftext|>", "<|padding|>"],
                minimum_frequency=1,
            )
            self.assertTrue(release["uint16_safe"])
            validation = validate_tokenizer(
                root / "tokenizer" / "tokenizer.json",
                ({"category": "tiny", "text": text} for text in corpus[:4]),
            )
            self.assertTrue(validation["ok"], validation["roundtrip_failures"])
            packed = pack_release(
                [
                    {"phase": "phase_a", "source_id": "tiny", "doc_id": "1", "text": corpus[0]},
                    {"phase": "phase_b", "source_id": "tiny", "doc_id": "2", "text": corpus[1]},
                    {"phase": "phase_c", "source_id": "tiny", "doc_id": "3", "text": corpus[2]},
                ] * 10,
                tokenizer_path=root / "tokenizer" / "tokenizer.json",
                output_root=root / "release",
                phase_targets={"phase_a": 20, "phase_b": 12, "phase_c": 8},
                shard_tokens=10,
            )
            self.assertEqual(packed["target_tokens"], 40)
            for shard in packed["shards"]:
                self.assertEqual(Path(shard["binary"]).stat().st_size, shard["tokens"] * 2)


class ParallelCpuBuildTests(unittest.TestCase):
    """The whole-node fan-out and the sharded sample/verify stages."""

    CATEGORIES = ("web", "code", "math", "science")

    def _manifest(self) -> dict:
        sources = [
            (f"src_{category}_{index}", category)
            for category in self.CATEGORIES
            for index in range(3)
        ]
        return {
            "tokenizer": {
                "sample_target_bytes": 200_000,
                "min_sample_bytes_per_category": 10_000,
            },
            "categories": [
                {"id": category, "phase_tokens": {"phase_a": 100 + 10 * index}}
                for index, category in enumerate(self.CATEGORIES)
            ],
            "sources": [
                {"id": source, "category": category, "phase_tokens": {"phase_a": 50 + 7 * index}}
                for index, (source, category) in enumerate(sources)
            ],
        }

    def _write_corpus(self, root: Path, files: int, seed: int, skew: bool) -> dict[Path, list[dict]]:
        import io
        import random

        import zstandard as zstd

        manifest = self._manifest()
        sources = [(source["id"], source["category"]) for source in manifest["sources"]]
        rng = random.Random(seed)
        eligible = root / "eligible" / "final"
        eligible.mkdir(parents=True, exist_ok=True)
        rows_by_file: dict[Path, list[dict]] = {}
        for index in range(files):
            # Concentrating some sources into a few files stresses the
            # apportionment far harder than a uniform corpus does.
            pool = sources[:2] if skew and index % 5 == 0 else sources
            rows = []
            for _ in range(rng.randint(120, 300)):
                source_id, category = rng.choice(pool)
                text = "".join(rng.choice("abcdefghij ") for _ in range(rng.randint(50, 400)))
                rows.append({"text": text, "metadata": {"source_id": source_id, "category": category}})
            path = eligible / f"task-{index:06d}.jsonl.zst"
            with path.open("wb") as raw:
                with zstd.ZstdCompressor(level=1).stream_writer(raw) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                        for row in rows:
                            handle.write(json.dumps(row) + "\n")
            rows_by_file[path] = rows
        return rows_by_file

    def _serial_reference(self, manifest: dict, rows_by_file: dict[Path, list[dict]]) -> dict[str, int]:
        """The original single-process sampler, for behavioural comparison."""

        _categories, source_targets, _target = stage_runner._tokenizer_sample_targets(manifest)
        written = {source: 0 for source in source_targets}
        for path in sorted(rows_by_file):
            for row in rows_by_file[path]:
                source_id = row["metadata"]["source_id"]
                if source_id not in source_targets or written[source_id] >= source_targets[source_id]:
                    continue
                written[source_id] += len(row["text"].encode("utf-8"))
        return written

    def _profile(self, root: Path) -> dict:
        return {
            "manifest": "unused",
            "storage": {
                "lustre_root": str(root),
                "directories": {
                    "state": "state",
                    "eligible": "eligible",
                    "tokenizer": "tokenizer",
                },
            },
        }

    def test_sharded_tokenizer_sample_matches_the_serial_stratification(self) -> None:
        manifest = self._manifest()
        for files, seed, skew in ((12, 1, False), (40, 2, False), (25, 3, True), (60, 4, True)):
            with self.subTest(files=files, skew=skew), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                rows_by_file = self._write_corpus(root, files, seed, skew)
                profile = self._profile(root)
                state = StateStore(root / "state")
                state.write("build.inputs.json", payload={"input_count": files, "inputs": []})
                _categories, targets, _target = stage_runner._tokenizer_sample_targets(manifest)
                serial = self._serial_reference(manifest, rows_by_file)

                with mock.patch.object(stage_runner, "_manifest", return_value=manifest):
                    for task_index in range(files):
                        stage_runner._tokenizer_sample_scan(profile, task_index)
                    stage_runner._tokenizer_sample_plan(profile)
                    for task_index in range(files):
                        stage_runner._tokenizer_sample(profile, task_index)
                    parts = stage_runner._tokenizer_sample_parts(profile, state)

                sharded = {source: 0 for source in targets}
                for part in parts:
                    for row in stage_runner._iter_rows(part):
                        sharded[row["source_id"]] += len(row["text"].encode("utf-8"))

                self.assertEqual(len(parts), files, "every shard must publish a part")
                largest_document = max(
                    len(row["text"].encode("utf-8"))
                    for rows in rows_by_file.values()
                    for row in rows
                )
                for source, target in targets.items():
                    # Whole documents are emitted, so the contract is "reach the
                    # target, overshoot by less than one document" - exactly what
                    # the serial sampler guarantees, and independent of how many
                    # shards hold the source.
                    self.assertGreaterEqual(sharded[source], target, source)
                    self.assertLess(sharded[source] - target, largest_document, source)
                    self.assertLess(serial[source] - target, largest_document, source)

    def test_tokenizer_sample_plan_fails_closed_when_a_source_is_exhausted(self) -> None:
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self._write_corpus(root, 3, 9, False)
            profile = self._profile(root)
            StateStore(root / "state").write(
                "build.inputs.json", payload={"input_count": 3, "inputs": []}
            )
            with mock.patch.object(stage_runner, "_manifest", return_value=manifest):
                for task_index in range(3):
                    stage_runner._tokenizer_sample_scan(profile, task_index)
                with self.assertRaises(RuntimeError) as caught:
                    stage_runner._tokenizer_sample_plan(profile)
        self.assertIn("exhausted before its stratified source targets", str(caught.exception))

    def test_verify_only_trusts_shard_receipts_bound_to_the_current_policy(self) -> None:
        binding = stage_runner._verify_shard_binding(
            selection={"token_count_contract_sha256": "tcc"},
            selection_sha="sel",
            tokenizer_contract={"tokenizer_sha256": "tok"},
            maximum_exposures=4,
        )
        for field, changed in (
            ("selection", {"token_count_contract_sha256": "other"}),
            ("selection_sha", "different"),
            ("tokenizer_contract", {"tokenizer_sha256": "retrained"}),
            ("maximum_exposures", 5),
        ):
            arguments = {
                "selection": {"token_count_contract_sha256": "tcc"},
                "selection_sha": "sel",
                "tokenizer_contract": {"tokenizer_sha256": "tok"},
                "maximum_exposures": 4,
            }
            arguments[field] = changed
            self.assertNotEqual(
                binding,
                stage_runner._verify_shard_binding(**arguments),
                f"{field} must invalidate a per-shard verification receipt",
            )

    def test_verify_reverifies_shards_whose_receipt_is_stale_or_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state = StateStore(root / "state")
            shards = [
                {"global_index": index, "phase": "phase_a", "phase_index": index,
                 "target_tokens": 10, "sha256": f"schedule-{index}"}
                for index in range(4)
            ]
            selection = {"shards": shards, "token_count_contract_sha256": "tcc"}
            context = {
                "manifest": {}, "selection": selection,
                "selection_path": root / "SELECTION.json", "selection_sha": "sel",
                "selection_artifacts": {"tokenizer_contract": {"t": 1}},
                "selection_contract": {"maximum_document_exposures": 4},
                "tokenizer_contract": {"t": 1}, "maximum_exposures": 4,
                "binding": stage_runner._verify_shard_binding(
                    selection=selection, selection_sha="sel",
                    tokenizer_contract={"t": 1}, maximum_exposures=4,
                ),
            }
            # Shard 0 fresh, shard 1 stale binding, shard 2 absent, shard 3 marked
            # not-present. Only shard 0 may be taken on trust.
            for index, payload in (
                (0, {"schema": "metis.verify-shard/v1", "task_index": 0, "shard_present": True,
                     "binding_sha256": context["binding"], "pack_report": {"receipt": 0}}),
                (1, {"schema": "metis.verify-shard/v1", "task_index": 1, "shard_present": True,
                     "binding_sha256": "stale", "pack_report": {"receipt": 1}}),
                (3, {"schema": "metis.verify-shard/v1", "task_index": 3, "shard_present": False,
                     "binding_sha256": context["binding"]}),
            ):
                state.complete("verify_shard", f"task-{index:06d}", payload)

            reverified: list[int] = []

            def fake_payload(_profile, _state, *, shard, **_kwargs):
                reverified.append(int(shard["global_index"]))
                return {"recomputed": int(shard["global_index"])}

            profile = {
                "storage": {"lustre_root": str(root),
                            "directories": {"state": "state", "selected": "selected",
                                            "release": "release"}},
                "gates": {},
            }
            with mock.patch.object(stage_runner, "_verify_selection_context", return_value=context), \
                 mock.patch.object(stage_runner, "_manifest", return_value={}), \
                 mock.patch.object(stage_runner, "_verify_shard_payload", side_effect=fake_payload), \
                 self.assertRaises(Exception):
                # Aggregation past the shard loop needs the full selection
                # contract; the loop's trust decisions are what is under test.
                stage_runner._verify(profile)

        self.assertEqual(reverified, [1, 2, 3], "stale, missing and absent receipts must be redone")

    def test_grouped_arrays_run_every_task_exactly_once(self) -> None:
        from metis_data.slurm import _contiguous_chunks

        for indices, maximum, group in (
            (range(20000), 1000, 48),
            (range(20000), 1000, 1),
            ([3, 4, 5, 900, 901, 5000], 1000, 48),
            (range(49), 1000, 48),
            (range(200000), 1000, 48),
            (range(1000), 1000, 7),
        ):
            with self.subTest(count=len(list(indices)), group=group):
                executed: list[int] = []
                for chunk in _contiguous_chunks(indices, maximum, group):
                    entries = -(-len(chunk) // group)
                    self.assertLessEqual(entries, maximum)
                    for array_id in range(entries):
                        first = chunk.start + array_id * group
                        # This mirrors stage.sbatch plus the runner's clamp.
                        executed.extend(i for i in range(first, first + group) if i < chunk.stop)
                self.assertEqual(len(executed), len(set(executed)), "a task ran twice")
                self.assertEqual(set(executed), set(indices), "a task never ran")

    def test_stage_wrapper_maps_array_id_through_the_group_stride(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_python = root / "python"
            fake_python.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
            fake_python.chmod(0o755)
            environment = {
                **os.environ,
                "METIS_PYTHON": str(fake_python),
                # Slurm runs a staged copy of this script, so it cannot find the
                # checkout on its own. See tests/test_slurm_script_staging.py.
                "METIS_ROOT": str(Path(__file__).resolve().parents[1]),
                "METIS_PROFILE": str(root / "portage-cpu.yaml"),
                "METIS_STAGE": "normalize",
                "METIS_TASK_OFFSET": "3000",
                "METIS_TASKS_PER_JOB": "48",
                "METIS_TASK_LIMIT": "20000",
                "SLURM_ARRAY_TASK_ID": "17",
            }
            result = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "slurm" / "metis16" / "stage.sbatch")],
                check=True, capture_output=True, text=True, env=environment,
            )
            arguments = result.stdout.splitlines()
            self.assertEqual(arguments[arguments.index("--task-index") + 1], str(3000 + 17 * 48))
            self.assertEqual(arguments[arguments.index("--task-count") + 1], "48")
            self.assertEqual(arguments[arguments.index("--task-limit") + 1], "20000")

    def test_in_node_fanout_is_parallel_idempotent_and_isolates_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            profile = {"storage": {"lustre_root": str(root), "directories": {"state": "state"}}}
            state = StateStore(root / "state")
            failing = {"index": -1}

            def fake_run_stage(active_profile, stage, task_index):
                if task_index == failing["index"]:
                    raise RuntimeError(f"synthetic failure on {task_index}")
                time.sleep(0.3)
                payload = {"stage": stage, "task_index": task_index}
                StateStore(Path(active_profile["storage"]["lustre_root"]) / "state").complete(
                    stage, f"task-{task_index:06d}", payload
                )
                return payload

            def group(first: int, count: int, limit: int) -> tuple[int, float]:
                started = time.monotonic()
                code = stage_runner._run_task_group(
                    profile, "normalize", first_index=first, task_count=count,
                    task_limit=limit, workers=0,
                )
                return code, time.monotonic() - started

            with mock.patch.object(stage_runner, "run_stage", side_effect=fake_run_stage):
                code, elapsed = group(0, 12, 12)
                self.assertEqual(code, 0)
                self.assertLess(elapsed, 12 * 0.3, "tasks did not run concurrently")
                self.assertTrue(all(state.is_complete("normalize", f"task-{i:06d}") for i in range(12)))

                code, elapsed = group(0, 12, 12)
                self.assertEqual(code, 0)
                self.assertLess(elapsed, 0.3, "completed tasks were re-run")

                # A trailing group must never reach past its submission's chunk.
                self.assertEqual(group(12, 12, 16)[0], 0)
                self.assertFalse(state.is_complete("normalize", "task-000016"))

                failing["index"] = 42
                self.assertEqual(group(40, 8, 48)[0], 1, "a failed task must fail the job")
                self.assertFalse(state.is_complete("normalize", "task-000042"))
                for index in (40, 41, 43, 44, 45, 46, 47):
                    self.assertTrue(state.is_complete("normalize", f"task-{index:06d}"))

                failing["index"] = -1
                self.assertEqual(group(40, 8, 48)[0], 0)
                self.assertTrue(state.is_complete("normalize", "task-000042"))


if __name__ == "__main__":
    unittest.main()
