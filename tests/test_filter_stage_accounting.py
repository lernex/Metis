from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from metis_data.stage_runner import _datatrove_task_counts, _stat_total


def _write(root: Path, rank: int, steps: list) -> Path:
    stats = root / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    (stats / f"{rank:05d}.json").write_text(json.dumps(steps), encoding="utf-8")
    return root


# The shape DataTrove actually writes, trimmed to the fields that matter.
REAL_STEPS = [
    {
        "name": "READER: Jsonl",
        "stats": {
            "input_files": 1,
            "doc_len": {"total": 21000044756, "n": 779046},
            "documents": {"total": 779046, "n": 1},
        },
    },
    {
        "name": "DECONT: Metis benchmark decontamination",
        "stats": {
            "total": 779046,
            "forwarded": 766536,
            "doc_len": {"total": 20614959793},
            "dropped": 12510,
            "dropped_benchmark_short_ngram": 6062,
            "dropped_benchmark_contiguous_run": 2241,
        },
    },
    {
        "name": "WRITER: Jsonl",
        "stats": {
            "total": 766536,
            "doc_len": {"total": 20614959793},
        },
    },
]


class DatatroveTaskCountsTests(unittest.TestCase):
    """A completion marker should outlive the corpus it describes.

    The paired cleanup retires the predecessor once a stage verifies, so a
    receipt that records only "finished" makes "how much did this pass remove"
    permanently unanswerable -- there is nothing left to compare against.
    """

    def test_reads_the_real_stats_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write(Path(tmp), 2034, REAL_STEPS)
            counts = _datatrove_task_counts(root, 2034)
        self.assertEqual(counts["records_in"], 779_046)
        self.assertEqual(counts["records_out"], 766_536)
        self.assertEqual(counts["records_removed"], 12_510)
        self.assertEqual(counts["bytes_in"], 21_000_044_756)
        self.assertEqual(counts["bytes_out"], 20_614_959_793)
        self.assertEqual(counts["bytes_removed"], 385_084_963)
        self.assertEqual(
            counts["removed_by_reason"],
            {
                "dropped_benchmark_contiguous_run": 2241,
                "dropped_benchmark_short_ngram": 6062,
            },
        )

    def test_removed_is_consistent_with_the_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write(Path(tmp), 1, REAL_STEPS)
            counts = _datatrove_task_counts(root, 1)
        self.assertGreaterEqual(
            counts["records_removed"], sum(counts["removed_by_reason"].values())
        )

    def test_missing_stats_degrade_quietly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_datatrove_task_counts(Path(tmp), 0), {})

    def test_unreadable_stats_degrade_quietly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats = Path(tmp) / "stats"
            stats.mkdir()
            (stats / "00000.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(_datatrove_task_counts(Path(tmp), 0), {})

    def test_a_stage_that_writes_nothing_is_not_reported_as_total_loss(self) -> None:
        """Signature stages read documents and emit no corpus; that is not removal."""

        steps = [
            {"name": "READER", "stats": {"documents": {"total": 500}, "doc_len": {"total": 1000}}},
            {"name": "SIG", "stats": {"signatures": 500}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = _write(Path(tmp), 0, steps)
            counts = _datatrove_task_counts(root, 0)
        self.assertEqual(counts["records_in"], 500)
        self.assertEqual(counts["records_out"], 0)
        self.assertNotIn("removed_by_reason", counts)

    def test_stat_total_accepts_both_encodings(self) -> None:
        self.assertEqual(_stat_total({"total": 7}), 7)
        self.assertEqual(_stat_total(7), 7)
        self.assertEqual(_stat_total(None), 0)
        self.assertEqual(_stat_total("x"), 0)
        self.assertEqual(_stat_total({}), 0)


if __name__ == "__main__":
    unittest.main()
