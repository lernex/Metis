from __future__ import annotations

import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path

import zstandard as zstd

from metis_data.stage_runner import _iter_rows


class BinaryOrjsonReaderTests(unittest.TestCase):
    """Decoding these shards faster must not decode them differently.

    The selection stages stream 1.2 TB of token-count shards through this
    reader on one core, so the decode is the stage. It now reads the
    decompressed stream as bytes and parses with orjson, which is about twice
    stdlib json through a TextIOWrapper -- but rows have to come back exactly as
    before, because selection hashes their fields.
    """

    def _roundtrip(self, rows: list[dict], suffix: str) -> list[dict]:
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / f"rows{suffix}"
            if suffix.endswith(".zst"):
                with path.open("wb") as raw:
                    with zstd.ZstdCompressor(level=3).stream_writer(raw) as comp:
                        with io.TextIOWrapper(comp, encoding="utf-8") as handle:
                            for row in rows:
                                handle.write(json.dumps(row) + "\n")
            elif suffix.endswith(".gz"):
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
            else:
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            return list(_iter_rows(path))

    def test_rows_survive_every_container(self) -> None:
        rows = [
            {"doc_id": "a", "source_id": "s1", "token_count": 12, "text": "plain"},
            {"doc_id": "b", "source_id": "s2", "token_count": 0, "text": ""},
        ]
        for suffix in (".jsonl.zst", ".jsonl.gz", ".jsonl"):
            with self.subTest(suffix=suffix):
                self.assertEqual(self._roundtrip(rows, suffix), rows)

    def test_unicode_is_not_mangled_by_the_binary_read(self) -> None:
        rows = [
            {"doc_id": "u1", "text": "héllo ✅ 中文 🚀 ünïcôde"},
            {"doc_id": "u2", "text": "tab\there newline\\nescaped"},
            {"doc_id": "u3", "text": "\u00e9" * 500},
        ]
        self.assertEqual(self._roundtrip(rows, ".jsonl.zst"), rows)

    def test_blank_lines_are_skipped_not_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "rows.jsonl.zst"
            with path.open("wb") as raw:
                with zstd.ZstdCompressor(level=3).stream_writer(raw) as comp:
                    with io.TextIOWrapper(comp, encoding="utf-8") as handle:
                        handle.write(json.dumps({"doc_id": "a"}) + "\n")
                        handle.write("\n")
                        handle.write("   \n")
                        handle.write(json.dumps({"doc_id": "b"}) + "\n")
            self.assertEqual(list(_iter_rows(path)), [{"doc_id": "a"}, {"doc_id": "b"}])

    def test_non_dict_payloads_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "rows.jsonl"
            path.write_text('{"doc_id": "a"}\n[1,2,3]\n"scalar"\n{"doc_id": "b"}\n',
                            encoding="utf-8")
            self.assertEqual(list(_iter_rows(path)), [{"doc_id": "a"}, {"doc_id": "b"}])

    def test_numeric_fidelity(self) -> None:
        """token_count feeds quota arithmetic; it must not become a float."""

        rows = [{"doc_id": "n", "token_count": 9007199254740993, "ratio": 0.1}]
        got = self._roundtrip(rows, ".jsonl.zst")
        self.assertEqual(got, rows)
        self.assertIsInstance(got[0]["token_count"], int)


if __name__ == "__main__":
    unittest.main()
