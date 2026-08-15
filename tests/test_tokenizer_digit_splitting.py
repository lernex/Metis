from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer

from metis_data.tokenizer import train_tokenizer

CORPUS = [
    "The year 2026 cost 147832 dollars and 55 cents",
    "x = 1234567 + 89 * 4321",
    "invoice 90210 total 31415926 balance 271828",
    "port 8080 pid 12345 offset 65536 length 1024",
] * 60


def _train(tmp: str, *, split_digits: bool) -> Tokenizer:
    train_tokenizer(
        iter(CORPUS),
        output_dir=tmp,
        vocabulary_size=800,
        special_tokens=["<|endoftext|>"],
        minimum_frequency=1,
        split_digits=split_digits,
    )
    return Tokenizer.from_file(str(Path(tmp) / "tokenizer.json"))


class DigitSplittingTests(unittest.TestCase):
    """Multi-digit literals must not become single ids when splitting is on.

    The byte-level regex matches runs of digits, so BPE is free to merge a whole
    number into one token. A model then has no positional handle on the digits
    and place value is memorised per literal rather than learned once.
    """

    def test_digits_are_separate_tokens_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tok = _train(tmp, split_digits=True)
            for number in ("147832", "1234567", "31415926"):
                with self.subTest(number=number):
                    pieces = tok.encode(number).tokens
                    self.assertEqual(len(pieces), len(number))

    def test_default_still_merges_digit_runs(self) -> None:
        """The default must reproduce 1.6, which was trained without this."""

        with tempfile.TemporaryDirectory() as tmp:
            tok = _train(tmp, split_digits=False)
            self.assertLess(len(tok.encode("147832").tokens), len("147832"))

    def test_round_trip_is_lossless_either_way(self) -> None:
        text = "invoice 90210 total 31415926 balance 271828"
        for split in (True, False):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as tmp:
                tok = _train(tmp, split_digits=split)
                self.assertEqual(tok.decode(tok.encode(text).ids), text)

    def test_letters_are_unaffected(self) -> None:
        """Digit splitting is an arithmetic fix, not a character-level one."""

        with tempfile.TemporaryDirectory() as tmp:
            tok = _train(tmp, split_digits=True)
            # "dollars" appears often enough to merge; splitting digits does not
            # make word pieces character-level.
            self.assertLess(len(tok.encode("dollars").tokens), len("dollars"))


if __name__ == "__main__":
    unittest.main()
