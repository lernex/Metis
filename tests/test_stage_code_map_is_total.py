"""The stage-module map must name stages that actually exist.

``stage_code_sha256`` falls back to binding every module when a stage is not in
``STAGE_MODULES``. That fallback is deliberate -- an unmapped stage should
over-bind rather than silently reuse work the changed code would not have
produced -- but it makes a typo'd key indistinguishable from a deliberate
choice at runtime, because both merely over-bind and over-binding never
produces a wrong artifact.

It cost an hour on 1.6. The map held ``"context"`` while the stages are called
``context_select``, ``context_prepare``, ``context_pack`` and
``context_verify``, so the entry was never read and all four of the longest
stages in the pipeline bound to the whole package.
"""

import unittest

from metis_data.slurm import BUILD_GRAPH
from metis_data.stage_code import COMMON_MODULES, STAGE_MODULES, stage_code_sha256


class StageCodeMapTests(unittest.TestCase):
    def test_every_build_stage_is_mapped(self) -> None:
        # The direction that matters. An unmapped build stage binds the whole
        # package; the map already spells out empty tuples for stages with no
        # dedicated modules, so absence means oversight, not "nothing to bind".
        built = {str(stage) for stage, _ in BUILD_GRAPH}
        unmapped = sorted(built - set(STAGE_MODULES))
        self.assertEqual(
            unmapped,
            [],
            f"build stages missing from STAGE_MODULES: {unmapped}. Each one "
            "binds every module in the package, so any unrelated edit moves "
            "its execution contract and invalidates finished work.",
        )

    def test_context_stages_bind_narrowly(self) -> None:
        # The specific regression. Each context stage must resolve through the
        # map rather than the bind-to-everything fallback, which is observable
        # as its hash being unaffected by an unrelated module.
        for stage in (
            "context_select",
            "context_prepare",
            "context_pack",
            "context_verify",
        ):
            self.assertIn(stage, STAGE_MODULES, f"{stage} falls back to all modules")
            self.assertIn("context_extension.py", STAGE_MODULES[stage])

    def test_mapped_stages_do_not_bind_the_whole_package(self) -> None:
        # A mapped stage binds COMMON_MODULES plus its own; if that ever equals
        # the full package the map has stopped meaning anything.
        for stage, names in STAGE_MODULES.items():
            bound = set(COMMON_MODULES) | set(names)
            self.assertLess(
                len(bound),
                40,
                f"{stage} binds {len(bound)} modules, which is close enough to "
                "everything that the map is not narrowing anything",
            )

    def test_hashes_are_stable_and_distinct(self) -> None:
        # Two stages with different module sets must not collide, and repeated
        # calls must agree.
        self.assertEqual(
            stage_code_sha256("context_pack"), stage_code_sha256("context_pack")
        )
        self.assertNotEqual(
            stage_code_sha256("context_pack"), stage_code_sha256("normalize")
        )


if __name__ == "__main__":
    unittest.main()
