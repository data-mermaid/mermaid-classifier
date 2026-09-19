"""Invariants of the suite's own layout."""

import unittest

from support.paths import REPO_ROOT

TESTS_ROOT = REPO_ROOT / "tests"


class SuiteLayoutTest(unittest.TestCase):
    def test_every_directory_holding_tests_is_a_package(self):
        """unittest discovery descends into packages only.

        A directory without __init__.py contributes nothing to a full run,
        while `python -m unittest <pkg>.<module>` still executes it -- so a new
        test package passes when its author names it, passes in CI, and never
        runs there. An intermediate directory -- one holding only subpackages,
        no test_*.py of its own -- needs __init__.py just as much: discovery
        cannot descend past it to reach the tests beneath, so every ancestor
        from a test file up to the tests root is checked, not just the leaf.
        """
        missing = sorted(
            {
                str(p.relative_to(TESTS_ROOT))
                for f in TESTS_ROOT.rglob("test_*.py")
                for p in (f.parent, *f.parent.parents)
                if p != TESTS_ROOT and TESTS_ROOT in p.parents and not (p / "__init__.py").is_file()
            }
        )

        self.assertEqual(
            missing,
            [],
            f"test packages missing __init__.py, so discovery skips them: {missing}",
        )
