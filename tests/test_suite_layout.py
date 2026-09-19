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
        runs there.
        """
        missing = sorted(
            str(d.relative_to(TESTS_ROOT))
            for d in TESTS_ROOT.rglob("*")
            if d.is_dir()
            and d.name != "__pycache__"
            and any(d.glob("test_*.py"))
            and not (d / "__init__.py").is_file()
        )

        self.assertEqual(
            missing,
            [],
            f"test packages missing __init__.py, so discovery skips them: {missing}",
        )
