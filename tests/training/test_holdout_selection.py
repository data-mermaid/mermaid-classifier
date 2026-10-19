"""Group holdout selection: pure greedy test-group pick plus fold assignment."""

from __future__ import annotations

import math
import random
import unittest

import pandas as pd

from mermaid_classifier.training.holdout_selection import (
    SelectionParams,
    build_coverage_report,
    coverage_warnings,
    select_holdout,
)

CORALNET = "coralnet"
MERMAID = "mermaid"


def _counts(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["class_key", "site", "project_id", "n"])


def _synthetic_counts(seed: int = 1) -> pd.DataFrame:
    """40 CoralNet sources, 20 MERMAID projects and an 'unknown' MERMAID group,
    each holding a random subset of 8 classes."""
    rng = random.Random(seed)
    classes = [f"c{i}" for i in range(8)]
    rows: list[tuple[str, str, str, int]] = []
    groups = [(CORALNET, str(i)) for i in range(40)]
    groups += [(MERMAID, f"p{i}") for i in range(20)]
    for site, gid in groups:
        for c in rng.sample(classes, rng.randint(3, 8)):
            rows.append((c, site, gid, rng.randint(5, 60)))
    for c in classes:
        rows.append((c, MERMAID, "unknown", 500))
    return _counts(rows)


def _site_groups(df: pd.DataFrame, site: str) -> set[tuple[str, str]]:
    sub = df[(df["site"] == site) & (df["project_id"] != "unknown")]
    return {(site, g) for g in sub["project_id"].unique()}


LOOSE = SelectionParams(
    test_frac=0.10, folds=5, min_test_points=30, max_class_group_loss=0.25, seed=0
)


class SelectionTest(unittest.TestCase):
    def test_same_seed_gives_identical_assignments(self):
        df = _synthetic_counts()
        a = select_holdout(df, LOOSE, beta_sources=set())
        b = select_holdout(df, LOOSE, beta_sources=set())
        self.assertEqual(a.assignments, b.assignments)

    def test_row_order_does_not_change_the_selection(self):
        df = _synthetic_counts()
        shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
        a = select_holdout(df, LOOSE, beta_sources=set())
        b = select_holdout(shuffled, LOOSE, beta_sources=set())
        self.assertEqual(a.assignments, b.assignments)

    def test_each_site_selects_about_test_frac_of_its_own_groups(self):
        df = _synthetic_counts()
        result = select_holdout(df, LOOSE, beta_sources=set())
        for site, expected in ((CORALNET, 4), (MERMAID, 2)):
            groups = _site_groups(df, site)
            tests = {g for g in groups if result.assignments[g] == "test"}
            self.assertEqual(len(tests), expected, site)

    def test_beta_sources_are_never_test_even_when_most_valuable(self):
        rows = [("rare", CORALNET, "1", 500)]  # the only source with 'rare'
        for i in range(2, 22):
            rows.append(("common", CORALNET, str(i), 50))
            rows.append(("rare", CORALNET, str(i), 1))
        df = _counts(rows)
        params = SelectionParams(0.10, 3, 30, 1.0, 0)
        result = select_holdout(df, params, beta_sources={"1"})
        self.assertNotEqual(result.assignments[(CORALNET, "1")], "test")
        # Beta sources still train: they land in a fold.
        self.assertIn(result.assignments[(CORALNET, "1")], {"0", "1", "2"})

    def test_beta_exclusion_does_not_touch_mermaid_projects(self):
        rows = [("a", MERMAID, f"p{i}", 40) for i in range(20)]
        result = select_holdout(
            _counts(rows), SelectionParams(0.10, 2, 30, 1.0, 0), beta_sources={"p0", "p1"}
        )
        tests = [g for g, f in result.assignments.items() if f == "test"]
        self.assertEqual(len(tests), 2)

    def test_unknown_group_is_never_test_and_never_in_a_fold(self):
        df = _synthetic_counts()
        result = select_holdout(df, LOOSE, beta_sources=set())
        self.assertNotIn((MERMAID, "unknown"), result.assignments)

    def test_max_class_group_loss_bounds_test_groups_per_class(self):
        df = _synthetic_counts()
        params = SelectionParams(0.30, 5, 10_000, 0.25, 0)
        result = select_holdout(df, params, beta_sources=set())
        assignments = result.assignments
        for cls, rows in df.groupby("class_key"):
            groups = {(s, p) for s, p in zip(rows["site"], rows["project_id"], strict=True)}
            in_test = sum(1 for g in groups if assignments.get(g) == "test")
            self.assertLessEqual(in_test, math.floor(0.25 * len(groups)), cls)

    def test_a_class_in_one_group_keeps_that_group_out_of_test(self):
        rows = [("solo", CORALNET, "0", 100)]
        rows += [("common", CORALNET, str(i), 50) for i in range(1, 20)]
        rows += [("common", CORALNET, "0", 50)]
        result = select_holdout(
            _counts(rows), SelectionParams(0.10, 2, 30, 0.25, 0), beta_sources=set()
        )
        self.assertNotEqual(result.assignments[(CORALNET, "0")], "test")

    def test_mermaid_coverage_is_judged_without_coralnet_test_points(self):
        rows = [("x", CORALNET, str(i), 100) for i in range(10)]
        rows += [("x", MERMAID, "p0", 50), ("y", MERMAID, "p0", 5)]
        rows += [("y", MERMAID, f"p{i}", 50) for i in range(1, 10)]
        selection = select_holdout(_counts(rows), LOOSE, set())
        self.assertEqual(selection.assignments[(MERMAID, "p0")], "test")

    def test_every_non_test_group_lands_in_exactly_one_fold(self):
        df = _synthetic_counts()
        result = select_holdout(df, LOOSE, beta_sources=set())
        every = _site_groups(df, CORALNET) | _site_groups(df, MERMAID)
        self.assertEqual(set(result.assignments), every)
        folds = {f for f in result.assignments.values() if f != "test"}
        self.assertEqual(folds, {"0", "1", "2", "3", "4"})

    def test_folds_balance_points_for_equal_groups(self):
        rows = [("a", CORALNET, str(i), 10) for i in range(8)]
        result = select_holdout(
            _counts(rows), SelectionParams(0.0, 2, 30, 1.0, 0), beta_sources=set()
        )
        per_fold = pd.Series(result.assignments).value_counts()
        self.assertEqual(per_fold.to_dict(), {"0": 4, "1": 4})


class CoverageReportTest(unittest.TestCase):
    def setUp(self):
        self.df = _counts(
            [
                ("a", CORALNET, "1", 40),
                ("a", CORALNET, "2", 10),
                ("a", CORALNET, "3", 20),
                ("b", CORALNET, "1", 5),
                ("b", CORALNET, "3", 7),
            ]
        )
        self.assignments = {
            (CORALNET, "1"): "test",
            (CORALNET, "2"): "0",
            (CORALNET, "3"): "1",
        }

    def test_report_matches_hand_computed_values(self):
        report = build_coverage_report(self.df, self.assignments, folds=2).set_index("class_key")
        a, b = report.loc["a"], report.loc["b"]
        self.assertEqual(
            (a.total_groups, a.total_points, a.test_groups, a.test_points, a.train_groups),
            (3, 70, 1, 40, 2),
        )
        self.assertEqual((a.points_fold_0, a.points_fold_1), (10, 20))
        self.assertEqual(
            (b.total_groups, b.total_points, b.test_groups, b.test_points, b.train_groups),
            (2, 12, 1, 5, 1),
        )
        self.assertEqual((b.points_fold_0, b.points_fold_1), (0, 7))

    def test_warnings_flag_thin_test_coverage_and_empty_folds(self):
        report = build_coverage_report(self.df, self.assignments, folds=2)
        warnings = coverage_warnings(report, min_test_points=30, folds=2)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("'b'" in w and "test" in w for w in warnings))
        self.assertTrue(any("'b'" in w and "fold 0" in w for w in warnings))
        self.assertFalse(any("'a'" in w for w in warnings))

    def test_warnings_apply_min_test_points_per_site(self):
        df = _counts(
            [
                ("a", CORALNET, "1", 40),
                ("a", MERMAID, "m1", 5),
                ("a", MERMAID, "m2", 10),
                ("c", CORALNET, "1", 40),
            ]
        )
        assignments = {(CORALNET, "1"): "test", (MERMAID, "m1"): "test", (MERMAID, "m2"): "0"}
        report = build_coverage_report(df, assignments, folds=1).set_index("class_key")
        self.assertEqual(
            (report.loc["a"].test_points_coralnet, report.loc["a"].test_points_mermaid), (40, 5)
        )
        warnings = [
            w
            for w in coverage_warnings(report.reset_index(), min_test_points=30, folds=1)
            if "test points" in w
        ]
        self.assertEqual(len(warnings), 1)
        self.assertIn("'a'", warnings[0])
        self.assertIn("mermaid", warnings[0])


if __name__ == "__main__":
    unittest.main()
