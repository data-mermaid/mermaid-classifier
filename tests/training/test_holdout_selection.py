"""Group holdout selection: pure greedy test-group pick plus fold assignment."""

from __future__ import annotations

import dataclasses
import math
import random
import unittest

import pandas as pd

from mermaid_classifier.training.holdout_selection import (
    SelectionParams,
    SiteShare,
    build_coverage_report,
    coverage_warnings,
    select_holdout,
    select_val_groups,
    site_test_shares,
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


class ValSelectionTest(unittest.TestCase):
    def _test_groups(self, df: pd.DataFrame) -> set[tuple[str, str]]:
        sel = select_holdout(df, LOOSE, beta_sources=set())
        return {g for g, f in sel.assignments.items() if f == "test"}

    def test_val_is_disjoint_from_test_and_never_unknown(self):
        df = _synthetic_counts()
        tests = self._test_groups(df)
        val = select_val_groups(df, tests, LOOSE).assignments
        self.assertTrue(val)
        self.assertFalse(set(val) & tests)
        self.assertNotIn((MERMAID, "unknown"), val)

    def test_each_site_selects_about_val_frac_of_its_non_test_groups(self):
        df = _synthetic_counts()
        tests = self._test_groups(df)
        val = select_val_groups(df, tests, LOOSE).assignments
        # 36 non-test CoralNet and 18 non-test MERMAID groups, 10% rounded half up.
        for site, expected in ((CORALNET, 4), (MERMAID, 2)):
            self.assertEqual(len([g for g in val if g[0] == site]), expected, site)

    def test_every_class_keeps_a_non_val_non_test_group(self):
        df = _synthetic_counts()
        tests = self._test_groups(df)
        val = set(select_val_groups(df, tests, LOOSE).assignments)
        for class_key, sub in df.groupby("class_key"):
            gs = {(str(s), str(p)) for s, p in zip(sub["site"], sub["project_id"], strict=True)}
            self.assertTrue(gs - val - tests, class_key)

    def test_a_class_confined_to_one_group_never_loses_it_to_val(self):
        rows = [("rare", CORALNET, "1", 500)]
        for i in range(2, 22):
            rows.append(("common", CORALNET, str(i), 50))
        val = select_val_groups(_counts(rows), set(), LOOSE).assignments
        self.assertNotIn((CORALNET, "1"), val)

    def test_deterministic_and_row_order_independent(self):
        df = _synthetic_counts()
        tests = self._test_groups(df)
        shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
        a = select_val_groups(df, tests, LOOSE)
        b = select_val_groups(shuffled, tests, LOOSE)
        self.assertEqual(a.assignments, b.assignments)
        self.assertEqual(a.assignments, select_val_groups(df, tests, LOOSE).assignments)

    def test_shortfall_names_the_site_and_speaks_of_val_groups(self):
        df = _synthetic_counts()
        tests = self._test_groups(df)
        tight = dataclasses.replace(LOOSE, max_test_point_frac=0.001)
        result = select_val_groups(df, tests, tight)
        self.assertIn(CORALNET, result.short_sites)
        self.assertTrue(result.warnings)
        for w in result.warnings:
            self.assertNotIn("test groups", w)
            self.assertNotIn("max_test_point_frac", w)
        self.assertTrue(any("max_val_point_frac" in w for w in result.warnings))

    def test_no_shortfall_when_every_site_reaches_its_target(self):
        df = _synthetic_counts()
        result = select_val_groups(df, self._test_groups(df), LOOSE)
        self.assertEqual(result.short_sites, [])


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


def _one_huge_many_small() -> pd.DataFrame:
    """MERMAID: p0 holds 1000 points, p1..p19 hold 10 each (site total 1190, 10% = 119)."""
    rows = [("a", MERMAID, "p0", 1000)]
    rows += [("a", MERMAID, f"p{i}", 10) for i in range(1, 20)]
    return _counts(rows)


def _test_points(df: pd.DataFrame, assignments: dict[tuple[str, str], str], site: str) -> int:
    sub = df[df["site"] == site]
    return sum(
        int(n)
        for p, n in zip(sub["project_id"], sub["n"], strict=True)
        if assignments.get((site, p)) == "test"
    )


class PointCapTest(unittest.TestCase):
    CAPPED = SelectionParams(0.10, 3, 30, 1.0, 0, max_test_point_frac=0.10)

    def test_huge_group_stays_out_of_test_when_it_would_exceed_the_cap(self):
        result = select_holdout(_one_huge_many_small(), self.CAPPED, beta_sources=set())
        self.assertNotEqual(result.assignments[(MERMAID, "p0")], "test")
        tests = [g for g, f in result.assignments.items() if f == "test"]
        self.assertEqual(len(tests), 2)

    def test_without_the_cap_the_huge_group_is_picked_for_its_gain(self):
        for cap in (None, 0.0):
            params = dataclasses.replace(self.CAPPED, max_test_point_frac=cap)
            result = select_holdout(_one_huge_many_small(), params, beta_sources=set())
            self.assertEqual(result.assignments[(MERMAID, "p0")], "test", cap)

    def test_realised_point_share_is_within_the_cap_for_every_site(self):
        df = _synthetic_counts()
        params = dataclasses.replace(LOOSE, test_frac=0.30, max_test_point_frac=0.10)
        result = select_holdout(df, params, beta_sources=set())
        for site in (CORALNET, MERMAID):
            total = int(df[df["site"] == site]["n"].sum())
            self.assertLessEqual(_test_points(df, result.assignments, site), 0.10 * total, site)

    def test_cap_counts_the_unknown_group_in_the_site_total(self):
        rows = [("a", MERMAID, f"p{i}", 100) for i in range(10)]
        rows.append(("a", MERMAID, "unknown", 9000))
        # Site total 10000, cap 10% = 1000: all ten known groups (100 each) may be test.
        params = SelectionParams(0.50, 2, 30, 1.0, 0, max_test_point_frac=0.10)
        result = select_holdout(_counts(rows), params, beta_sources=set())
        tests = [g for g, f in result.assignments.items() if f == "test"]
        self.assertEqual(len(tests), 5)

    def test_group_target_still_limits_when_the_cap_is_loose(self):
        rows = [("a", MERMAID, f"p{i}", 10) for i in range(20)]
        params = SelectionParams(0.10, 2, 30, 1.0, 0, max_test_point_frac=0.90)
        result = select_holdout(_counts(rows), params, beta_sources=set())
        tests = [g for g, f in result.assignments.items() if f == "test"]
        self.assertEqual(len(tests), 2)

    def test_cap_that_leaves_no_test_group_warns(self):
        rows = [("a", MERMAID, "p0", 100), ("a", MERMAID, "p1", 100)]
        params = SelectionParams(0.50, 2, 30, 1.0, 0, max_test_point_frac=0.10)
        result = select_holdout(_counts(rows), params, beta_sources=set())
        self.assertFalse(any(f == "test" for f in result.assignments.values()))
        self.assertTrue(any("max_test_point_frac" in w and "mermaid" in w for w in result.warnings))

    def test_capped_selection_is_deterministic_and_row_order_independent(self):
        df = _synthetic_counts()
        params = dataclasses.replace(LOOSE, max_test_point_frac=0.10)
        shuffled = df.sample(frac=1.0, random_state=5).reset_index(drop=True)
        a = select_holdout(df, params, beta_sources=set())
        b = select_holdout(shuffled, params, beta_sources=set())
        self.assertEqual(a.assignments, b.assignments)
        self.assertEqual(a.warnings, b.warnings)

    def test_site_test_shares_reports_groups_and_points(self):
        df = _one_huge_many_small()
        assignments = {(MERMAID, "p1"): "test", (MERMAID, "p2"): "test"}
        assignments.update({(MERMAID, f"p{i}"): "0" for i in [0, *range(3, 20)]})
        shares = site_test_shares(df, assignments)
        self.assertEqual(shares[MERMAID], SiteShare(2, 20, 20, 1190))


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
