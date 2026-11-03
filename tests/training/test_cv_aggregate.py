"""CV aggregation: fold concatenation, per-class metrics, cluster bootstrap."""

from __future__ import annotations

import random
import unittest

import numpy as np
import pandas as pd

from mermaid_classifier.training.cv_aggregate import (
    cluster_bootstrap_intervals,
    cluster_counts,
    combine_cv_runs,
    concat_folds,
    point_estimates,
    summarize,
    to_markdown,
    unknown_class_counts,
)


def _rows(
    gt: list[str],
    pred: list[str],
    *,
    site: str = "coralnet",
    project: str = "1",
    start: int = 0,
    known: bool = True,
) -> pd.DataFrame:
    n = len(gt)
    return pd.DataFrame(
        {
            "bucket": "b",
            "feature_vector": [f"{site}/{project}/i{start + i}" for i in range(n)],
            "row": 1,
            "col": 1,
            "site": site,
            "project_id": project,
            "gt_bagf": gt,
            "pred_bagf": pred,
            "pred_score": 0.9,
            "in_model_classes": known,
        }
    )


class ConcatFoldsTest(unittest.TestCase):
    def test_overlapping_point_across_runs_is_rejected(self):
        a = _rows(["x", "y"], ["x", "y"])
        b = _rows(["x"], ["x"])  # same key as a's first row
        with self.assertRaisesRegex(ValueError, "more than one run"):
            concat_folds({"run_a": a, "run_b": b})

    def test_disjoint_folds_are_concatenated_with_run_id(self):
        a = _rows(["x"], ["x"], project="1")
        b = _rows(["y"], ["y"], project="2")
        out = concat_folds({"run_a": a, "run_b": b})
        self.assertEqual(sorted(out["run_id"]), ["run_a", "run_b"])
        self.assertEqual(len(out), 2)


class CombineCvRunsTest(unittest.TestCase):
    GROUPS = b"site,group_id,fold\ncoralnet,1,0\ncoralnet,2,1\ncoralnet,3,test\n"

    def _runs(self, folds=("0", "1"), groups=None, projects=("1", "2")):
        groups = groups or [self.GROUPS] * len(folds)
        return {
            f"run{i}": (fold, groups[i], _rows(["x"], ["x"], project=projects[i]))
            for i, fold in enumerate(folds)
        }

    def test_complete_distinct_folds_are_combined(self):
        out = combine_cv_runs(self._runs())
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(out["run_id"]), ["run0", "run1"])

    def test_test_run_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "holdout_fold"):
            combine_cv_runs(self._runs(folds=("0", "test")))

    def test_missing_or_non_integer_fold_param_is_rejected(self):
        for bad in (None, "abc", ""):
            with (
                self.subTest(bad=bad),
                self.assertRaisesRegex(ValueError, "holdout_fold"),
            ):
                combine_cv_runs(self._runs(folds=("0", bad)))

    def test_runs_covering_part_of_the_selections_folds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly folds 0..1"):
            combine_cv_runs(self._runs(folds=("0",), projects=("1",)))

    def test_selection_with_a_fold_gap_is_rejected(self):
        gapped = b"site,group_id,fold\ncoralnet,1,0\ncoralnet,2,2\ncoralnet,3,test\n"
        with self.assertRaisesRegex(ValueError, "0..K-1"):
            combine_cv_runs(self._runs(groups=[gapped, gapped]))

    def test_selection_with_no_cv_folds_is_rejected(self):
        only_test = b"site,group_id,fold\ncoralnet,3,test\n"
        with self.assertRaisesRegex(ValueError, "0..K-1"):
            combine_cv_runs(self._runs(groups=[only_test, only_test]))

    def test_duplicate_fold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "more than one run"):
            combine_cv_runs(self._runs(folds=("0", "0")))

    def test_differing_holdout_groups_are_rejected(self):
        other = self.GROUPS.replace(b"coralnet,2,1", b"coralnet,2,0")
        with self.assertRaisesRegex(ValueError, "holdout_groups.csv"):
            combine_cv_runs(self._runs(groups=[self.GROUPS, other]))

    def test_group_in_two_runs_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coralnet:1"):
            combine_cv_runs(self._runs(projects=("1", "1")))


class HandComputedMetricsTest(unittest.TestCase):
    def setUp(self):
        gt = ["a", "a", "a", "b", "b", "c"]
        pred = ["a", "a", "b", "b", "c", "c"]
        self.df = _rows(gt, pred)

    def test_per_class_recall_precision_f1(self):
        counts = cluster_counts(self.df["gt_bagf"], self.df["pred_bagf"], self.df["project_id"])
        est = point_estimates(counts)
        self.assertEqual(counts.classes, ["a", "b", "c"])
        np.testing.assert_allclose(est["recall"], [2 / 3, 1 / 2, 1.0])
        np.testing.assert_allclose(est["precision"], [1.0, 1 / 2, 1 / 2])
        np.testing.assert_allclose(est["f1"], [0.8, 0.5, 2 / 3])
        self.assertAlmostEqual(est["accuracy"], 4 / 6)
        self.assertAlmostEqual(est["macro_recall"], 13 / 18)
        self.assertAlmostEqual(est["macro_precision"], 2 / 3)
        self.assertAlmostEqual(est["macro_f1"], (0.8 + 0.5 + 2 / 3) / 3)

    def test_unknown_class_points_are_excluded_and_counted(self):
        extra = _rows(["zzz", "zzz"], ["a", "a"], start=100, known=False)
        df = pd.concat([self.df, extra], ignore_index=True)
        summary = summarize(df, reps=20, seed=0)
        acc = summary[(summary["scope"] == "all") & (summary["metric"] == "accuracy")][
            "estimate"
        ].iloc[0]
        self.assertAlmostEqual(acc, 4 / 6)
        counts = unknown_class_counts(df).set_index("scope")
        self.assertEqual(counts.loc["all", "n_unknown_class"], 2)
        self.assertEqual(counts.loc["all", "n_points"], 8)


def _synthetic_df(seed: int = 4) -> pd.DataFrame:
    rng = random.Random(seed)
    classes = ["a", "b", "c"]
    frames = []
    for i in range(24):
        site = "coralnet" if i < 16 else "mermaid"
        gt = [rng.choice(classes) for _ in range(30)]
        acc = rng.uniform(0.4, 0.95)
        pred = [g if rng.random() < acc else rng.choice(classes) for g in gt]
        frames.append(_rows(gt, pred, site=site, project=str(i)))
    return pd.concat(frames, ignore_index=True)


class BootstrapTest(unittest.TestCase):
    def test_intervals_contain_point_estimates(self):
        df = _synthetic_df()
        counts = cluster_counts(df["gt_bagf"], df["pred_bagf"], df["site"] + ":" + df["project_id"])
        est = point_estimates(counts)
        lo, hi = cluster_bootstrap_intervals(counts, reps=500, seed=0)
        for name in ("recall", "precision", "f1"):
            self.assertTrue(np.all(lo[name] <= est[name]), name)
            self.assertTrue(np.all(est[name] <= hi[name]), name)
        for name in ("accuracy", "macro_recall", "macro_precision", "macro_f1"):
            self.assertLessEqual(lo[name], est[name], name)
            self.assertLessEqual(est[name], hi[name], name)

    def test_same_seed_is_deterministic_and_seeds_differ(self):
        df = _synthetic_df()
        counts = cluster_counts(df["gt_bagf"], df["pred_bagf"], df["site"] + ":" + df["project_id"])
        a = cluster_bootstrap_intervals(counts, reps=200, seed=7)
        b = cluster_bootstrap_intervals(counts, reps=200, seed=7)
        c = cluster_bootstrap_intervals(counts, reps=200, seed=8)
        self.assertEqual(a[0]["accuracy"], b[0]["accuracy"])
        self.assertNotEqual(a[0]["accuracy"], c[0]["accuracy"])

    def test_clusters_are_resampled_whole(self):
        # Two clusters, one perfect and one always wrong: a point-level
        # bootstrap would give a tight interval around 0.5; a cluster
        # bootstrap reaches both 0 and 1.
        df = pd.concat(
            [
                _rows(["a"] * 50, ["a"] * 50, project="1"),
                _rows(["a"] * 50, ["b"] * 50, project="2", start=100),
            ],
            ignore_index=True,
        )
        counts = cluster_counts(df["gt_bagf"], df["pred_bagf"], df["project_id"])
        lo, hi = cluster_bootstrap_intervals(counts, reps=400, seed=0)
        self.assertEqual((lo["accuracy"], hi["accuracy"]), (0.0, 1.0))


class SummaryTest(unittest.TestCase):
    def test_site_scopes_use_only_their_own_points(self):
        df = _synthetic_df()
        summary = summarize(df, reps=50, seed=0)
        self.assertEqual(set(summary["scope"]), {"all", "coralnet", "mermaid"})
        for site in ("coralnet", "mermaid"):
            sub = df[df["site"] == site]
            expected = float((sub["gt_bagf"] == sub["pred_bagf"]).mean())
            got = summary[(summary["scope"] == site) & (summary["metric"] == "accuracy")]
            self.assertAlmostEqual(got["estimate"].iloc[0], expected)

    def test_markdown_lists_each_scope(self):
        summary = summarize(_synthetic_df(), reps=20, seed=0)
        md = to_markdown(summary)
        for scope in ("all", "coralnet", "mermaid"):
            self.assertIn(f"## {scope}", md)
        self.assertIn("| class |", md)


if __name__ == "__main__":
    unittest.main()
