"""Paired comparison of two runs on one holdout: alignment, paired bootstrap, rollups."""

from __future__ import annotations

import random
import unittest

import numpy as np
import pandas as pd

from mermaid_classifier.training.cv_aggregate import bootstrap_weights, cluster_counts
from mermaid_classifier.training.run_compare import (
    OVERALL_KEY,
    align_holdout,
    check_group_disjointness,
    paired_summary,
    real_mix_weights,
    to_markdown_paired,
    top_level_map,
    with_class_names,
)


def _holdout(
    gt: list[str],
    pred: list[str],
    *,
    images: list[str] | None = None,
    projects: list[str] | None = None,
    site: str = "coralnet",
    known: list[bool] | None = None,
) -> pd.DataFrame:
    """A holdout table in the logged schema; labels are BA ids with no growth form."""
    n = len(gt)
    images = images or [f"i{i}" for i in range(n)]
    projects = projects or ["1"] * n
    return pd.DataFrame(
        {
            "row": [str(i) for i in range(n)],
            "col": "1",
            "image_id": images,
            "site": site,
            "bucket": "b",
            "project_id": projects,
            "feature_vector": [
                f"{p}/{im}.featurevector" for p, im in zip(projects, images, strict=True)
            ],
            "benthic_attribute_id": gt,
            "growth_form_id": "",
            "pred_bagf": [f"{p}::" for p in pred],
            "in_model_classes": ["true" if k else "false" for k in (known or [True] * n)],
        }
    )


def _rows(summary: pd.DataFrame, metric: str, level: str = "class") -> pd.DataFrame:
    sub = summary[
        (summary["scope"] == "all") & (summary["metric"] == metric) & (summary["level"] == level)
    ]
    return sub.set_index("class_key")


# Two clusters: project 1 holds images i0 (1 point) and i1 (3 points); project 2
# holds image i2 (2 points).
GT = ["a", "a", "a", "b", "b", "c"]
PRED_A = ["a", "a", "b", "b", "c", "c"]
PRED_B = ["a", "a", "a", "b", "b", "b"]
IMAGES = ["i0", "i1", "i1", "i1", "i2", "i2"]
PROJECTS = ["1", "1", "1", "1", "2", "2"]


def _hand_pair() -> pd.DataFrame:
    a = _holdout(GT, PRED_A, images=IMAGES, projects=PROJECTS)
    b = _holdout(GT, PRED_B, images=IMAGES, projects=PROJECTS)
    aligned, _ = align_holdout(a, b)
    return aligned


class AlignHoldoutTest(unittest.TestCase):
    def test_point_only_in_run_a_is_rejected(self):
        a = _holdout(["a", "b"], ["a", "b"])
        b = _holdout(["a"], ["a"])
        with self.assertRaisesRegex(ValueError, "only in A"):
            align_holdout(a, b)

    def test_relabelled_point_is_rejected(self):
        a = _holdout(["a", "b"], ["a", "b"])
        b = _holdout(["a", "c"], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "relabelled"):
            align_holdout(a, b)

    def test_allow_mismatch_keeps_the_agreeing_intersection(self):
        a = _holdout(["a", "b", "c"], ["a", "b", "c"])
        b = _holdout(["a", "x"], ["a", "b"])  # row 1 relabelled, row 2 missing
        aligned, report = align_holdout(a, b, allow_mismatch=True)
        self.assertEqual(aligned["row"].tolist(), ["0"])
        self.assertEqual((report.only_a, report.only_b, report.relabelled), (1, 0, 1))
        self.assertEqual(report.aligned, 1)

    def test_points_unknown_to_either_run_are_dropped(self):
        a = _holdout(["a", "b"], ["a", "b"], known=[True, False])
        b = _holdout(["a", "b"], ["a", "b"])
        aligned, report = align_holdout(a, b)
        self.assertEqual(len(aligned), 1)
        self.assertEqual(report.unknown, 1)


class PairedSummaryTest(unittest.TestCase):
    def test_identical_runs_give_zero_diff_and_zero_interval(self):
        df = _synthetic(seed=1)
        aligned, _ = align_holdout(df, df)
        summary = paired_summary(aligned, reps=100, seed=0)
        defined = summary.dropna(subset=["est_a"])
        self.assertFalse(defined.empty)
        np.testing.assert_array_equal(defined["diff"], 0.0)
        np.testing.assert_array_equal(defined["diff_ci_low"].dropna(), 0.0)
        np.testing.assert_array_equal(defined["diff_ci_high"].dropna(), 0.0)

    def test_per_class_diff_matches_hand_computation(self):
        summary = paired_summary(_hand_pair(), reps=50, seed=0)
        recall = _rows(summary, "recall")
        np.testing.assert_allclose(recall.loc[["a::", "b::", "c::"], "est_a"], [2 / 3, 1 / 2, 1])
        np.testing.assert_allclose(recall.loc[["a::", "b::", "c::"], "diff"], [1 / 3, 1 / 2, -1])
        precision = _rows(summary, "precision")
        np.testing.assert_allclose(precision.loc[["b::", "c::"], "diff"], [2 / 3 - 1 / 2, -1 / 2])
        self.assertEqual(recall.loc["a::", "support"], 3)
        accuracy = _rows(summary, "accuracy").loc[OVERALL_KEY]
        self.assertAlmostEqual(accuracy["diff"], 5 / 6 - 4 / 6)

    def test_pooled_cover_bias_matches_hand_computation(self):
        bias = _rows(paired_summary(_hand_pair(), reps=50, seed=0), "cover_bias_pooled")
        np.testing.assert_allclose(bias.loc[["a::", "b::", "c::"], "est_a"], [-1 / 6, 0, 1 / 6])
        np.testing.assert_allclose(bias.loc[["a::", "b::", "c::"], "est_b"], [0, 1 / 6, -1 / 6])

    def test_per_image_cover_bias_matches_hand_computation(self):
        # Run A: i0 exact; i1 true (a 2/3, b 1/3) vs pred (a 1/3, b 2/3);
        # i2 true (b 1/2, c 1/2) vs pred (c 1). Mean over three images.
        bias = _rows(paired_summary(_hand_pair(), reps=50, seed=0), "cover_bias_image")
        np.testing.assert_allclose(
            bias.loc[["a::", "b::", "c::"], "est_a"], [-1 / 9, -1 / 18, 1 / 6]
        )
        np.testing.assert_allclose(bias.loc[["a::", "b::", "c::"], "est_b"], [0, 1 / 6, -1 / 6])

    def test_paired_interval_is_narrower_than_unpaired(self):
        df_a = _synthetic(seed=2)
        rng = random.Random(9)
        df_b = df_a.copy()
        flip = [rng.random() < 0.05 for _ in range(len(df_b))]
        df_b.loc[flip, "pred_bagf"] = "a::"
        aligned, _ = align_holdout(df_a, df_b)
        reps = 400
        acc = _rows(paired_summary(aligned, reps=reps, seed=0), "accuracy").loc[OVERALL_KEY]
        paired_width = acc["diff_ci_high"] - acc["diff_ci_low"]

        clusters = aligned["site"] + ":" + aligned["project_id"]
        classes = sorted(set(aligned["gt_bagf"]) | set(aligned["pred_a"]) | set(aligned["pred_b"]))
        ca = cluster_counts(aligned["gt_bagf"], aligned["pred_a"], clusters, classes=classes)
        cb = cluster_counts(aligned["gt_bagf"], aligned["pred_b"], clusters, classes=classes)
        wa = bootstrap_weights(len(ca.n), reps, 1)
        wb = bootstrap_weights(len(cb.n), reps, 2)
        diff = (wb @ cb.tp.sum(1)) / (wb @ cb.n) - (wa @ ca.tp.sum(1)) / (wa @ ca.n)
        low, high = np.percentile(diff, [2.5, 97.5])
        self.assertLess(paired_width, (high - low) / 2)

    def test_markdown_lists_scopes_and_classes(self):
        md = to_markdown_paired(paired_summary(_hand_pair(), reps=20, seed=0))
        self.assertIn("## all", md)
        self.assertIn("a::", md)


class TopLevelTest(unittest.TestCase):
    ROLLUP = pd.DataFrame(
        {
            "from_ba_id": ["a", "b", "c"],
            "from_gf_id": ["", "", ""],
            "to_ba_id": ["T1", "T1", "T2"],
            "to_gf_id": ["", "", ""],
        }
    )

    def test_classes_map_to_targets_and_targets_to_themselves(self):
        mapping = top_level_map(["a::", "b::", "c::", "T1::"], self.ROLLUP)
        self.assertEqual(mapping, {"a::": "T1::", "b::": "T1::", "c::": "T2::", "T1::": "T1::"})

    def test_listed_top_level_id_without_rollup_rows_maps_to_itself(self):
        mapping = top_level_map(["a::", "R::"], self.ROLLUP, top_level_ids={"R", "T1"})
        self.assertEqual(mapping, {"a::": "T1::", "R::": "R::"})

    def test_unmapped_class_is_rejected_by_name(self):
        with self.assertRaisesRegex(ValueError, "z::"):
            top_level_map(["a::", "z::"], self.ROLLUP)

    def test_summary_relabels_gt_and_pred_at_top_level(self):
        aligned = _hand_pair()
        mapping = top_level_map(["a::", "b::", "c::"], self.ROLLUP)
        summary = paired_summary(aligned, reps=20, seed=0, class_map=mapping)
        self.assertEqual(set(summary["level"]), {"top_level"})
        recall = _rows(summary, "recall", level="top_level")
        # Run A: T1 points (a a a b b) predicted a a b b c -> 4/5; T2 (c) -> c, 1.
        np.testing.assert_allclose(recall.loc[["T1::", "T2::"], "est_a"], [4 / 5, 1.0])


class ClassNamesTest(unittest.TestCase):
    def test_names_label_rows_and_markdown_falling_back_to_the_key(self):
        summary = with_class_names(
            paired_summary(_hand_pair(), reps=20, seed=0), {"a::": "Algae A"}
        )
        names = summary.drop_duplicates("class_key").set_index("class_key")["class_name"]
        self.assertEqual(names["a::"], "Algae A")
        self.assertEqual(names["b::"], "b::")
        md = to_markdown_paired(summary)
        self.assertIn("| Algae A |", md)
        self.assertIn("| b:: |", md)


class RealMixWeightsTest(unittest.TestCase):
    COUNTS = pd.DataFrame(
        {
            "benthic_attribute_id": ["a", "b", "c"],
            "growth_form_id": ["", "", ""],
            "available": [30, 2, 1],
            "realized": [3, 2, 1],
        }
    )

    def test_weighted_class_totals_equal_available_on_an_intersection(self):
        # Two of a's three realized points survive alignment.
        gt = ["a::", "a::", "b::", "b::", "c::"]
        weights = real_mix_weights(self.COUNTS, gt)
        totals = pd.Series(weights).groupby(gt).sum()
        np.testing.assert_allclose(totals.loc[["a::", "b::", "c::"]], [30, 2, 1])

    def test_recall_is_unchanged_and_precision_changes(self):
        aligned = _hand_pair()
        weights = real_mix_weights(self.COUNTS, aligned["gt_bagf"].tolist())
        plain = paired_summary(aligned, reps=20, seed=0)
        mixed = paired_summary(aligned, reps=20, seed=0, point_weights=weights)
        np.testing.assert_allclose(_rows(mixed, "recall")["est_a"], _rows(plain, "recall")["est_a"])
        # Run A predicts b for one a-point (weight 10) and one b-point (weight 1).
        self.assertAlmostEqual(_rows(plain, "precision").loc["b::", "est_a"], 1 / 2)
        self.assertAlmostEqual(_rows(mixed, "precision").loc["b::", "est_a"], 1 / 11)

    def test_class_missing_from_counts_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "z::"):
            real_mix_weights(self.COUNTS, ["a::", "z::"])


class GroupDisjointnessTest(unittest.TestCase):
    def test_group_in_both_train_and_val_is_flagged(self):
        stats = pd.DataFrame(
            {
                "site": ["coralnet", "coralnet", "coralnet"],
                "project_id": [1, 2, 3],
                "num_images": [1, 1, 1],
                "num_annotations": [8, 3, 4],
                "train": [5, 0, 0],
                "ref": [0, 0, 0],
                "val": [3, 3, 0],
                "holdout": [0, 0, 4],
                "dropped": [0, 0, 0],
            }
        )
        val = {("coralnet", "1"), ("coralnet", "2")}
        test = {("coralnet", "3")}
        problems = check_group_disjointness(stats, val, test)
        self.assertEqual(len(problems), 1)
        self.assertIn("coralnet:1", problems[0])
        self.assertEqual(check_group_disjointness(stats.iloc[1:], val, test), [])


def _synthetic(seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    classes = ["a", "b", "c"]
    frames = []
    for i in range(20):
        gt = [rng.choice(classes) for _ in range(30)]
        acc = rng.uniform(0.4, 0.95)
        pred = [g if rng.random() < acc else rng.choice(classes) for g in gt]
        images = [f"p{i}i{j // 10}" for j in range(30)]
        frame = _holdout(gt, pred, images=images, projects=[str(i)] * 30)
        frame["site"] = "coralnet" if i < 14 else "mermaid"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    unittest.main()
