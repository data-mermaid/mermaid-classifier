"""Tests for mermaid_classifier.pyspacer.metrics.calibration."""

import unittest

import matplotlib.pyplot as plt

from mermaid_classifier.pyspacer.metrics import MetricGroupResult
from mermaid_classifier.pyspacer.metrics.calibration import (
    _adaptive_ece,
    compute_calibration,
)
from pyspacer.metrics_test_helpers import (
    make_ctx,
)


class AdaptiveECETest(unittest.TestCase):
    """Tests for the private _adaptive_ece function."""

    def test_perfectly_calibrated(self):
        """Confidence matching actual accuracy gives ECE near 0."""
        # 50 correct with confidence 0.9, 50 wrong with confidence 0.1.
        # Low-conf bin: avg_conf=0.1, avg_acc=0.0 -> gap=0.1
        # High-conf bin: avg_conf=0.9, avg_acc=1.0 -> gap=0.1
        # ECE = 0.1 * 0.5 + 0.1 * 0.5 = 0.1 (well-calibrated for 2 bins)
        n = 100
        gt = [0] * n
        # First 50 wrong (low conf), last 50 correct (high conf).
        est = [1] * 50 + [0] * 50
        scores = [0.1] * 50 + [0.9] * 50
        ece, bin_data = _adaptive_ece(scores, gt, est, n_bins=2)
        self.assertAlmostEqual(ece, 0.1, places=5)

    def test_maximally_overconfident(self):
        """All predictions wrong with confidence 1.0 gives ECE close to 1.0."""
        n = 100
        gt = [0] * n
        est = [1] * n  # all wrong
        scores = [1.0] * n
        ece, bin_data = _adaptive_ece(scores, gt, est, n_bins=2)
        self.assertAlmostEqual(ece, 1.0, places=5)


class ComputeCalibrationTest(unittest.TestCase):
    """Tests for compute_calibration."""

    def test_returns_expected_artifacts(self):
        """Result contains ece scalar, per_bin_details df, reliability_diagram
        figure, and per_category_ece df."""
        ctx = make_ctx(
            gt_indices=[0, 1, 0, 1],
            est_indices=[0, 1, 1, 0],
            classes=["A1::", "B1::"],
            scores=[0.9, 0.8, 0.7, 0.6],
        )
        result = compute_calibration(ctx)

        self.assertIsInstance(result, MetricGroupResult)

        scalar_names = {s.name for s in result.scalars}
        self.assertIn("ece", scalar_names)

        df_paths = {df.artifact_path for df in result.dataframes}
        self.assertIn("calibration/per_bin_details", df_paths)
        self.assertIn("calibration/per_category_ece", df_paths)

        fig_paths = {f.artifact_path for f in result.figures}
        self.assertIn("calibration/reliability_diagram.png", fig_paths)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_per_category_ece_respects_min_samples(self):
        """With fewer than 30 samples per category, per_category_ece is empty."""
        # Only 4 samples total, well below the 30-sample minimum.
        ctx = make_ctx(
            gt_indices=[0, 0, 1, 1],
            est_indices=[0, 1, 1, 0],
            classes=["A1::", "B1::"],
            scores=[0.9, 0.6, 0.8, 0.55],
        )
        result = compute_calibration(ctx)

        per_cat_df_result = next(
            df for df in result.dataframes if df.artifact_path == "calibration/per_category_ece"
        )
        self.assertEqual(len(per_cat_df_result.df), 0)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_per_category_ece_populated_with_enough_samples(self):
        """With >30 samples per category, per_category_ece has rows."""
        n_per_class = 40
        # Class 0 = A1:: (top-level A), Class 1 = B1:: (top-level B)
        gt_indices = [0] * n_per_class + [1] * n_per_class
        est_indices = [0] * n_per_class + [1] * n_per_class
        scores = [0.9] * n_per_class + [0.8] * n_per_class

        ctx = make_ctx(
            gt_indices=gt_indices,
            est_indices=est_indices,
            classes=["A1::", "B1::"],
            scores=scores,
        )
        result = compute_calibration(ctx)

        per_cat_df_result = next(
            df for df in result.dataframes if df.artifact_path == "calibration/per_category_ece"
        )
        self.assertGreater(len(per_cat_df_result.df), 0)
        expected_columns = {"category", "ece", "accuracy", "avg_confidence", "n_samples"}
        self.assertEqual(set(per_cat_df_result.df.columns), expected_columns)

        for fig_result in result.figures:
            plt.close(fig_result.fig)


if __name__ == "__main__":
    unittest.main()
