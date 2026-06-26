"""Tests for mermaid_classifier.pyspacer.metrics.calibration."""

import unittest

import matplotlib.pyplot as plt

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import MetricGroupResult
from mermaid_classifier.pyspacer.metrics.calibration import (
    _adaptive_ece,
    compute_calibration,
)
from pyspacer.metrics_test_helpers import (
    MockBALibrary,
    MockGFLibrary,
    format_metric,
    make_val_results,
)


def _make_ctx(gt_indices, est_indices, classes, scores=None):
    """Build a MetricsContext from simple index lists."""
    val_results = make_val_results(gt_indices, est_indices, classes, scores)
    return MetricsContext(
        val_results=val_results,
        ba_library=MockBALibrary(),
        gf_library=MockGFLibrary(),
        format_func=format_metric,
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

    def test_bin_count(self):
        """n_bins=5 with 100 samples returns at most 5 non-empty bins."""
        n = 100
        gt = list(range(n))
        est = list(range(n))
        scores = [float(i) / n for i in range(1, n + 1)]
        ece, bin_data = _adaptive_ece(scores, gt, est, n_bins=5)
        self.assertLessEqual(len(bin_data), 5)
        self.assertGreater(len(bin_data), 0)

    def test_bin_structure(self):
        """Each bin dict has the expected keys."""
        n = 50
        gt = list(range(n))
        est = list(range(n))
        scores = [0.8] * n
        ece, bin_data = _adaptive_ece(scores, gt, est, n_bins=3)
        expected_keys = {"avg_confidence", "avg_accuracy", "count", "conf_min", "conf_max"}
        for b in bin_data:
            self.assertEqual(set(b.keys()), expected_keys)


class ComputeCalibrationTest(unittest.TestCase):
    """Tests for compute_calibration."""

    def test_returns_expected_artifacts(self):
        """Result contains ece scalar, per_bin_details df, reliability_diagram
        figure, and per_category_ece df."""
        ctx = _make_ctx(
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

    def test_ece_is_float(self):
        """The ece scalar is a float between 0 and 1."""
        ctx = _make_ctx(
            gt_indices=[0, 0, 1, 1],
            est_indices=[0, 1, 1, 0],
            classes=["A1::", "B1::"],
            scores=[0.9, 0.6, 0.8, 0.55],
        )
        result = compute_calibration(ctx)

        ece_scalar = next(s for s in result.scalars if s.name == "ece")
        self.assertIsInstance(ece_scalar.value, float)
        self.assertGreaterEqual(ece_scalar.value, 0.0)
        self.assertLessEqual(ece_scalar.value, 1.0)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_per_category_ece_respects_min_samples(self):
        """With fewer than 30 samples per category, per_category_ece is empty."""
        # Only 4 samples total, well below the 30-sample minimum.
        ctx = _make_ctx(
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

        ctx = _make_ctx(
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

    def test_per_bin_details_columns(self):
        """per_bin_details DataFrame has the expected columns."""
        ctx = _make_ctx(
            gt_indices=[0, 1, 0, 1],
            est_indices=[0, 1, 0, 1],
            classes=["A1::", "B1::"],
            scores=[0.9, 0.8, 0.7, 0.6],
        )
        result = compute_calibration(ctx)

        per_bin_df_result = next(
            df for df in result.dataframes if df.artifact_path == "calibration/per_bin_details"
        )
        expected_columns = {
            "bin",
            "conf_min",
            "conf_max",
            "avg_confidence",
            "avg_accuracy",
            "gap",
            "count",
        }
        self.assertEqual(set(per_bin_df_result.df.columns), expected_columns)

        for fig_result in result.figures:
            plt.close(fig_result.fig)


if __name__ == "__main__":
    unittest.main()
