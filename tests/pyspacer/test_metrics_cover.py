"""Tests for mermaid_classifier.pyspacer.metrics.cover."""

import unittest

import matplotlib.pyplot as plt

from mermaid_classifier.pyspacer.metrics import MetricGroupResult
from mermaid_classifier.pyspacer.metrics.cover import compute_cover
from pyspacer.metrics_test_helpers import (
    make_ctx,
)


class _MockValLabels:
    """Mock of ImageLabels for val set with known per-image point counts."""

    def __init__(self, image_sizes):
        """image_sizes: list of ints, each is n_points for one image."""
        self._data = {}
        for i, n in enumerate(image_sizes):
            key = f"img_{i}"
            self._data[key] = [(0, 0, "dummy")] * n

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]


class _MockDataset:
    """Mock of TrainingDataset with only labels.val."""

    def __init__(self, val_labels):
        self.labels = type("obj", (object,), {"val": val_labels})()


class ComputeCoverTest(unittest.TestCase):
    """Tests for compute_cover."""

    def _make_ctx(self, gt_indices, est_indices, classes, image_sizes):
        return make_ctx(
            gt_indices,
            est_indices,
            classes,
            dataset=_MockDataset(_MockValLabels(image_sizes)),
        )

    def test_returns_expected_artifacts(self):
        """Result contains the expected scalar names, DataFrame, and figure."""
        classes = ["A1::", "A::"]
        # Vary cover across images so the figure is generated.
        # Image 0: all class 0; Image 1: all class 1; Image 2: 50/50
        gt_indices = [0] * 4 + [1] * 4 + [0, 0, 1, 1]
        est_indices = [0] * 4 + [1] * 4 + [0, 0, 1, 1]

        ctx = self._make_ctx(gt_indices, est_indices, classes, [4, 4, 4])
        result = compute_cover(ctx)

        self.assertIsInstance(result, MetricGroupResult)

        expected_scalar_names = {
            "cover_mean_abs_bias_pct",
            "cover_mean_rmse_pct",
            "cover_mean_mae_pct",
            "cover_median_r_squared",
        }
        actual_scalar_names = {s.name for s in result.scalars}
        self.assertEqual(actual_scalar_names, expected_scalar_names)

        self.assertEqual(len(result.dataframes), 1)
        self.assertEqual(result.dataframes[0].artifact_path, "cover/per_class_cover_metrics")

        self.assertEqual(len(result.figures), 1)
        self.assertEqual(result.figures[0].artifact_path, "cover/per_class_bias.png")

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_cover_vectors_sum_to_one(self):
        """Mean true cover values across all classes should sum to ~100%."""
        classes = ["A1::", "A::"]
        gt_indices = [0, 0, 1, 1, 0, 0, 1, 1]
        est_indices = [0, 0, 1, 1, 0, 0, 1, 1]

        ctx = self._make_ctx(gt_indices, est_indices, classes, [4, 4])
        result = compute_cover(ctx)

        df = result.dataframes[0].df
        total = df["mean_true_cover_pct"].sum()
        self.assertAlmostEqual(total, 100.0, places=6)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_biased_predictions(self):
        """Systematic overprediction of class 0 produces positive bias for class 0."""
        classes = ["A1::", "A::"]
        # 5 images, 10 points each.
        # True: 5 class 0, 5 class 1 per image (50/50 cover).
        # Predicted: 7 class 0, 3 class 1 per image.
        gt_per_image = [0] * 5 + [1] * 5
        est_per_image = [0] * 7 + [1] * 3
        gt_indices = gt_per_image * 5
        est_indices = est_per_image * 5

        ctx = self._make_ctx(gt_indices, est_indices, classes, [10, 10, 10, 10, 10])
        result = compute_cover(ctx)

        df = result.dataframes[0].df
        # all_classes is sorted(['A1::', 'A::']) = ['A1::', 'A::']
        # class index 0 = 'A1::' is what's being over-predicted
        row_class0 = df[df["bagf_id"] == "A1::"].iloc[0]
        row_class1 = df[df["bagf_id"] == "A::"].iloc[0]

        # Over-predicted class should have positive bias
        self.assertGreater(row_class0["bias_pct"], 0.0)
        # Under-predicted class should have negative bias
        self.assertLess(row_class1["bias_pct"], 0.0)

        for fig_result in result.figures:
            plt.close(fig_result.fig)


if __name__ == "__main__":
    unittest.main()
