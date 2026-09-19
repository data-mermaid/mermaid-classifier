"""Tests for mermaid_classifier.pyspacer.metrics.classification."""

import unittest

import matplotlib.pyplot as plt

from mermaid_classifier.pyspacer.metrics._context import (
    MetricsContext,
    MetricsContextError,
)
from mermaid_classifier.pyspacer.metrics._results import MetricGroupResult
from mermaid_classifier.pyspacer.metrics.classification import (
    compute_balanced_accuracy_mcc,
    compute_confusion_matrices,
    compute_precision_recall_f1,
)
from pyspacer.metrics_test_helpers import (
    MockBALibrary,
    MockGFLibrary,
    format_metric,
    make_val_results,
)


def _make_ctx(gt_indices, est_indices, classes):
    """Build a MetricsContext from simple index lists."""
    val_results = make_val_results(gt_indices, est_indices, classes)
    return MetricsContext(
        val_results=val_results,
        ba_library=MockBALibrary(),
        gf_library=MockGFLibrary(),
        format_func=format_metric,
    )


class ComputeConfusionMatricesTest(unittest.TestCase):
    """Tests for compute_confusion_matrices."""

    def test_returns_metric_group_result(self):
        ctx = _make_ctx(
            gt_indices=[0, 1, 0, 1],
            est_indices=[0, 1, 1, 0],
            classes=["A1::", "B1::"],
        )
        result = compute_confusion_matrices(ctx)

        self.assertIsInstance(result, MetricGroupResult)
        # Two confusion matrices: frequencies and percents
        self.assertEqual(len(result.dataframes), 2)
        self.assertEqual(len(result.figures), 2)
        self.assertEqual(result.dataframes[0].artifact_path, "confusion_matrix/frequencies")
        self.assertEqual(result.dataframes[1].artifact_path, "confusion_matrix/percents")

        # Clean up figures
        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_normalized_diagonal_values(self):
        """Perfect predictions should have 100 on the diagonal."""
        ctx = _make_ctx(
            gt_indices=[0, 0, 1, 1],
            est_indices=[0, 0, 1, 1],
            classes=["A1::", "B1::"],
        )
        result = compute_confusion_matrices(ctx)

        # Percents matrix is the second dataframe
        df = result.dataframes[1].df
        num_classes = 2
        for i in range(num_classes):
            # First column is the label column '-', data starts at col 1
            self.assertEqual(df.iloc[i, i + 1], 100)

        for fig_result in result.figures:
            plt.close(fig_result.fig)


class ComputePrecisionRecallF1Test(unittest.TestCase):
    """Tests for compute_precision_recall_f1."""

    def test_returns_metric_group_result(self):
        ctx = _make_ctx(
            gt_indices=[0, 0, 1, 1],
            est_indices=[0, 0, 1, 1],
            classes=["A1::", "B1::"],
        )
        result = compute_precision_recall_f1(ctx)

        self.assertIsInstance(result, MetricGroupResult)
        # Scalars: precision_macro, recall_macro, f1_macro
        self.assertEqual(len(result.scalars), 3)
        scalar_names = {s.name for s in result.scalars}
        self.assertEqual(scalar_names, {"precision_macro", "recall_macro", "f1_macro"})

        # Per-label DataFrame
        self.assertEqual(len(result.dataframes), 1)
        self.assertEqual(result.dataframes[0].artifact_path, "metrics_per_label")
        df = result.dataframes[0].df
        self.assertEqual(len(df), 2)  # Two classes

        # Overall dict
        self.assertEqual(len(result.dicts), 1)
        self.assertEqual(result.dicts[0].artifact_path, "metrics_overall.yaml")
        self.assertIn("precision_macro", result.dicts[0].data)

    def test_perfect_predictions(self):
        ctx = _make_ctx(
            gt_indices=[0, 0, 1, 1],
            est_indices=[0, 0, 1, 1],
            classes=["A1::", "B1::"],
        )
        result = compute_precision_recall_f1(ctx)

        scalars = {s.name: s.value for s in result.scalars}
        self.assertEqual(scalars["precision_macro"], 1.0)
        self.assertEqual(scalars["recall_macro"], 1.0)
        self.assertEqual(scalars["f1_macro"], 1.0)

        # Per-label should all be 1.0
        df = result.dataframes[0].df
        for _, row in df.iterrows():
            self.assertEqual(row["precision"], 1.0)
            self.assertEqual(row["recall"], 1.0)
            self.assertEqual(row["f1_score"], 1.0)

    def test_all_wrong_predictions(self):
        """When all predictions are wrong, macro F1 should be 0.0,
        not raise ZeroDivisionError."""
        ctx = _make_ctx(
            gt_indices=[0, 0, 1, 1],
            est_indices=[1, 1, 0, 0],
            classes=["A1::", "B1::"],
        )
        result = compute_precision_recall_f1(ctx)

        scalars = {s.name: s.value for s in result.scalars}
        self.assertEqual(scalars["precision_macro"], 0.0)
        self.assertEqual(scalars["recall_macro"], 0.0)
        self.assertEqual(scalars["f1_macro"], 0.0)

        # Per-label should all be 0.0
        df = result.dataframes[0].df
        for _, row in df.iterrows():
            self.assertEqual(row["precision"], 0.0)
            self.assertEqual(row["recall"], 0.0)
            self.assertEqual(row["f1_score"], 0.0)


class ComputeBalancedAccuracyMccTest(unittest.TestCase):
    """Tests for compute_balanced_accuracy_mcc."""

    def test_perfect_predictions(self):
        ctx = _make_ctx(
            gt_indices=[0, 0, 1, 1, 2, 2],
            est_indices=[0, 0, 1, 1, 2, 2],
            classes=["A1::", "B1::", "A2::"],
        )
        result = compute_balanced_accuracy_mcc(ctx)

        self.assertIsInstance(result, MetricGroupResult)
        self.assertEqual(len(result.scalars), 2)

        scalars_by_name = {s.name: s.value for s in result.scalars}
        self.assertEqual(scalars_by_name["balanced_accuracy"], 1.0)
        self.assertEqual(scalars_by_name["mcc"], 1.0)

    def test_all_wrong_binary(self):
        ctx = _make_ctx(
            gt_indices=[0, 0, 1, 1],
            est_indices=[1, 1, 0, 0],
            classes=["A1::", "B1::"],
        )
        result = compute_balanced_accuracy_mcc(ctx)

        scalars_by_name = {s.name: s.value for s in result.scalars}
        self.assertEqual(scalars_by_name["balanced_accuracy"], 0.0)
        self.assertEqual(scalars_by_name["mcc"], -1.0)

    def test_imbalanced_classes(self):
        # 8 samples of class 0, 2 samples of class 1.
        # All predicted as class 0. Balanced accuracy should be 0.5
        # (50% recall on class 0, 0% recall on class 1, averaged).
        ctx = _make_ctx(
            gt_indices=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            est_indices=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            classes=["A1::", "B1::"],
        )
        result = compute_balanced_accuracy_mcc(ctx)

        scalars_by_name = {s.name: s.value for s in result.scalars}
        self.assertEqual(scalars_by_name["balanced_accuracy"], 0.5)
        # MCC is 0 when predicting a single class
        self.assertEqual(scalars_by_name["mcc"], 0.0)


class MetricsContextValidationTest(unittest.TestCase):
    """Tests for MetricsContext.validate()."""

    def _make_ctx_with_val_results(self, val_results, ba_library=None):
        """Build a MetricsContext with a pre-built ValResults."""
        return MetricsContext(
            val_results=val_results,
            ba_library=ba_library or MockBALibrary(),
            gf_library=MockGFLibrary(),
            format_func=format_metric,
        )

    def test_mismatched_label_ids_raises(self):
        """Class IDs not resolvable by ba_library should fail validation."""
        val_results = make_val_results(gt_indices=[0], est_indices=[0], classes=["unknown::"])
        ctx = self._make_ctx_with_val_results(val_results)
        with self.assertRaises(MetricsContextError):
            ctx.validate()

    def test_valid_context_passes(self):
        """A well-formed context should pass validation without error."""
        ctx = _make_ctx(
            gt_indices=[0, 1],
            est_indices=[0, 1],
            classes=["A1::", "B1::"],
        )
        ctx.validate()  # Should not raise


if __name__ == "__main__":
    unittest.main()
