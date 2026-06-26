"""Tests for mermaid_classifier.pyspacer.metrics.classification."""

import unittest

import matplotlib.pyplot as plt
from spacer.data_classes import ValResults

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


class _MockBALibrary:
    """Minimal mock of BenthicAttributeLibrary for testing."""

    def bagf_id_to_name(self, bagf_id, gf_library):
        return f"name_{bagf_id}"


class _MockGFLibrary:
    """Minimal mock of GrowthFormLibrary for testing."""

    pass


def _make_val_results(gt_indices, est_indices, classes):
    """Helper to build a ValResults with dummy scores."""
    scores = [1.0] * len(gt_indices)
    return ValResults(
        scores=scores,
        gt=gt_indices,
        est=est_indices,
        classes=classes,
    )


def _format_metric(value):
    return round(float(value), 3)


def _make_ctx(gt_indices, est_indices, classes):
    """Build a MetricsContext from simple index lists."""
    val_results = _make_val_results(gt_indices, est_indices, classes)
    return MetricsContext(
        val_results=val_results,
        ba_library=_MockBALibrary(),
        gf_library=_MockGFLibrary(),
        format_func=_format_metric,
    )


class ComputeConfusionMatricesTest(unittest.TestCase):
    """Tests for compute_confusion_matrices."""

    def test_returns_metric_group_result(self):
        ctx = _make_ctx(
            gt_indices=[0, 1, 0, 1],
            est_indices=[0, 1, 1, 0],
            classes=["a::", "b::"],
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
            classes=["a::", "b::"],
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

    def test_dataframe_shape(self):
        """DataFrame should have N+1 columns and N rows."""
        classes = ["a::", "b::", "c::"]
        ctx = _make_ctx(
            gt_indices=[0, 0, 1, 1, 2, 2],
            est_indices=[0, 1, 1, 1, 2, 0],
            classes=classes,
        )
        result = compute_confusion_matrices(ctx)

        for df_result in result.dataframes:
            df = df_result.df
            # N+1 columns (label column + N classes)
            self.assertEqual(len(df.columns), len(classes) + 1)
            # N rows
            self.assertEqual(len(df), len(classes))

        for fig_result in result.figures:
            plt.close(fig_result.fig)


class ComputePrecisionRecallF1Test(unittest.TestCase):
    """Tests for compute_precision_recall_f1."""

    def test_returns_metric_group_result(self):
        ctx = _make_ctx(
            gt_indices=[0, 0, 1, 1],
            est_indices=[0, 0, 1, 1],
            classes=["a::", "b::"],
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
            classes=["a::", "b::"],
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
            classes=["a::", "b::"],
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

    def test_per_label_has_bagf_fields(self):
        ctx = _make_ctx(
            gt_indices=[0],
            est_indices=[0],
            classes=["a::"],
        )
        result = compute_precision_recall_f1(ctx)

        df = result.dataframes[0].df
        self.assertIn("bagf_name", df.columns)
        self.assertIn("bagf_id", df.columns)
        self.assertEqual(df.iloc[0]["bagf_id"], "a::")


class ComputeBalancedAccuracyMccTest(unittest.TestCase):
    """Tests for compute_balanced_accuracy_mcc."""

    def test_perfect_predictions(self):
        ctx = _make_ctx(
            gt_indices=[0, 0, 1, 1, 2, 2],
            est_indices=[0, 0, 1, 1, 2, 2],
            classes=["a::", "b::", "c::"],
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
            classes=["a::", "b::"],
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
            classes=["a::", "b::"],
        )
        result = compute_balanced_accuracy_mcc(ctx)

        scalars_by_name = {s.name: s.value for s in result.scalars}
        self.assertEqual(scalars_by_name["balanced_accuracy"], 0.5)
        # MCC is 0 when predicting a single class
        self.assertEqual(scalars_by_name["mcc"], 0.0)

    def test_no_figures_or_dataframes(self):
        ctx = _make_ctx(
            gt_indices=[0, 1],
            est_indices=[0, 1],
            classes=["a::", "b::"],
        )
        result = compute_balanced_accuracy_mcc(ctx)

        self.assertEqual(len(result.figures), 0)
        self.assertEqual(len(result.dataframes), 0)
        self.assertEqual(len(result.dicts), 0)


class MetricsContextValidationTest(unittest.TestCase):
    """Tests for MetricsContext.validate()."""

    def _make_ctx_with_val_results(self, val_results, ba_library=None):
        """Build a MetricsContext with a pre-built ValResults."""
        return MetricsContext(
            val_results=val_results,
            ba_library=ba_library or _MockBALibrary(),
            gf_library=_MockGFLibrary(),
            format_func=_format_metric,
        )

    def test_empty_gt_raises(self):
        """Empty ground truth should fail validation."""
        # Build a valid ValResults, then clear gt/est to simulate
        # an edge case (ValResults.__init__ validates non-empty).
        val_results = _make_val_results(gt_indices=[0], est_indices=[0], classes=["a::"])
        val_results.gt = []
        val_results.est = []
        ctx = self._make_ctx_with_val_results(val_results)
        with self.assertRaises(MetricsContextError):
            ctx.validate()

    def test_mismatched_label_ids_raises(self):
        """Class IDs not resolvable by ba_library should fail validation."""

        class _BrokenBALibrary:
            def bagf_id_to_name(self, bagf_id, gf_library):
                raise KeyError(f"Unknown ID: {bagf_id}")

        val_results = _make_val_results(gt_indices=[0], est_indices=[0], classes=["unknown::"])
        ctx = self._make_ctx_with_val_results(val_results, ba_library=_BrokenBALibrary())
        with self.assertRaises(MetricsContextError):
            ctx.validate()

    def test_valid_context_passes(self):
        """A well-formed context should pass validation without error."""
        ctx = _make_ctx(
            gt_indices=[0, 1],
            est_indices=[0, 1],
            classes=["a::", "b::"],
        )
        ctx.validate()  # Should not raise

    def test_out_of_range_class_index_raises(self):
        """Class indices beyond len(classes) should fail validation."""
        # Build valid ValResults, then inject an out-of-range index.
        val_results = _make_val_results(gt_indices=[0], est_indices=[0], classes=["a::"])
        val_results.gt = [0, 5]
        val_results.est = [0, 0]
        val_results.scores = [1.0, 1.0]
        ctx = self._make_ctx_with_val_results(val_results)
        with self.assertRaises(MetricsContextError):
            ctx.validate()


if __name__ == "__main__":
    unittest.main()
