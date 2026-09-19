"""Tests for mermaid_classifier.pyspacer.metrics.taxonomic."""

import unittest

import matplotlib.pyplot as plt

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import MetricGroupResult
from mermaid_classifier.pyspacer.metrics.taxonomic import compute_taxonomic
from pyspacer.metrics_test_helpers import (
    MockBALibrary,
    MockGFLibrary,
    format_metric,
    make_val_results,
)


def _make_ctx(gt_indices, est_indices, classes, gf_library=None):
    """Build a MetricsContext from simple index lists."""
    val_results = make_val_results(gt_indices, est_indices, classes)
    return MetricsContext(
        val_results=val_results,
        ba_library=MockBALibrary(),
        gf_library=gf_library or MockGFLibrary(),
        format_func=format_metric,
    )


class ErrorAttributionTest(unittest.TestCase):
    """Tests for cross_branch_error_rate and within_branch_error_rate scalars."""

    def test_perfect_predictions(self):
        """No errors -> both error rates are 0.0 and attribution df is empty."""
        ctx = _make_ctx(
            gt_indices=[0, 1, 2, 3],
            est_indices=[0, 1, 2, 3],
            classes=["A1::", "A2::", "B1::", "B2::"],
        )
        result = compute_taxonomic(ctx)

        scalars = {s.name: s.value for s in result.scalars}
        self.assertEqual(scalars["cross_branch_error_rate"], 0.0)
        self.assertEqual(scalars["within_branch_error_rate"], 0.0)

        attribution_dfs = [
            dr for dr in result.dataframes if dr.artifact_path == "taxonomic/error_attribution"
        ]
        self.assertEqual(len(attribution_dfs), 1)
        self.assertEqual(len(attribution_dfs[0].df), 0)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_cross_branch_errors(self):
        """A1->B1 predictions -> cross_branch_error_rate > 0."""
        ctx = _make_ctx(
            gt_indices=[0, 0, 0, 0],
            est_indices=[2, 2, 2, 2],
            classes=["A1::", "A2::", "B1::", "B2::"],
        )
        result = compute_taxonomic(ctx)

        scalars = {s.name: s.value for s in result.scalars}
        self.assertGreater(scalars["cross_branch_error_rate"], 0.0)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_within_branch_errors(self):
        """A1->A2 predictions -> within_branch_error_rate > 0."""
        ctx = _make_ctx(
            gt_indices=[0, 0, 0, 0],
            est_indices=[1, 1, 1, 1],
            classes=["A1::", "A2::", "B1::", "B2::"],
        )
        result = compute_taxonomic(ctx)

        scalars = {s.name: s.value for s in result.scalars}
        self.assertGreater(scalars["within_branch_error_rate"], 0.0)

        for fig_result in result.figures:
            plt.close(fig_result.fig)


class GrowthFormDifferentiationTest(unittest.TestCase):
    """Tests for GF-related scalars: gf_accuracy_gf_relevant and
    within_ba_gf_accuracy."""

    def test_no_growth_forms(self):
        """All classes have empty GF -> gf_accuracy_gf_relevant == 0.0."""
        ctx = _make_ctx(
            gt_indices=[0, 1, 2, 3],
            est_indices=[0, 1, 2, 3],
            classes=["A1::", "A2::", "B1::", "B2::"],
        )
        result = compute_taxonomic(ctx)

        scalars = {s.name: s.value for s in result.scalars}
        self.assertEqual(scalars["gf_accuracy_gf_relevant"], 0.0)

        for fig_result in result.figures:
            plt.close(fig_result.fig)


class ComputeTaxonomicIntegrationTest(unittest.TestCase):
    """Integration tests verifying compute_taxonomic returns all expected
    artifacts."""

    _CLASSES = ["A1::", "A2::", "B1::", "B2::"]

    _EXPECTED_SCALARS = {
        "cross_branch_error_rate",
        "within_branch_error_rate",
        "gf_accuracy_gf_relevant",
        "within_ba_gf_accuracy",
    }

    _EXPECTED_DATAFRAMES = {
        "taxonomic/error_attribution",
        "taxonomic/top_level_confusions",
        "taxonomic/gf_precision_recall_f1",
    }

    _EXPECTED_FIGURES = {
        "taxonomic/error_attribution.png",
        "taxonomic/top_level_confusion.png",
        "taxonomic/gf_confusion.png",
    }

    def test_returns_all_expected_artifacts(self):
        """compute_taxonomic must return all required scalars, dataframes,
        and figures."""
        # Use classes with growth forms and imperfect predictions so that
        # error attribution and GF differentiation figures are generated.
        classes = ["A1::gf1", "A2::gf2", "B1::", "B2::"]
        ctx = _make_ctx(
            gt_indices=[0, 0, 1, 1, 2, 2, 3, 3],
            est_indices=[0, 1, 1, 0, 2, 3, 3, 2],
            classes=classes,
            gf_library=MockGFLibrary(),
        )
        result = compute_taxonomic(ctx)

        self.assertIsInstance(result, MetricGroupResult)

        scalar_names = {s.name for s in result.scalars}
        for expected in self._EXPECTED_SCALARS:
            self.assertIn(expected, scalar_names, msg=f"Missing scalar: {expected}")

        df_paths = {dr.artifact_path for dr in result.dataframes}
        for expected in self._EXPECTED_DATAFRAMES:
            self.assertIn(expected, df_paths, msg=f"Missing dataframe: {expected}")

        fig_paths = {fr.artifact_path for fr in result.figures}
        for expected in self._EXPECTED_FIGURES:
            self.assertIn(expected, fig_paths, msg=f"Missing figure: {expected}")

        for fig_result in result.figures:
            plt.close(fig_result.fig)


if __name__ == "__main__":
    unittest.main()
