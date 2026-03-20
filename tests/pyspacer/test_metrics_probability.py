"""Tests for mermaid_classifier.pyspacer.metrics.probability."""

import math
import unittest

import matplotlib.pyplot as plt
import numpy as np
from spacer.data_classes import ValResults

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import MetricGroupResult
from mermaid_classifier.pyspacer.metrics.probability import compute_probability
from metrics_test_helpers import MockBALibrary, MockGFLibrary, MockClf, format_metric


def _make_ctx(classes, gt_labels, proba):
    """Build a MetricsContext from classes, ground-truth labels, and proba matrix."""
    n = len(gt_labels)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    gt_indices = [class_to_idx[g] for g in gt_labels]
    est_indices = gt_indices  # exact predictions (irrelevant to probability metrics)

    val_results = ValResults(
        scores=[0.9] * n,
        gt=gt_indices,
        est=est_indices,
        classes=classes,
    )
    return MetricsContext(
        val_results=val_results,
        ba_library=MockBALibrary(),
        gf_library=MockGFLibrary(),
        format_func=format_metric,
        clf=MockClf(classes),
        val_proba=proba,
        val_gt_labels=gt_labels,
    )


class ComputeProbabilityTest(unittest.TestCase):
    """Tests for compute_probability."""

    def test_perfect_probability_matrix(self):
        """Perfect probability assignments yield log_loss near zero."""
        classes = ['A1::', 'B1::']
        gt_labels = ['A1::', 'A1::', 'B1::', 'B1::']
        proba = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        ctx = _make_ctx(classes, gt_labels, proba)
        result = compute_probability(ctx)

        scalars = {s.name: s.value for s in result.scalars}
        self.assertAlmostEqual(scalars['log_loss'], 0.0, places=5)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_uniform_probability_matrix(self):
        """Uniform probabilities yield log_loss = log(n_classes)."""
        classes = ['A1::', 'B1::']
        gt_labels = ['A1::', 'A1::', 'B1::', 'B1::']
        proba = np.full((4, 2), 0.5)
        ctx = _make_ctx(classes, gt_labels, proba)
        result = compute_probability(ctx)

        scalars = {s.name: s.value for s in result.scalars}
        expected = math.log(2)
        self.assertAlmostEqual(scalars['log_loss'], expected, places=5)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_returns_expected_artifacts(self):
        """Result contains the log_loss scalar and per_category_log_loss DataFrame."""
        classes = ['A1::', 'B1::']
        gt_labels = ['A1::', 'A1::', 'B1::', 'B1::']
        proba = np.full((4, 2), 0.5)
        ctx = _make_ctx(classes, gt_labels, proba)
        result = compute_probability(ctx)

        self.assertIsInstance(result, MetricGroupResult)

        scalar_names = {s.name for s in result.scalars}
        self.assertIn('log_loss', scalar_names)

        df_paths = {d.artifact_path for d in result.dataframes}
        self.assertIn('probability/per_category_log_loss', df_paths)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_per_category_requires_min_samples(self):
        """Fewer than 30 samples per category yields an empty per_category DataFrame."""
        classes = ['A1::', 'B1::']
        # 4 samples total — well below the 30-sample minimum per category.
        gt_labels = ['A1::', 'A1::', 'B1::', 'B1::']
        proba = np.full((4, 2), 0.5)
        ctx = _make_ctx(classes, gt_labels, proba)
        result = compute_probability(ctx)

        df_result = next(
            d for d in result.dataframes
            if d.artifact_path == 'probability/per_category_log_loss'
        )
        self.assertEqual(len(df_result.df), 0)
        self.assertListEqual(
            list(df_result.df.columns), ['category', 'log_loss', 'n_samples'])

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_per_category_with_enough_samples(self):
        """35+ samples per category yields a populated per_category DataFrame."""
        classes = ['A1::', 'B1::']
        n_per_class = 35
        gt_labels = (['A1::'] * n_per_class) + (['B1::'] * n_per_class)
        n = len(gt_labels)
        # Uniform probabilities — simple and well-defined log loss.
        proba = np.full((n, 2), 0.5)
        ctx = _make_ctx(classes, gt_labels, proba)
        result = compute_probability(ctx)

        df_result = next(
            d for d in result.dataframes
            if d.artifact_path == 'probability/per_category_log_loss'
        )
        df = df_result.df
        # Both top-level categories (TopA for A1, TopB for B1) should appear.
        self.assertGreater(len(df), 0)
        self.assertIn('category', df.columns)
        self.assertIn('log_loss', df.columns)
        self.assertIn('n_samples', df.columns)
        self.assertEqual(df['n_samples'].sum(), n)

        for fig_result in result.figures:
            plt.close(fig_result.fig)


if __name__ == '__main__':
    unittest.main()
