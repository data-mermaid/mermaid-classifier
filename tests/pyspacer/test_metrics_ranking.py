"""Tests for mermaid_classifier.pyspacer.metrics.ranking."""

import unittest

import matplotlib.pyplot as plt
import numpy as np
from spacer.data_classes import ValResults

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import MetricGroupResult
from mermaid_classifier.pyspacer.metrics.ranking import (
    _compute_topk_mrr,
    compute_ranking,
)
from metrics_test_helpers import MockBALibrary, MockGFLibrary, MockClf, format_metric


class ComputeTopKMRRTest(unittest.TestCase):
    """Tests for _compute_topk_mrr."""

    def setUp(self):
        self.classes = ['A1::', 'A2::', 'B1::']

    def test_perfect_predictions(self):
        gt_labels = ['A1::', 'A2::', 'B1::']
        proba = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])
        result = _compute_topk_mrr(proba, gt_labels, self.classes)

        self.assertAlmostEqual(result['topk'][1], 1.0)
        self.assertAlmostEqual(result['mrr'], 1.0)

    def test_worst_predictions(self):
        gt_labels = ['A1::', 'A2::', 'B1::']
        proba = np.array([
            [0.05, 0.45, 0.5],
            [0.5, 0.05, 0.45],
            [0.45, 0.5, 0.05],
        ])
        result = _compute_topk_mrr(proba, gt_labels, self.classes)

        self.assertAlmostEqual(result['topk'][1], 0.0)

    def test_topk_monotonically_increasing(self):
        gt_labels = ['A1::', 'A2::', 'B1::']
        proba = np.array([
            [0.6, 0.3, 0.1],
            [0.1, 0.5, 0.4],
            [0.2, 0.3, 0.5],
        ])
        result = _compute_topk_mrr(proba, gt_labels, self.classes)

        ks = sorted(result['topk'].keys())
        for i in range(len(ks) - 1):
            self.assertLessEqual(
                result['topk'][ks[i]], result['topk'][ks[i + 1]])


class ComputeRankingTest(unittest.TestCase):
    """Tests for compute_ranking."""

    def setUp(self):
        classes = ['A1::', 'A2::', 'B1::']
        n = 4
        gt_labels = ['A1::', 'A2::', 'B1::', 'A1::']
        proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
        ])
        val_results = ValResults(
            scores=[0.8] * n,
            gt=[0, 1, 2, 0],
            est=[0, 1, 2, 0],
            classes=classes,
        )
        self.ctx = MetricsContext(
            val_results=val_results,
            ba_library=MockBALibrary(),
            gf_library=MockGFLibrary(),
            format_func=format_metric,
            clf=MockClf(classes),
            val_proba=proba,
            val_gt_labels=gt_labels,
        )

    def tearDown(self):
        plt.close('all')

    def test_returns_expected_scalars(self):
        result = compute_ranking(self.ctx)

        self.assertIsInstance(result, MetricGroupResult)
        scalar_names = {s.name for s in result.scalars}
        expected = {
            'top_1_accuracy',
            'top_3_accuracy',
            'top_5_accuracy',
            'top_10_accuracy',
            'mrr',
            'hierarchical_top_5_mean_similarity',
        }
        self.assertEqual(scalar_names, expected)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_returns_expected_dataframes(self):
        result = compute_ranking(self.ctx)

        artifact_paths = {df.artifact_path for df in result.dataframes}
        self.assertIn('ranking/per_category_topk', artifact_paths)
        self.assertIn('ranking/hierarchical_topk', artifact_paths)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_hierarchical_topk_partial_credit(self):
        """Sibling predictions (A1 vs A2) yield hierarchical similarity > 0."""
        classes = ['A1::', 'A2::', 'B1::']
        # All samples are A1:: but predicted as A2:: (siblings)
        n = 4
        gt_labels = ['A1::'] * n
        proba = np.array([
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
        ])
        val_results = ValResults(
            scores=[0.8] * n,
            gt=[0] * n,
            est=[1] * n,
            classes=classes,
        )
        ctx = MetricsContext(
            val_results=val_results,
            ba_library=MockBALibrary(),
            gf_library=MockGFLibrary(),
            format_func=format_metric,
            clf=MockClf(classes),
            val_proba=proba,
            val_gt_labels=gt_labels,
        )
        result = compute_ranking(ctx)

        scalars = {s.name: s.value for s in result.scalars}
        # Top prediction is A2:: (sibling of A1::), so hierarchical
        # similarity at top-1 is > 0 (siblings share parent A).
        self.assertGreater(scalars['hierarchical_top_5_mean_similarity'], 0.0)

        for fig_result in result.figures:
            plt.close(fig_result.fig)


if __name__ == '__main__':
    unittest.main()
