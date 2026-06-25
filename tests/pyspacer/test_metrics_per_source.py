"""Tests for mermaid_classifier.pyspacer.metrics.per_source."""

import unittest
from collections import OrderedDict

import matplotlib.pyplot as plt
from spacer.data_classes import DataLocation

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import MetricGroupResult
from mermaid_classifier.pyspacer.metrics.per_source import compute_per_source
from .metrics_test_helpers import (
    MockBALibrary,
    MockGFLibrary,
    format_metric,
    make_val_results,
)


def _make_loc(key: str) -> DataLocation:
    return DataLocation(storage_type='filesystem', key=key)


class _MockValLabels:
    """Mock val ImageLabels: maps DataLocation -> list of (row, col, label)."""

    def __init__(self, image_specs):
        """image_specs: list of (DataLocation, n_points)."""
        self._data = OrderedDict()
        for loc, n in image_specs:
            self._data[loc] = [(0, 0, 'dummy')] * n

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]


class _MockDataset:
    """Mock TrainingDataset exposing labels.val and feature_loc_to_source."""

    def __init__(self, image_specs, source_map):
        val_labels = _MockValLabels(image_specs)
        self.labels = type('obj', (object,), {'val': val_labels})()
        self.feature_loc_to_source = source_map


def _make_ctx(image_specs, source_map, gt_indices, est_indices, classes):
    val_results = make_val_results(gt_indices, est_indices, classes)
    dataset = _MockDataset(image_specs, source_map)
    return MetricsContext(
        val_results=val_results,
        ba_library=MockBALibrary(),
        gf_library=MockGFLibrary(),
        format_func=format_metric,
        dataset=dataset,
    )


class ComputePerSourceTest(unittest.TestCase):
    """Tests for compute_per_source."""

    def test_two_sources_split_correctly(self):
        """Two sources with different per-source accuracies map to two
        rows with distinct metrics."""
        # Image 1 (source A): 4 points, all correct.
        # Image 2 (source B): 4 points, half wrong (still within branch).
        loc_a = _make_loc('img_a')
        loc_b = _make_loc('img_b')
        image_specs = [(loc_a, 4), (loc_b, 4)]
        source_map = {
            loc_a: ('coralnet', '109'),
            loc_b: ('coralnet', '155'),
        }
        gt = [0, 0, 1, 1, 0, 0, 1, 1]
        est = [0, 0, 1, 1, 1, 1, 0, 0]  # source B all wrong
        classes = ['A1::', 'A2::']  # both within branch A

        ctx = _make_ctx(image_specs, source_map, gt, est, classes)
        result = compute_per_source(ctx)

        df = next(
            d.df for d in result.dataframes
            if d.artifact_path == 'per_source/metrics'
        )
        self.assertEqual(len(df), 2)

        row_a = df[df['source_key'] == 'coralnet:109'].iloc[0]
        row_b = df[df['source_key'] == 'coralnet:155'].iloc[0]

        self.assertEqual(row_a['num_val_images'], 1)
        self.assertEqual(row_a['num_val_annotations'], 4)
        self.assertAlmostEqual(row_a['accuracy'], 1.0)
        self.assertAlmostEqual(row_b['accuracy'], 0.0)
        # Both errors stay within branch A -> cross_branch_error_rate = 0
        self.assertAlmostEqual(row_b['cross_branch_error_rate'], 0.0)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_cross_branch_error_per_source(self):
        """A->B confusions on one source -> that source has positive
        cross_branch_error_rate."""
        loc_a = _make_loc('img_a')
        loc_b = _make_loc('img_b')
        image_specs = [(loc_a, 2), (loc_b, 2)]
        source_map = {
            loc_a: ('coralnet', '109'),
            loc_b: ('mermaid', 'all'),
        }
        # Source A: A1 -> A2 (within branch).
        # Source B: B1 -> A1 (cross branch).
        gt = [0, 0, 2, 2]
        est = [1, 1, 0, 0]
        classes = ['A1::', 'A2::', 'B1::']

        ctx = _make_ctx(image_specs, source_map, gt, est, classes)
        result = compute_per_source(ctx)

        df = next(
            d.df for d in result.dataframes
            if d.artifact_path == 'per_source/metrics'
        )
        row_a = df[df['source_key'] == 'coralnet:109'].iloc[0]
        row_b = df[df['source_key'] == 'mermaid:all'].iloc[0]

        self.assertAlmostEqual(row_a['cross_branch_error_rate'], 0.0)
        self.assertAlmostEqual(row_b['cross_branch_error_rate'], 1.0)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_returns_expected_artifacts(self):
        """compute_per_source returns the dataframe, scalars, and figure."""
        loc_a = _make_loc('img_a')
        loc_b = _make_loc('img_b')
        image_specs = [(loc_a, 4), (loc_b, 4)]
        source_map = {
            loc_a: ('coralnet', '109'),
            loc_b: ('coralnet', '155'),
        }
        gt = [0, 0, 1, 1, 0, 0, 1, 1]
        est = [0, 1, 1, 0, 0, 0, 1, 1]
        classes = ['A1::', 'A2::']

        ctx = _make_ctx(image_specs, source_map, gt, est, classes)
        result = compute_per_source(ctx)

        self.assertIsInstance(result, MetricGroupResult)

        df_paths = {d.artifact_path for d in result.dataframes}
        self.assertIn('per_source/metrics', df_paths)

        scalar_names = {s.name for s in result.scalars}
        self.assertEqual(
            scalar_names,
            {'per_source/n_sources',
             'per_source/min_accuracy',
             'per_source/max_accuracy'},
        )

        fig_paths = {f.artifact_path for f in result.figures}
        self.assertIn('per_source/accuracy_by_source.png', fig_paths)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_sorted_by_annotation_count_desc(self):
        """Source with more val annotations appears first."""
        loc_big = _make_loc('img_big')
        loc_small = _make_loc('img_small')
        image_specs = [(loc_small, 2), (loc_big, 6)]
        source_map = {
            loc_small: ('coralnet', '155'),
            loc_big: ('coralnet', '109'),
        }
        gt = [0, 1, 0, 0, 1, 1, 0, 1]
        est = [0, 1, 0, 0, 1, 1, 0, 1]
        classes = ['A1::', 'A2::']

        ctx = _make_ctx(image_specs, source_map, gt, est, classes)
        result = compute_per_source(ctx)

        df = next(
            d.df for d in result.dataframes
            if d.artifact_path == 'per_source/metrics'
        )
        self.assertEqual(df.iloc[0]['source_key'], 'coralnet:109')
        self.assertEqual(df.iloc[0]['num_val_annotations'], 6)
        self.assertEqual(df.iloc[1]['source_key'], 'coralnet:155')
        self.assertEqual(df.iloc[1]['num_val_annotations'], 2)

        for fig_result in result.figures:
            plt.close(fig_result.fig)

    def test_no_dataset_returns_empty(self):
        """compute_per_source with no dataset is a no-op."""
        val_results = make_val_results([0, 1], [0, 1], ['A1::', 'A2::'])
        ctx = MetricsContext(
            val_results=val_results,
            ba_library=MockBALibrary(),
            gf_library=MockGFLibrary(),
            format_func=format_metric,
            dataset=None,
        )
        result = compute_per_source(ctx)
        # Coordinator already gates on dataset, but defend in depth:
        # if called without one, return an empty result rather than crash.
        self.assertEqual(len(result.dataframes), 0)
        self.assertEqual(len(result.scalars), 0)
        self.assertEqual(len(result.figures), 0)

    def test_index_count_mismatch_raises(self):
        """If val annotation count diverges from sum of per-image n_points,
        the function fails loudly rather than producing wrong metrics."""
        loc_a = _make_loc('img_a')
        image_specs = [(loc_a, 4)]
        source_map = {loc_a: ('coralnet', '109')}
        # gt has 5 entries but image has only 4 points -> mismatch.
        gt = [0, 0, 1, 1, 0]
        est = [0, 0, 1, 1, 0]
        classes = ['A1::', 'A2::']

        ctx = _make_ctx(image_specs, source_map, gt, est, classes)
        with self.assertRaises(ValueError):
            compute_per_source(ctx)


if __name__ == '__main__':
    unittest.main()
