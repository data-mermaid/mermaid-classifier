"""
Tests for mermaid_classifier.pyspacer.materialized_data.

Verifies materialization roundtrip, MemmapFeatureDataset iteration,
shuffling, label reconstruction, and error handling.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from spacer.data_classes import DataLocation

from mermaid_classifier.pyspacer.materialized_data import (
    MemmapFeatureDataset,
    cleanup_materialized,
    load_materialized_batches,
    materialize_split,
)
from pyspacer.npz_test_helpers import make_npz


def _make_mock_split(cache_dir, n_images=5, points_per_image=10,
                     feature_dim=20, n_classes=3):
    """Build a mock ImageLabels split backed by real .npz files."""
    rng = np.random.RandomState(42)
    classes = [f'class_{i}' for i in range(n_classes)]
    label_to_idx = {label: idx for idx, label in enumerate(classes)}

    data = {}
    all_features = []
    all_labels = []

    for img_idx in range(n_images):
        key = f'features/img_{img_idx}.npz'
        loc = DataLocation('filesystem', key=key)

        rows = list(range(points_per_image))
        cols = [0] * points_per_image
        feat = rng.randn(points_per_image, feature_dim).astype(np.float32)

        make_npz(cache_dir, key, rows, cols, feat)

        annotations = []
        for p in range(points_per_image):
            label = rng.choice(classes)
            annotations.append((rows[p], cols[p], label))
            all_features.append(feat[p])
            all_labels.append(label)

        data[loc] = annotations

    mock_labels = mock.Mock()
    mock_labels._data = data
    mock_labels.keys = lambda: data.keys()
    mock_labels.label_count = len(all_labels)

    return mock_labels, label_to_idx, all_features, all_labels, classes


class MaterializeSplitTest(unittest.TestCase):
    """Tests for materialize_split."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.tmp_dir, 'cache')
        self.cache_dir.mkdir()
        self.output_dir = Path(self.tmp_dir, 'materialized')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_roundtrip(self):
        """Materialized features match original data."""
        mock_labels, label_to_idx, expected_feats, expected_labels, classes = \
            _make_mock_split(self.cache_dir)

        feat_path, label_path, n = materialize_split(
            mock_labels, label_to_idx, self.cache_dir, self.output_dir,
            'train', feature_dim=20)

        self.assertEqual(n, 50)
        self.assertTrue(feat_path.exists())
        self.assertTrue(label_path.exists())

        feat_mmap = np.memmap(feat_path, dtype='float32', mode='r',
                              shape=(n, 20))
        label_mmap = np.memmap(label_path, dtype='int64', mode='r',
                               shape=(n,))

        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        for i in range(n):
            np.testing.assert_array_almost_equal(
                feat_mmap[i], expected_feats[i])
            self.assertEqual(
                idx_to_label[int(label_mmap[i])], expected_labels[i])

    def test_reuse_existing(self):
        """If sentinel exists, files are reused without re-materialization."""
        mock_labels, label_to_idx, _, _, _ = \
            _make_mock_split(self.cache_dir)

        # First materialization
        materialize_split(
            mock_labels, label_to_idx, self.cache_dir, self.output_dir,
            'train', feature_dim=20)

        feat_path = self.output_dir / 'train_features.dat'
        sentinel = self.output_dir / 'train.done'
        self.assertTrue(sentinel.exists())
        mtime1 = feat_path.stat().st_mtime

        # Second materialization should reuse
        materialize_split(
            mock_labels, label_to_idx, self.cache_dir, self.output_dir,
            'train', feature_dim=20)

        mtime2 = feat_path.stat().st_mtime
        self.assertEqual(mtime1, mtime2)

    def test_no_reuse_without_sentinel(self):
        """Files without sentinel are not reused (handles interrupted writes)."""
        mock_labels, label_to_idx, _, _, _ = \
            _make_mock_split(self.cache_dir)

        # First materialization
        materialize_split(
            mock_labels, label_to_idx, self.cache_dir, self.output_dir,
            'train', feature_dim=20)

        feat_path = self.output_dir / 'train_features.dat'
        sentinel = self.output_dir / 'train.done'

        # Remove sentinel to simulate an interrupted prior run
        sentinel.unlink()
        mtime1 = feat_path.stat().st_mtime

        # Should re-materialize (not reuse)
        import time
        time.sleep(0.01)  # ensure mtime differs
        materialize_split(
            mock_labels, label_to_idx, self.cache_dir, self.output_dir,
            'train', feature_dim=20)

        mtime2 = feat_path.stat().st_mtime
        self.assertNotEqual(mtime1, mtime2)
        self.assertTrue(sentinel.exists())

    def test_empty_split_raises(self):
        """Empty split raises ValueError."""
        mock_labels = mock.Mock()
        mock_labels.label_count = 0
        with self.assertRaises(ValueError):
            materialize_split(
                mock_labels, {}, self.cache_dir, self.output_dir,
                'empty', feature_dim=20)

    def test_missing_rowcol_raises(self):
        """Annotation with missing (row, col) raises RowColumnMismatchError."""
        from spacer.exceptions import RowColumnMismatchError

        key = 'features/img_bad.npz'
        loc = DataLocation('filesystem', key=key)
        # Feature file has rows [0, 1], but annotation references row 99
        make_npz(self.cache_dir, key,
                  rows=[0, 1], cols=[0, 0],
                  feat=np.zeros((2, 20), dtype=np.float32))

        mock_labels = mock.Mock()
        mock_labels._data = {loc: [(99, 0, 'class_0')]}
        mock_labels.keys = lambda: [loc]
        mock_labels.label_count = 1

        with self.assertRaises(RowColumnMismatchError):
            materialize_split(
                mock_labels, {'class_0': 0},
                self.cache_dir, self.output_dir,
                'bad', feature_dim=20)

    def test_parallel_workers(self):
        """Parallel materialization produces same result as sequential."""
        mock_labels, label_to_idx, _, _, _ = \
            _make_mock_split(self.cache_dir, n_images=10)

        out_seq = Path(self.tmp_dir, 'seq')
        out_par = Path(self.tmp_dir, 'par')

        materialize_split(
            mock_labels, label_to_idx, self.cache_dir, out_seq,
            'train', feature_dim=20, max_workers=1)
        materialize_split(
            mock_labels, label_to_idx, self.cache_dir, out_par,
            'train', feature_dim=20, max_workers=4)

        n = mock_labels.label_count
        feat_seq = np.memmap(
            out_seq / 'train_features.dat', dtype='float32',
            mode='r', shape=(n, 20))
        feat_par = np.memmap(
            out_par / 'train_features.dat', dtype='float32',
            mode='r', shape=(n, 20))
        np.testing.assert_array_equal(feat_seq, feat_par)


class MemmapFeatureDatasetTest(unittest.TestCase):
    """Tests for MemmapFeatureDataset."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.tmp_dir, 'cache')
        self.cache_dir.mkdir()
        self.output_dir = Path(self.tmp_dir, 'materialized')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _materialize(self, n_images=5, points_per_image=10, feature_dim=20):
        mock_labels, label_to_idx, _, _, classes = \
            _make_mock_split(self.cache_dir, n_images=n_images,
                             points_per_image=points_per_image,
                             feature_dim=feature_dim)
        feat_path, label_path, n = materialize_split(
            mock_labels, label_to_idx, self.cache_dir, self.output_dir,
            'train', feature_dim=feature_dim)
        return feat_path, label_path, n, feature_dim, label_to_idx, classes

    def test_iteration_yields_all_samples(self):
        """All samples are yielded exactly once per epoch."""
        feat_path, label_path, n, dim, _, _ = self._materialize()
        dataset = MemmapFeatureDataset(
            feat_path, label_path, n, dim, batch_size=15)

        total = sum(x.shape[0] for x, _ in dataset)
        self.assertEqual(total, n)

    def test_batch_shapes(self):
        """Batch tensors have correct shapes and dtypes."""
        feat_path, label_path, n, dim, _, _ = self._materialize()
        dataset = MemmapFeatureDataset(
            feat_path, label_path, n, dim, batch_size=20)

        batches = list(dataset)
        # 50 samples / 20 batch = 2 full + 1 remainder of 10
        self.assertEqual(len(batches), 3)

        for x, y in batches[:2]:
            self.assertEqual(x.shape, (20, dim))
            self.assertEqual(y.shape, (20,))
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(y.dtype, torch.int64)

        x_last, y_last = batches[-1]
        self.assertEqual(x_last.shape, (10, dim))

    def test_shuffling(self):
        """Different seeds produce different orderings."""
        feat_path, label_path, n, dim, _, _ = self._materialize()

        ds1 = MemmapFeatureDataset(
            feat_path, label_path, n, dim, batch_size=n,
            random_seed=0)
        ds2 = MemmapFeatureDataset(
            feat_path, label_path, n, dim, batch_size=n,
            random_seed=1)

        y1 = next(iter(ds1))[1].numpy()
        y2 = next(iter(ds2))[1].numpy()

        # Very unlikely to be identical with different seeds
        self.assertFalse(np.array_equal(y1, y2))

    def test_deterministic_with_same_seed(self):
        """Same seed produces identical ordering."""
        feat_path, label_path, n, dim, _, _ = self._materialize()

        ds1 = MemmapFeatureDataset(
            feat_path, label_path, n, dim, batch_size=n,
            random_seed=42)
        ds2 = MemmapFeatureDataset(
            feat_path, label_path, n, dim, batch_size=n,
            random_seed=42)

        y1 = next(iter(ds1))[1].numpy()
        y2 = next(iter(ds2))[1].numpy()
        np.testing.assert_array_equal(y1, y2)


class LoadMaterializedBatchesTest(unittest.TestCase):
    """Tests for load_materialized_batches."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.tmp_dir, 'cache')
        self.cache_dir.mkdir()
        self.output_dir = Path(self.tmp_dir, 'materialized')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_label_reconstruction(self):
        """Labels are correctly reconstructed from indices."""
        mock_labels, label_to_idx, _, expected_labels, classes = \
            _make_mock_split(self.cache_dir)

        feat_path, label_path, n = materialize_split(
            mock_labels, label_to_idx, self.cache_dir, self.output_dir,
            'train', feature_dim=20)

        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        all_labels = []
        for _, y_batch in load_materialized_batches(
                feat_path, label_path, n, 20, 15, idx_to_label):
            all_labels.extend(y_batch)

        self.assertEqual(all_labels, expected_labels)

    def test_batch_sizes(self):
        """Batches have correct sizes including remainder."""
        mock_labels, label_to_idx, _, _, _ = \
            _make_mock_split(self.cache_dir)

        feat_path, label_path, n = materialize_split(
            mock_labels, label_to_idx, self.cache_dir, self.output_dir,
            'train', feature_dim=20)

        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        batches = list(load_materialized_batches(
            feat_path, label_path, n, 20, 15, idx_to_label))

        # 50 / 15 = 3 full + 1 remainder of 5
        self.assertEqual(len(batches), 4)
        self.assertEqual(len(batches[0][0]), 15)
        self.assertEqual(len(batches[-1][0]), 5)


class CleanupMaterializedTest(unittest.TestCase):
    """Tests for cleanup_materialized."""

    def test_cleanup_removes_directory(self):
        tmp = tempfile.mkdtemp()
        Path(tmp, 'test.dat').touch()
        cleanup_materialized(tmp)
        self.assertFalse(Path(tmp).exists())

    def test_cleanup_nonexistent_noop(self):
        """Cleaning up a non-existent dir does not raise."""
        cleanup_materialized('/tmp/does_not_exist_abc123')
