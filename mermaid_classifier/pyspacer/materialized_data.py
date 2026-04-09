"""
Pre-materialize training data into memory-mapped numpy files.

After feature vectors are downloaded to a local cache, this module extracts
matched (feature, label) pairs into contiguous memmap files.  Training then
reads from these mmaps directly -- no per-image ImageFeatures/PointFeatures
overhead, no GIL contention from Python object creation.
"""

import shutil
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from spacer.exceptions import RowColumnMismatchError

logger = getLogger(__name__)


def _sentinel_path(output_dir, split_name):
    return Path(output_dir) / f'{split_name}.done'


def materialize_split(labels, label_to_idx, cache_dir, output_dir,
                      split_name, feature_dim=4096, max_workers=4):
    """Extract matched (feature, label) pairs into contiguous memmap files.

    For a given ImageLabels split (e.g. labels.train):
    - Counts total annotations N
    - Pre-allocates two memmap files:
        {split_name}_features.dat -- shape (N, feature_dim), float32
        {split_name}_labels.dat   -- shape (N,), int64
    - Loads .npz files in parallel, writing directly to non-overlapping
      memmap offsets

    Returns (features_path, labels_path, N).
    """
    N = labels.label_count
    if N == 0:
        raise ValueError(f"Split '{split_name}' has no annotations")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / f'{split_name}_features.dat'
    labels_path = output_dir / f'{split_name}_labels.dat'
    sentinel = _sentinel_path(output_dir, split_name)

    # Only reuse if the sentinel exists (written after successful flush).
    # Size alone is insufficient: w+ pre-allocates full size before writing.
    if sentinel.exists():
        logger.info(
            f"Materialized {split_name} already complete"
            f" ({N} samples), reusing")
        return features_path, labels_path, N

    # Remove stale files from a prior interrupted run
    for p in (features_path, labels_path, sentinel):
        p.unlink(missing_ok=True)

    # Pre-allocate memmap files
    feat_mmap = np.memmap(
        features_path, dtype='float32', mode='w+',
        shape=(N, feature_dim))
    label_mmap = np.memmap(
        labels_path, dtype='int64', mode='w+', shape=(N,))

    # Pre-compute per-image write offsets (non-overlapping)
    keys = list(labels.keys())
    offsets = []
    offset = 0
    for key in keys:
        n_annos = len(labels._data[key])
        offsets.append(offset)
        offset += n_annos

    def _process_image(args):
        key, write_offset = args
        annotations = labels._data[key]
        npz_path = Path(cache_dir, key.key)
        npz = np.load(str(npz_path))

        rows = npz['rows']
        cols = npz['cols']
        feat = npz['feat']

        # Build (row, col) -> index lookup
        rc_to_idx = {}
        for i in range(len(rows)):
            rc_to_idx[(int(rows[i]), int(cols[i]))] = i

        for j, (row, col, label) in enumerate(annotations):
            feat_idx = rc_to_idx.get((row, col))
            if feat_idx is None:
                raise RowColumnMismatchError(
                    f"{key.key}: annotation ({row}, {col}) not found"
                    f" in feature file")
            dest = write_offset + j
            feat_mmap[dest] = feat[feat_idx]
            label_mmap[dest] = label_to_idx[label]

    work_items = list(zip(keys, offsets))

    try:
        if max_workers <= 1:
            for item in work_items:
                _process_image(item)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(_process_image, work_items))

        feat_mmap.flush()
        label_mmap.flush()
    except BaseException:
        # Clean up partial files so the next run doesn't reuse them
        del feat_mmap, label_mmap
        features_path.unlink(missing_ok=True)
        labels_path.unlink(missing_ok=True)
        raise

    # Sentinel written only after successful flush
    sentinel.write_text('')

    del feat_mmap, label_mmap

    logger.info(
        f"Materialized {split_name}: {N} samples, "
        f"features={features_path}, labels={labels_path}")

    return features_path, labels_path, N


class MemmapFeatureDataset(IterableDataset):
    """Drop-in replacement for StreamingFeatureDataset using memmap files.

    Yields (x_tensor, y_tensor) batches from pre-materialized data.
    Memory usage is O(batch_size) -- memmap pages are released after copy.
    """

    def __init__(self, features_path, labels_path, n_samples, feature_dim,
                 batch_size, random_seed=None):
        self.features_path = features_path
        self.labels_path = labels_path
        self.n_samples = n_samples
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.random_seed = random_seed

    def __iter__(self):
        feat_mmap = np.memmap(
            self.features_path, dtype='float32', mode='r',
            shape=(self.n_samples, self.feature_dim))
        label_mmap = np.memmap(
            self.labels_path, dtype='int64', mode='r',
            shape=(self.n_samples,))

        try:
            rng = np.random.RandomState(self.random_seed)
            indices = rng.permutation(self.n_samples)

            for start in range(0, self.n_samples, self.batch_size):
                end = min(start + self.batch_size, self.n_samples)
                batch_idx = indices[start:end]

                # .copy() releases memmap page references
                x = torch.from_numpy(feat_mmap[batch_idx].copy())
                y = torch.from_numpy(label_mmap[batch_idx].copy())

                yield x, y
        finally:
            del feat_mmap, label_mmap


def load_materialized_batches(features_path, labels_path, n_samples,
                              feature_dim, batch_size, idx_to_label):
    """Generator yielding (x_array, y_list) matching ImageLabels.load_data_in_batches().

    Used for ref accuracy and calibration where callers need numpy arrays
    and string labels, not tensor indices.
    """
    feat_mmap = np.memmap(
        features_path, dtype='float32', mode='r',
        shape=(n_samples, feature_dim))
    label_mmap = np.memmap(
        labels_path, dtype='int64', mode='r',
        shape=(n_samples,))

    try:
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            x_batch = feat_mmap[start:end].copy()
            y_batch = [
                idx_to_label[int(label_mmap[i])]
                for i in range(start, end)
            ]

            yield x_batch, y_batch
    finally:
        del feat_mmap, label_mmap


def cleanup_materialized(output_dir):
    """Remove the materialized data directory."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info(f"Cleaned up materialized data: {output_dir}")
