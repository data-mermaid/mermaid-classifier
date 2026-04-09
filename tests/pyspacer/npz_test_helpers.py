"""Shared test helpers for creating pyspacer .npz feature files."""

from pathlib import Path

import numpy as np


def make_npz(cache_dir, key, rows, cols, feat):
    """Create a .npz file in pyspacer format.

    Args:
        cache_dir: Root directory for the cache.
        key: Relative path within cache_dir (e.g. 'features/img_0.npz').
        rows: Row coordinates for each feature point.
        cols: Column coordinates for each feature point.
        feat: 2D array of feature vectors (n_points, feature_dim).
    """
    path = Path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends .npz automatically, so strip if present
    save_path = str(path)
    if save_path.endswith('.npz'):
        save_path = save_path[:-4]
    np.savez(
        save_path, meta='{}',
        rows=np.array(rows), cols=np.array(cols),
        feat=np.array(feat, dtype=np.float32))
