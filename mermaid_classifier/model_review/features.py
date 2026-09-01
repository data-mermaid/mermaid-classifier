"""Load the pre-extracted PySpacer feature vectors for a set of review points.

Every reference model in a review project scores the same points, and the models
share one EfficientNet extractor, so the features are read from S3 once and each
model is handed the same matrix. `stack_features` returns the point keys alongside
the matrix: row alignment between models holds by construction rather than by a join.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mermaid_classifier.model_review.sample import ReviewPoint

# Identifies a point across the app: tasks, predictions, exports and synthesis all key on it.
PointKey = tuple[str, int, int]  # (image_id, row, col)


def default_feature_loader(bucket: str, key: str) -> Any:
    from spacer.data_classes import DataLocation, ImageFeatures

    return ImageFeatures.load(DataLocation("s3", bucket_name=bucket, key=key))


def stack_features(
    points: Sequence[ReviewPoint],
    load_features: Callable[[str, str], Any] = default_feature_loader,
) -> tuple[list[PointKey], NDArray[np.float32]]:
    """Point keys and the ``(n_points, dim)`` matrix whose rows align with them.

    Points sharing a feature file are grouped so each file is read once.
    """
    by_image: dict[tuple[str, str], list[ReviewPoint]] = defaultdict(list)
    for p in points:
        by_image[(p.bucket, p.feature_key)].append(p)

    keys: list[PointKey] = []
    rows: list[Any] = []
    for (bucket, key), img_points in by_image.items():
        feats = load_features(bucket, key)
        for p in img_points:
            keys.append((p.image_id, p.row, p.col))
            rows.append(feats.get_array((p.row, p.col)))
    if not rows:
        raise ValueError("no review points to load features for")
    return keys, np.vstack(rows).astype(np.float32)
