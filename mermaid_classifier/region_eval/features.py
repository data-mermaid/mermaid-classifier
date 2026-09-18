"""The probe set's feature cache: one aligned matrix, downloaded once.

Every probe point already has a 1280-d EfficientNet vector on S3, extracted
with the extractor every model version shares. Caching them means a new model
is scored on the probe without touching S3 again, and means two models are
scored on byte-identical inputs.

Matching is by `(row, col)` against the arrays the feature file stores, never
by position: the annotation order in the parquet and the point order in the
archive are independently produced, so a positional match would quietly pair a
point's ground truth with its neighbour's vector. A point whose `(row, col)`
the file does not carry is dropped and counted, and so is every point of an
image whose file is missing altogether. A file that downloads but does not
parse raises instead, because a truncated archive silently shrinking the probe
is the one failure the counts could not tell you about.

Features are cast to float32. The extractor writes float64, which doubles the
cache for precision no downstream head uses.

Downloads run in parallel into `download_dir` through
`download_features_parallel`, the training pipeline's own S3 downloader.
`download_dir` is a scratch directory that does not outlive the run.
"""

import dataclasses
import io
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mermaid_classifier.common.s3_utils import download_features_parallel
from mermaid_classifier.region_eval.probe_set import probe_content_hash

logger = logging.getLogger(__name__)

FEATURE_DIM = 1280
DEFAULT_WORKERS = 32
DEFAULT_FEATURE_BUCKET = "coral-reef-training"
DEFAULT_FEATURE_PREFIX = "mermaid/"
DEFAULT_FEATURE_SUFFIX = "_featurevector"

METADATA_COLUMNS = ("image_id", "point_id", "row", "col", "gt_label", "region_id", "held_out")

# A callable taking an image id and returning that image's feature-file bytes.
# Anything it raises means "no file for this image"; the caller counts it.
FeatureLoader = Callable[[str], bytes]


@dataclasses.dataclass(frozen=True)
class FeatureCache:
    """Probe features and the metadata they line up with, row for row.

    `features[i]` is the vector for the point described by every other array
    at index `i`. The three missing counts partition what was requested but is
    not here.
    """

    features: NDArray[np.float32]
    image_ids: tuple[str, ...]
    point_ids: tuple[str, ...]
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    gt_labels: tuple[str, ...]
    region_ids: tuple[str, ...]
    held_out: NDArray[np.bool_]
    n_points_requested: int
    n_points_missing_row_col: int
    n_points_missing_image: int
    missing_image_ids: tuple[str, ...]


def read_feature_file(
    payload: bytes,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
    """Parse one `.featurevector` npz into its rows, cols and feature matrix.

    The three arrays must describe the same points; a disagreement means the
    archive is truncated or was written by something else.
    """
    archive = np.load(io.BytesIO(payload), allow_pickle=False)
    rows = np.asarray(archive["rows"]).astype(np.int64)
    cols = np.asarray(archive["cols"]).astype(np.int64)
    features = np.asarray(archive["feat"]).astype(np.float32)
    if features.ndim != 2 or not len(rows) == len(cols) == features.shape[0]:
        raise ValueError(
            f"feature file disagrees with itself: {len(rows)} rows,"
            f" {len(cols)} cols, feat shape {features.shape}"
        )
    return rows, cols, features


def build_feature_cache(
    probe_rows: pd.DataFrame,
    download_dir: Path,
    *,
    bucket: str = DEFAULT_FEATURE_BUCKET,
    prefix: str = DEFAULT_FEATURE_PREFIX,
    suffix: str = DEFAULT_FEATURE_SUFFIX,
    workers: int = DEFAULT_WORKERS,
    feature_dim: int = FEATURE_DIM,
) -> FeatureCache:
    """Fetch every probe point's feature vector into one aligned matrix.

    Output rows follow `probe_rows` order, minus the points no vector was
    found for. Every wanted image's file is downloaded into `download_dir`
    through `download_features_parallel` before anything is read, so a test
    that pre-populates the directory never reaches S3.
    """
    missing_columns = [column for column in METADATA_COLUMNS if column not in probe_rows.columns]
    if missing_columns:
        raise ValueError(f"probe rows are missing column(s): {', '.join(missing_columns)}")

    image_ids = [str(value) for value in probe_rows["image_id"]]
    point_ids = [str(value) for value in probe_rows["point_id"]]
    rows = np.asarray(probe_rows["row"], dtype=np.int64)
    cols = np.asarray(probe_rows["col"], dtype=np.int64)

    wanted: dict[str, list[int]] = {}
    for position, image_id in enumerate(image_ids):
        wanted.setdefault(image_id, []).append(position)

    local_paths = {image_id: download_dir / f"{image_id}{suffix}" for image_id in wanted}
    s3_keys = {
        (bucket, f"{prefix}{image_id}{suffix}"): str(local_paths[image_id]) for image_id in wanted
    }
    download_features_parallel(s3_keys, max_workers=workers)

    def load(image_id: str) -> bytes:
        return local_paths[image_id].read_bytes()

    n_points = len(image_ids)
    features = np.zeros((n_points, feature_dim), dtype=np.float32)
    filled = np.zeros(n_points, dtype=bool)
    missing_images: set[str] = set()
    for image_id, positions in wanted.items():
        matched = _image_vectors(
            load, image_id, positions, rows=rows, cols=cols, feature_dim=feature_dim
        )
        if matched is None:
            missing_images.add(image_id)
            continue
        for position, vector in matched:
            features[position] = vector
            filled[position] = True

    absent_image = np.array([image_id in missing_images for image_id in image_ids], dtype=bool)
    n_missing_image = int((~filled & absent_image).sum())
    n_missing_row_col = int((~filled & ~absent_image).sum())
    if n_missing_row_col:
        logger.warning(
            "%d probe point(s) had no matching (row, col) in their feature file; dropping them",
            n_missing_row_col,
        )
    logger.info(
        "cached %d/%d probe point(s); %d image(s) had no feature file",
        int(filled.sum()),
        n_points,
        len(missing_images),
    )

    kept = np.flatnonzero(filled)
    return FeatureCache(
        features=features[kept],
        image_ids=tuple(image_ids[position] for position in kept),
        point_ids=tuple(point_ids[position] for position in kept),
        rows=rows[kept],
        cols=cols[kept],
        gt_labels=tuple(str(value) for value in probe_rows["gt_label"].to_numpy()[kept]),
        region_ids=tuple(str(value) for value in probe_rows["region_id"].to_numpy()[kept]),
        held_out=np.asarray(probe_rows["held_out"], dtype=bool)[kept],
        n_points_requested=n_points,
        n_points_missing_row_col=n_missing_row_col,
        n_points_missing_image=n_missing_image,
        missing_image_ids=tuple(sorted(missing_images)),
    )


def write_feature_cache(cache: FeatureCache, rows: pd.DataFrame, path: Path) -> None:
    """Write the cache as one npz whose arrays stay index-aligned.

    `rows` is the probe selection the cache was built to cover -- not
    necessarily every row it carries, since an image with no feature file
    drops out -- and its hash is what lets `load_probe` tell this cache apart
    from one built for a different selection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=cache.features,
        image_id=np.array(cache.image_ids, dtype=np.str_),
        point_id=np.array(cache.point_ids, dtype=np.str_),
        row=cache.rows,
        col=cache.cols,
        gt_label=np.array(cache.gt_labels, dtype=np.str_),
        region_id=np.array(cache.region_ids, dtype=np.str_),
        held_out=cache.held_out,
        content_hash=np.array(probe_content_hash(rows), dtype=np.str_),
    )


def _image_vectors(
    loader: FeatureLoader,
    image_id: str,
    positions: Sequence[int],
    *,
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    feature_dim: int,
) -> list[tuple[int, NDArray[np.float32]]] | None:
    """One image's probe vectors, or None when it has no feature file.

    Only the fetch is forgiving. A file that arrives and does not parse, or
    that carries a different feature dimension, raises: that is a wrong
    extractor or a corrupt object, and shrinking the probe over it would
    change what the score means.
    """
    try:
        payload = loader(image_id)
    except Exception as error:
        logger.warning("no feature file for image %s: %s", image_id, error)
        return None

    file_rows, file_cols, features = read_feature_file(payload)
    if features.shape[1] != feature_dim:
        raise ValueError(
            f"image {image_id} has feature dimension {features.shape[1]}, expected {feature_dim}"
        )

    index = {
        (int(row), int(col)): position
        for position, (row, col) in enumerate(zip(file_rows, file_cols, strict=True))
    }
    matched: list[tuple[int, NDArray[np.float32]]] = []
    for position in positions:
        source = index.get((int(rows[position]), int(cols[position])))
        if source is not None:
            matched.append((position, features[source]))
    return matched
