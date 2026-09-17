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

Downloads run through a thread pool in batches, each batch checkpointed to a
shard so an interrupted build restarts where it stopped. The shard records
which images were missing as well as which points were filled, so a resumed
build reports the same counts as an uninterrupted one. It also records a hash
of the `(image_id, point_id, row, col)` its rows were built for: positions are
indices into one run's probe rows, so a shard reused across a changed
selection would land the previous selection's vectors on this one's points --
the very misalignment the (row, col) match exists to prevent. A shard whose
hash does not match the batch is discarded and the batch downloaded again.
"""

import dataclasses
import hashlib
import io
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FEATURE_DIM = 1280
DEFAULT_WORKERS = 32
DEFAULT_BATCH_SIZE = 500
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


def s3_feature_loader(
    *,
    bucket: str = DEFAULT_FEATURE_BUCKET,
    prefix: str = DEFAULT_FEATURE_PREFIX,
    suffix: str = DEFAULT_FEATURE_SUFFIX,
    region_name: str = "us-east-1",
    workers: int = DEFAULT_WORKERS,
) -> FeatureLoader:
    """A loader that fetches one image's feature file from S3.

    One client serves every worker: botocore clients are thread-safe, and the
    connection pool is sized to the pool that will share it.
    """
    client = boto3.client(
        "s3",
        region_name=region_name,
        config=Config(
            max_pool_connections=workers + 8,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )

    def load(image_id: str) -> bytes:
        key = f"{prefix}{image_id}{suffix}"
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()

    return load


def build_feature_cache(
    probe_rows: pd.DataFrame,
    loader: FeatureLoader,
    *,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shard_dir: Path | None = None,
    feature_dim: int = FEATURE_DIM,
) -> FeatureCache:
    """Fetch every probe point's feature vector into one aligned matrix.

    Output rows follow `probe_rows` order, minus the points no vector was
    found for.
    """
    missing = [column for column in METADATA_COLUMNS if column not in probe_rows.columns]
    if missing:
        raise ValueError(f"probe rows are missing column(s): {', '.join(missing)}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    image_ids = [str(value) for value in probe_rows["image_id"]]
    point_ids = [str(value) for value in probe_rows["point_id"]]
    rows = np.asarray(probe_rows["row"], dtype=np.int64)
    cols = np.asarray(probe_rows["col"], dtype=np.int64)

    wanted: dict[str, list[int]] = {}
    for position, image_id in enumerate(image_ids):
        wanted.setdefault(image_id, []).append(position)
    batches = _batches(sorted(wanted), batch_size)

    n_points = len(image_ids)
    features = np.zeros((n_points, feature_dim), dtype=np.float32)
    filled = np.zeros(n_points, dtype=bool)
    missing_images: set[str] = set()

    for index, batch in enumerate(batches):
        shard = None if shard_dir is None else shard_dir / f"batch_{index:05d}.npz"
        key = _shard_points_key(
            [position for image_id in batch for position in wanted[image_id]],
            image_ids=image_ids,
            point_ids=point_ids,
            rows=rows,
            cols=cols,
        )
        if shard is not None and shard.exists():
            restored = _restore_shard(shard, key)
            if restored is None:
                logger.warning(
                    "shard %s was built for a different set of probe points;"
                    " discarding it and downloading this batch again",
                    shard.name,
                )
            else:
                positions, vectors, shard_missing = restored
                features[positions] = vectors
                filled[positions] = True
                missing_images.update(shard_missing)
                continue

        found, batch_missing = _run_batch(
            loader,
            batch,
            wanted=wanted,
            rows=rows,
            cols=cols,
            workers=workers,
            feature_dim=feature_dim,
        )
        positions = np.array([position for position, _ in found], dtype=np.int64)
        vectors = (
            np.stack([vector for _, vector in found])
            if found
            else np.zeros((0, feature_dim), dtype=np.float32)
        )
        features[positions] = vectors
        filled[positions] = True
        missing_images.update(batch_missing)
        if shard is not None:
            shard.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                shard,
                positions=positions,
                vectors=vectors,
                missing_image_ids=np.array(sorted(batch_missing), dtype=np.str_),
                points_key=np.array(key, dtype=np.str_),
            )
        logger.info(
            "feature batch %d/%d: %d points, %d image(s) without a feature file",
            index + 1,
            len(batches),
            len(found),
            len(batch_missing),
        )

    absent_image = np.array([image_id in missing_images for image_id in image_ids], dtype=bool)
    n_missing_image = int((~filled & absent_image).sum())
    n_missing_row_col = int((~filled & ~absent_image).sum())
    if n_missing_row_col:
        logger.warning(
            "%d probe point(s) had no matching (row, col) in their feature file; dropping them",
            n_missing_row_col,
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


def write_feature_cache(cache: FeatureCache, path: Path) -> None:
    """Write the cache as one npz whose arrays stay index-aligned."""
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
    )


def _shard_points_key(
    positions: Sequence[int],
    *,
    image_ids: Sequence[str],
    point_ids: Sequence[str],
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
) -> str:
    """A hash of the points one shard covers, in the order it stores them.

    Identifies the points rather than their row indices, which belong to one
    run's probe rows and mean nothing in another's.
    """
    digest = hashlib.sha256()
    for position in positions:
        digest.update(
            f"{image_ids[position]}\x1f{point_ids[position]}\x1f"
            f"{int(rows[position])}\x1f{int(cols[position])}\x1e".encode()
        )
    return digest.hexdigest()


def _restore_shard(
    shard: Path, key: str
) -> tuple[NDArray[np.int64], NDArray[np.float32], set[str]] | None:
    """One shard's filled positions, vectors and missing images, or None.

    None is a shard whose key is not this batch's, and a shard carrying no key
    at all: neither says its rows are these points.
    """
    cached = np.load(shard, allow_pickle=False)
    if "points_key" not in cached.files or str(cached["points_key"]) != key:
        return None
    return (
        cached["positions"].astype(np.int64),
        np.asarray(cached["vectors"], dtype=np.float32),
        {str(value) for value in cached["missing_image_ids"]},
    )


def _batches(image_ids: Sequence[str], batch_size: int) -> list[Sequence[str]]:
    return [image_ids[start : start + batch_size] for start in range(0, len(image_ids), batch_size)]


def _run_batch(
    loader: FeatureLoader,
    batch: Sequence[str],
    *,
    wanted: dict[str, list[int]],
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    workers: int,
    feature_dim: int,
) -> tuple[list[tuple[int, NDArray[np.float32]]], set[str]]:
    """Download and match one batch of images, in parallel."""
    found: list[tuple[int, NDArray[np.float32]]] = []
    missing: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _image_vectors,
                loader,
                image_id,
                wanted[image_id],
                rows=rows,
                cols=cols,
                feature_dim=feature_dim,
            )
            for image_id in batch
        ]
        for image_id, future in zip(batch, futures, strict=True):
            matched = future.result()
            if matched is None:
                missing.add(image_id)
            else:
                found += matched
    return found, missing


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
