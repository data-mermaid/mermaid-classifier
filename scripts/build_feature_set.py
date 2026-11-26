"""Write pyspacer feature vectors for a dataset version's images into a feature family.

Reads one pinned segdata dataset version (``--dataset``, ``--version``,
``--manifest-sha256``) from the dataset bucket, runs pyspacer's EfficientNet
extractor at each selected image's annotated points, and writes one file per
image to

    features/<store>/<family>/<image_id>.featurevector

The family's ``_extractor_spec.json`` is claimed before any vector is written,
so a family never holds vectors from two extractors. Every vector is put with
``If-None-Match: *``: the bucket refuses unconditional puts, and a file that
already exists is never replaced.

A family's files are keyed by image id alone, so dataset versions that share
an image store share them. Within a family an image's points must therefore not
change across versions: ``segdata features attach`` refuses a vector that lacks
one of the version's points.

This script does not publish a feature-set index. ``segdata features attach``
adopts the family's files for a version and publishes the index.

Resumability
------------
A re-run skips every image whose family file already exists, so the bucket is
the record of progress. ``--shard i/n`` splits the selected images by a stable
hash of the image id, so parallel workers never write the same image.

Example
-------
    uv run python scripts/build_feature_set.py \\
        --dataset coralnet --version 20261124-cfc389d \\
        --manifest-sha256 <sha256> --family efficientnet-b0-v1 \\
        --cache-dir ~/.cache/segdata --aws-profile wcs
"""

from __future__ import annotations

# -- SSO bootstrap ----------------------------------------------------
# Must run before importing mermaid_classifier: its Settings object is
# created at import time and reads SPACER_AWS_* env vars then. Mirrors
# scripts/classifier_train.py lines 1-14 exactly.
import os
import sys


def _bootstrap_aws_env(profile: str | None) -> None:

    if profile is None:
        # Ambient mode: boto3's default chain refreshes the task-role token;
        # pinning a profile or frozen SPACER_AWS_* creds would expire mid-job.
        os.environ.pop("AWS_PROFILE", None)
        os.environ.pop("SPACER_AWS_ACCESS_KEY_ID", None)
        os.environ.pop("SPACER_AWS_SECRET_ACCESS_KEY", None)
        os.environ.pop("SPACER_AWS_SESSION_TOKEN", None)
        return

    os.environ["AWS_PROFILE"] = profile
    session = boto3.Session()
    credentials = session.get_credentials()
    if credentials is None:
        return
    creds = credentials.get_frozen_credentials()
    os.environ["SPACER_AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["SPACER_AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    if creds.token:
        os.environ["SPACER_AWS_SESSION_TOKEN"] = creds.token


def _early_profile_from_argv(argv: list[str]) -> str | None:
    """Resolve the AWS profile (or None for ambient creds) from argv.

    Honors --no-aws-bootstrap (signals ambient credentials) and
    --aws-profile (otherwise). Default is 'wcs'.
    """
    if "--no-aws-bootstrap" in argv:
        return None
    for i, a in enumerate(argv):
        if a == "--aws-profile" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--aws-profile="):
            return a.split("=", 1)[1]
    return "wcs"


# -- main module imports ----------------------------------------------
import argparse
import csv
import hashlib
import json
import logging
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import boto3
import pyarrow as pa
import pyarrow.compute as _pc
from botocore.exceptions import ClientError
from segdata_spec import DATASET_BUCKET
from segdata_spec.features import claim_extractor, feature_file_key
from segdata_spec.manifest import ImageStore, Pin
from segdata_spec.read import open_tables
from segdata_spec.sync import sync_tables

logger = logging.getLogger("build_feature_set")

# pyarrow.compute's functions are generated at import, so the type checker cannot see them.
pc: Any = _pc

DEFAULT_AWS_PROFILE = "wcs"
DEFAULT_MAX_IO_WORKERS = 16
# SageMaker uploads this directory when a processing job ends; logs default here when it exists.
PROCESSING_OUTPUT_DIR = Path("/opt/ml/processing/output")
PROGRESS_LOG_EVERY = 500

# (shard index, shard count), 0 <= index < count.
Shard = tuple[int, int]


# ---- CLI -------------------------------------------------------------


def parse_shard(text: str) -> Shard:
    """Parse ``i/n`` into (i, n), requiring n >= 1 and 0 <= i < n."""
    try:
        index_text, count_text = text.split("/")
        index, count = int(index_text), int(count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--shard must be i/n, got {text!r}") from exc
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError(f"--shard needs n >= 1 and 0 <= i < n, got {text!r}")
    return index, count


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. coralnet.")
    p.add_argument("--version", required=True, help="Dataset version, e.g. 20261124-cfc389d.")
    p.add_argument(
        "--manifest-sha256", required=True, help="sha256 of the version's manifest.json (the pin)."
    )
    p.add_argument("--family", required=True, help="Feature family to write into.")
    p.add_argument("--bucket", default=DATASET_BUCKET, help="Dataset bucket.")
    p.add_argument(
        "--cache-dir", type=Path, required=True, help="Local segdata cache for the tables."
    )
    group_sel = p.add_mutually_exclusive_group()
    group_sel.add_argument("--groups", help="Comma-separated dataset groups to extract.")
    group_sel.add_argument("--groups-csv", type=Path, help="File with one dataset group per line.")
    p.add_argument(
        "--shard",
        type=parse_shard,
        help="i/n: extract only the i-th of n disjoint shards of the selected images.",
    )
    p.add_argument(
        "--weights",
        help="EfficientNet weights URI (s3://... or path). Default: settings.weights_location.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help='Torch device for the forward pass. "auto" picks '
        "mps if available, else cuda, else cpu. On Apple "
        "Silicon, mps is ~5-10x faster than cpu.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Patches per forward-pass batch (pyspacer default "
        "is 10; bump to 64-128 on mps for higher throughput).",
    )
    p.add_argument(
        "--verify-numerics",
        action="store_true",
        help="Before processing, compare chosen-device features "
        "against CPU features on a random batch. Adds ~10s "
        "startup; catches MPS/CUDA numerical regressions.",
    )
    p.add_argument(
        "--max-io-workers",
        type=int,
        default=DEFAULT_MAX_IO_WORKERS,
        help="Concurrent image downloads ahead of the extractor.",
    )
    p.add_argument("--aws-profile", default=DEFAULT_AWS_PROFILE)
    p.add_argument(
        "--no-aws-bootstrap",
        action="store_true",
        help="Skip SSO bootstrap; use ambient AWS credentials "
        "(e.g. SageMaker task role, EC2 instance profile).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Sync the tables and log planned work; claim, read and write nothing in S3.",
    )
    p.add_argument(
        "--error-log",
        type=Path,
        help="Append-only CSV for per-image failures. Default: build_feature_set_errors_<ts>.csv in "
        "/opt/ml/processing/output when it exists, else the current directory.",
    )
    p.add_argument(
        "--progress-log",
        type=Path,
        help="Append-only JSONL for (image, outcome). "
        "Default: build_feature_set_progress_<ts>.jsonl, in the same directory as the error log.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def groups_from_args(args: argparse.Namespace) -> set[str] | None:
    """Return the groups --groups or --groups-csv names, or None to select every group.

    Raises:
        ValueError: The option is given but names no group.
    """
    if args.groups is not None:
        raw = args.groups.split(",")
    elif args.groups_csv is not None:
        raw = args.groups_csv.read_text().splitlines()
    else:
        return None
    groups = {g.strip() for g in raw if g.strip()}
    if not groups:
        raise ValueError("--groups/--groups-csv named no groups")
    return groups


# ---- Selection -------------------------------------------------------


def shard_of(image_id: str, count: int) -> int:
    """Return the shard ``image_id`` belongs to, stable across processes and runs."""
    digest = hashlib.sha256(image_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def select_images(
    images: Any,  # pyarrow.Table (untyped)
    groups: set[str] | None,
    shard: Shard | None,
) -> Any:  # pyarrow.Table
    """Keep the rows of ``images`` in ``groups`` (all when None) and in ``shard`` (all when None).

    Raises:
        ValueError: ``groups`` names a group the version does not have.
    """
    if groups is not None:
        absent = groups - set(pc.unique(images.column("group")).to_pylist())
        if absent:
            raise ValueError(f"groups not in the dataset version: {sorted(absent)}")
        images = images.filter(
            pc.is_in(images.column("group"), value_set=pa.array(sorted(groups), pa.string()))
        )
    if shard is not None:
        index, count = shard
        ids: list[str] = images.column("image_id").to_pylist()
        images = images.filter(pa.array([shard_of(i, count) == index for i in ids], pa.bool_()))
    return images


def check_image_ids(image_ids: Iterable[str]) -> None:
    """Refuse ids that are not plain file names: empty, a leading ``.``, ``/`` or ``..``.

    Raises:
        ValueError: Names up to ten offending ids.
    """
    bad = [i for i in image_ids if not i or i.startswith(".") or "/" in i or ".." in i]
    if bad:
        raise ValueError(f"{len(bad)} image id(s) are not plain file names: {bad[:10]}")


def rowcols_by_image(
    points: Any,  # pyarrow.Table (untyped)
    image_ids: Iterable[str],
) -> dict[str, list[tuple[int, int]]]:
    """Return each image's distinct (row, col) points, sorted; an image with none maps to []."""
    wanted = set(image_ids)
    out: dict[str, set[tuple[int, int]]] = {i: set() for i in wanted}
    mine = points.filter(
        pc.is_in(points.column("image_id"), value_set=pa.array(sorted(wanted), pa.string()))
    )
    for image_id, row, col in zip(
        mine.column("image_id").to_pylist(),
        mine.column("row").to_pylist(),
        mine.column("col").to_pylist(),
        strict=True,
    ):
        out[image_id].add((int(row), int(col)))
    return {i: sorted(rcs) for i, rcs in out.items()}


# ---- S3 --------------------------------------------------------------


def _client_error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def is_precondition_failed(exc: ClientError) -> bool:
    return (
        _client_error_code(exc) == "PreconditionFailed"
        or exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 412
    )


# The family prefix and file names follow segdata-spec v0.2.0's key layout
# (features/<store>/<family>/<image_id>[~<suffix>].featurevector), which has no public prefix helper.
def feature_family_prefix(image_store: ImageStore, bucket: str, family: str) -> str:
    """Return ``features/<store>/<family>/``, the prefix every family file is under.

    Derived from ``feature_file_key`` so the store and family are validated as segdata-spec does.
    """
    return feature_file_key(image_store, bucket, family, "_").rsplit("/", 1)[0] + "/"


def list_existing_image_ids(client: Any, bucket: str, prefix: str) -> set[str]:
    """Return the image ids that have an unsuffixed ``.featurevector`` under ``prefix``."""
    existing: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(prefix) :]
            if "/" not in name and "~" not in name and name.endswith(".featurevector"):
                existing.add(name.removesuffix(".featurevector"))
    return existing


def download_image(client: Any, bucket: str, key: str, etag: str, dest: Path) -> Path:
    """Write ``s3://bucket/key`` to ``dest``, requiring its ETag to still be ``etag``.

    Raises:
        ClientError: 412 PreconditionFailed when the object changed, or any other S3 error.
    """
    data = client.get_object(Bucket=bucket, Key=key, IfMatch=etag)["Body"].read()
    try:
        dest.write_bytes(data)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return dest


def put_feature(client: Any, bucket: str, key: str, body: bytes) -> bool:
    """Put ``body`` at ``key`` unless an object is there; return whether it was written.

    The put is conditional (``If-None-Match: *``) and carries a CRC64NVME checksum.
    """
    try:
        # segdata features attach refuses a family file without a CRC64NVME checksum.
        client.put_object(
            Bucket=bucket, Key=key, Body=body, IfNoneMatch="*", ChecksumAlgorithm="CRC64NVME"
        )
    except ClientError as exc:
        if is_precondition_failed(exc):
            return False
        raise
    return True


# ---- Device-aware extractor (cached model, chosen torch device) -----


def resolve_device(name: str) -> str:
    """Resolve --device argument to a concrete torch device string."""
    import torch

    if name == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but torch.backends.mps.is_available() is False.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False.")
    return name


def _build_device_caching_extractor_class():
    """Build the subclass lazily so we don't import torch/spacer until after
    the AWS bootstrap (and SPACER_* env vars) are set."""
    from spacer.extractors import EfficientNetExtractor

    class _DeviceCachingExtractor(EfficientNetExtractor):
        """EfficientNetExtractor variant that runs the forward pass on a
        chosen torch device and caches the loaded model across images.

        Pyspacer's stock TorchExtractor.patches_to_features rebuilds the
        network from disk on every call AND only knows about cuda. We
        override both behaviors:
        - The model is built once on first use, moved to the target
          device, and reused for every subsequent image.
        - Each batch tensor is moved to the device before the forward
          pass; outputs are copied back to CPU before .tolist().
        """

        def __init__(self, *, device: str, batch_size: int, **kwargs: Any):
            super().__init__(**kwargs)
            import torch

            self._device = torch.device(device)
            self._batch_size = int(batch_size)
            self._cached_net = None
            self._cached_loaded_remote = False

        def _ensure_net(self) -> tuple[Any, bool]:  # net: untyped pyspacer/torch model
            if self._cached_net is not None:
                return self._cached_net, False
            weights_ds, loaded_remote = self.load_datastream("weights")
            # Parent's load_weights builds an untrained model on CPU and
            # loads the state_dict into it. We then move it to our device.
            net = type(self).__mro__[1].load_weights(weights_ds)
            net = net.to(self._device)
            net.eval()
            self._cached_net = net
            self._cached_loaded_remote = loaded_remote
            return net, loaded_remote

        def patches_to_features(
            self, patch_list: Any
        ) -> Any:  # patch_list: pyspacer untyped interface
            import numpy as np
            import torch
            from spacer.extractors.torch_extractors import transformation

            net, loaded_remote = self._ensure_net()
            transformer = transformation()
            n = len(patch_list)
            bs = self._batch_size
            num_batches = int(np.ceil(n / bs))
            feats: list[list[float]] = []
            for b in range(num_batches):
                batch_imgs = patch_list[b * bs : (b + 1) * bs]
                batch_t = torch.stack([transformer(i) for i in batch_imgs]).to(
                    self._device, non_blocking=True
                )
                with torch.no_grad():
                    out = net.extract_features(batch_t)
                # Bring back to CPU before .tolist() -- MPS tensors can't
                # be iterated as Python floats directly on some torch builds.
                feats.extend(out.detach().to("cpu").tolist())
                del batch_t, out
            # Release MPS/CUDA allocator cache so the OS can reclaim memory
            # between images. Without this on Apple Silicon the allocator
            # fragments over a long run and pushes the system into swap.
            if self._device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
            elif self._device.type == "cuda":
                torch.cuda.empty_cache()
            return feats, loaded_remote

    return _DeviceCachingExtractor


def verify_device_numerics(
    extractor: Any,  # _DeviceCachingExtractor (untyped pyspacer subclass)
    weights_loc: Any,  # spacer DataLocation (untyped)
    batch_size: int,
    device: str,
    n_patches: int = 8,
    threshold: float = 0.999,
) -> None:
    """Sanity check: this extractor's features match stock pyspacer's on a
    fixed random batch. Raises RuntimeError if min cosine similarity drops
    below ``threshold``.

    The reference is a plain ``EfficientNetExtractor`` on CPU at pyspacer's
    own batch size, because that is what serving runs. Both axes this script
    can move -- device and batch size -- are then covered by one comparison;
    a CPU run at ``--batch-size 128`` changes BLAS reduction order just as a
    device switch does, so this runs even when the device is already cpu.
    """
    import numpy as np
    import torch  # noqa: F401  # pyright: ignore[reportUnusedImport]  # forces same import order as production path
    from PIL import Image
    from spacer.extractors import EfficientNetExtractor

    rng = np.random.default_rng(seed=42)
    patches = [
        Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(n_patches)
    ]

    reference = EfficientNetExtractor(data_locations={"weights": weights_loc})

    device_feats, _ = extractor.patches_to_features(patches)
    reference_feats, _ = reference.patches_to_features(patches)  # pyright: ignore[reportArgumentType]  # same PIL stub issue

    A = np.asarray(device_feats)
    B = np.asarray(reference_feats)
    sims = (A * B).sum(axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
    logger.info(
        "Numerics check (%s @ batch %d vs stock pyspacer @ batch %d on cpu, "
        "%d random patches): min_cos=%.6f median=%.6f max_abs_diff=%.4g",
        device,
        batch_size,
        EfficientNetExtractor.BATCH_SIZE,
        n_patches,
        float(sims.min()),
        float(np.median(sims)),
        float(np.abs(A - B).max()),
    )
    if sims.min() < threshold:
        raise RuntimeError(
            f"Numerics check FAILED for {device} @ batch {batch_size}: min "
            f"cosine similarity {sims.min():.6f} < {threshold} against stock "
            f"pyspacer. Features extracted this way would not be comparable "
            f"with the ones inference produces."
        )


def extract_to_file(
    extractor: Any,  # _DeviceCachingExtractor (untyped pyspacer subclass)
    image_id: str,
    image_path: Path,
    rowcols: list[tuple[int, int]],
    feature_path: Path,
) -> None:
    """Run pyspacer on a local image and write the feature file to ``feature_path``.

    Both locations are on the filesystem: pyspacer's own S3 write is
    unconditional, which the dataset bucket denies.
    """
    from spacer.data_classes import DataLocation
    from spacer.messages import ExtractFeaturesMsg
    from spacer.tasks import extract_features

    extract_features(
        ExtractFeaturesMsg(
            job_token=image_id,
            extractor=extractor,
            rowcols=rowcols,
            image_loc=DataLocation(storage_type="filesystem", key=str(image_path)),
            feature_loc=DataLocation(storage_type="filesystem", key=str(feature_path)),
        )
    )


# ---- Run -------------------------------------------------------------


@dataclass
class RunCounters:
    images_selected: int = 0
    images_ok: int = 0
    images_skipped: int = 0
    images_failed: int = 0
    started: float = field(default_factory=time.monotonic)

    def exit_code(self) -> int:
        """1 when any image failed, else 0."""
        return 0 if self.images_failed == 0 else 1


@dataclass(frozen=True)
class ImageJob:
    image_id: str
    key: str
    etag: str
    rowcols: list[tuple[int, int]]


def load_tables(client: Any, pin: Pin, cache_dir: Path, bucket: str) -> Any:
    """Sync the pinned version's manifest and tables (no images) and open them.

    Raises:
        PinMismatch: The published manifest differs from ``pin``.
        ManifestNotFound: The version has no manifest.
    """
    sync_tables(client, {pin.name: pin}, cache_dir, bucket=bucket)
    return open_tables(cache_dir, bucket, pin)


def plan_jobs(
    images: Any,  # pyarrow.Table (untyped)
    points: Any,  # pyarrow.Table (untyped)
    existing: set[str],
) -> tuple[list[ImageJob], list[str], list[str]]:
    """Split the selected images into jobs, ids whose family file exists, and ids with no points."""
    ids = images.column("image_id").to_pylist()
    rowcols = rowcols_by_image(points, ids)
    jobs: list[ImageJob] = []
    exists: list[str] = []
    no_points: list[str] = []
    for image_id, key, etag in zip(
        ids, images.column("key").to_pylist(), images.column("etag").to_pylist(), strict=True
    ):
        if image_id in existing:
            exists.append(image_id)
        elif not rowcols[image_id]:
            no_points.append(image_id)
        else:
            jobs.append(ImageJob(image_id, key, etag, rowcols[image_id]))
    return jobs, exists, no_points


def _prefetched[T, R](
    executor: ThreadPoolExecutor, items: Iterable[T], fetch: Callable[[T], R], window: int
) -> Iterator[tuple[T, Future[R]]]:
    """Yield (item, future) in order, keeping up to ``window`` fetches in flight."""
    pending: deque[tuple[T, Future[R]]] = deque()
    source = iter(items)
    for item in source:
        pending.append((item, executor.submit(fetch, item)))
        if len(pending) >= window:
            yield pending.popleft()
    while pending:
        yield pending.popleft()


def scratch_stem(image_id: str) -> str:
    """Return the scratch file stem for ``image_id``: its sha256, never the id itself."""
    return hashlib.sha256(image_id.encode("utf-8")).hexdigest()


def progress_line(done: int, total: int, elapsed_s: float) -> str:
    """Describe ``done`` of ``total`` images with the rate so far and the ETA at that rate."""
    rate = done / elapsed_s if elapsed_s > 0 else 0.0
    eta_s = (total - done) / rate if rate > 0 else float("inf")
    return f"{done}/{total} images, {rate:.2f} img/s, ETA {eta_s / 60:.1f} min"


def process_job(
    client: Any,
    bucket: str,
    image_path: Path,
    job: ImageJob,
    feature_key: str,
    extractor: Any,
    workdir: Path,
) -> bool:
    """Extract one downloaded image and put its vector; return whether it was written."""
    feature_path = workdir / f"{scratch_stem(job.image_id)}.featurevector"
    try:
        extract_to_file(extractor, job.image_id, image_path, job.rowcols, feature_path)
        return put_feature(client, bucket, feature_key, feature_path.read_bytes())
    finally:
        feature_path.unlink(missing_ok=True)


def run(
    *,
    client: Any,
    pin: Pin,
    bucket: str,
    family: str,
    cache_dir: Path,
    groups: set[str] | None,
    shard: Shard | None,
    extractor: Any,  # _DeviceCachingExtractor (untyped pyspacer subclass)
    extractor_spec: Mapping[str, Any],
    dry_run: bool,
    max_io_workers: int,
    progress_writer: Any,  # IO[str] | None
    error_writer: Any,  # csv._writer | None
) -> RunCounters:
    """Write a family file for every selected image of ``pin`` that lacks one.

    Raises:
        PinMismatch: The dataset manifest differs from ``pin``; nothing is claimed or read.
        ExtractorMismatch: The family is claimed by another extractor; no image is read.
        ValueError: The version has no points table, ``groups`` names an absent group, or a
            selected image id is not a plain file name; raised before the claim.
    """
    counters = RunCounters()
    tables = load_tables(client, pin, cache_dir, bucket)
    if tables.points is None:
        raise ValueError(f"{pin.name} {pin.version} is a dense dataset: it has no points to embed")
    store: ImageStore = tables.manifest.image_store
    selected = select_images(tables.images, groups, shard)
    check_image_ids(selected.column("image_id").to_pylist())
    counters.images_selected = selected.num_rows
    logger.info("Selected %d image(s) of %s %s.", selected.num_rows, pin.name, pin.version)

    if not dry_run:
        claim_extractor(client, bucket, store, family, extractor_spec)
    existing = list_existing_image_ids(client, bucket, feature_family_prefix(store, bucket, family))
    jobs, exists, no_points = plan_jobs(selected, tables.points, existing)
    for image_id in exists:
        counters.images_skipped += 1
        record_progress(progress_writer, image_id, "skipped", reason="exists")
    for image_id in no_points:
        counters.images_skipped += 1
        record_progress(progress_writer, image_id, "skipped", reason="no_points")
    logger.info(
        "%d to extract, %d already present, %d without points.",
        len(jobs),
        len(exists),
        len(no_points),
    )

    if dry_run:
        for job in jobs:
            counters.images_ok += 1
            record_progress(progress_writer, job.image_id, "ok", dry_run=True)
        return counters

    with (
        tempfile.TemporaryDirectory(prefix="build_feature_set_") as tmp,
        ThreadPoolExecutor(max_workers=max_io_workers) as executor,
    ):
        workdir = Path(tmp)

        def fetch(job: ImageJob) -> Path:
            suffix = PurePosixPath(job.key).suffix
            dest = workdir / f"{scratch_stem(job.image_id)}{suffix}"
            return download_image(client, store.bucket, store.prefix + job.key, job.etag, dest)

        started = time.monotonic()
        for done, (job, future) in enumerate(
            _prefetched(executor, jobs, fetch, max(1, max_io_workers) * 2), start=1
        ):
            _run_one(
                client,
                bucket,
                store,
                family,
                extractor,
                workdir,
                job,
                future,
                counters,
                progress_writer,
                error_writer,
            )
            if done % PROGRESS_LOG_EVERY == 0 or done == len(jobs):
                logger.info(progress_line(done, len(jobs), time.monotonic() - started))
    return counters


def _run_one(
    client: Any,
    bucket: str,
    store: ImageStore,
    family: str,
    extractor: Any,
    workdir: Path,
    job: ImageJob,
    future: Future[Path],
    counters: RunCounters,
    progress_writer: Any,
    error_writer: Any,
) -> None:
    """Finish one image: wait for its download, extract, put, and record the outcome."""
    image_path: Path | None = None
    try:
        image_path = future.result()
        key = feature_file_key(store, bucket, family, job.image_id)
        written = process_job(client, bucket, image_path, job, key, extractor, workdir)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        counters.images_failed += 1
        logger.warning("%s: failed: %s", job.image_id, exc)
        record_failure(error_writer, job.image_id, type(exc).__name__, str(exc))
        record_progress(progress_writer, job.image_id, "failed", error_type=type(exc).__name__)
        return
    finally:
        if image_path is not None:
            image_path.unlink(missing_ok=True)
    if written:
        counters.images_ok += 1
        record_progress(progress_writer, job.image_id, "ok")
    else:
        counters.images_skipped += 1
        record_progress(progress_writer, job.image_id, "skipped", reason="exists_at_put")


# ---- Logging helpers ------------------------------------------------


def record_progress(writer: Any, image_id: str, outcome: str, **extra: object) -> None:
    """Append one JSONL record; ``writer`` is an IO[str] or None."""
    if writer is None:
        return
    rec = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image_id": image_id,
        "outcome": outcome,
        **extra,
    }
    writer.write(json.dumps(rec) + "\n")
    writer.flush()


def record_failure(writer: Any, image_id: str, error_type: str, error_msg: str) -> None:
    """Append one CSV failure row; ``writer`` is a csv writer or None."""
    if writer is None:
        return
    writer.writerow(
        [datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), image_id, error_type, error_msg]
    )


# ---- main -----------------------------------------------------------


def _make_s3_client(max_io_workers: int) -> Any:
    from botocore.config import Config

    # Prefetch threads, the main thread's put and its listing share the pool.
    return boto3.client(
        "s3",
        config=Config(
            max_pool_connections=max_io_workers + 2,
            retries={"mode": "adaptive", "max_attempts": 8},
            connect_timeout=10,
            read_timeout=60,
        ),
    )


def default_log_dir() -> Path:
    """Return the processing output dir inside a SageMaker job, else the current directory."""
    return PROCESSING_OUTPUT_DIR if PROCESSING_OUTPUT_DIR.is_dir() else Path(".")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    _bootstrap_aws_env(_early_profile_from_argv(argv))

    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    # Imports that depend on env-resolved settings.
    from mermaid_classifier.pyspacer.extraction import (
        extractor_spec_from_weights,
        weights_data_location,
    )
    from mermaid_classifier.pyspacer.settings import settings

    weights_uri = args.weights or settings.weights_location
    if not weights_uri:
        logger.error(
            "No EfficientNet weights configured. Pass --weights or set WEIGHTS_LOCATION in .env."
        )
        return 2
    weights_loc = weights_data_location(weights_uri)

    device = resolve_device(args.device)
    logger.info("Compute device: %s (batch_size=%d)", device, args.batch_size)
    extractor_cls = _build_device_caching_extractor_class()
    extractor = extractor_cls(
        data_locations={"weights": weights_loc},
        device=device,
        batch_size=args.batch_size,
    )
    if args.verify_numerics:
        verify_device_numerics(extractor, weights_loc, args.batch_size, device)

    pin = Pin(name=args.dataset, version=args.version, manifest_sha256=args.manifest_sha256)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir = default_log_dir()
    error_log_path = args.error_log or log_dir / f"build_feature_set_errors_{ts}.csv"
    progress_log_path = args.progress_log or log_dir / f"build_feature_set_progress_{ts}.jsonl"

    with (
        open(error_log_path, "a", newline="") as error_f,
        open(progress_log_path, "a") as progress_f,
    ):
        error_writer = csv.writer(error_f)
        if error_f.tell() == 0:
            error_writer.writerow(["ts", "image_id", "error_type", "error_msg"])
        logger.info("Errors -> %s", error_log_path)
        logger.info("Progress -> %s", progress_log_path)
        if args.dry_run:
            logger.warning("DRY RUN -- no S3 claim, image read or feature write will occur.")

        counters = run(
            client=_make_s3_client(args.max_io_workers),
            pin=pin,
            bucket=args.bucket,
            family=args.family,
            cache_dir=args.cache_dir,
            groups=groups_from_args(args),
            shard=args.shard,
            extractor=extractor,
            extractor_spec=extractor_spec_from_weights(weights_uri).to_dict(),
            dry_run=args.dry_run,
            max_io_workers=args.max_io_workers,
            progress_writer=progress_f,
            error_writer=error_writer,
        )
        logger.info(
            "Done. selected=%d ok=%d skipped=%d failed=%d elapsed=%.1fs",
            counters.images_selected,
            counters.images_ok,
            counters.images_skipped,
            counters.images_failed,
            time.monotonic() - counters.started,
        )
    return counters.exit_code()


if __name__ == "__main__":
    sys.exit(main())
