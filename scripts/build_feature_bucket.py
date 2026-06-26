"""Build a CoralNet-layout feature-vector bucket for the updated source set.

Reads images and per-source annotations from
``s3://<source-image-bucket>/<source-prefix>/s{source_id}/``
(the segmentation team's scrape), runs pyspacer's EfficientNet extractor
against every annotated image, and writes the results into a target
bucket laid out as:

    s{source_id}/annotations.csv
    s{source_id}/features/i{image_id}.featurevector

After a successful build, point ``CORALNET_TRAIN_DATA_BUCKET`` in the
classifier's ``.env`` at the new bucket to retrain on the updated set.

Resumability
------------
Every write is idempotent at the (source, image) level. With
``--skip-existing`` (default) a re-run skips any image whose feature
file already exists in the target bucket -- the bucket itself is the
source of truth for progress. ``--force`` overwrites everything.

Example
-------
    uv run python scripts/build_feature_bucket.py \\
        --source-bucket <source-image-bucket> \\
        --target-bucket <target-feature-bucket> \\
        --sources-csv /path/to/sources.csv \\
        --aws-profile wcs
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
        # Container / ambient mode: let every consumer (pyspacer, our
        # own boto3 calls, ...) use boto3's default credential chain.
        # That chain auto-refreshes from the SageMaker task-role
        # endpoint as the session token nears expiry. Pinning
        # AWS_PROFILE or copying frozen creds into SPACER_AWS_* here
        # would freeze a 1-hour token and break long-running jobs.
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
import json
import logging
import re
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError

try:
    from tqdm import tqdm
except ImportError:  # tqdm is convenient but not required.

    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else iter(())


logger = logging.getLogger("build_feature_bucket")

DEFAULT_SOURCE_BUCKET = "source-image-bucket"
DEFAULT_SOURCE_PREFIX = "coralnet-public-images/"
DEFAULT_AWS_PROFILE = "wcs"
DEFAULT_MAX_IO_WORKERS = 16

SOURCE_ID_COLUMN_CANDIDATES = ("id", "Source ID", "source_id")

FEATURE_KEY_RE = re.compile(r".*/i([^/]+)\.featurevector$")

# `image_list.csv` Name values look like "001-CA2M-1.JPG - Confirmed".
# `annotations.csv` Name values are the bare filename without the suffix.
# Strip the trailing status to make them join.
IMAGE_LIST_STATUS_SUFFIX_RE = re.compile(r"\s+-\s+(?:Confirmed|Unconfirmed|Unclassified)\s*$")

# `Image Page` column in image_list.csv looks like "/image/1719202/view/".
IMAGE_PAGE_ID_RE = re.compile(r"/image/(\d+)/")


# ---- CLI -------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target-bucket", required=True, help="Destination bucket. Must already exist.")
    src_group = p.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--sources-csv", type=Path, help="CSV with a source-ID column.")
    src_group.add_argument("--source-ids", help="Comma-separated source IDs (testing).")
    p.add_argument(
        "--source-id-column",
        help="Override CSV column name for source ID. "
        f"Default: auto-detect {SOURCE_ID_COLUMN_CANDIDATES}.",
    )
    p.add_argument("--source-bucket", default=DEFAULT_SOURCE_BUCKET)
    p.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
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
    p.add_argument("--max-io-workers", type=int, default=DEFAULT_MAX_IO_WORKERS)
    p.add_argument("--aws-profile", default=DEFAULT_AWS_PROFILE)
    p.add_argument(
        "--no-aws-bootstrap",
        action="store_true",
        help="Skip SSO bootstrap; use ambient AWS credentials "
        "(e.g. SageMaker task role, EC2 instance profile).",
    )
    skip_group = p.add_mutually_exclusive_group()
    skip_group.add_argument(
        "--skip-existing", dest="skip_existing", action="store_true", default=True
    )
    skip_group.add_argument(
        "--force",
        dest="skip_existing",
        action="store_false",
        help="Re-extract even if feature file already exists.",
    )
    p.add_argument("--dry-run", action="store_true", help="Log planned work; do not write to S3.")
    p.add_argument(
        "--error-log",
        type=Path,
        help="Append-only CSV for per-image failures. "
        "Default: build_feature_bucket_errors_<ts>.csv",
    )
    p.add_argument(
        "--progress-log",
        type=Path,
        help="Append-only JSONL for (source, image, outcome). "
        "Default: build_feature_bucket_progress_<ts>.jsonl",
    )
    p.add_argument("--log-level", default="INFO")
    return p


# ---- Source-ID CSV parsing ------------------------------------------


def detect_id_column(columns: Iterable[str], override: str | None) -> str:
    cols = list(columns)
    if override:
        if override not in cols:
            raise ValueError(f"--source-id-column={override!r} not found in CSV columns: {cols}")
        return override
    for candidate in SOURCE_ID_COLUMN_CANDIDATES:
        if candidate in cols:
            return candidate
    raise ValueError(
        f"Could not find a source-ID column in CSV. "
        f"Tried {SOURCE_ID_COLUMN_CANDIDATES}; got columns: {cols}. "
        f"Pass --source-id-column to override."
    )


def load_source_ids_from_csv(path: Path, override: str | None) -> list[str]:
    df = pd.read_csv(path)
    col = detect_id_column(df.columns, override)
    # Cast to string and strip whitespace; drop blanks.
    ids = [str(v).strip() for v in df[col].tolist()]
    ids = [v for v in ids if v and v.lower() != "nan"]
    # Normalize numeric "123.0" -> "123" while leaving non-numeric IDs alone.
    out = []
    for v in ids:
        try:
            out.append(str(int(float(v))))
        except ValueError:
            out.append(v)
    # Preserve order, drop duplicates.
    seen, deduped = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def load_source_ids_from_args(args: argparse.Namespace) -> list[str]:
    if args.sources_csv:
        return load_source_ids_from_csv(args.sources_csv, args.source_id_column)
    raw = [s.strip() for s in args.source_ids.split(",")]
    return [s for s in raw if s]


# ---- S3 helpers -----------------------------------------------------


_NOT_FOUND_CODES = ("404", "NoSuchKey", "NotFound")


def _client_error_code(exc: ClientError) -> str:
    return exc.response.get("Error", {}).get("Code", "")


def head_ok(s3, bucket: str, key: str) -> bool:
    try:
        s3.meta.client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if _client_error_code(exc) in _NOT_FOUND_CODES:
            return False
        raise


def filter_to_available_sources(
    s3,
    source_bucket: str,
    source_prefix: str,
    source_ids: list[str],
    max_workers: int,
) -> list[str]:
    """Drop source IDs whose annotations.csv is missing in the source bucket.

    Distinguishes 404 (truly missing -- skip) from other errors like 403
    (AccessDenied -- bail out so the user knows it's a permissions issue,
    not a data issue).
    """
    client = s3.meta.client

    def probe(sid: str) -> tuple[str, bool | ClientError]:
        key = f"{source_prefix}s{sid}/annotations.csv"
        try:
            client.head_object(Bucket=source_bucket, Key=key)
            return sid, True
        except ClientError as exc:
            if _client_error_code(exc) in _NOT_FOUND_CODES:
                return sid, False
            return sid, exc

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = dict(ex.map(probe, source_ids))

    other_errors = {sid: r for sid, r in results.items() if isinstance(r, ClientError)}
    if other_errors:
        sample_sid, sample_exc = next(iter(other_errors.items()))
        code = _client_error_code(sample_exc)
        msg = sample_exc.response.get("Error", {}).get("Message", "")
        raise RuntimeError(
            f"HEAD on s3://{source_bucket}/{source_prefix}s{{id}}/annotations.csv "
            f"failed with non-404 error for {len(other_errors)}/{len(source_ids)} "
            f"source(s). Example: source s{sample_sid} -> {code}: {msg}. "
            f"This is usually a permissions problem (the IAM principal can't "
            f"HEAD the source bucket) -- not a data problem."
        )

    present = [sid for sid in source_ids if results[sid] is True]
    missing = [sid for sid in source_ids if results[sid] is False]
    if missing:
        logger.warning(
            "%d source(s) skipped -- no annotations.csv in s3://%s/%s: %s",
            len(missing),
            source_bucket,
            source_prefix,
            ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""),
        )
    return present


def list_existing_feature_image_ids(s3, target_bucket: str, source_id: str) -> set[str]:
    """Return the set of image IDs that already have a feature file in target."""
    prefix = f"s{source_id}/features/"
    paginator = s3.meta.client.get_paginator("list_objects_v2")
    existing: set[str] = set()
    for page in paginator.paginate(Bucket=target_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            m = FEATURE_KEY_RE.match(obj["Key"])
            if m:
                existing.add(m.group(1))
    return existing


# ---- DataLocation helpers -------------------------------------------


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

        def __init__(self, *, device: str, batch_size: int, **kwargs):
            super().__init__(**kwargs)
            import torch

            self._device = torch.device(device)
            self._batch_size = int(batch_size)
            self._cached_net = None
            self._cached_loaded_remote = False

        def _ensure_net(self):
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

        def patches_to_features(self, patch_list):
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
    extractor,
    weights_loc,
    batch_size: int,
    device: str,
    n_patches: int = 8,
    threshold: float = 0.999,
) -> None:
    """Sanity check: feature vectors on the chosen device match CPU output
    on a fixed random batch. Raises RuntimeError if min cosine similarity
    drops below ``threshold``."""
    if device == "cpu":
        return

    import numpy as np
    import torch  # noqa: F401  (forces the same import order)
    from PIL import Image

    rng = np.random.default_rng(seed=42)
    patches = [
        Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(n_patches)
    ]

    cls = _build_device_caching_extractor_class()
    cpu_extractor = cls(
        data_locations={"weights": weights_loc},
        device="cpu",
        batch_size=batch_size,
    )

    device_feats, _ = extractor.patches_to_features(patches)
    cpu_feats, _ = cpu_extractor.patches_to_features(patches)

    A = np.asarray(device_feats)
    B = np.asarray(cpu_feats)
    sims = (A * B).sum(axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
    logger.info(
        "Device numerics check (%s vs cpu, %d random patches): "
        "min_cos=%.6f median=%.6f max_abs_diff=%.4g",
        device,
        n_patches,
        float(sims.min()),
        float(np.median(sims)),
        float(np.abs(A - B).max()),
    )
    if sims.min() < threshold:
        raise RuntimeError(
            f"Device numerics check FAILED on {device}: min cosine "
            f"similarity {sims.min():.6f} < {threshold}. The features "
            f"would not be safe to mix with previously CPU-extracted ones."
        )


def parse_weights_location(uri: str):
    """Parse an s3://bucket/key or filesystem path into a DataLocation."""
    from spacer.data_classes import DataLocation

    if uri.startswith("s3://"):
        rest = uri[len("s3://") :]
        bucket, _, key = rest.partition("/")
        if not bucket or not key:
            raise ValueError(f"Bad S3 URI for weights: {uri!r}")
        return DataLocation(storage_type="s3", key=key, bucket_name=bucket)
    return DataLocation(storage_type="filesystem", key=uri)


def build_extract_msg(
    source_id: str,
    image_id: str,
    rowcols: list[tuple[int, int]],
    extractor,
    source_bucket: str,
    source_prefix: str,
    target_bucket: str,
):
    from spacer.data_classes import DataLocation
    from spacer.messages import ExtractFeaturesMsg

    return ExtractFeaturesMsg(
        job_token=f"s{source_id}_i{image_id}",
        extractor=extractor,
        rowcols=rowcols,
        image_loc=DataLocation(
            storage_type="s3",
            key=f"{source_prefix}s{source_id}/images/{image_id}.jpg",
            bucket_name=source_bucket,
        ),
        feature_loc=DataLocation(
            storage_type="s3",
            key=f"s{source_id}/features/i{image_id}.featurevector",
            bucket_name=target_bucket,
        ),
    )


# ---- Per-source processing ------------------------------------------


@dataclass
class RunCounters:
    sources_done: int = 0
    sources_skipped: int = 0
    images_ok: int = 0
    images_skipped: int = 0
    images_failed: int = 0
    annotations_copied: int = 0
    annotations_skipped: int = 0
    started: float = field(default_factory=time.monotonic)


@dataclass
class PreparedSource:
    transformed_csv: bytes
    images: dict[str, list[tuple[int, int]]]
    n_unmapped_rows: int = 0


def build_name_to_image_id_mapping(
    s3,
    source_bucket: str,
    source_prefix: str,
    source_id: str,
) -> dict[str, str]:
    """Read s{id}/image_list.csv and build {filename: numeric_image_id}.

    image_list.csv has columns Name, Image Page, Image URL where:
      Name        = "001-CA2M-1.JPG - Confirmed"     (filename + status)
      Image Page  = "/image/1719202/view/"           (numeric ID in path)
    annotations.csv carries the bare filename ("001-CA2M-1.JPG"), so we
    strip the status suffix before keying.
    """
    key = f"{source_prefix}s{source_id}/image_list.csv"
    body = s3.Object(source_bucket, key).get()["Body"].read()
    df = pd.read_csv(BytesIO(body))
    if "Name" not in df.columns or "Image Page" not in df.columns:
        raise ValueError(
            f"s{source_id}/image_list.csv missing required columns; got {list(df.columns)}"
        )
    df = df[["Name", "Image Page"]].dropna()
    df["image_id"] = df["Image Page"].astype(str).str.extract(IMAGE_PAGE_ID_RE.pattern)[0]
    df["name_norm"] = (
        df["Name"].astype(str).map(lambda n: IMAGE_LIST_STATUS_SUFFIX_RE.sub("", n).strip())
    )
    df = df.dropna(subset=["image_id"])
    return dict(zip(df["name_norm"], df["image_id"], strict=False))


def prepare_source(
    s3,
    source_bucket: str,
    source_prefix: str,
    source_id: str,
) -> PreparedSource | None:
    """Read source annotations.csv (and image_list.csv if needed), augment
    with an `Image ID` column when only `Name` is present, group rowcols
    by image_id, and produce the transformed CSV body to upload.

    Returns None if the source is unusable (missing files/columns).
    """
    ann_key = f"{source_prefix}s{source_id}/annotations.csv"
    body = s3.Object(source_bucket, ann_key).get()["Body"].read()
    df = pd.read_csv(BytesIO(body))

    required_basic = {"Row", "Column", "Label ID"}
    missing_basic = required_basic - set(df.columns)
    if missing_basic:
        logger.warning(
            "s%s: annotations.csv missing required columns %s; skipping source",
            source_id,
            missing_basic,
        )
        return None

    n_unmapped = 0
    if "Image ID" not in df.columns:
        if "Name" not in df.columns:
            logger.warning(
                "s%s: annotations.csv has neither 'Image ID' nor 'Name'; skipping source", source_id
            )
            return None
        try:
            name_to_id = build_name_to_image_id_mapping(s3, source_bucket, source_prefix, source_id)
        except ClientError as exc:
            code = _client_error_code(exc)
            if code in _NOT_FOUND_CODES:
                logger.warning(
                    "s%s: image_list.csv missing -- cannot map Name->Image ID; skipping source",
                    source_id,
                )
                return None
            raise
        norm = df["Name"].astype(str).map(lambda n: IMAGE_LIST_STATUS_SUFFIX_RE.sub("", n).strip())
        df["Image ID"] = norm.map(name_to_id)
        n_unmapped = int(df["Image ID"].isna().sum())
        if n_unmapped:
            logger.warning(
                "s%s: dropping %d/%d annotation row(s) with unmappable Name",
                source_id,
                n_unmapped,
                len(df),
            )
            df = df.dropna(subset=["Image ID"]).copy()
        df["Image ID"] = df["Image ID"].astype(int).astype(str)
    else:
        df["Image ID"] = df["Image ID"].astype(str)

    # Group rowcols by Image ID.
    sub = df[["Image ID", "Row", "Column"]].copy()
    sub["Row"] = sub["Row"].astype(int)
    sub["Column"] = sub["Column"].astype(int)
    images: dict[str, list[tuple[int, int]]] = {}
    for image_id, g in sub.groupby("Image ID", sort=True):
        pairs = sorted({(r, c) for r, c in zip(g["Row"], g["Column"], strict=False)})
        images[str(image_id)] = pairs

    # Emit transformed CSV bytes (Image ID added; everything else preserved).
    buf = BytesIO()
    df.to_csv(buf, index=False)
    return PreparedSource(transformed_csv=buf.getvalue(), images=images, n_unmapped_rows=n_unmapped)


def upload_annotations_csv(
    s3,
    source_id: str,
    transformed_csv: bytes,
    target_bucket: str,
    skip_existing: bool,
    dry_run: bool,
) -> str:
    """Return 'uploaded' or 'skipped'."""
    tgt_key = f"s{source_id}/annotations.csv"
    if skip_existing and head_ok(s3, target_bucket, tgt_key):
        return "skipped"
    if dry_run:
        return "uploaded"
    s3.meta.client.put_object(Bucket=target_bucket, Key=tgt_key, Body=transformed_csv)
    return "uploaded"


def process_source(
    *,
    source_id: str,
    extractor,
    s3,
    args: argparse.Namespace,
    counters: RunCounters,
    progress_writer,
    error_writer,
) -> None:
    # Lazy import: spacer is only needed inside the GPU loop.
    from spacer.tasks import extract_features

    try:
        prepared = prepare_source(s3, args.source_bucket, args.source_prefix, source_id)
    except ClientError as exc:
        logger.error("s%s: failed to read source: %s", source_id, exc)
        record_failure(
            error_writer, source_id, image_id="", error_type=type(exc).__name__, error_msg=str(exc)
        )
        counters.sources_skipped += 1
        return

    if prepared is None or not prepared.images:
        counters.sources_skipped += 1
        return

    try:
        ann_outcome = upload_annotations_csv(
            s3,
            source_id,
            prepared.transformed_csv,
            target_bucket=args.target_bucket,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
    except ClientError as exc:
        logger.error("s%s: failed to upload annotations.csv: %s", source_id, exc)
        record_failure(
            error_writer, source_id, image_id="", error_type=type(exc).__name__, error_msg=str(exc)
        )
        counters.sources_skipped += 1
        return

    if ann_outcome == "uploaded":
        counters.annotations_copied += 1
    else:
        counters.annotations_skipped += 1

    existing = (
        list_existing_feature_image_ids(s3, args.target_bucket, source_id)
        if args.skip_existing
        else set()
    )

    grouped = prepared.images
    image_ids = sorted(grouped.keys())
    inner = tqdm(image_ids, desc=f"s{source_id}", leave=False, unit="img")
    for image_id in inner:
        rowcols = grouped[image_id]
        if not rowcols:
            counters.images_skipped += 1
            record_progress(progress_writer, source_id, image_id, "skipped", reason="no_rowcols")
            continue
        if image_id in existing:
            counters.images_skipped += 1
            record_progress(progress_writer, source_id, image_id, "skipped", reason="exists")
            continue

        if args.dry_run:
            counters.images_ok += 1
            record_progress(progress_writer, source_id, image_id, "ok", dry_run=True)
            continue

        msg = build_extract_msg(
            source_id=source_id,
            image_id=image_id,
            rowcols=rowcols,
            extractor=extractor,
            source_bucket=args.source_bucket,
            source_prefix=args.source_prefix,
            target_bucket=args.target_bucket,
        )
        try:
            extract_features(msg)
            counters.images_ok += 1
            record_progress(progress_writer, source_id, image_id, "ok")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            counters.images_failed += 1
            logger.warning("s%s i%s: extract_features failed: %s", source_id, image_id, exc)
            record_failure(error_writer, source_id, image_id, type(exc).__name__, str(exc))
            record_progress(
                progress_writer, source_id, image_id, "failed", error_type=type(exc).__name__
            )

    counters.sources_done += 1


# ---- Logging helpers ------------------------------------------------


def record_progress(writer, source_id: str, image_id: str, outcome: str, **extra) -> None:
    if writer is None:
        return
    rec = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_id": source_id,
        "image_id": image_id,
        "outcome": outcome,
        **extra,
    }
    writer.write(json.dumps(rec) + "\n")
    writer.flush()


def record_failure(writer, source_id: str, image_id: str, error_type: str, error_msg: str) -> None:
    if writer is None:
        return
    writer.writerow(
        [
            datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            source_id,
            image_id,
            error_type,
            error_msg,
        ]
    )


# ---- main -----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    _bootstrap_aws_env(_early_profile_from_argv(argv))

    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    # Imports that depend on env-resolved settings.
    from spacer.aws import get_s3_resource

    from mermaid_classifier.pyspacer.settings import settings

    weights_uri = args.weights or settings.weights_location
    if not weights_uri:
        logger.error(
            "No EfficientNet weights configured. Pass --weights or set WEIGHTS_LOCATION in .env."
        )
        return 2
    weights_loc = parse_weights_location(weights_uri)

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

    s3 = get_s3_resource()

    source_ids = load_source_ids_from_args(args)
    logger.info("Requested %d source(s).", len(source_ids))
    source_ids = filter_to_available_sources(
        s3, args.source_bucket, args.source_prefix, source_ids, args.max_io_workers
    )
    logger.info("%d source(s) have annotations.csv and will be processed.", len(source_ids))

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    error_log_path = args.error_log or Path(f"build_feature_bucket_errors_{ts}.csv")
    progress_log_path = args.progress_log or Path(f"build_feature_bucket_progress_{ts}.jsonl")

    counters = RunCounters()
    with (
        open(error_log_path, "a", newline="") as error_f,
        open(progress_log_path, "a") as progress_f,
    ):
        error_writer = csv.writer(error_f)
        if error_f.tell() == 0:
            error_writer.writerow(["ts", "source_id", "image_id", "error_type", "error_msg"])

        logger.info("Errors -> %s", error_log_path)
        logger.info("Progress -> %s", progress_log_path)
        if args.dry_run:
            logger.warning("DRY RUN -- no S3 writes will occur.")

        try:
            for source_id in tqdm(source_ids, desc="sources", unit="src"):
                process_source(
                    source_id=source_id,
                    extractor=extractor,
                    s3=s3,
                    args=args,
                    counters=counters,
                    progress_writer=progress_f,
                    error_writer=error_writer,
                )
        except KeyboardInterrupt:
            logger.warning("Interrupted by user; printing partial summary.")
        finally:
            elapsed = time.monotonic() - counters.started
            logger.info(
                "Done. sources_done=%d sources_skipped=%d "
                "images_ok=%d images_skipped=%d images_failed=%d "
                "annotations_copied=%d annotations_skipped=%d elapsed=%.1fs",
                counters.sources_done,
                counters.sources_skipped,
                counters.images_ok,
                counters.images_skipped,
                counters.images_failed,
                counters.annotations_copied,
                counters.annotations_skipped,
                elapsed,
            )

    return 0 if counters.images_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
