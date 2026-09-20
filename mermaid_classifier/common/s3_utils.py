"""Shared S3 helpers: bucket/key URI parsing, object hashing, and parallel
feature downloads.

One home for the URI parsing `scripts/build_region_probe.py` and
`scripts/evaluate_region_probe.py` each carried their own copy of, now that
`region_eval/score.py` needs the same parsing to accept a probe directory
published to S3 alongside a local one; and for `download_features_parallel`,
which both the training pipeline (`pyspacer/dataset.py`) and the region-eval
feature cache (`region_eval/features.py`) use to fetch `.featurevector` files.
It lives here rather than in `pyspacer/` so a lightweight consumer can reach
it without importing that subpackage's training-only logging and settings
setup.
"""

import concurrent.futures
import hashlib
import logging
import os
from pathlib import Path
from typing import TypeGuard
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an s3://bucket/key URI into (bucket, key)."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"not an s3://bucket/key URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def is_s3_uri(value: Path | str) -> TypeGuard[str]:
    """True if `value` is an s3://bucket/key-shaped string.

    A `Path` never is: `Path("s3://bucket/key")` collapses the double slash
    to `s3:/bucket/key`, so this check must run before anything wraps the
    value in `Path`.
    """
    return isinstance(value, str) and value.startswith("s3://")


def sha256_of_uri(uri: str, chunk_size: int = 1 << 20) -> str:
    """Hex sha256 of the object at an s3://bucket/key URI or filesystem path.

    Routed through pyspacer's storage layer so the same call works over s3,
    filesystem and memory locations — the same property that lets the serving
    lane run without AWS.
    """
    # Deferred, as with _download below: parse_s3_uri/is_s3_uri callers stay
    # free of spacer's import cost.
    from spacer.storage import storage_factory

    if is_s3_uri(uri):
        bucket, key = parse_s3_uri(uri)
        storage = storage_factory("s3", bucket)
    else:
        key = uri
        storage = storage_factory("filesystem")
    # storage_factory and RemoteStorage._load_remote carry no return
    # annotations upstream, so both of these type as optional.
    stream = storage.load(key)  # pyright: ignore[reportOptionalMemberAccess]

    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):  # pyright: ignore[reportOptionalMemberAccess]
        digest.update(chunk)
    return digest.hexdigest()


def download_features_parallel(
    s3_keys: dict[tuple[str, str], str],
    max_workers: int = 50,
) -> set[tuple[str, str]]:
    """
    Download feature vectors from S3 in parallel.

    Args:
        s3_keys: Mapping of (bucket, key) → local_path for each file
            to download.
        max_workers: Number of concurrent download threads.

    Returns:
        Set of (bucket, key) tuples that failed to download.
    """
    # Deferred: only the nested _download below needs it, so a caller of
    # parse_s3_uri/is_s3_uri alone stays free of spacer's import cost.
    from spacer.aws import get_s3_resource

    total = len(s3_keys)
    if total == 0:
        return set()

    logger.info(f"Downloading {total} feature vectors with {max_workers} workers...")

    # Pre-create all unique parent directories.
    unique_dirs = {os.path.dirname(local_path) for local_path in s3_keys.values()}
    for d in unique_dirs:
        os.makedirs(d, exist_ok=True)

    failed: set[tuple[str, str]] = set()
    succeeded = 0

    def _download(item: tuple[tuple[str, str], str]) -> None:
        (bucket, key), local_path = item
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return
        s3 = get_s3_resource()
        part_path = local_path + ".part"
        s3.Object(bucket, key).download_file(part_path)  # pyright: ignore[reportAttributeAccessIssue]  # boto3 S3 resource is untyped
        os.rename(part_path, local_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download, item): item for item in s3_keys.items()}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            (bucket, key), _local_path = futures[future]
            try:
                future.result()
                succeeded += 1
            except Exception as e:
                failed.add((bucket, key))
                logger.warning(f"Failed to download s3://{bucket}/{key}: {e}")
            if i % 1000 == 0 or i == total:
                logger.info(
                    f"Download progress: {i}/{total} ({succeeded} ok, {len(failed)} failed)"
                )

    return failed
