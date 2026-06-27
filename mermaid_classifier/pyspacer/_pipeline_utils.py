"""
Pipeline utility helpers shared across the training pipeline modules.

- section_profiling: context manager that records timing and memory usage
  for a named section of code.
- download_features_parallel: parallel S3 feature-vector downloader.
"""

import concurrent.futures
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta

import psutil
from spacer.aws import get_s3_resource

from mermaid_classifier.pyspacer.utils import logging_config_for_script

logger = logging_config_for_script("train")


@contextmanager
def section_profiling(profiled_sections: list[dict[str, object]], section_name: str):
    """
    Performance-profile a wrapped section of code and save the stats
    (time, memory) as part of the passed structure.
    """
    approx_start_date = datetime.now()
    # This is more accurate, but doesn't have time-of-day info.
    start_time = time.perf_counter()

    yield

    seconds_elapsed = time.perf_counter() - start_time
    section_profile: dict[str, object] = {
        # Name for this section of code.
        "name": section_name,
        # Number of seconds.
        "seconds": format(seconds_elapsed, ".1f"),
        # Hours, minutes, seconds, ns.
        "hms": str(timedelta(seconds=seconds_elapsed)),
        # Date and time, to see if the sections we've chosen skip any
        # substantial time blocks that we should also be monitoring.
        "approx_start": approx_start_date.strftime("%b %d %H:%M:%S"),
        "memory_usage_at_end": f"{psutil.virtual_memory().percent}%",
    }
    profiled_sections.append(section_profile)

    logger.debug(
        f"{section_name} -"
        f" Elapsed time = {section_profile['hms']},"
        f" Memory usage at end = {section_profile['memory_usage_at_end']}"
    )


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
