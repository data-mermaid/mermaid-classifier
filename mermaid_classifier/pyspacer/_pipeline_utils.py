"""
Pipeline utility helpers shared across the training pipeline modules.

- section_profiling: context manager that records timing and memory usage
  for a named section of code.

The parallel S3 feature-vector downloader lives in
``mermaid_classifier.common.s3_utils`` instead: it has no need for the
training-only logging setup below, and a lightweight consumer (region-eval)
must be able to reach it without pulling that setup in as an import-time side
effect.
"""

import time
from contextlib import contextmanager
from datetime import datetime, timedelta

import psutil

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
