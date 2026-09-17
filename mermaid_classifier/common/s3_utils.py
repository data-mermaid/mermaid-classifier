"""Shared s3://bucket/key URI helpers.

One home for the parsing `scripts/build_region_probe.py` and
`scripts/evaluate_region_probe.py` each carried their own copy of, now that
`region_eval/score.py` needs the same parsing to accept a probe directory
published to S3 alongside a local one.
"""

from pathlib import Path
from typing import TypeGuard
from urllib.parse import urlparse


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
