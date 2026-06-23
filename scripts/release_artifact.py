"""Release a trained MLflow classifier as an immutable version vN.

Fetches the parity-proven portable artifact (model.pt + model.json) by MLflow
model ID, re-validates it (load + manifest checks), assembles the per-version
S3 layout (model.pt, model.json, efficientnet.pt), and prints the resulting
S3 URIs. The GitHub workflow wraps this with OIDC auth and `gh release create`.

Run: uv run python scripts/release_artifact.py --mlflow-model-id m-... --version vN
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

_VERSION_RE = re.compile(r'^v\d+$')


def validate_version(version: str) -> None:
    """Raise ValueError unless version matches ^v\\d+$ (e.g. v3)."""
    if not _VERSION_RE.fullmatch(version):
        raise ValueError(
            f"version must match ^v\\d+$ (e.g. v3); got {version!r}")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an s3://bucket/key URI into (bucket, key)."""
    parsed = urlparse(uri)
    if parsed.scheme != 's3' or not parsed.netloc or not parsed.path.strip('/'):
        raise ValueError(f"not an s3://bucket/key URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip('/')
