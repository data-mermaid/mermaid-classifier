"""Release a trained MLflow classifier as an immutable version vN.

Fetches the parity-proven portable artifact (model.pt + model.json) by MLflow
model ID, re-validates it (load + manifest checks), assembles the per-version
S3 layout (model.pt, model.json, efficientnet.pt), and prints the resulting
S3 URIs. The GitHub workflow wraps this with OIDC auth and `gh release create`.

Run: uv run python scripts/release_artifact.py --mlflow-model-id m-... --version vN
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from botocore.exceptions import ClientError

from mermaid_classifier.pyspacer.inference import (
    SCHEMA_VERSION, TASK_NAME, load_predictor,
)

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


def validate_artifact(model_pt: Path, model_json: Path) -> dict:
    """Re-validate the artifact at release time (the release "parity gate").

    load_predictor raises ManifestError on schema_version / input_dim /
    class-count mismatch. We then add the release-only checks load_predictor
    does not make: task identity, non-empty classes, and provenance presence.
    Returns the parsed manifest.
    """
    load_predictor(model_pt, model_json)  # ManifestError on graph/manifest skew
    manifest = json.loads(Path(model_json).read_text())

    if manifest.get('task') != TASK_NAME:
        raise ValueError(
            f"manifest task={manifest.get('task')!r} != {TASK_NAME!r}")
    if not manifest.get('classes'):
        raise ValueError("manifest has empty/missing 'classes'")
    if not manifest.get('trained_with'):
        raise ValueError("manifest missing 'trained_with' provenance")
    # SCHEMA_VERSION is enforced by load_predictor; assert it stays referenced
    # so a future loader change that drops the check is caught here too.
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version={manifest.get('schema_version')!r}"
            f" != {SCHEMA_VERSION}")
    return manifest


_NOT_FOUND_CODES = {'404', 'NoSuchKey', 'NotFound'}


def s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    """True if the object exists; False on 404/NotFound; re-raise otherwise."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response.get('Error', {}).get('Code') in _NOT_FOUND_CODES:
            return False
        raise


def assemble_s3_layout(
    s3_client,
    *,
    dest_bucket: str,
    dest_prefix: str,
    version: str,
    model_pt: Path,
    model_json: Path,
    weights_uri: str,
) -> dict[str, str]:
    """Upload model.pt + model.json and copy the extractor weights into
    s3://<dest_bucket>/<dest_prefix>/<version>/. Returns {name: s3_uri}."""
    base_key = f"{dest_prefix}/{version}"

    def _uri(name: str) -> str:
        return f"s3://{dest_bucket}/{base_key}/{name}"

    for path, name in [(model_pt, 'model.pt'), (model_json, 'model.json')]:
        s3_client.upload_file(str(path), dest_bucket, f"{base_key}/{name}")

    src_bucket, src_key = parse_s3_uri(weights_uri)
    s3_client.copy_object(
        Bucket=dest_bucket,
        Key=f"{base_key}/efficientnet.pt",
        CopySource={'Bucket': src_bucket, 'Key': src_key},
    )

    return {name: _uri(name)
            for name in ('model.pt', 'model.json', 'efficientnet.pt')}
