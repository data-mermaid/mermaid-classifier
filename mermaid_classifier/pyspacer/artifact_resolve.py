"""Resolve a classifier location to local (model.pt, model.json) paths.

Accepts an MLflow model ID (m-...), an S3 directory URI, or a local
directory.
"""

import atexit
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.artifacts
from spacer.data_classes import DataLocation
from spacer.storage import storage_factory

from mermaid_classifier.pyspacer.utils import mlflow_connect

# Accepts 30-32 hex digits (not just the full 32) so a copy-paste that drops a
# digit still matches as a run ID rather than silently falling through as a path.
MLFLOW_MODEL_ID_REGEX = re.compile(r"m-[a-f0-9]{30,32}")


def mlflow_model_id_to_artifact_uris(model_id: str) -> tuple[str, str]:

    time_taken = mlflow_connect()
    print(f"Time to connect to MLflow tracking: {time_taken}")

    logged_model = mlflow.get_logged_model(model_id)
    base = logged_model.artifact_location
    # log_artifact_model logs model.pt / model.json as pyfunc artifacts,
    # which MLflow stores under the model's artifacts/ subdirectory.
    return f"{base}/artifacts/model.pt", f"{base}/artifacts/model.json"


def _download_pair_to_tempdir(
    pt_loc: DataLocation,
    json_loc: DataLocation,
) -> tuple[Path, Path]:
    """Download an S3 model.pt + model.json pair into one temp dir."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="clf_artifact_"))
    # mkdtemp() isn't auto-cleaned, and the pair must outlive this call (the
    # caller loads them immediately after), so reclaim the dir at process exit.
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    paths = []
    for loc, name in [(pt_loc, "model.pt"), (json_loc, "model.json")]:
        storage = storage_factory(loc.storage_type, loc.bucket_name)
        stream = storage.load(loc.key)  # pyright: ignore[reportOptionalMemberAccess]  # storage_factory always returns Storage for valid storage types; None is unreachable
        dest = tmp_dir / name
        # getbuffer() writes the stream's bytes without an extra full copy.
        dest.write_bytes(stream.getbuffer())  # pyright: ignore[reportOptionalMemberAccess]  # storage.load returns BytesIO, not None
        paths.append(dest)
    return paths[0], paths[1]


def parse_location_str(location: str | None) -> DataLocation | None:

    if not location:
        return None

    # Un-proxies an mlflow-artifacts:/ URI by hand, since MLflow's own
    # un-proxying methods don't resolve it for us.
    uri = urlparse(location)
    if uri.scheme == "mlflow-artifacts":
        # So far we only handle the case where artifacts are stored in
        # the local filesystem cwd.
        # If other cases come up in practice, add to this code to handle
        # those cases.
        location = "mlartifacts" + uri.path

    try:
        # S3 URI
        # Example:
        # s3://my-bucket/my-folder/model.pkl
        uri = urlparse(location)
        if uri.scheme == "s3":
            return DataLocation(
                "s3",
                bucket_name=uri.netloc,
                # url.path probably begins with a slash, which isn't
                # what we want when ultimately passing this to boto's
                # Object().
                key=uri.path.strip("/"),
            )
    except ValueError:
        pass

    # Default to assuming a filesystem path
    return DataLocation("filesystem", key=location)


def resolve_classifier_artifact(location: str) -> tuple[Path, Path]:
    """Resolve a classifier location to local (model.pt, model.json) paths.

    Accepts an MLflow model ID (m-...), an S3 directory URI, or a local
    directory. The S3/filesystem forms point at the *directory* containing
    model.pt + model.json (migrated from the old single-pickle meaning).
    """
    if MLFLOW_MODEL_ID_REGEX.fullmatch(location):
        pt_uri, json_uri = mlflow_model_id_to_artifact_uris(location)
        local_pt = mlflow.artifacts.download_artifacts(artifact_uri=pt_uri)
        local_json = mlflow.artifacts.download_artifacts(artifact_uri=json_uri)
        return Path(local_pt), Path(local_json)

    # Directory mode: S3 URI or local filesystem directory.
    base = location.rstrip("/")
    pt_loc = parse_location_str(f"{base}/model.pt")
    json_loc = parse_location_str(f"{base}/model.json")
    assert pt_loc is not None and json_loc is not None  # guaranteed: non-empty strings passed

    if pt_loc.storage_type == "s3":
        return _download_pair_to_tempdir(pt_loc, json_loc)
    return Path(pt_loc.key), Path(json_loc.key)
