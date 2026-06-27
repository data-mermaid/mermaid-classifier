"""Run a classifier training job locally from a committed config.

Loads a committed training config — the same ``training_config.yaml`` + sibling
CSVs the SageMaker path consumes — and runs it locally, so local and SageMaker
training share a single source of truth and cannot drift. This mirrors
``scripts/sagemaker_train_entrypoint.py``: load config → ``apply_env()`` →
``build_options()`` → run.

Run from the repo root with AWS SSO access to the training buckets:

    uv run python scripts/classifier_train.py
    uv run python scripts/classifier_train.py --config-dir sagemaker/configs/example

Defaults to the committed ``coralnet_top108_best`` config. Config dirs are
repo-root-relative (``sagemaker/configs/<name>/``). Local AWS credentials come
from the ``wcs`` SSO profile; the MLflow tracking server comes from the ``.env``
in the cwd (copy ``.env.example`` from the repo root to ``.env``); the chosen config's ``env`` block
supplies bucket names / weights location. For the SageMaker equivalent see
``scripts/launch_training.py`` + ``docs/training_at_scale.md``; for the overall
script order see ``docs/workflow.md``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "sagemaker" / "configs" / "coralnet_top108_best"
CONFIG_FILENAME = "training_config.yaml"


def _resolve_local_aws_credentials() -> None:
    """Resolve AWS SSO credentials into the SPACER_AWS_* env vars (local-dev only).

    On SageMaker an instance role supplies credentials; locally we read the
    ``wcs`` SSO profile and expose them under the ``SPACER_AWS_*`` names that
    pyspacer's ``Settings()`` reads. Must run before pyspacer is imported.
    """
    os.environ["AWS_PROFILE"] = "wcs"
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise RuntimeError(
            "Could not resolve AWS credentials for the 'wcs' profile. Run "
            "`aws sso login --profile wcs` (or set the SPACER_AWS_* env vars) "
            "before training locally."
        )
    creds = credentials.get_frozen_credentials()
    os.environ["SPACER_AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["SPACER_AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    if creds.token:
        os.environ["SPACER_AWS_SESSION_TOKEN"] = creds.token
    else:
        # Clear any stale token from a previous session — a mismatched token
        # breaks auth even with a valid key/secret.
        os.environ.pop("SPACER_AWS_SESSION_TOKEN", None)


def _resolve_runner_factory():
    """Return ``MLflowTrainingRunner`` (heavy import; factored out so tests can patch it)."""
    from mermaid_classifier.pyspacer.runner import MLflowTrainingRunner

    return MLflowTrainingRunner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a committed training config locally.")
    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help=(
            "Directory containing training_config.yaml and its sibling CSVs, "
            "repo-root-relative (e.g. sagemaker/configs/example). "
            f"Default: {DEFAULT_CONFIG_DIR.relative_to(REPO_ROOT)}."
        ),
    )
    args = parser.parse_args(argv)
    # Interpret a relative --config-dir against the repo root (the documented
    # convention), not the cwd, so it resolves correctly regardless of where the
    # script is invoked from. The default is already absolute.
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = REPO_ROOT / config_dir
    config_dir = config_dir.resolve()

    # Resolve local AWS creds before any pyspacer import (Settings() reads env
    # at import time).
    _resolve_local_aws_credentials()

    # Load the committed config and apply its env block BEFORE importing
    # pyspacer — same ordering as scripts/sagemaker_train_entrypoint.py.
    from mermaid_classifier.sagemaker.config import TrainingRunConfig

    config = TrainingRunConfig.from_yaml_path(config_dir / CONFIG_FILENAME)
    config.apply_env()

    dataset_options, training_options, mlflow_options = config.build_options(config_dir=config_dir)

    runner_class = _resolve_runner_factory()
    runner_class(
        dataset_options=dataset_options,
        training_options=training_options,
        mlflow_options=mlflow_options,
    ).run()


if __name__ == "__main__":
    main()
