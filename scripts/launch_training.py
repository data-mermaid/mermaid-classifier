"""Launch a mermaid-classifier training run as a SageMaker TrainingJob.

Reads a per-run YAML and submits CreateTrainingJob via sagemaker-core's
TrainingJob. The script is script-agnostic: it doesn't know what
the named `job.entrypoint` does. See
`mermaid-api/iac/sagemaker-launcher-convention.md` for the schema and
canonical ARNs.

Example
-------
    uv run python scripts/launch_training.py \\
        --run-config sagemaker/runs/my-run.yaml \\
        --config-dir sagemaker/configs/my-run/ \\
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-2OMU4VP53ZS2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from sagemaker.core.resources import Tag, TrainingJob
from sagemaker.core.shapes import (
    AlgorithmSpecification,
    Channel,
    DataSource,
    OutputDataConfig,
    ResourceConfig,
    S3DataSource,
    StoppingCondition,
)

from mermaid_classifier.sagemaker.launcher_config import parse_run_config

# Canonical ARNs (see convention doc). Hardcoded by intent; the
# launcher is bound to the dev account.
ACCOUNT = "554812291621"
REGION = "us-east-1"
EXEC_ROLE = f"arn:aws:iam::{ACCOUNT}:role/dev-sm-execution-role"
STAGING_BUCKET = "dev-datamermaid-sm-data"
ECR_HOST = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"
KNOWN_SHORT_REPOS = {"mermaid-classifier-jobs"}  # this repo's ECR

log = logging.getLogger("launch_training")


def expand_image_uri(image: str) -> str:
    """Expand a short-form `<repo>:<tag>` into a full ECR URI.

    Full URIs (containing `.dkr.ecr.`) pass through unchanged.
    Short forms must reference this repo's known ECR; otherwise raises.
    """
    if ".dkr.ecr." in image:
        return image
    if ":" not in image:
        raise ValueError(f"Image {image!r} must be `<repo>:<tag>` or a full ECR URI.")
    repo, _ = image.split(":", 1)
    if repo not in KNOWN_SHORT_REPOS:
        raise ValueError(
            f"Unknown short-form repo {repo!r}. Known: {sorted(KNOWN_SHORT_REPOS)}. "
            f"Use a full ECR URI to override."
        )
    return f"{ECR_HOST}/{image}"


def make_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _upload_config_dir(config_dir: Path, run_id: str, s3_client: Any) -> str:
    key_prefix = f"runs/{run_id}/config"
    for p in sorted(config_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(config_dir)
        s3_client.upload_file(str(p), STAGING_BUCKET, f"{key_prefix}/{rel}")
    return f"s3://{STAGING_BUCKET}/{key_prefix}/"


def _s3_channel(name: str, s3_uri: str, input_mode: str) -> Channel:
    return Channel(
        channel_name=name,
        input_mode=input_mode,
        data_source=DataSource(
            s3_data_source=S3DataSource(
                s3_data_type="S3Prefix",
                s3_uri=s3_uri,
                s3_data_distribution_type="FullyReplicated",
            )
        ),
    )


def build_training_job_kwargs(
    *,
    cfg: Any,
    run_id: str,
    staging_bucket: str,
    mlflow_uri: str,
) -> dict[str, Any]:
    job = cfg.job
    env = {
        **job.env,
        # Launcher-owned: sits after job.env so a YAML env block cannot
        # redirect these. CONTAINER_ENTRYPOINT_SCRIPT is consumed by the
        # container entrypoint shim (docker/jobs/training-entrypoint.sh) to
        # dispatch to the named script (falls back to its default if unset).
        "MLFLOW_TRACKING_SERVER": mlflow_uri,
        "AWS_DEFAULT_REGION": REGION,
        "CONTAINER_ENTRYPOINT_SCRIPT": job.entrypoint,
    }
    max_runtime = job.max_runtime_hours * 3600
    stopping = {"max_runtime_in_seconds": max_runtime}
    if job.use_spot:
        stopping["max_wait_time_in_seconds"] = max_runtime + 3600

    channels = [_s3_channel("config", f"s3://{staging_bucket}/runs/{run_id}/config/", "File")]
    if cfg.training:
        channels += [
            _s3_channel(name, ch.s3_uri, ch.input_mode)
            for name, ch in cfg.training.channels.items()
        ]

    kwargs: dict[str, Any] = {
        "training_job_name": run_id,
        "role_arn": EXEC_ROLE,
        "algorithm_specification": AlgorithmSpecification(
            training_image=expand_image_uri(job.image),
            training_input_mode="File",
        ),
        "resource_config": ResourceConfig(
            instance_type=job.instance_type,
            instance_count=job.instance_count,
            volume_size_in_gb=job.volume_gb,
        ),
        "stopping_condition": StoppingCondition(**stopping),
        "output_data_config": OutputDataConfig(
            s3_output_path=f"s3://{staging_bucket}/runs/{run_id}/output/"
        ),
        "input_data_config": channels,
        "environment": env,
        "tags": [Tag(key=k, value=v) for k, v in job.tags.items()],
    }
    if job.use_spot:
        kwargs["enable_managed_spot_training"] = True
    return kwargs


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _cloudwatch_url(run_id: str) -> str:
    return (
        f"https://{REGION}.console.aws.amazon.com/cloudwatch/home"
        f"?region={REGION}#logsV2:log-groups/log-group/"
        f"$252Faws$252Fsagemaker$252FTrainingJobs"
        f"/log-events/{run_id}"
    )


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = parse_run_config(args.run_config.read_text(), kind="training", strict=False)

    if not args.config_dir.is_dir():
        log.error("Config dir does not exist: %s", args.config_dir)
        sys.exit(2)

    run_id = make_run_id(cfg.job.name_prefix)
    cw_url = _cloudwatch_url(run_id)

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN -- not submitting")
        print("=" * 60)
        print(f"run_id:           {run_id}")
        print(f"image:            {expand_image_uri(cfg.job.image)}")
        print(f"entrypoint:       {cfg.job.entrypoint}")
        print(f"instance:         {cfg.job.instance_type} (x{cfg.job.instance_count})")
        print(f"volume_gb:        {cfg.job.volume_gb}")
        print(f"max_runtime:      {cfg.job.max_runtime_hours}h")
        print(f"output:           s3://{STAGING_BUCKET}/runs/{run_id}/output/")
        print(f"CloudWatch:       {cw_url}")
        return

    boto_session = boto3.Session(region_name=REGION)

    _upload_config_dir(args.config_dir.resolve(), run_id, boto_session.client("s3"))

    kwargs = build_training_job_kwargs(
        cfg=cfg,
        run_id=run_id,
        staging_bucket=STAGING_BUCKET,
        mlflow_uri=args.mlflow_tracking_uri,
    )

    log.info("Run ID:       %s", run_id)
    log.info("CloudWatch:   %s", cw_url)
    log.info("MLflow:       %s", args.mlflow_tracking_uri)

    log.info("Submitting TrainingJob...")
    job = TrainingJob.create(**kwargs, session=boto_session, region=REGION)
    if job is None:
        raise RuntimeError(f"TrainingJob.create returned no job for {run_id}")
    # sagemaker-core's log client ignores the passed session and reads the ambient region.
    os.environ["AWS_DEFAULT_REGION"] = REGION
    job.wait(logs=True)
    log.info("Job %s reached terminal state.", run_id)
    log.info("CloudWatch: %s", cw_url)


if __name__ == "__main__":
    main()
