"""Launch a mermaid-classifier training run as a SageMaker TrainingJob.

Wraps the SageMaker Python SDK's Estimator. The user supplies a local
config directory containing training_config.yaml plus sibling CSVs.
The launcher:

  1. Validates the YAML against the Pydantic schema.
  2. Uploads the directory to a run-scoped S3 prefix.
  3. Builds an Estimator pointing at the ECR image.
  4. Calls fit(wait=True, logs="All") so CloudWatch logs stream live.

Example
-------
python scripts/launch_training_sagemaker.py \\
    --config-dir sagemaker/configs/my-run \\
    --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:ACCT:mlflow-app/APP \\
    --role-arn arn:aws:iam::ACCT:role/MermaidTrainer \\
    --ecr-image-uri ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-training:latest \\
    --staging-bucket my-staging-bucket
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# These imports are heavy; they happen at module load. The launcher
# always needs them (unlike the entrypoint, which sequences imports).
import boto3
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.session import Session


log = logging.getLogger("launch_training_sagemaker")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _validate_config_dir(config_dir: Path) -> None:
    if not config_dir.is_dir():
        log.error("Config dir does not exist: %s", config_dir)
        sys.exit(2)
    yaml_path = config_dir / "training_config.yaml"
    if not yaml_path.is_file():
        log.error("Missing training_config.yaml in %s", config_dir)
        sys.exit(2)

    # Schema validation. Don't suppress the Pydantic ValidationError --
    # its message includes the field path, which is exactly what the
    # user needs.
    from pydantic import ValidationError
    from mermaid_classifier.sagemaker.config import TrainingRunConfig
    try:
        config = TrainingRunConfig.from_yaml_path(yaml_path)
    except ValidationError as e:
        log.error("Invalid config YAML:\n%s", e)
        sys.exit(2)

    # Verify referenced CSVs exist next to the YAML.
    referenced = [
        config.dataset.coralnet_sources_csv,
        config.dataset.label_rollup_spec_csv,
        config.dataset.included_labels_csv,
        config.dataset.excluded_labels_csv,
    ]
    for filename in referenced:
        if filename is None:
            continue
        candidate = config_dir / filename
        if not candidate.is_file():
            log.error(
                "Referenced CSV missing: %s (expected at %s)",
                filename, candidate,
            )
            sys.exit(2)


def _make_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _upload_config_dir(
    *,
    config_dir: Path,
    staging_bucket: str,
    run_id: str,
    session: Session,
) -> str:
    """Upload the config dir to s3://<bucket>/runs/<run_id>/config/.

    Returns the S3 URI prefix (with trailing slash) the Estimator should
    mount as its 'config' channel.
    """
    key_prefix = f"runs/{run_id}/config"
    log.info(
        "Uploading %s to s3://%s/%s/",
        config_dir, staging_bucket, key_prefix,
    )
    session.upload_data(
        path=str(config_dir),
        bucket=staging_bucket,
        key_prefix=key_prefix,
    )
    return f"s3://{staging_bucket}/{key_prefix}/"


def _build_environment(args: argparse.Namespace) -> dict[str, str]:
    return {
        "MLFLOW_TRACKING_SERVER": args.mlflow_tracking_uri,
        "AWS_DEFAULT_REGION": args.region,
        "FEATURE_CACHE_DIR": "/tmp/feature_cache",
        "SPACER_EXTRACTORS_CACHE_DIR": "/tmp/spacer_extractors",
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
    }


def _print_dry_run(args, environment) -> None:
    print("=" * 60)
    print("DRY RUN -- not submitting")
    print("=" * 60)
    print(f"config_dir:           {args.config_dir}")
    print(f"staging_bucket:       {args.staging_bucket}")
    print(f"role_arn:             {args.role_arn}")
    print(f"ecr_image_uri:        {args.ecr_image_uri}")
    print(f"region:               {args.region}")
    print(f"instance_type:        {args.instance_type}")
    print(f"instance_count:       {args.instance_count}")
    print(f"volume_size_gb:       {args.volume_size_gb}")
    print(f"max_runtime_hours:    {args.max_runtime_hours}")
    print(f"use_spot:             {args.use_spot}")
    print(f"job_name_prefix:      {args.job_name_prefix}")
    print("environment:")
    for k, v in sorted(environment.items()):
        print(f"  {k}={v}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--ecr-image-uri", required=True)
    parser.add_argument("--staging-bucket", required=True,
                        help="S3 bucket for run-scoped config uploads "
                             "and SageMaker output artifacts.")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="ml.m5.4xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--volume-size-gb", type=int, default=200)
    parser.add_argument("--max-runtime-hours", type=int, default=24)
    parser.add_argument("--use-spot", action="store_true")
    parser.add_argument("--job-name-prefix",
                        default="mermaid-classifier")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate, print the planned config, "
                             "skip upload/submit.")
    args = parser.parse_args(argv)

    args.config_dir = args.config_dir.resolve()
    _validate_config_dir(args.config_dir)

    environment = _build_environment(args)

    if args.dry_run:
        _print_dry_run(args, environment)
        return

    boto_session = boto3.Session(region_name=args.region)
    sm_session = Session(boto_session=boto_session)

    run_id = _make_run_id(args.job_name_prefix)
    config_s3 = _upload_config_dir(
        config_dir=args.config_dir,
        staging_bucket=args.staging_bucket,
        run_id=run_id,
        session=sm_session,
    )

    output_path = f"s3://{args.staging_bucket}/runs/{run_id}/output/"
    cw_url = (
        f"https://{args.region}.console.aws.amazon.com/cloudwatch/home"
        f"?region={args.region}#logsV2:log-groups/log-group/"
        f"$252Faws$252Fsagemaker$252FTrainingJobs"
        f"/log-events/{run_id}"
    )
    log.info("Run ID:          %s", run_id)
    log.info("Output S3:       %s", output_path)
    log.info("CloudWatch:      %s", cw_url)
    log.info("MLflow:          %s", args.mlflow_tracking_uri)

    estimator_kwargs = dict(
        image_uri=args.ecr_image_uri,
        role=args.role_arn,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        volume_size=args.volume_size_gb,
        max_run=args.max_runtime_hours * 3600,
        output_path=output_path,
        environment=environment,
        sagemaker_session=sm_session,
        base_job_name=args.job_name_prefix,
    )
    if args.use_spot:
        estimator_kwargs["use_spot_instances"] = True
        estimator_kwargs["max_wait"] = (
            args.max_runtime_hours * 3600 + 3600)

    estimator = Estimator(**estimator_kwargs)

    inputs = {
        "config": TrainingInput(
            s3_data=config_s3,
            input_mode="File",
        ),
    }

    log.info("Submitting TrainingJob...")
    # logs="None" disables local CloudWatch log streaming. The launcher
    # IAM role doesn't have logs:DescribeLogStreams; tailing would crash
    # the launcher even though the job itself runs fine. wait=True still
    # blocks on completion via describe-training-job (which is permitted).
    estimator.fit(inputs=inputs, wait=True, logs="None",
                  job_name=run_id)
    log.info("Job %s reached terminal state.", run_id)
    log.info("CloudWatch logs: %s", cw_url)


if __name__ == "__main__":
    main()
