"""Fan out build_feature_bucket.py across N SageMaker Processing Jobs.

The companion script ``scripts/build_feature_bucket.py`` is GPU-bound and
single-process: it iterates sources sequentially. To process the full
192-source CoralNet set in hours rather than days, this launcher submits
N parallel SageMaker Processing Jobs, each running the same container
image on its own GPU instance against a sharded subset of source IDs.

There is no persistent infrastructure: only an ECR repo (holding the
image) and an IAM role exist between runs. Each Processing Job spins up
a GPU instance, runs the container, and self-terminates -- nothing to
clean up afterward.

Resumability matches the underlying script: ``--skip-existing`` is
always passed to each worker, so re-running the launcher after a partial
failure picks up exactly where it left off (the target bucket is the
source of truth).

Example
-------
    uv run python scripts/launch_feature_extraction_sagemaker.py \\
        --sources-csv "/path/to/CoralNetSourcesFirst192.csv" \\
        --target-bucket <target-feature-bucket> \\
        --weights-s3-uri s3://<target-feature-bucket>/efficientnet_weights.pt \\
        --ecr-image <compute-account-id>.dkr.ecr.us-east-1.amazonaws.com/mermaid-features:latest \\
        --role-arn arn:aws:iam::<compute-account-id>:role/MermaidFeatureExtractionRole \\
        --workers 16

Set up steps (ECR repo, image build/push, IAM role) live in
``docs/sagemaker_setup.md``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

# Reuse the CSV parser from the worker script so source-ID semantics
# stay identical across launcher and worker.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import build_feature_bucket as bfb  # noqa: E402

logger = logging.getLogger('launch_feature_extraction')

DEFAULT_INSTANCE_TYPE = 'ml.g5.xlarge'
DEFAULT_VOLUME_GB = 100
DEFAULT_MAX_RUNTIME_S = 12 * 3600
DEFAULT_POLL_INTERVAL_S = 60
DEFAULT_WORKERS = 16
DEFAULT_REGION = 'us-east-1'

TERMINAL_STATES = {'Completed', 'Failed', 'Stopped'}

# Rough on-demand SageMaker Processing pricing (us-east-1, 2026-05).
# Used only for --dry-run cost estimation; not load-bearing.
INSTANCE_HOURLY_USD = {
    'ml.g5.xlarge': 1.408,
    'ml.g5.2xlarge': 1.515,
    'ml.g4dn.xlarge': 0.7364,
    'ml.g4dn.2xlarge': 0.94,
}


# ---- chunking -------------------------------------------------------


def chunk_sources(source_ids: list[str], n_workers: int) -> list[list[str]]:
    """Round-robin source IDs (sorted numerically) into ``n_workers`` chunks.

    Equal source-count chunks. Image-count variance across sources averages
    out at ~12 sources/worker. Returns at most ``n_workers`` chunks; if
    there are fewer sources than workers, empty chunks are dropped.
    """
    if n_workers <= 0:
        raise ValueError(f'n_workers must be positive, got {n_workers}')
    chunks: list[list[str]] = [[] for _ in range(n_workers)]
    if not source_ids:
        return chunks
    # Sort by integer when possible; non-numeric IDs sort lexically after.
    def _sort_key(s: str):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)
    for i, sid in enumerate(sorted(source_ids, key=_sort_key)):
        chunks[i % n_workers].append(sid)
    return [c for c in chunks if c]


# ---- source-ID loading ----------------------------------------------


def load_source_ids(args: argparse.Namespace) -> list[str]:
    """Resolve source IDs from --sources-csv or --source-ids."""
    if args.sources_csv:
        return bfb.load_source_ids_from_csv(
            args.sources_csv, override=args.source_id_column)
    if args.source_ids:
        return [s.strip() for s in args.source_ids.split(',') if s.strip()]
    raise ValueError(
        'Must pass either --sources-csv or --source-ids.')


# ---- job-name + run-id generation -----------------------------------


def make_run_id() -> str:
    """Compact UTC ISO timestamp, e.g. 20260514T000000Z."""
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def make_job_name(run_id: str, worker_idx: int) -> str:
    """SageMaker job name. Must match [a-zA-Z0-9-]+ and be unique per run."""
    return f'mermaid-features-{run_id}-{worker_idx}'


# ---- request shaping ------------------------------------------------


def build_processing_job_request(
    *,
    job_name: str,
    worker_idx: int,
    source_ids: list[str],
    target_bucket: str,
    weights_uri: str,
    ecr_image: str,
    role_arn: str,
    instance_type: str,
    volume_gb: int,
    max_runtime_s: int,
    run_id: str,
) -> dict:
    """Return kwargs for boto3 ``sagemaker.create_processing_job``."""
    container_args = [
        '--target-bucket', target_bucket,
        '--source-ids', ','.join(source_ids),
        '--weights', weights_uri,
        '--device', 'cuda',
        # Inside the container the SageMaker task role provides AWS
        # credentials directly; skip the local SSO bootstrap path.
        # (SageMaker rejects empty strings in ContainerArguments, so a
        # dedicated flag is used instead of --aws-profile "".)
        '--no-aws-bootstrap',
        '--skip-existing',
    ]
    return {
        'ProcessingJobName': job_name,
        'RoleArn': role_arn,
        'AppSpecification': {
            'ImageUri': ecr_image,
            'ContainerArguments': container_args,
        },
        'ProcessingResources': {
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': instance_type,
                'VolumeSizeInGB': volume_gb,
            },
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': max_runtime_s,
        },
        'Tags': [
            {'Key': 'Project', 'Value': 'mermaid-features'},
            {'Key': 'RunId', 'Value': run_id},
            {'Key': 'WorkerIdx', 'Value': str(worker_idx)},
        ],
    }


# ---- submission + polling -------------------------------------------


def submit_jobs(client, requests: list[dict]) -> list[str]:
    """Submit all requests sequentially. Return job names; raise on any error.

    Submission is fast (milliseconds per call), so sequential is cheap.
    Parallel submission used to hammer the SageMaker CreateProcessingJob
    rate limit (~2 req/s burst); the boto3 client is also configured
    with adaptive retries (see :func:`make_sagemaker_client`) for
    defense in depth.
    """
    job_names: list[str] = []
    for req in requests:
        client.create_processing_job(**req)
        job_names.append(req['ProcessingJobName'])
    return job_names


def make_sagemaker_client(region: str):
    """Build a boto3 SageMaker client with adaptive retry tuned for fan-out."""
    from botocore.config import Config
    cfg = Config(
        region_name=region,
        retries={'mode': 'adaptive', 'max_attempts': 10},
    )
    return boto3.client('sagemaker', config=cfg)


def wait_for_completion(
    client,
    job_names: list[str],
    *,
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
) -> dict[str, str]:
    """Poll until every job is in a terminal state. Return name -> status."""
    status: dict[str, str] = {n: 'Pending' for n in job_names}
    while True:
        for name in job_names:
            if status[name] in TERMINAL_STATES:
                continue
            resp = client.describe_processing_job(ProcessingJobName=name)
            status[name] = resp['ProcessingJobStatus']
        unfinished = [n for n, s in status.items() if s not in TERMINAL_STATES]
        if not unfinished:
            return status
        logger.info(
            'Polling: %d/%d still running; sleeping %ds',
            len(unfinished), len(job_names), poll_interval_s)
        time.sleep(poll_interval_s)


# ---- cost estimation ------------------------------------------------


def estimate_cost(
    n_workers: int, instance_type: str, est_hours: float = 3.0,
) -> float:
    hourly = INSTANCE_HOURLY_USD.get(instance_type, 0.0)
    return n_workers * hourly * est_hours


# ---- CLI ------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split('\n\n', 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src_group = p.add_mutually_exclusive_group(required=True)
    src_group.add_argument('--sources-csv', type=Path,
                           help='CSV with a source-ID column.')
    src_group.add_argument('--source-ids',
                           help='Comma-separated source IDs (testing).')
    p.add_argument('--source-id-column',
                   help='Override CSV column name; default auto-detect.')
    p.add_argument('--target-bucket', required=True,
                   help='Destination bucket for feature vectors.')
    p.add_argument('--weights-s3-uri', required=True,
                   help='s3://... URI for EfficientNet weights (.pt).')
    p.add_argument('--ecr-image', required=True,
                   help='ECR image URI for the worker container.')
    p.add_argument('--role-arn', required=True,
                   help='IAM role ARN for the Processing Job (read source/'
                        'weights buckets, write target bucket, CloudWatch).')
    p.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                   help=f'Number of parallel jobs (default {DEFAULT_WORKERS}).')
    p.add_argument('--instance-type', default=DEFAULT_INSTANCE_TYPE,
                   help=f'SageMaker instance type (default {DEFAULT_INSTANCE_TYPE}).')
    p.add_argument('--volume-gb', type=int, default=DEFAULT_VOLUME_GB,
                   help='EBS volume size per instance (GB).')
    p.add_argument('--max-runtime-s', type=int, default=DEFAULT_MAX_RUNTIME_S,
                   help='Per-job runtime cap (safety net).')
    p.add_argument('--poll-interval-s', type=int, default=DEFAULT_POLL_INTERVAL_S,
                   help='Status polling interval.')
    p.add_argument('--region', default=DEFAULT_REGION)
    p.add_argument('--aws-profile', default='wcs',
                   help='Local AWS profile (for submitting jobs); '
                        'unrelated to in-container credentials.')
    p.add_argument('--dry-run', action='store_true',
                   help='Print planned chunking + cost; do not submit.')
    p.add_argument('--log-level', default='INFO')
    return p


def _print_plan(
    source_ids: list[str],
    chunks: list[list[str]],
    args: argparse.Namespace,
) -> None:
    cost = estimate_cost(
        n_workers=len(chunks), instance_type=args.instance_type)
    lines = [
        f'Planned run: {len(source_ids)} sources across {len(chunks)} worker(s)',
        f'  Instance type:  {args.instance_type}',
        f'  Target bucket:  {args.target_bucket}',
        f'  ECR image:      {args.ecr_image}',
        f'  Weights:        {args.weights_s3_uri}',
        f'  Max runtime:    {args.max_runtime_s}s',
        '',
        'Per-worker chunks:',
    ]
    for i, chunk in enumerate(chunks):
        sample = ','.join(chunk[:5])
        if len(chunk) > 5:
            sample += f',... ({len(chunk)} total)'
        lines.append(f'  worker {i:>2}: {len(chunk):>3} sources [{sample}]')
    lines.append('')
    if cost > 0:
        lines.append(
            f'Estimated cost (assumes ~3h/worker): ${cost:.2f} USD '
            f'(on-demand, {args.instance_type})')
    else:
        lines.append(
            f'Estimated cost: unknown (no pricing for {args.instance_type}).')
    sys.stdout.write('\n'.join(lines) + '\n')


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    source_ids = load_source_ids(args)
    if not source_ids:
        logger.error('No source IDs to process.')
        return 2

    chunks = chunk_sources(source_ids, n_workers=args.workers)
    if args.dry_run:
        _print_plan(source_ids, chunks, args)
        return 0

    # Local creds (the launcher itself talks to SageMaker via the user's
    # profile; in-container creds come from the task role and are
    # unrelated).
    if args.aws_profile:
        os.environ['AWS_PROFILE'] = args.aws_profile
    client = make_sagemaker_client(args.region)

    run_id = make_run_id()
    requests = [
        build_processing_job_request(
            job_name=make_job_name(run_id, i),
            worker_idx=i,
            source_ids=chunk,
            target_bucket=args.target_bucket,
            weights_uri=args.weights_s3_uri,
            ecr_image=args.ecr_image,
            role_arn=args.role_arn,
            instance_type=args.instance_type,
            volume_gb=args.volume_gb,
            max_runtime_s=args.max_runtime_s,
            run_id=run_id,
        )
        for i, chunk in enumerate(chunks)
    ]

    logger.info(
        'Submitting %d Processing Job(s); run id %s', len(requests), run_id)
    job_names = submit_jobs(client, requests)
    logger.info('Submitted: %s', ', '.join(job_names))

    status = wait_for_completion(
        client, job_names, poll_interval_s=args.poll_interval_s)

    # Summary.
    by_status: dict[str, list[str]] = {}
    for name, s in status.items():
        by_status.setdefault(s, []).append(name)
    sys.stdout.write(f'\nRun {run_id} complete:\n')
    for s in sorted(by_status):
        sys.stdout.write(f'  {s}: {len(by_status[s])}\n')
        for name in by_status[s]:
            sys.stdout.write(f'    - {name}\n')

    failed_or_stopped = (
        len(by_status.get('Failed', [])) + len(by_status.get('Stopped', []))
    )
    if failed_or_stopped:
        sys.stdout.write(
            f'\n{failed_or_stopped} job(s) did not complete cleanly. '
            f'Check CloudWatch Logs (group: /aws/sagemaker/ProcessingJobs) '
            f'and re-run -- --skip-existing makes the launcher idempotent.\n')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
