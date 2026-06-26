# Feature extraction at scale (SageMaker ProcessingJobs)

`scripts/build_feature_bucket.py` is GPU-bound and single-process: it
walks N CoralNet sources sequentially. To run in parallel, use
`scripts/launch_processing.py` with `processing.shard:` set — it
submits N parallel SageMaker ProcessingJobs, each handling a sharded
subset of source IDs.

> The launcher convention (role ARN, bucket, schema, ECR tagging) is
> defined in
> [`mermaid-api/iac/sagemaker-launcher-convention.md`](https://github.com/data-mermaid/mermaid-api/blob/dev/iac/sagemaker-launcher-convention.md).
> This doc is the classifier-specific runbook on top of that.

## Prerequisites

- AWS SSO setup with the `SageMaker` Identity Center permission set
  on account `554812291621` (see convention doc for `~/.aws/config`
  and the role-chain pattern).
- Docker installed locally.
- `uv` installed locally.

The IAM role, ECR repo, S3 staging bucket, and MLflow App are all
provisioned by IaC and already exist in the dev account; nothing
manual is required on the AWS side.

## Concrete values

| What | Value |
|---|---|
| AWS account | `554812291621` (dev) |
| Region | `us-east-1` |
| ECR repo | `554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-jobs` (use the `:features-*` tag prefix) |
| Launcher IAM role | `arn:aws:iam::554812291621:role/dev-mermaid-sagemaker-launcher-role` |
| Job execution role | `arn:aws:iam::554812291621:role/dev-sm-execution-role` |
| Staging S3 bucket | `dev-datamermaid-sm-data` (runs land under `runs/<run-id>/`) |

## One-time: build and push the features image

```bash
export AWS_PROFILE=wcs-launcher
ACCT=554812291621
IMG=$ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-jobs

aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin \
        $ACCT.dkr.ecr.us-east-1.amazonaws.com

docker buildx build --platform linux/amd64 \
    -t $IMG:features-latest \
    -f docker/jobs/features.Dockerfile .

docker push $IMG:features-latest
```

Smoke-check locally before pushing if you've changed anything:

```bash
bash docker/jobs/local_smoke.sh features
```

This builds the image and verifies the worker script is reachable
inside the container. Note the features image is ~6-7 GB (PyTorch +
CUDA base layers), so the first push uploads everything; subsequent
pushes ship only the changed `pip install` layer.

> **macOS + Rancher Desktop gotcha:** If `docker push` reports
> `denied: requested access to the resource is denied` even after a
> successful `docker login`, see the same workaround in
> [`training_at_scale.md`](training_at_scale.md#3-build-and-push-the-image)
> — add a `credHelpers` entry to `~/.docker/config.json` so docker
> uses the `ecr-login` helper instead of the cached keychain entry.

## Author a run config + config dir

The launcher reads:

- **`--run-config`**: a YAML in `sagemaker/runs/` describing the
  SageMaker job shape AND the shard plan (`job:` + `processing:`
  blocks).
- **`--config-dir`**: a directory in `sagemaker/configs/` whose
  contents are uploaded to S3 and contain the `sources.csv` (or
  whichever CSV your YAML's `processing.shard.items_from` names).

The example `sagemaker/runs/example-processing.yaml` is a starting
point:

```yaml
job:
  name_prefix: mermaid-features-example
  image: mermaid-classifier-jobs:features-latest
  entrypoint: scripts/build_feature_bucket.py
  instance_type: ml.g5.xlarge
  volume_gb: 100
  max_runtime_hours: 4

processing:
  container_args:
    - --target-bucket=2605-coralnet-public-sources
    - --weights=s3://2605-coralnet-public-sources/efficientnet_weights.pt
    - --device=cuda
    - --no-aws-bootstrap
    - --skip-existing
  shard:
    items_from: sources.csv
    workers: 4
    per_worker_arg: --source-ids
```

Copy and edit for your run:

```bash
cp sagemaker/runs/example-processing.yaml sagemaker/runs/my-extract.yaml
cp -r sagemaker/configs/example sagemaker/configs/my-extract
# Replace sagemaker/configs/my-extract/sources.csv with the real list.
# Edit sagemaker/runs/my-extract.yaml: tune instance_type, workers,
# max_runtime_hours.
```

## Dry-run the launcher

```bash
export AWS_PROFILE=wcs-launcher
uv run python scripts/launch_processing.py \
    --run-config sagemaker/runs/my-extract.yaml \
    --config-dir sagemaker/configs/my-extract/ \
    --dry-run
```

Expected: a `DRY RUN` block listing the run ID, worker count, image
URI, and the resolved per-worker `container_args` (each chunk's IDs
appended after `--source-ids`).

## Submit a sharded extraction

```bash
uv run python scripts/launch_processing.py \
    --run-config sagemaker/runs/my-extract.yaml \
    --config-dir sagemaker/configs/my-extract/
```

The launcher prints CloudWatch URLs and polls until all workers
complete. Pass `--no-wait` to fire-and-forget.

## Tuning

| Instance | GPU | $/hr | Notes |
|---|---|---|---|
| `ml.g4dn.xlarge` | T4 (16GB) | ~$0.74 | Cheapest GPU. Good for small/medium sources. |
| `ml.g5.xlarge` | A10G (24GB) | ~$1.41 | Default. ~2x faster than g4dn per dollar. |
| `ml.g5.2xlarge` | A10G (24GB), more CPU | ~$1.52 | Use if I/O is the bottleneck, not GPU. |

The `processing.shard.workers` setting trades parallelism for
concurrent-job quota. The default SageMaker ProcessingJob quota in a
new account is 4-8 simultaneous jobs per instance type; if you submit
more, the extras queue. Request a quota increase before going past 16.

## Debug a failed worker

CloudWatch log group: `/aws/sagemaker/ProcessingJobs`. Each worker's
log stream is named after its `ProcessingJobName`
(`<run-id>-<worker-idx>`). The launcher role grants
`logs:DescribeLogStreams / GetLogEvents / FilterLogEvents /
StartLiveTail` on `/aws/sagemaker/*`, so tailing works directly from
the `wcs-launcher` profile:

```bash
export AWS_PROFILE=wcs-launcher
aws logs tail /aws/sagemaker/ProcessingJobs \
    --log-stream-name-prefix <run-id>-0/ \
    --follow
```

If a worker hits the `job.max_runtime_hours` cap with
`--skip-existing` on, just re-run the launcher — completed sources
are skipped on the next pass (the target bucket is the source of
truth for resumability).
