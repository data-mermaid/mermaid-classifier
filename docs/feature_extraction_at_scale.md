# Feature extraction at scale (SageMaker ProcessingJobs)

Feature extraction is the **only GPU-accelerated step** in the pipeline: it runs
the fixed pretrained EfficientNet backbone over images to produce the feature
vectors that training consumes. The classifier trained on those vectors — and
inference with it — runs entirely on CPU (see [training_at_scale.md](training_at_scale.md)).

`scripts/build_feature_set.py` is GPU-bound and runs on a single
machine: it extracts the images of one pinned dataset version (`segdata-spec`,
bucket `mermaid-ic-datasets`) with threaded S3 I/O (`--max-io-workers`) but a
single process for the GPU forward pass. It writes
`features/<store>/<family>/<image_id>.featurevector`, claiming the family's
extractor first and skipping files that exist. It publishes no index:
`segdata features attach` (segdata-etl) adopts the family files and publishes
the feature-set index. To parallelize across machines, use
`scripts/launch_processing.py` with `processing.shard:` set — it submits N
parallel SageMaker ProcessingJobs. Each worker handles a disjoint share of the
images (`--shard i/n`, split by a stable hash of the image id).

Training reads these families through the feature-set pins in its config's
`datasets:` entries.

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
inside the container. The features image is Python 3.12 on an NVIDIA CUDA 12.8
base with cu128 torch, installed by
`uv sync --frozen --extra training --extra cuda --no-group cpu`. It is several
GB, so the first push uploads everything; subsequent pushes ship only the
changed layers. Every other sync in this repo keeps CPU torch through the
default `cpu` dependency group.

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
  contents are uploaded to S3. It holds a CSV only when your YAML's
  `processing.shard.items_from` names one; `--shard i/n` runs need none.

The example `sagemaker/runs/example-processing.yaml` is a starting
point:

```yaml
job:
  name_prefix: mermaid-features-example
  image: mermaid-classifier-jobs:features-latest
  entrypoint: scripts/build_feature_set.py
  instance_type: ml.g5.xlarge
  volume_gb: 100
  max_runtime_hours: 4

processing:
  container_args:
    - --dataset=coralnet
    - --version=20261124-cfc389d
    - --manifest-sha256=<manifest sha256 of the pinned version>
    - --family=efficientnet-b0-v1
    - --cache-dir=/opt/ml/processing/segdata-cache
    - --weights=s3://2605-coralnet-public-sources/efficientnet_weights.pt
    - --device=cuda
    - --no-aws-bootstrap
  shard:
    workers: 4
    per_worker_arg: --shard
```

Without `shard.items_from`, worker `i` of N receives `--shard i/N`. With it,
the launcher splits the CSV's items round-robin and appends each worker's
comma-joined chunk after `per_worker_arg`.

Copy and edit for your run:

```bash
cp sagemaker/runs/example-processing.yaml sagemaker/runs/my-extract.yaml
cp -r sagemaker/configs/example sagemaker/configs/my-extract
# Edit sagemaker/runs/my-extract.yaml: set --version and --manifest-sha256
# to the pinned dataset version, --family, and tune instance_type, workers,
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
URI, and the resolved per-worker `container_args` (`--shard i/N` per worker,
or each CSV chunk's items appended after `per_worker_arg`).

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

If a worker hits the `job.max_runtime_hours` cap, re-run the launcher —
images whose family file exists are skipped on the next pass (the feature
family in the bucket is the source of truth for resumability).
