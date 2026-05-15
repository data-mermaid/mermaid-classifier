# Training at Scale (SageMaker TrainingJob)

Run the mermaid-classifier training pipeline as a SageMaker
TrainingJob. Use this for runs that don't fit on a laptop -- e.g.,
the 192-source production run.

For day-to-day local iteration use `scripts/classifier_train.py`. The
SageMaker path is for production / large runs only.

## One-time setup

You need to do these once per AWS account.

### 1. ECR repo

Create the container registry that will hold the training image:

    aws ecr create-repository \
        --repository-name mermaid-classifier-training \
        --region us-east-1 \
        --profile wcs

### 2. IAM role for training jobs

Create an IAM role (e.g. `MermaidClassifierTrainerRole`) that SageMaker
will assume during the job. Trust policy:

    {
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }

Inline permissions:

- `s3:GetObject` + `s3:ListBucket` on the training-data bucket
  (`coral-reef-training` and any others the YAML's `env` overrides).
- `s3:GetObject` + `s3:PutObject` + `s3:ListBucket` on the staging
  bucket you pass to `--staging-bucket` (for config upload + output).
- `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`,
  `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer` on `*` (image pull).
- `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`
  on `/aws/sagemaker/TrainingJobs*`.
- `sagemaker-mlflow:*` on the MLflow tracking server ARN, OR the
  equivalent IAM allow for whatever MLflow auth you use.

### 3. Build and push the image

From `mermaid-classifier/`:

    ACCOUNT_ID=<your-account-id>
    REGION=us-east-1
    IMAGE=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/mermaid-classifier-training:latest

    aws ecr get-login-password --region ${REGION} --profile wcs \
        | docker login --username AWS --password-stdin \
            ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

    docker buildx build --platform linux/amd64 \
        -t ${IMAGE} \
        -f docker/training/Dockerfile .

    docker push ${IMAGE}

Re-run this every time you change code. The image tag is the unit of
versioning -- promote a build by tagging it (`:prod`, `:2026-05-15`,
etc.) and pushing.

### 4. Verify locally first

Before paying SageMaker:

    bash docker/training/local_smoke.sh

This builds the image, runs the example config with `-v` mounts, and
asserts that the container reaches the `build_options` stage. Without
real AWS creds it cannot complete training, but it catches packaging
and entrypoint mistakes.

## Launching a training run

### 5. Author a config dir

Copy `sagemaker/configs/example/` to a new name:

    cp -r sagemaker/configs/example sagemaker/configs/my-run

Edit `training_config.yaml` to point at your sources/rollups/included-labels
CSVs (paths are resolved as siblings of the YAML, so just `sources.csv`
not `/abs/path/sources.csv`). Replace the placeholder CSVs with the
real ones for your experiment.

### 6. SageMaker smoke run

Before the production run, do a 1-source / 1-epoch run on the
cheapest instance to prove the whole pipeline works end-to-end:

    python scripts/launch_training_sagemaker.py \
        --config-dir sagemaker/configs/example \
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:ACCT:mlflow-app/APP \
        --role-arn arn:aws:iam::ACCT:role/MermaidClassifierTrainerRole \
        --ecr-image-uri ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-training:latest \
        --staging-bucket my-staging-bucket \
        --instance-type ml.m5.2xlarge

Expect: ~10-20 minutes wall-clock, ~$0.15, an MLflow run with metrics
+ artifacts, CloudWatch logs streaming live to your terminal.

### 7. Production run

For the full 192-source run, the defaults are tuned to be a sensible
starting point:

    python scripts/launch_training_sagemaker.py \
        --config-dir sagemaker/configs/my-run \
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:ACCT:mlflow-app/APP \
        --role-arn arn:aws:iam::ACCT:role/MermaidClassifierTrainerRole \
        --ecr-image-uri ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-training:latest \
        --staging-bucket my-staging-bucket

This uses:

- `ml.m5.4xlarge` -- 16 vCPU, 64 GB RAM, ~$0.92/hr. PyTorch on CPU
  parallelizes BLAS across all cores; 64 GB is comfortable for the
  ~3-4M subsampled annotations at 192 sources.
- 200 GB volume.
- 24h max runtime (hard kill).

`--dry-run` prints the planned Estimator config and exits without
submitting.

## Tuning for cost vs speed

| Instance | vCPU | RAM | $/hr | When to use |
|----------|------|-----|------|-------------|
| `ml.m5.2xlarge` | 8 | 32 GB | ~$0.46 | 20-source runs, smoke tests |
| `ml.m5.4xlarge` (default) | 16 | 64 GB | ~$0.92 | 192-source runs |
| `ml.c5.9xlarge` | 36 | 72 GB | ~$1.94 | "Fast mode" -- roughly halves wall-clock at ~2x hourly. Often net-cheaper per run if the job otherwise hits the 24h limit. |

Pricing is approximate (us-east-1 on-demand, 2026). Check the AWS
SageMaker pricing page for current numbers.

`--use-spot` enables spot pricing (~70% discount) but the trainer has
no checkpointing, so a spot interruption restarts from epoch zero.
Use only for runs you're OK retrying.

## Debugging a failed job

1. Open the CloudWatch URL printed by the launcher. The container's
   first ~50 lines dump Python version, package versions, the loaded
   YAML, the resolved env vars (secrets redacted), and the
   `/opt/ml/input/data/` listing.
2. Scan for `[stage:XX] FAIL` -- this tells you which phase failed
   (`load_config`, `apply_env`, `build_options`, `runner_run`).
3. The full traceback is logged immediately after the FAIL marker.
4. The launcher writes the exact config that ran to
   `s3://<staging-bucket>/runs/<run-id>/config/`, so you can re-run
   the smoke test against that exact bundle:

       aws s3 sync s3://<staging-bucket>/runs/<run-id>/config/ /tmp/repro/
       SAGEMAKER_CONFIG_DIR=/tmp/repro python scripts/sagemaker_train_entrypoint.py

   (This skips Docker; runs the entrypoint directly against your
   local pyspacer install.)

## MLflow

The container needs to reach the MLflow tracking server. The server
URI flows through the launcher's `--mlflow-tracking-uri` flag into the
`MLFLOW_TRACKING_SERVER` env var on the Estimator. The training run
appears under whatever `experiment_name` is set in the YAML.

If MLflow auth is via the SageMaker IAM integration (server URI is an
`arn:aws:sagemaker:...:mlflow-app/...`), the training role needs
`sagemaker-mlflow:*` on that ARN. Without it the runner will hang or
fail at the first `mlflow.log_param` call.
