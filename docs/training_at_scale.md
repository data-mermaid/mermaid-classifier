# Training at Scale (SageMaker TrainingJob)

Run the mermaid-classifier training pipeline as a SageMaker
TrainingJob. Use this for runs that don't fit on a laptop -- e.g.,
the 192-source production run.

For day-to-day local iteration use `scripts/classifier_train.py`. The
SageMaker path is for production / large runs only.

## Prerequisites

You should already have:

- An AWS SSO account in the MERMAID org with the `SageMaker`
  permission set assigned to account `554812291621`.
- Docker installed locally.
- `uv` installed locally (`pip install uv` or via Homebrew).

Everything else -- IAM roles, the ECR repository, the S3 staging
bucket, the MLflow App -- is provisioned by IaC in the
[mermaid-api](https://github.com/data-mermaid/mermaid-api) repo and
already exists in the dev account.

## Concrete values you'll need

| What | Value |
|---|---|
| AWS account | `554812291621` (dev) |
| Region | `us-east-1` |
| ECR repo | `554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-training` |
| Launcher IAM role | `arn:aws:iam::554812291621:role/dev-mermaid-classifier-launcher-role` |
| Training execution role | `arn:aws:iam::554812291621:role/dev-sm-execution-role` |
| Staging S3 bucket | `dev-datamermaid-sm-data` (runs land under `runs/<run-id>/`) |
| MLflow tracking URI | `arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-2OMU4VP53ZS2` (look up by name if it changes -- see step 2) |

## 1. AWS profile setup

You don't run training directly as your SSO role. You log in via SSO,
then role-chain into the launcher role, which has scoped permissions
for ECR push/pull, SageMaker training-job submission, and S3 writes
under the staging bucket's `runs/` prefix.

Add the following to `~/.aws/config`:

```ini
[profile wcs-sso]
sso_start_url      = https://<your-aws-sso-portal>.awsapps.com/start
sso_region         = us-east-1
sso_account_id     = 554812291621
sso_role_name      = SageMaker
region             = us-east-1
output             = json

[profile wcs-launcher]
source_profile     = wcs-sso
role_arn           = arn:aws:iam::554812291621:role/dev-mermaid-classifier-launcher-role
region             = us-east-1
duration_seconds   = 28800
```

Log in once per work session:

```bash
aws sso login --profile wcs-sso
aws sts get-caller-identity --profile wcs-launcher
```

The second call should return an `Arn` ending in
`.../dev-mermaid-classifier-launcher-role/...`. Export `AWS_PROFILE`
to avoid repeating the flag:

```bash
export AWS_PROFILE=wcs-launcher
```

## 2. Look up the MLflow tracking URI

MERMAID's MLflow is deployed as a SageMaker MLflow App. Discover the
ARN once and reuse it:

```bash
aws sagemaker list-mlflow-apps --region us-east-1 \
    --query 'MlflowAppSummaries[].[Name,Arn]' --output table
```

You're looking for the one named `pyspacer`. Copy its `Arn` -- you'll
pass it as `--mlflow-tracking-uri`.

## 3. Build and push the image

From the `mermaid-classifier/` repo root:

```bash
ACCOUNT=554812291621
REGION=us-east-1
IMAGE=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/mermaid-classifier-training:latest

aws ecr get-login-password --region ${REGION} \
    | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

docker buildx build --platform linux/amd64 -t ${IMAGE} -f docker/training/Dockerfile .
docker push ${IMAGE}
```

Re-run this every time you change code that affects the container
(Dockerfile, `mermaid_classifier/`, `scripts/sagemaker_train_entrypoint.py`,
`docker/training/entrypoint.sh`, or `pyproject.toml` / `uv.lock`). The
image tag is the unit of versioning -- promote a build by tagging it
(`:prod`, `:2026-05-15`, etc.) and pushing.

### Optional local smoke

```bash
bash docker/training/local_smoke.sh
```

This builds the image, runs the example config with `-v` mounts, and
asserts the container reaches the `build_options` stage. Useful for
catching packaging and entrypoint mistakes before pushing.

**On Apple Silicon (ARM) Macs**, the smoke run goes through Docker
Desktop's qemu emulation and often segfaults inside torch/pyarrow on
import. If you see a `qemu: uncaught target signal 11` after stage
`build_options ENTER`, that's emulation noise -- the image is fine and
will run cleanly on the native x86 SageMaker instance. Treat reaching
`[stage:build_options] ENTER` as success.

## 4. Author a config dir

Copy `sagemaker/configs/example/` to a new name:

```bash
cp -r sagemaker/configs/example sagemaker/configs/my-run
```

Edit `training_config.yaml` to point at your sources / rollups /
included-labels CSVs (paths are resolved as siblings of the YAML, so
just `sources.csv`, not `/abs/path/sources.csv`). Replace the
placeholder CSVs with the real ones for your experiment.

## 5. Dry-run the launcher

Before paying SageMaker, validate the config and print the planned
Estimator without uploading or submitting:

```bash
export EXEC_ROLE=arn:aws:iam::554812291621:role/dev-sm-execution-role
export STAGING=dev-datamermaid-sm-data
export MLFLOW_ARN=arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-2OMU4VP53ZS2

uv run --extra pyspacer --extra sagemaker python scripts/launch_training_sagemaker.py \
    --config-dir sagemaker/configs/my-run \
    --role-arn ${EXEC_ROLE} \
    --ecr-image-uri ${IMAGE} \
    --staging-bucket ${STAGING} \
    --mlflow-tracking-uri ${MLFLOW_ARN} \
    --instance-type ml.m5.2xlarge \
    --dry-run
```

Expected: a `DRY RUN` block listing the planned config and exiting
cleanly. SDK regex-deprecation warnings from `sagemaker_core` /
`smdebug_rulesconfig` are harmless.

## 6. SageMaker smoke run

Before the production run, do a 1-source / 1-epoch run on the cheapest
instance to prove the whole pipeline works end-to-end. Drop
`--dry-run`:

```bash
uv run --extra pyspacer --extra sagemaker python scripts/launch_training_sagemaker.py \
    --config-dir sagemaker/configs/example \
    --role-arn ${EXEC_ROLE} \
    --ecr-image-uri ${IMAGE} \
    --staging-bucket ${STAGING} \
    --mlflow-tracking-uri ${MLFLOW_ARN} \
    --instance-type ml.m5.2xlarge
```

Expect: ~10-20 minutes wall-clock, ~$0.15, an MLflow run with metrics
+ artifacts, CloudWatch logs streaming live to your terminal.

> The launcher uses `estimator.fit(wait=True, logs="All")` to tail
> CloudWatch logs to your terminal. The launcher role does not
> currently include CloudWatch Logs read permissions, so this step
> fails with `AccessDeniedException` on `logs:DescribeLogStreams`
> *after the training job has already been submitted*. The job itself
> is unaffected -- monitor it via the CloudWatch URL the launcher
> printed, or via `aws logs tail` from an admin profile (see step 8).

## 7. Production run

For the full 192-source run, the defaults are tuned to be a sensible
starting point:

```bash
uv run --extra pyspacer --extra sagemaker python scripts/launch_training_sagemaker.py \
    --config-dir sagemaker/configs/my-run \
    --role-arn ${EXEC_ROLE} \
    --ecr-image-uri ${IMAGE} \
    --staging-bucket ${STAGING} \
    --mlflow-tracking-uri ${MLFLOW_ARN}
```

This uses:

- `ml.m5.4xlarge` -- 16 vCPU, 64 GB RAM, ~$0.92/hr. PyTorch on CPU
  parallelizes BLAS across all cores; 64 GB is comfortable for the
  ~3-4M subsampled annotations at 192 sources.
- 200 GB volume.
- 24h max runtime (hard kill).

## 8. Monitor a running job

The launcher prints a CloudWatch URL and run ID when it submits.
After that, three options:

**Browser** -- open the printed CloudWatch URL.

**`aws logs tail`** -- requires CloudWatch Logs read permissions
(your SSO `SageMaker` permission set has them via
`AmazonSageMakerFullAccess`; the launcher role currently does not):

```bash
aws logs tail /aws/sagemaker/TrainingJobs \
    --profile wcs-sso --region us-east-1 \
    --log-stream-name-prefix <run-id>/ \
    --follow
```

**Status only**:

```bash
aws sagemaker describe-training-job \
    --training-job-name <run-id> \
    --query '[TrainingJobStatus,SecondaryStatus,FailureReason]' \
    --output table
```

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

   ```bash
   aws s3 sync s3://${STAGING}/runs/<run-id>/config/ /tmp/repro/
   SAGEMAKER_CONFIG_DIR=/tmp/repro uv run --extra pyspacer python scripts/sagemaker_train_entrypoint.py
   ```

   (This skips Docker; runs the entrypoint directly against your
   local pyspacer install.)

## MLflow

The container needs to reach the MLflow tracking server. The URI
flows through the launcher's `--mlflow-tracking-uri` flag into the
`MLFLOW_TRACKING_SERVER` env var on the Estimator. The training run
appears under whatever `experiment_name` is set in the YAML.

For SageMaker-managed MLflow Apps (ARN of the form
`arn:aws:sagemaker:...:mlflow-app/...`), the training execution role
needs `sagemaker-mlflow:*` -- already attached to
`dev-sm-execution-role` via the IaC.

Artifact uploads (`mlflow.log_artifact(...)`) land in the MLflow
App's backing S3 bucket; the execution role's S3 grants cover this.
If a run logs metrics/params successfully but fails on
`log_artifact`, the artifact bucket isn't covered by the execution
role's S3 policy -- escalate to the IaC owner.
