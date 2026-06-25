# Training at Scale (SageMaker TrainingJob)

Run the mermaid-classifier training pipeline as a SageMaker
TrainingJob. Use this for runs that don't fit on a laptop -- e.g.,
the 192-source production run.

For day-to-day local iteration use `scripts/classifier_train.py`. The
SageMaker path is for production / large runs only.

> The launcher convention (role ARN, bucket, schema, ECR tagging) is
> defined in
> [`mermaid-api/iac/sagemaker-launcher-convention.md`](https://github.com/data-mermaid/mermaid-api/blob/dev/iac/sagemaker-launcher-convention.md).
> This doc is the classifier-specific runbook on top of that.

## Prerequisites

You should already have:

- An AWS SSO account in the MERMAID org with the `SageMaker`
  permission set assigned to account `554812291621`.
- Docker installed locally.
- `uv` installed locally (`pip install uv` or via Homebrew).

Everything else -- IAM role, the ECR repository, the S3 staging
bucket, the MLflow App -- is provisioned by IaC in the
[mermaid-api](https://github.com/data-mermaid/mermaid-api) repo and
already exists in the dev account.

## Concrete values you'll need

| What | Value |
|---|---|
| AWS account | `554812291621` (dev) |
| Region | `us-east-1` |
| ECR repo | `554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-jobs` |
| Launcher IAM role | `arn:aws:iam::554812291621:role/dev-mermaid-sagemaker-launcher-role` |
| Training execution role | `arn:aws:iam::554812291621:role/dev-sm-execution-role` |
| Staging S3 bucket | `dev-datamermaid-sm-data` (runs land under `runs/<run-id>/`) |
| MLflow tracking URI | `arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-2OMU4VP53ZS2` (look up by name if it changes -- see step 2) |

## 1. AWS profile setup

You don't run training directly as your SSO role. You log in via SSO,
then role-chain into the launcher role, which has scoped permissions
for ECR push/pull on `mermaid-*-jobs`, SageMaker
Training/ProcessingJob submission, CloudWatch Logs tailing on
`/aws/sagemaker/*`, and S3 writes under the staging bucket's `runs/`
prefix.

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
role_arn           = arn:aws:iam::554812291621:role/dev-mermaid-sagemaker-launcher-role
region             = us-east-1
duration_seconds   = 28800
```

Log in once per work session:

```bash
aws sso login --profile wcs-sso
aws sts get-caller-identity --profile wcs-launcher
```

The second call should return an `Arn` ending in
`.../dev-mermaid-sagemaker-launcher-role/...`. Export `AWS_PROFILE`
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

You're looking for the one named `pyspacer` (the classifier's MLflow
app). Copy its `Arn` -- you'll pass it as `--mlflow-tracking-uri`.
The segmentation team uses a different app (`mermaidseg`); the
launcher role grants `sagemaker-mlflow:*` on both.

## 3. Build and push the image

From the `mermaid-classifier/` repo root:

```bash
ACCOUNT=554812291621
REGION=us-east-1
IMAGE=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/mermaid-classifier-jobs:training-latest

aws ecr get-login-password --region ${REGION} \
    | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

docker buildx build --platform linux/amd64 -t ${IMAGE} -f docker/jobs/training.Dockerfile .
docker push ${IMAGE}
```

Re-run this every time you change code that affects the container
(`docker/jobs/training.Dockerfile`, `mermaid_classifier/`,
`scripts/sagemaker_train_entrypoint.py`,
`docker/jobs/training-entrypoint.sh`, or `pyproject.toml` /
`uv.lock`). The image tag is the unit of versioning -- promote a
build by tagging it (`:training-prod-2026-05-25`, `:training-smoke`,
`:user-<name>-...`) and pushing. See
[`mermaid-api/iac/sagemaker-launcher-convention.md`](https://github.com/data-mermaid/mermaid-api/blob/dev/iac/sagemaker-launcher-convention.md#ecr-tagging-convention)
for the full tagging conventions.

> **macOS + Rancher Desktop gotcha:** If `docker push` reports
> `denied: requested access to the resource is denied` even after a
> successful `docker login`, the issue is that Rancher's docker
> stores ECR credentials in the macOS Keychain under a long-lived
> entry that can outlast the SSO session that created it. Add this
> to `~/.docker/config.json` (alongside `credsStore`) so docker
> instead fetches a fresh ECR token per push using your ambient AWS
> profile:
>
> ```json
> "credHelpers": {
>   "554812291621.dkr.ecr.us-east-1.amazonaws.com": "ecr-login"
> }
> ```
>
> Then run pushes with `AWS_PROFILE=wcs-launcher docker push ...`.

### Optional local smoke

```bash
bash docker/jobs/local_smoke.sh training
```

This builds the image, runs the example config with `-v` mounts, and
asserts the container reaches the `load_config` stage. Useful for
catching packaging and entrypoint mistakes before pushing.

**On Apple Silicon (ARM) Macs**, the smoke run goes through
emulation and often segfaults inside torch/pyarrow on import. If you
see a `qemu: uncaught target signal 11` after stage `build_options
ENTER`, that's emulation noise -- the image is fine and will run
cleanly on the native x86 SageMaker instance. Treat reaching
`[stage:load_config] OK` as success.

## 4. Author a run config + config dir

The launcher reads two things:

- **`--run-config`**: a YAML in `sagemaker/runs/` describing the
  SageMaker job shape (`job:` block: image, entrypoint, instance
  type, etc.).
- **`--config-dir`**: a directory in `sagemaker/configs/` whose
  contents are uploaded to S3 and mounted at
  `/opt/ml/input/data/config/` inside the container. Holds
  `training_config.yaml` (classifier-specific dataset/training/mlflow
  blocks) plus any CSVs it references.

Copy `sagemaker/configs/example/` to a new name:

```bash
cp -r sagemaker/configs/example sagemaker/configs/my-run
cp sagemaker/runs/example-training.yaml sagemaker/runs/my-run.yaml
```

Edit:
- `sagemaker/runs/my-run.yaml` to adjust the SageMaker shape
  (`job.image`, `job.instance_type`, `job.max_runtime_hours`, etc.).
- `sagemaker/configs/my-run/training_config.yaml` to point at your
  sources / rollups / included-labels CSVs (paths are resolved as
  siblings of the YAML, so just `sources.csv`, not
  `/abs/path/sources.csv`). Replace the placeholder CSVs with the
  real ones for your experiment.

## 5. Dry-run the launcher

Before paying SageMaker, validate the config and print the planned
Estimator without uploading or submitting:

```bash
export MLFLOW_ARN=arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-2OMU4VP53ZS2

uv run python scripts/launch_training.py \
    --run-config sagemaker/runs/my-run.yaml \
    --config-dir sagemaker/configs/my-run/ \
    --mlflow-tracking-uri ${MLFLOW_ARN} \
    --dry-run
```

Expected: a `DRY RUN` block listing the planned config and exiting
cleanly.

## 6. SageMaker smoke run

Before the production run, do a 1-source / 1-epoch run on the cheapest
instance to prove the whole pipeline works end-to-end. Drop
`--dry-run`:

```bash
uv run python scripts/launch_training.py \
    --run-config sagemaker/runs/example-training.yaml \
    --config-dir sagemaker/configs/example/ \
    --mlflow-tracking-uri ${MLFLOW_ARN}
```

Expect: ~10-20 minutes wall-clock, ~$0.15, an MLflow run with metrics
+ artifacts, CloudWatch logs streaming live to your terminal via
`estimator.fit(wait=True, logs="All")`.

## 7. Production run

For the full 192-source run, the defaults in
`sagemaker/runs/example-training.yaml` are tuned to be a sensible
starting point. Edit `job.instance_type` and `job.max_runtime_hours`
in your run YAML as needed. The submitter command is the same:

```bash
uv run python scripts/launch_training.py \
    --run-config sagemaker/runs/my-run.yaml \
    --config-dir sagemaker/configs/my-run/ \
    --mlflow-tracking-uri ${MLFLOW_ARN}
```

Defaults assumed in the example:

- `ml.m5.4xlarge` -- 16 vCPU, 64 GB RAM, ~$0.92/hr. PyTorch on CPU
  parallelizes BLAS across all cores; 64 GB is comfortable for the
  ~3-4M subsampled annotations at 192 sources.
- 200 GB volume.
- 4h max runtime (raise to 24h for the production 192-source run).

## 8. Monitor a running job

The launcher prints a CloudWatch URL and run ID when it submits.
After that, three options:

**Browser** -- open the printed CloudWatch URL.

**`aws logs tail`** -- the launcher role grants
`logs:DescribeLogStreams / GetLogEvents / FilterLogEvents /
StartLiveTail` on `/aws/sagemaker/*`, so tailing works directly from
the `wcs-launcher` profile:

```bash
aws logs tail /aws/sagemaker/TrainingJobs \
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

## 9. Generate a report locally from a SageMaker run

`scripts/generate_report.py` produces a self-contained HTML report from
any MLflow run. To point it at the SageMaker MLflow App (rather than
the default local `sqlite:///mlflow.db` in `.env`), pass the App ARN
via `--mlflow-tracking-uri`. The same flag exists on
`scripts/evaluate_model.py`.

```bash
aws sso login --profile wcs-sso
export AWS_PROFILE=wcs-launcher
aws sts get-caller-identity   # confirm role-chained identity

# (Re)discover the ARN if needed:
aws sagemaker list-mlflow-apps --region us-east-1 \
    --query 'MlflowAppSummaries[].[Name,Arn]' --output table

cd mermaid-classifier
uv run python scripts/generate_report.py \
    --run-id <mlflow-run-id> \
    --mlflow-tracking-uri ${MLFLOW_ARN} \
    --output report_<run-id-prefix>.html
```

The launcher role (`dev-mermaid-sagemaker-launcher-role`) already
carries the required `sagemaker-mlflow:*` grant on the App ARN, so the
same profile used to launch training works for reading runs back.

If you see `No module named sagemaker_mlflow`, run `uv sync` to install
the plugin (it's already in `uv.lock` under the `pyspacer` extra). If
you see `AccessDeniedException` on `sagemaker-mlflow:CallMlflowApi`,
your active profile is not the launcher role.

## Tuning for cost vs speed

| Instance | vCPU | RAM | $/hr | When to use |
|----------|------|-----|------|-------------|
| `ml.m5.2xlarge` | 8 | 32 GB | ~$0.46 | 20-source runs, smoke tests |
| `ml.m5.4xlarge` (default) | 16 | 64 GB | ~$0.92 | 192-source runs |
| `ml.c5.9xlarge` | 36 | 72 GB | ~$1.94 | "Fast mode" -- roughly halves wall-clock at ~2x hourly. Often net-cheaper per run if the job otherwise hits the 24h limit. |

Pricing is approximate (us-east-1 on-demand, 2026). Check the AWS
SageMaker pricing page for current numbers.

Set `job.use_spot: true` in the run YAML to enable spot pricing (~70%
discount) but the trainer has no checkpointing, so a spot interruption
restarts from epoch zero. Use only for runs you're OK retrying.

## Debugging a failed job

1. Open the CloudWatch URL printed by the launcher. The container's
   first ~50 lines dump Python version, package versions, the loaded
   YAML, the resolved env vars (secrets redacted), and the
   `/opt/ml/input/data/` listing.
2. Scan for `[stage:XX] FAIL` -- this tells you which phase failed
   (`load_config`, `apply_env`, `build_options`, `runner_run`).
3. The full traceback is logged immediately after the FAIL marker.
4. The launcher writes the exact config that ran to
   `s3://dev-datamermaid-sm-data/runs/<run-id>/config/`, so you can
   re-run the smoke test against that exact bundle:

   ```bash
   aws s3 sync s3://dev-datamermaid-sm-data/runs/<run-id>/config/ /tmp/repro/
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
