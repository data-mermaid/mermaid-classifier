# SageMaker Training Launcher for mermaid-classifier

**Date:** 2026-05-15
**Branch:** `rollup`
**Status:** Design approved; awaiting implementation plan

## Context

Today, training the mermaid-classifier runs locally via `scripts/classifier_train.py` — a Python script whose configuration lives as hardcoded dataclass instances inside the file. This works for the current 20-source workflow on an M1 laptop, but scaling to 192 sources (≈10× the data, ~3–4M subsampled annotations after rollup) makes local runs impractical: estimated 12+ hour wall-clock, 32–64 GB RAM, and a machine the user does not want to dedicate.

This work adds a path to launch the *same* training pipeline as a SageMaker TrainingJob, with reasonable defaults for the resource parameters, while preserving the existing notebook-style entry point so anyone in a SageMaker notebook can still construct `MLflowTrainingRunner` directly with hardcoded options.

The intended outcome: launch a 192-source training run from a single command, with live CloudWatch log streaming, on a CPU-only instance, for under ~$20 of compute per run, and with a setup runbook so a teammate can repeat it.

## Goals

- Launch SageMaker TrainingJobs that run `MLflowTrainingRunner.run()` end-to-end inside a container.
- Keep `scripts/classifier_train.py` untouched — it remains the notebook/local example.
- Add a parallel **config-driven** path: a YAML file describes the run, the container reads it, calls the same runner.
- Make failures debuggable: live log streaming, structured stage markers, env dump at startup, full tracebacks on exception.
- Defaults tuned for CPU-only PyTorch training at 20–192 source scale, with cost as a first-class consideration.

## Non-goals

- Distributed / multi-node training. Single node only.
- GPU training. (The classifier is PyTorch on CPU; feature extraction has its own GPU image on a different branch.)
- Hyperparameter sweeps, automated retries, or spot-resume checkpointing.
- Changes to the training pipeline itself (`MLflowTrainingRunner`, dataset loading, subsampling, weighting). This launcher is purely orchestration.

## Architecture

```
USER LAPTOP                              S3                        SAGEMAKER CONTAINER
-----------                              --                        -------------------
<config-dir>/
  ├─ sources.csv
  ├─ rollups.csv
  ├─ included_labels.csv      ─upload→   s3://<staging>/runs/      ─mount→  /opt/ml/input/data/config/
  └─ training_config.yaml                <run-id>/config/                     ├─ sources.csv
                                                                              ├─ rollups.csv
                                                                              ├─ included_labels.csv
                                                                              └─ training_config.yaml
launch_training_sagemaker.py
  ├─ validates YAML
  ├─ uploads dir to S3                                              entrypoint.sh
  ├─ builds Estimator                                                 → python sagemaker_train_entrypoint.py
  └─ estimator.fit(wait=True)  ─submit→  SageMaker TrainingJob       ├─ reads training_config.yaml
                                         pulls ECR image             ├─ builds DatasetOptions/TrainingOptions/MLflowOptions
                                                                     └─ MLflowTrainingRunner.run()
                                                                          ├─ downloads .fv from coral-reef-training
                                                                          ├─ trains PyTorch classifier (CPU)
                                                                          └─ logs everything to MLflow tracking server
```

**Key invariants:**

- `scripts/classifier_train.py` is unchanged. It remains the hardcoded notebook/local example.
- Both entry points (notebook and SageMaker) construct the same `DatasetOptions` / `TrainingOptions` / `MLflowOptions` and call the same `MLflowTrainingRunner.run()`. Only the *source* of those options differs (Python literal vs. YAML).
- Code is **baked into the Docker image at build time**; nothing is uploaded via the SDK's `source_dir`. Image tag = pinned code state.
- `/opt/ml/model` stays empty. MLflow's artifact store is the single source of truth for trained models and metrics. SageMaker produces an empty `model.tar.gz` at `OutputDataConfig.S3OutputPath` — this is expected.
- One input channel only (`config`). Training data (CoralNet/MERMAID annotations, feature vectors, EfficientNet weights) is fetched live from S3 by existing pyspacer code using the container's IAM execution role.

## Components

| Path | Purpose |
|------|---------|
| `scripts/launch_training_sagemaker.py` | Launcher. Argparse, validates local YAML, syncs config dir to S3, builds `sagemaker.estimator.Estimator`, calls `.fit(wait=True, logs="All")`. Prints job ARN, CloudWatch URL, and MLflow run URL. |
| `scripts/sagemaker_train_entrypoint.py` | Container entrypoint. Reads `/opt/ml/input/data/config/training_config.yaml`, validates via Pydantic, builds `DatasetOptions` / `TrainingOptions` / `MLflowOptions`, calls `MLflowTrainingRunner.run()`. |
| `mermaid_classifier/sagemaker/__init__.py` | New subpackage. |
| `mermaid_classifier/sagemaker/config.py` | Pydantic `TrainingRunConfig` schema mirroring the three option dataclasses, with discriminated unions for `subsample_strategy` and `weighting_strategy`. Single source of truth for the YAML schema. |
| `docker/training/Dockerfile` | `FROM python:3.10-slim`. Installs OS deps for pyarrow/duckdb, installs CPU-only PyTorch first via `--index-url https://download.pytorch.org/whl/cpu`, then `pip install -e .[pyspacer]`, then `COPY` repo. `ENTRYPOINT ["/opt/ml/code/docker/training/entrypoint.sh"]`. |
| `docker/training/entrypoint.sh` | Thin shim: `cd /opt/ml/code && exec python -u scripts/sagemaker_train_entrypoint.py "$@"`. |
| `docker/training/local_smoke.sh` | Local smoke recipe (see Testing). |
| `sagemaker/configs/example/training_config.yaml` | Reference YAML covering all the options `classifier_train.py` currently sets, with comments. |
| `sagemaker/configs/example/{sources,rollups,included_labels}.csv` | Tiny example CSVs (1–2 sources) suitable for smoke tests. |
| `docs/training_at_scale.md` | One-time setup runbook. |
| `tests/sagemaker/__init__.py` | |
| `tests/sagemaker/test_config.py` | Unit tests for the Pydantic schema. |
| `tests/sagemaker/test_entrypoint.py` | Unit tests for the entrypoint (mocked runner). |

`mermaid_classifier/sagemaker/config.py` is the architectural keystone — the only place where YAML structure ↔ Python dataclass mapping lives. The entrypoint imports it; unit tests cover it; the example YAML conforms to it. When the dataclasses grow new fields, you update `config.py` and the example YAML; nothing else changes.

**Unchanged:** `scripts/classifier_train.py`, all of `mermaid_classifier/pyspacer/`, `mermaid_classifier/training/`, `mermaid_classifier/common/`.

## Data flow & I/O details

**Inputs into the SageMaker job (one channel):**

The Estimator declares a single input channel named `config`:

```python
estimator.fit(
    inputs={"config": "s3://<staging-bucket>/runs/<run-id>/config/"},
    wait=True,
    logs="All",
)
```

SageMaker mounts that S3 prefix at `/opt/ml/input/data/config/` inside the container. The entrypoint finds:

```
/opt/ml/input/data/config/
  ├─ training_config.yaml
  ├─ sources.csv
  ├─ rollups.csv
  └─ included_labels.csv
```

CSV paths inside `training_config.yaml` are written as **bare filenames** (e.g., `sources_csv: sources.csv`). The entrypoint resolves them as siblings of the YAML file. The launcher never has to rewrite paths.

**Other inputs (not via SageMaker channels):**

- CoralNet annotations + feature vectors: `s3://coral-reef-training/...` — pulled live by existing pyspacer code via s3fs/boto3, identical to local runs. Uses the container's IAM execution role.
- MERMAID annotations: same bucket, same code path.
- EfficientNet weights: pulled by pyspacer using `WEIGHTS_LOCATION` env var, set on the Estimator from the YAML's `env` block.

**Environment variables (set on the Estimator via `environment=`):**

- `MLFLOW_TRACKING_SERVER` — required, from launcher CLI flag `--mlflow-tracking-uri`. No committed default (URIs include account-specific ARNs).
- `WEIGHTS_LOCATION` — required, S3 URI to the EfficientNet weights.
- `AWS_DEFAULT_REGION` — `us-east-1` (default).
- `FEATURE_CACHE_DIR` — `/tmp/feature_cache` (or `/opt/ml/input/data/feature_cache` if a cache channel is added later).
- `SPACER_EXTRACTORS_CACHE_DIR` — `/tmp/spacer_extractors`.
- `MLFLOW_HTTP_REQUEST_MAX_RETRIES` — `2` (per the existing codebase note).

**Hyperparameters:** intentionally empty. The YAML is the single source of training config. We do not split options between YAML and SageMaker hyperparameters.

**Outputs:**

- **MLflow** (primary): trained classifier, val metrics, profiling CSV, etc. Identical to local runs.
- **CloudWatch Logs**: stdout/stderr streamed live to the launcher via `fit(logs="All")` AND retained at `/aws/sagemaker/TrainingJobs/<job-name>` for later inspection.
- **`OutputDataConfig.S3OutputPath`**: `s3://<staging-bucket>/runs/<run-id>/output/` — receives an empty `model.tar.gz` plus SageMaker's own `output.tar.gz` (failure traces). Used for forensics on failed runs.

**Run ID convention:** `<job-name-prefix>-<UTC-timestamp>`, e.g. `mermaid-classifier-20260515T141200Z`. The same string is used for the SageMaker `TrainingJobName`, the S3 prefix under `runs/`, and the default MLflow run name. One ID, three places — trivial cross-system correlation.

## Resource defaults

The PyTorch classifier on CPU is multi-threaded (BLAS via MKL/OpenMP), so vCPUs scale training throughput. RAM is the harder ceiling, but PyTorch's tensor reuse keeps the working set tighter than sklearn's copy-heavy `partial_fit`. Sweet spot: compute-balanced instances over memory-heavy.

| Launcher flag | Default | Rationale |
|---------------|---------|-----------|
| `--instance-type` | `ml.m5.4xlarge` | 16 vCPU, 64 GB RAM, ~$0.92/hr. 16 cores feed the PyTorch BLAS pool; 64 GB is comfortable for ~3–4M subsampled rows at 192 sources. |
| `--instance-count` | `1` | Single-node only — no distributed training code path. |
| `--volume-size-gb` | `200` | Feature vector cache for 192 sources ≈ 70 GB; 200 GB gives headroom for spill, SageMaker overhead, and PyTorch tensor scratch. |
| `--max-runtime-hours` | `24` | Hard kill. Easy to bump per-run for outliers. |
| `--use-spot` | `False` | Trainer has no PyTorch-state checkpointing — a spot interruption restarts from epoch zero. Opt-in flag for runs you're OK retrying. |
| `--job-name-prefix` | `mermaid-classifier` | Prepended to UTC timestamp for the run ID. |

**Alternatives documented in the runbook:**

- `ml.m5.2xlarge` (8 vCPU, 32 GB, ~$0.46/hr) — cheaper for 20-source runs and smoke tests.
- `ml.c5.9xlarge` (36 vCPU, 72 GB, ~$1.94/hr) — fast mode: roughly halves wall-clock at ~2× hourly cost, often net-cheaper per run.

## PyTorch-specific notes

1. **Dockerfile install order.** Install CPU-only PyTorch first using `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -e .[pyspacer]`. Without this, transitive resolution pulls the default ~2 GB CUDA torch wheel — wasted ECR space and pull time on a CPU-only image.

2. **Thread tuning.** Do **not** set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `TORCH_NUM_THREADS`. PyTorch defaults (= vCPU count) are correct. Documented as an explicit non-decision.

3. **`SPACER_BATCH_SIZE`.** Leave auto-calculation enabled. The auto-calc reads available RAM at runtime and adapts. User can override per-run via the YAML if they want determinism.

## Error handling & debuggability

**Pre-submit (in launcher, fail before any AWS spend):**

1. Config dir exists and contains `training_config.yaml`.
2. YAML parses and validates against the Pydantic schema → human-readable errors with field paths.
3. All sibling CSVs referenced in the YAML exist locally.
4. `--mlflow-tracking-uri`, `--role-arn`, `--ecr-image-uri`, `--staging-bucket` all provided.
5. `--dry-run` flag prints the full Estimator config and S3 upload plan, exits without uploading or submitting.

**Submit-time:** catch `ClientError` from boto/SDK, re-raise with a one-line summary identifying which step failed (S3 upload vs. `CreateTrainingJob` vs. log streaming).

**Log streaming:** `fit(wait=True, logs="All")` streams container stdout to the launcher in real time. Launcher also prints the CloudWatch URL up front so the user can open it in a browser if the terminal disconnects.

**Exit:** non-zero on any terminal state other than `Completed`.

**Container-side (where most debugging happens):**

1. **Structured stage logging.** The entrypoint logs `[stage:NN/<name>] ENTER` / `EXIT` / `FAIL` markers around each phase: `load_config`, `validate_inputs`, `build_options`, `runner.run`. So in CloudWatch you can immediately see which stage failed.

2. **First-line dump.** Right after `__main__` starts, log: Python version, package versions for pyspacer/mlflow/duckdb/torch, the loaded YAML config (pretty-printed), the resolved env vars (with secrets redacted), and a `/opt/ml/input/data/` directory listing. The first ~50 log lines must tell you exactly what the container thinks it's running.

3. **Wrap `runner.run()`** in `try / except Exception as e: log.exception(...); sys.exit(1)` so SageMaker marks the job `Failed` (instead of hanging on an uncaught exception) AND the full traceback is in CloudWatch.

4. **No silent retries.** If something fails, fail loudly. Re-runs are explicit.

5. **`python -u`** in the entrypoint so stdout isn't buffered — log lines hit CloudWatch within seconds, not minutes.

## Testing

**1. Unit tests** (fast, `cd tests && python -m unittest`):

- `tests/sagemaker/test_config.py`
  - Roundtrip: load `sagemaker/configs/example/training_config.yaml` → assert it parses and constructs valid `DatasetOptions` / `TrainingOptions` / `MLflowOptions`.
  - Each subsample strategy variant loads correctly via the discriminated union.
  - Each weighting strategy variant loads correctly.
  - Negative cases: missing required field → `ValidationError` with offending field path. Bad type → clear error. Unknown strategy → enum-mismatch error.
- `tests/sagemaker/test_entrypoint.py`
  - Patch `MLflowTrainingRunner` with a mock.
  - Write a fixture config dir to `tmp_path`, override `SAGEMAKER_CONFIG_DIR` env var to point there.
  - Run `main()`; assert the mock runner was instantiated with dataclasses populated from YAML, and `.run()` was called once.
  - Assert stages logged in order.
  - Failure path: make `runner.run` raise; assert exit code 1 and traceback in captured logs.

**2. Local Docker smoke test** (`docker/training/local_smoke.sh`, manual):

- Builds the image locally for `linux/amd64` via `docker buildx`.
- Runs the container with a bind-mount: `-v $PWD/sagemaker/configs/example:/opt/ml/input/data/config:ro`.
- Passes `WEIGHTS_LOCATION` and `MLFLOW_TRACKING_SERVER=file:./mlruns` (local MLflow), uses host AWS creds via `-v ~/.aws:/root/.aws:ro`.
- Trains on the example `sources.csv` (1–2 sources) for 1 epoch.
- Asserts container exit 0 and that a local `mlruns/` dir was created.

This proves the image and entrypoint work *before* the first SageMaker submission. No SageMaker bill required.

**3. SageMaker smoke test** (runbook recipe, manual):

- Example config with `sources.csv` of 1–2 sources, `epochs: 1`.
- Launcher with `--instance-type ml.m5.2xlarge`.
- Expected: completes in ~10–20 minutes, ~$0.15. Job ARN, CloudWatch URL, MLflow URL all printed.

Run this once after a new AWS account setup, before any 192-source production run.

**Out of scope:**

- The `MLflowTrainingRunner` pipeline — covered by existing tests in `tests/pyspacer/` and `tests/training/`.
- Live AWS API calls in unit tests — boto/SDK is mocked.

## Verification

End-to-end verification after implementation:

1. `cd mermaid-classifier/tests && python -m unittest sagemaker.test_config sagemaker.test_entrypoint` — all green.
2. `bash docker/training/local_smoke.sh` — exits 0, creates `mlruns/`, log shows ENTER/EXIT for each stage.
3. Launcher dry-run: `python scripts/launch_training_sagemaker.py --config-dir sagemaker/configs/example --dry-run --mlflow-tracking-uri <ARN> --role-arn <ARN> --ecr-image-uri <URI> --staging-bucket <BUCKET>` — prints intended Estimator config and S3 upload plan, exits 0.
4. SageMaker smoke (1-source, 1-epoch): completes in <20 min, MLflow run shows up, CloudWatch logs show full stage sequence.
5. 20-source production parity: a launcher-driven run reaches the same metrics as a local `classifier_train.py` run on identical inputs (within run-to-run noise).
6. 192-source production run: completes within `--max-runtime-hours`, produces an MLflow run with expected artifacts, total cost in the expected band.

## Open questions to resolve during implementation

- Exact Pydantic representation of `subsample_strategy` and `weighting_strategy`. The pyspacer code uses concrete classes; the YAML needs discriminated unions keyed on a `strategy: <name>` field. The example YAML should cover at least the variants used in `classifier_train.py` today.
- Whether the example CSVs (`sources.csv` with 1–2 sources, etc.) should be committed as small fixtures or generated on demand by a helper. Preference: commit as fixtures so the example is self-contained.
- Whether the launcher should optionally write the upload manifest (the list of files uploaded to S3) into the local config dir, for traceability outside of S3.
