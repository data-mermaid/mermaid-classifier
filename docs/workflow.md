# Script workflow

A map of the scripts in `scripts/` and the order to run them. Start here as a
newcomer; defer to the linked runbooks for full detail. See also
**[../README.md](../README.md)** for install and overview, and
**[../CLAUDE.md](../CLAUDE.md)** for architecture and conventions.

---

## Local vs SageMaker

Most steps have both a local path and a SageMaker path:

- **Local** is simpler and faster to iterate on but requires AWS SSO credentials
  and sufficient local RAM for the training data. Suitable for smoke runs and
  small source sets.
- **SageMaker** scales to the full dataset; see
  [feature_extraction_at_scale.md](feature_extraction_at_scale.md) and
  [training_at_scale.md](training_at_scale.md) for setup and runbooks.

---

## Step 1 — Generate a training config

**Script:** `generate_training_config.py`

**When:** Whenever you need a new or updated CoralNet→MERMAID label-mapping
directory (rollup CSV + included-labels CSV). Run this once per taxonomy
revision. The output is written in-repo under `sagemaker/configs/` (default
`coralnet_top108`) so it can be committed and consumed repo-root-relative by
downstream scripts; pass `--output-dir` to choose a different config name. Its
raw inputs — the curated CoralNet source list and the Drive-exported label
mapping — live in the surrounding workspace (not this repo); override them with
`--sources-csv` / `--labels-csv`.

```bash
uv run python scripts/generate_training_config.py \
    --output-dir sagemaker/configs/<config-name>
```

---

## Step 2 — Extract / build features

Choose one path based on scale:

| Path | Script | When |
| - | - | - |
| **Local** | `build_feature_bucket.py` | Building or updating a CoralNet-layout feature-vector bucket for a small/medium source set. Idempotent and resumable. |
| **SageMaker** | `launch_processing.py` | Full-dataset feature extraction via parallel SageMaker ProcessingJobs (optional sharding). See [feature_extraction_at_scale.md](feature_extraction_at_scale.md). |

> `extract_reference_features.py` is a one-off maintenance script: it stacks
> real EfficientNet feature vectors into a `.npy` for the TorchScript-vs-sklearn
> parity gate. Run it only when regenerating the parity-gate fixture.

---

## Step 3 — Train

Choose one path based on scale:

| Path | Script | When |
| - | - | - |
| **Local** | `classifier_train.py` | Loads a committed config (default `sagemaker/configs/coralnet_top108_best`; `--config-dir` picks another) and runs it locally via `MLflowTrainingRunner` — the *same* `training_config.yaml` the SageMaker path consumes, so local and SageMaker share one recipe. Requires AWS SSO. |
| **SageMaker** | `launch_training.py` + `sagemaker_train_entrypoint.py` | Launches a SageMaker TrainingJob via an Estimator. The entrypoint reads a YAML config and runs `MLflowTrainingRunner` inside the container. See [training_at_scale.md](training_at_scale.md). |

---

## Step 4 — Inspect results

**Script:** `generate_report.py`

Renders a self-contained HTML report from an MLflow run, covering
classification metrics, calibration, cover, ranking, taxonomic breakdown, and
per-source results. The only way to get a browsable HTML report outside the
MLflow UI.

```bash
uv run python scripts/generate_report.py
```

---

## Step 5 — Release

**Script:** `release_artifact.py`

Cuts an immutable `vN` classifier release: validates the trained artifact, pushes
`model.pt` + `model.json` + `efficientnet.pt` to
`s3://mermaid-config/classifier/<vN>/`, and creates a GitHub release `vN`.
Re-running an existing `vN` fails by design.

```bash
uv run python scripts/release_artifact.py
```

See the "Releasing a classifier version" section in
[../README.md](../README.md#releasing-a-classifier-version) for the full release
workflow (including the downstream inference-image build).
