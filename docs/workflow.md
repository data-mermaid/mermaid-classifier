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

## Step 2 — Extract features (build the feature-vector bucket)

Training reads **feature vectors**, not raw images. This step runs pyspacer's
EfficientNet extractor over every annotated image in a set of CoralNet sources
and writes a CoralNet-layout feature-vector bucket
(`s{source_id}/features/i{image_id}.featurevector`) that training then points at
via `CORALNET_TRAIN_DATA_BUCKET`. It's the heavy, scale-sensitive step — a large
CoralNet source set is a lot of images to run through the extractor.

The two scripts do the *same* extraction; they differ in how they run it:

| Script | Role |
| - | - |
| `build_feature_bucket.py` | The extractor itself. Single-process: walks the given CoralNet sources sequentially and writes their feature vectors. Idempotent and resumable (a re-run skips images already extracted), so it can grind through a large source set across restarts. Run it directly to build or update the bucket. |
| `launch_processing.py` | Runs that same extraction **in parallel** for throughput: fans out N sharded SageMaker ProcessingJobs over the source set. Reach for this when a single process would take too long. See [feature_extraction_at_scale.md](feature_extraction_at_scale.md). |

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
