# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is one project in a multi-repo workspace. The workspace-root `../CLAUDE.md`
is also in scope and already covers: the **`aws-mcp`-only** AWS policy (never the
`aws` CLI), the **`github`-MCP-only** GitHub policy (never the `gh` CLI; this repo
is `data-mermaid/mermaid-classifier`), `uv` as the package manager, BA+GF taxonomy
basics, and the cross-project data flow. Don't re-derive those here.

## Commands

Python 3.12 (`.python-version`, `requires-python = ">=3.12"`). Use `uv`.

```bash
uv sync --extra training          # full dev/test stack (superset of [inference])
uv sync --extra inference         # serving-only: just pyspacer + pinned sklearn
uv sync --frozen --extra training # what CI runs; fails if uv.lock is stale

# Tests — unittest, NOT pytest. Must run from the tests/ dir.
cd tests && uv run python -m unittest -v
cd tests && uv run python -m unittest -v pyspacer.test_train.ReadCoralNetDataTest
cd tests && uv run python -m unittest -v pyspacer.test_train.ReadCoralNetDataTest.test_method

# Scripts are run from the repo root (see each module docstring for args):
uv run python scripts/classifier_train.py
uv run python scripts/generate_report.py        # MLflow run -> self-contained HTML report
uv run python scripts/generate_training_config.py
uv run python scripts/release_artifact.py
```

CI runs two workflows on every PR — `tests.yml` (unittest suite, Linux/3.12) and
`lint.yml` (ruff lint + format check + basedpyright). Both must pass. Run locally
via the `lint` group: `uv run --group lint ruff check .` / `ruff format .` /
`basedpyright`. basedpyright is strict over `mermaid_classifier/` + `scripts/`
with the untyped-third-party "unknown type" rules disabled (see
`[tool.basedpyright]`); `tests/` are linted/formatted but not type-checked.
No pre-commit hooks.

## Big picture

A library for training and serving PySpacer-based coral-image classifiers, plus
MERMAID utilities. Flat package layout (no `src/`). Importable code lives under
`mermaid_classifier/`; `scripts/` are CLI drivers; `tests/` mirror the package.

### The two dependency lanes (`[inference]` vs `[training]`) — load-bearing

This split is an architectural invariant, not just packaging. `[inference]` is
deliberately minimal (`pyspacer` + a **pinned** `scikit-learn==1.5.2`) so serving
images stay light. `[training]` is a superset adding MLflow, DuckDB, pandas,
pydantic-settings, etc.

- `mermaid_classifier/pyspacer/inference/` (export/loader/head) and
  `torch_classifier.py` must import **only** torch/numpy/stdlib — never the
  training-only settings layer. Importing the package has no settings side
  effect; training entry points call env setup explicitly. Breaking this leaks
  training deps into the serving lane. `test_inference_decoupling.py` guards it.

### Portable, pickle-free artifact + the sklearn parity gate

Trained models ship as a TorchScript head + `model.json` manifest, **not** a
pickle (`inference/export.py` → `export_artifact`, `inference/loader.py` →
`load_predictor`). `scikit-learn` is pinned in lockstep across both extras
because `CalibratedClassifierCV` calibration semantics can shift between
releases. `PARITY_PROVEN_SKLEARN` (`pyspacer/inference/__init__.py`) records the
version the TorchScript-vs-sklearn parity was proven against; a mismatch raises
`SklearnPinError` at export and fails a guard test (`test_sklearn_pin.py`). If
you bump sklearn, you must re-prove parity and update the pin + constant together.

### Training pipeline (`pyspacer/dataset.py`, `pyspacer/runner.py`)

`TrainingDataset` (`dataset.py`) → `TrainingRunner` / `MLflowTrainingRunner`
(`runner.py`). Flow: load CoralNet per-source CSVs from S3 + MERMAID Parquet via
DuckDB → map CoralNet label IDs to MERMAID BA+GF (`CoralNetMermaidMapping`) →
filter/rollup (`LabelFilter`, `LabelRollupSpec`, `CNSourceFilter`, all `CsvSpec`
subclasses in `label_specs.py`) → validate `.fv` feature vectors exist on S3 →
train/ref/val split via PySpacer's `preprocess_labels` → train via
`MermaidTrainer` (`trainer.py`, a PySpacer `ClassifierTrainer` subclass doing
batched calibration + per-epoch MLflow callbacks; ref and train data are never
both in memory) → log model, metrics, confusion matrices to MLflow. The
`DatasetOptions` / `TrainingOptions` / `MLflowOptions` dataclasses live in
`options.py`; cross-cutting helpers (`section_profiling`,
`download_features_parallel`) in `_pipeline_utils.py`.

### Pluggable strategies (`mermaid_classifier/training/`)

Newer than the README. Two strategy families for the long-tailed coral
taxonomy:
- `sample_weighting/` — class-imbalance weighting via the effective-number
  formulation (Cui et al. 2019). A single `compute_class_weights` factory
  (`effective_number.py` + `options.py`); no registry — there is one strategy.
- `subsample/` — registry-based dataset subsampling: strategies `stratified`
  and `balanced`, listed in `SUBSAMPLE_STRATEGIES` and dispatched via the
  `_ALLOCATORS` dict in `registry.py` (config in `options.py`).

### Metrics (`pyspacer/metrics/`)

Post-training metric groups (classification, calibration, cover, probability,
ranking, taxonomic, per_source) orchestrated by `MetricsCoordinator` /
`MetricsContext`. HTML reports render from MLflow runs via
`scripts/generate_report.py` + `scripts/report_template.html.j2`.

## Conventions and gotchas

- **DuckDB is the ETL engine**, not pandas. SQL transforms via helpers in
  `common/duckdb_utils.py` (temp-table context managers, column transforms,
  batched iteration).
- **Empty growth forms are `''`, never NULL** in DuckDB — NULL breaks JOINs.
  BA+GF separator is `::` and the trailing `::` stays even with no GF
  (`Hard coral::`). Tests assert this.
- **Settings**: pydantic `Settings` reads an `.env` from the cwd; names are
  lowercase in code, UPPERCASE in `.env`. See `pyspacer_example/.env` for the
  full set (`CORALNET_TRAIN_DATA_BUCKET`, `WEIGHTS_LOCATION`, `AWS_ANONYMOUS`,
  `MLFLOW_TRACKING_SERVER`, `SPACER_BATCH_SIZE`, …). `SPACER_BATCH_SIZE` is
  auto-derived from available RAM when unset.
- **Test isolation**: `override_settings()` / `SettingsOverride` patch settings;
  `NoInitDataset` bypasses the S3/API-hitting `TrainingDataset.__init__`;
  `CoralNetMermaidMapping._download_mapping` is mocked.

## Releasing a classifier version

Trigger the **Release classifier version** GitHub workflow (`workflow_dispatch`,
`.github/workflows/release.yml`) with an MLflow model ID and a `vN` tag. It
fetches `model.pt` + `model.json`, re-validates (load + manifest gate), pushes to
`s3://mermaid-config/classifier/<vN>/`, and cuts release `vN`. Versions are
**immutable** — re-running an existing `vN` fails. The inference function image
is built per model version and tagged `vN-K` (`vN` = model version, `K` =
serving build): cutting model `vN` is followed by building the inference image
`vN-1` in mermaid-inference (which bakes `CLASSIFIER_VERSION=vN` and pins the
matching pyspacer/sklearn). Bump the build `K` for a code/library fix; bump the
model version `vN` for a retrain. `model.json`'s `trained_with` records the
torch/sklearn/pyspacer the model was built with, and the function fails loudly
at load if its runtime doesn't match.

## Pointers

- `README.md` — installation matrix, SageMaker vs local tradeoffs, release detail.
- `docs/` — MLflow setup, SageMaker runbooks, feature-extraction/training-at-scale.
- `docker/jobs/` — Dockerfiles for SageMaker training (CPU) / feature extraction (GPU).
