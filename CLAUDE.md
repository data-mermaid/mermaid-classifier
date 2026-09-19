# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is one project in a multi-repo workspace. The workspace-root `../CLAUDE.md`
is also in scope and already covers: the **`gh` CLI** GitHub policy (this repo is
`data-mermaid/mermaid-classifier`), the **`aws-mcp`** AWS policy (the `aws` CLI is
denied), `uv` as the package manager, BA+GF taxonomy basics, and the cross-project
data flow. Don't re-derive those here.

## Commands

Python 3.12 (`.python-version`, `requires-python = ">=3.12"`). Use `uv`.

```bash
uv sync --extra training          # full dev/test stack (superset of [inference])
uv sync --extra inference         # serving-only: just pyspacer + pinned sklearn
uv sync --extra training --extra sagemaker   # adds the SageMaker SDK, without
                                  # which sagemaker_launcher/test_launch_training.py skips
uv sync --frozen --extra training --extra sagemaker  # what CI runs; fails if uv.lock is stale

# Tests — unittest, NOT pytest. Must run from the tests/ dir.
cd tests && uv run python -m unittest -v
cd tests && uv run python -m unittest -v pyspacer.test_train.ReadCoralNetDataTest
cd tests && uv run python -m unittest -v pyspacer.test_train.ReadCoralNetDataTest.test_method

# Scripts are run from the repo root (see each module docstring for args):
uv run python scripts/classifier_train.py            # local training run
uv run python scripts/generate_report.py             # MLflow run -> self-contained HTML report
uv run python scripts/generate_training_config.py    # writes a sagemaker/configs/<name>/ dir
uv run python scripts/release_artifact.py            # validate + stage a vN artifact

# SageMaker (host side) and its in-container counterpart:
uv run python scripts/launch_training.py             # submit a TrainingJob
uv run python scripts/launch_processing.py           # submit ProcessingJob(s)
#   scripts/sagemaker_train_entrypoint.py runs inside the container: YAML -> MLflowTrainingRunner

# CoralNet data preparation:
uv run python scripts/build_coralnet_manifest.py     # ETL parquets -> raw-image manifest parquet
uv run python scripts/build_feature_bucket.py        # CoralNet-layout feature-vector bucket
uv run python scripts/extract_reference_features.py  # stack .fv files into a reference matrix

# Region-mismatch probe (see region_eval below). A published version is immutable --
# --publish refuses a prefix that already holds anything -- so cutting a new probe
# publishes under a new version (v2 here); evaluation scores against the published v1.
AWS_PROFILE=wcs-admin uv run python scripts/build_region_probe.py --out-dir region_probe/v2 --seed 1 --publish s3://dev-datamermaid-sm-sources/region_probe/v2/
AWS_PROFILE=wcs-admin uv run python scripts/evaluate_region_probe.py --probe-dir s3://dev-datamermaid-sm-sources/region_probe/v1/ --model v1=../models/v1 --out-dir region_probe/reports
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

### The two dependency lanes (`[inference]` vs `[training]`)

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
`SklearnPinError` at export, which fails every test that exports an artifact —
45 of them across `test_portable_artifact.py`, `test_release_artifact.py`,
`test_mlflow_model.py`, `test_annotation_resolver.py` and `region_eval/test_score.py`.
If you bump sklearn, you must re-prove parity and update the pin + constant together.

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

Two strategy families for the long-tailed coral taxonomy:
- `sample_weighting/` — class-imbalance weighting via the effective-number
  formulation (Cui et al. 2019). A single `compute_class_weights` factory
  (`effective_number.py` + `options.py`); no registry — there is one strategy.
- `subsample/` — registry-based dataset subsampling: strategies `stratified`
  and `balanced`, listed in `SUBSAMPLE_STRATEGIES` and dispatched via the
  `_ALLOCATORS` dict in `registry.py` (config in `options.py`).

### Metrics (`pyspacer/metrics/`)

Post-training metric groups (classification, calibration, cover, probability,
ranking, taxonomic, per_source) orchestrated by
`MetricsCoordinator` / `MetricsContext`. The coordinator iterates a declarative
registry (`registry.py`: the ordered `METRIC_GROUPS` list +
`applicable_metric_groups`, which gates groups on available context like
`dataset`/`val_proba`), so adding a metric group is a one-line edit
there, not a coordinator change. HTML reports render from MLflow runs via
`scripts/generate_report.py` + `scripts/report_template.html.j2`.

### Region-mismatch evaluation (`mermaid_classifier/region_eval/`)

Measures how often the classifier applies a benthic-attribute label from a
region that label does not occur in, and prices the candidate fixes — a
region-blind permutation baseline, a masking counterfactual, within-branch
share, and confidence stratification, all in `decisions.py`. Scoring runs
against a **frozen** probe (`probe_set.py`): the sampled points, the region
map, display names, taxonomic ancestry, and corpus-wide annotation counts are
all pinned and hashed, so upstream taxonomy curation cannot move a published
score. `features.py` holds the feature cache both ways —
`write_feature_cache`/`FeatureCache`, `read_feature_cache`/`ProbeFeatures` —
plus `feature_coverage`. `score.py` loads a probe and scores a model against
it, reporting a drift diagnostic when the live region map has moved since the
probe was frozen; `report.py` renders the nine report artifacts from the
resulting `ModelScore`, depending on `score` in one direction only. The pure
modules — `metrics.py`, `triage.py`, `decisions.py`, `region_rules.py` — take
the region map as caller-supplied frozen data and never import the live
benthic-attribute library; `decisions.py`'s four mitigation functions and
`triage_events` all take the `ScoredPoints` that the otherwise-pure
`metrics.py` prepares. Reached only through the two CLI scripts named in the
Commands block above: `evaluate_region_probe.py`'s `--min-coverage` refuses to
score a feature cache below a given coverage fraction (no floor by default),
and `build_region_probe.py` writes `held_out_images.csv`, the exclusion list
the training pipeline's `ImageExclusionFilter` consumes. Nothing under
`pyspacer/` imports it.

`metrics.py` deliberately holds both the rate statistics and the tables that
present them: the per-region and per-direction builders call the estimators
rather than format precomputed values, so splitting the two would need the
estimators exported as private cross-module imports plus a deferred import to
break the resulting cycle — more machinery than the separation buys.

### SageMaker launcher and CoralNet ingest

- `mermaid_classifier/sagemaker/{config,launcher_config}.py` is the host-side config
  layer behind `scripts/launch_training.py` / `launch_processing.py`. It shares the one
  `training_config.yaml` with local runs, so there is no recipe duplication between
  lanes. Cross-repo conventions live in `../mermaid-api/iac/sagemaker-launcher-convention.md`.
- `mermaid_classifier/coralnet/manifest.py` (+ `scripts/build_coralnet_manifest.py`)
  builds the raw-image CoralNet manifest Parquet that dataset loading reads from S3.
- `pyspacer/swap_monitor.py`, `pyspacer/mlflow_model.py`, `pyspacer/annotation.py` and
  `common/plots.py` support the above; none is an entry point.

## Conventions and gotchas

- **A test package must not share a name with an installed dependency.**
  The suite runs from `tests/`, which puts it on `sys.path`, so a
  `tests/<name>/` package shadows `<name>` for the whole session. `tests/sagemaker/`
  would shadow the SageMaker SDK, and a module that guards itself with
  `find_spec` then skips silently even where the SDK is installed — which is
  why the launcher's config tests live in `tests/sagemaker_launcher/`.
- **A new test package needs an `__init__.py`.** Discovery descends into
  packages only, so a directory without one runs zero tests in the full suite
  while `unittest <pkg>.<module>` still passes — green for its author and green
  in CI, having executed nothing. `tests/test_suite_layout.py` fails if one is
  missing.
- **Shared fixtures live in `tests/support/`** — `paths` (repo root, and
  putting `scripts/` on `sys.path`), `settings`, `dataset`, `calibrated_model`,
  `coralnet_tables`. The rule is scope: a fixture used inside one test package
  stays there (`pyspacer/metrics_test_helpers.py`, `region_eval/fixtures.py`),
  and one crossing packages goes in `support/`. No test module imports from
  another test module.
- **`unittest -v <package>` silently runs 0 tests**: a bare package name
  (`region_eval`, `common`, …) exposes nothing to unittest's loader; name
  modules explicitly (`region_eval.test_metrics`). The full suite does
  discover them, so a green full-suite run is not evidence that a narrower
  invocation ran anything.
- **Importing from `pyspacer/metrics/` pulls in boto3, duckdb, matplotlib,
  mlflow, sklearn, and spacer** — its `__init__.py` imports
  `MetricsCoordinator`, which reaches every metric group. A lightweight
  consumer must not import a helper from there.
- **DuckDB is the ETL engine**, not pandas. SQL transforms via helpers in
  `common/duckdb_utils.py` (temp-table context managers, column transforms,
  batched iteration).
- **Empty growth forms are `''`, never NULL** in DuckDB — NULL breaks JOINs.
  BA+GF separator is `::` and the trailing `::` stays even with no GF
  (`Hard coral::`). Tests assert this.
- **Settings**: pydantic `Settings` reads an `.env` from the cwd; names are
  lowercase in code, UPPERCASE in `.env`. See `.env.example` (repo root) for the
  full set (`CORALNET_TRAIN_DATA_BUCKET`, `WEIGHTS_LOCATION`, `AWS_ANONYMOUS`,
  `MLFLOW_TRACKING_SERVER`, `SPACER_BATCH_SIZE`, …). `SPACER_BATCH_SIZE` is
  auto-derived from available RAM when unset.
- **Test isolation**: `support.settings.override_settings()` /
  `SettingsOverride` patch the settings singleton (always via the context
  manager or `addCleanup`, since an unrestored override leaks into every later
  test in the process); `support.dataset.NoInitDataset` bypasses the
  S3/API-hitting `TrainingDataset.__init__`;
  `CoralNetMermaidMapping._download_mapping` is mocked. The suite makes no
  outbound network connections, except DuckDB's `httpfs` extension install
  fallback (`dataset.py`'s `duck_conn`, `build_coralnet_manifest.py`'s
  `_configure_duckdb_s3`) when the extension isn't already installed locally —
  anything new that would must be stubbed.
- **Config dirs are repo-root-relative**: a committed training config is a
  `sagemaker/configs/<name>/` dir (`training_config.yaml` plus whichever of
  `sources.csv` / `rollups.csv` / `included_labels.csv` that run needs — the
  siblings are optional; `coralnet_all_plus_mermaid/` carries no `sources.csv`).
  Scripts run from the
  repo root; both `classifier_train.py` (local) and the SageMaker launcher load
  a config by repo-root-relative `--config-dir` and share that one
  `training_config.yaml` (single source of truth — no recipe duplication).
  `generate_training_config.py` writes there by default, but its raw *inputs*
  (the curated source list, the Drive-exported label mapping) live in the
  surrounding workspace, not this repo.

## Releasing a classifier version

Run the **Release classifier version** workflow (`.github/workflows/release.yml`);
`README.md` has the step-by-step. Three invariants matter when reading or changing
the release path:

- **Versions are immutable.** Re-running an existing `vN` fails.
- **Two version numbers, not one.** Model version `vN` (a retrain) and serving build
  `K` (a code/library fix) compose into the inference image tag `vN-K`. Cutting model
  `vN` is followed by building image `vN-1` in mermaid-inference, which bakes
  `CLASSIFIER_VERSION=vN`.
- **`model.json`'s `trained_with`** records the torch/sklearn/pyspacer the model was
  built with, and the inference function fails loudly at load if its runtime differs.

## Pointers

- `README.md` — installation matrix, SageMaker vs local tradeoffs, release detail.
- `docs/` — MLflow setup, SageMaker runbooks, feature-extraction/training-at-scale.
- `docker/jobs/` — Dockerfiles for SageMaker training (CPU) / feature extraction (GPU).
- `../docs/adr/` — workspace ADRs behind the compute-lane and artifact-format decisions.
