# mermaid-classifier

This Python repository enables data scientists to experiment with PySpacer-based classifiers. It also has MERMAID-relevant utilities which aren't specific to the type of classifier being developed.


## Overview

This project is set up as a Python package, and requires Python 3.12 or higher. Once you have the package installed in your Python environment, you can import anything from `mermaid_classifier` into your own Python modules, notebooks, etc.

### General utilities

These are found in `mermaid_classifier.common`. Once this package is installed (see Installation section), the utilities can be imported from there.

### PySpacer training and classification code

This is found in `mermaid_classifier.pyspacer`.

### Scripts

The `scripts/` directory holds command-line entry points that *drive* the package, as opposed to the importable `mermaid_classifier` library code described above. Each script is run with `uv run python scripts/<name>.py` from the repo root (see the script's module docstring for arguments):

- `classifier_train.py` — run a training job locally (the production recipe).
- `generate_report.py` — render a self-contained HTML report from an MLflow run (uses `report_template.html.j2`).
- `generate_training_config.py` — generate a training-config directory (CoralNet→MERMAID label mapping + rollup).
- `build_feature_bucket.py` — build a CoralNet-layout feature-vector bucket for an updated source set (idempotent / resumable).
- `extract_reference_features.py` — stack real EfficientNet feature vectors into a `.npy` for the TorchScript-vs-sklearn parity gate.
- `launch_processing.py` — launch SageMaker ProcessingJob(s) for feature extraction (optional sharding).
- `launch_training.py` — launch a SageMaker TrainingJob via an Estimator (script-agnostic wrapper).
- `sagemaker_train_entrypoint.py` — SageMaker container entrypoint: reads YAML config → runs `MLflowTrainingRunner`.
- `release_artifact.py` — release a trained MLflow classifier as immutable version `vN` to `s3://mermaid-config/classifier/<vN>/`.

See docs/workflow.md for the order to run these in.

### Documentation

See the [docs](docs) section for usage explanations.


## Installation

### Python package installation

This project uses [`uv`](https://docs.astral.sh/uv/). From a clone of the repo:

| Result | Command |
| - | - |
| Serving-only (load/run a trained classifier) | `uv sync --extra inference` |
| Full training pipeline (superset of inference) | `uv sync --extra training` |
| Exactly what CI installs (fails if `uv.lock` is stale) | `uv sync --frozen --extra training` |

The `inference` extra is intentionally minimal (just `pyspacer` + a pinned
`scikit-learn`) so serving images stay light. `training` is a superset adding
MLflow, DuckDB, pandas, the settings layer, etc. Add `--extra jupyterlab` for
JupyterLab support.

To consume this package from another project, `pip install` the git URL with the
extra you need, e.g. `pip install "mermaid-classifier[inference] @ git+https://github.com/data-mermaid/mermaid-classifier.git"`.

### Additional steps for PySpacer classifiers

1. If you're in JupyterLab, you need to have interactive matplotlib working to have pan, zoom, and save controls on annotation plots. If you want this, after pip-installing ipympl, you must [hard-refresh](https://www.howtogeek.com/672607/how-to-hard-refresh-your-web-browser-to-bypass-your-cache/) the browser tab that has the JupyterLab space open.

1. You'll need to specify configuration values, using either an `.env` file in the directory that you're running your script or notebook from, or by setting environment variables. See the `pyspacer_example` directory for a full example.

### Installation environment

Although MERMAID IC is primarily targeting an AWS SageMaker environment, this package can also be set up on a local dev machine.

AWS SageMaker advantages over local:

- Easily and securely access private S3 files through spaces, as long as the SageMaker domain is set up with an applicable Space execution role.
- Web-based IDE spaces with real-time collaboration.
- MLflow tracking servers can be shared by everyone who can access the SageMaker domain.
- Default distribution image already has many Python packages relevant to this project. This could be preferable over maintaining a 3 GB local venv.

Local env advantages over SageMaker:

- Don't have to worry about the AWS web session expiring every so often, and don't need constant internet to keep working.
- More IDE choices, not just VSCode (Code Editor spaces) or JupyterLab.
- Can run a local MLflow tracking server with very low startup and cost.
- Easier to customize and persist the packages that are installed in the environment.

If you're on a local dev machine and accessing public S3 files, the `AWS_ANONYMOUS` setting may be useful.


## Releasing a classifier version

Trigger the **Release classifier version** workflow (`workflow_dispatch`) with the
MLflow model ID and a `vN` tag. It fetches the trained `model.pt` + `model.json`,
re-validates them (load + manifest gate), pushes
`s3://mermaid-config/classifier/<vN>/{model.pt,model.json,efficientnet.pt}`, and cuts
GitHub release `vN`. Versions are immutable — re-running an existing `vN` fails.

- **Inference image is built per model version.** Cutting model `vN` is followed
  by building the inference image `vN-K` (model version + serving build) in
  mermaid-inference, which bakes `CLASSIFIER_VERSION=vN` and pins the matching
  pyspacer/sklearn. A code/library fix bumps the build `K`; a retrain bumps `vN`.
  The deployed function serves exactly its `vN`.

Required repo secrets: `AWS_RELEASE_ROLE_ARN` (OIDC assume-role with read on the
MLflow artifact store + read/write on `s3://mermaid-config/classifier/*`) and
`MLFLOW_TRACKING_URI`.

## For developers

Set up this project as an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/): first git-clone this repo, then use `pip install -e <path to repo>`.

### Unit tests

These can be run by, for example, changing the working directory to `tests` and then running `python -m unittest`.

### Linting, formatting & type checking

These are enforced on every PR by `.github/workflows/lint.yml` (alongside the
unittest workflow in `tests.yml`); both must pass to merge. There are no
pre-commit hooks — a forgotten local check simply fails the PR.

Run the checks locally with the `lint` dependency group:

    uv run --group lint ruff check .          # lint
    uv run --group lint ruff format .         # auto-format
    uv run --group lint ruff format --check . # CI's format gate (no writes)
    uv run --group lint basedpyright          # type check

basedpyright runs in strict mode over `mermaid_classifier/` + `scripts/`, with
the "unknown type" rules that fire on untyped third-party libraries
(torch/pyspacer/duckdb/mlflow/sklearn) disabled — see `[tool.basedpyright]` in
`pyproject.toml`. `tests/` are linted and formatted but not type-checked; the
legacy `v1/` directory is excluded from all checks.

### Design notes

This project is set up as a Python package with a [flat project layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).

Although this project isn't on PyPI, the fact that it's set up as a package makes it easier to:

- Import from this project, compared to an ad-hoc addition to `sys.path`, for example.

- Manage project dependencies.


## Architecture

### Package Layout (flat layout, not src/)

- `mermaid_classifier/common/` -- Shared utilities (MERMAID API clients, DuckDB helpers, CSV parsing, plotting)
- `mermaid_classifier/pyspacer/` -- PySpacer training and classification pipeline
- `tests/pyspacer/` -- Unit tests (unittest, not pytest)
- `pyspacer_example/` -- Jupyter notebook demo with sample `.env`
- `v1/` -- Legacy code, not incorporated into current version

### Training Pipeline (train.py)

The core pipeline flows through `TrainingDataset` -> `TrainingRunner` / `MLflowTrainingRunner`:

1. **Data loading**: `read_coralnet_data()` reads per-source CSVs from S3 via DuckDB; `read_mermaid_data()` reads a Parquet file. Both write to a DuckDB `annotations` table.
2. **Label mapping**: CoralNet label IDs are mapped to MERMAID BA+GF using `CoralNetMermaidMapping` (fetched from MERMAID API, cached after first load).
3. **Filtering & rollup**: `LabelFilter`, `LabelRollupSpec`, `CNSourceFilter` (all `CsvSpec` subclasses) control which labels/sources are included and how labels are consolidated.
4. **Feature vector validation**: Checks S3 for missing `.fv` files; drops or aborts based on `TRAINING_INPUTS_PERCENT_MISSING_ALLOWED`.
5. **Train/ref/val split**: Uses PySpacer's `preprocess_labels()` with `SplitMode`.
6. **Training**: Calls PySpacer's `train_classifier()` with `TrainClassifierMsg`.
7. **Logging**: `MLflowTrainingRunner` logs model, metrics (precision/recall/F1), confusion matrices, and profiling data to MLflow.

### Key Patterns

- **Configuration**: Pydantic `Settings` class reads from `.env` file in cwd. Setting names are lowercase in code, UPPERCASE in `.env`. See `pyspacer_example/.env` for all options.
- **DuckDB as ETL engine**: All data transformations use SQL via DuckDB (not pandas). Helpers in `duckdb_utils.py` provide context managers for temp tables, column transforms, batched iteration, etc.
- **NULL growth forms**: CoralNet labels without a growth form must be stored as empty string `''` in DuckDB (not NULL), because NULL breaks JOINs. Tests specifically verify this.
- **BA+GF separator**: `::` separates benthic attribute from growth form (e.g., `Acropora::Branching` or `Hard coral::` for no GF). Empty growth forms must still have the trailing `::`.
- **CsvSpec pattern**: `LabelFilter`, `LabelRollupSpec`, and `CNSourceFilter` all inherit from `CsvSpec`, which validates CSV columns and initializes from a file-like object.
- **Context managers for resource cleanup**: `section_profiling()` for timing, `make_confusion_matrix()` for matplotlib figures, `duckdb_temp_table_name()` for temp tables.

### Testing Patterns

- Tests use `unittest` (not pytest).
- `SettingsOverride` / `override_settings()` context manager patches Pydantic settings for test isolation.
- `NoInitDataset` bypasses expensive `TrainingDataset.__init__` (which hits S3 and MERMAID API) to test individual methods.
- `CoralNetMermaidMapping._download_mapping` is mocked to avoid live API calls.

### Configuration (Settings)

Key settings (set via `.env` or environment variables):

| Setting | Purpose |
|---|---|
| `CORALNET_TRAIN_DATA_BUCKET` / `MERMAID_TRAIN_DATA_BUCKET` | S3 buckets for training data |
| `WEIGHTS_LOCATION` | Path to EfficientNet extractor weights |
| `AWS_ANONYMOUS` | `True` for public S3 access without credentials |
| `MLFLOW_TRACKING_SERVER` | MLflow server URI (ARN or localhost) |
| `SPACER_EXTRACTORS_CACHE_DIR` | Cache dir for downloaded weights |
| `SPACER_BATCH_SIZE` | Override for training batch size; if unset, auto-calculated from available memory at training time |
| `MLFLOW_HTTP_REQUEST_MAX_RETRIES` | Default 7 is slow; 2 recommended |
