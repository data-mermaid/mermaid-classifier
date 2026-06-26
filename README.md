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

basedpyright runs in strict mode over `mermaid_classifier/` + `scripts/` (see `[tool.basedpyright]` in `pyproject.toml` for the exact rule set).

### Design notes

This project is set up as a Python package with a [flat project layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).

Although this project isn't on PyPI, the fact that it's set up as a package makes it easier to:

- Import from this project, compared to an ad-hoc addition to `sys.path`, for example.

- Manage project dependencies.


## Architecture

This project is a flat-layout Python package: importable code under
`mermaid_classifier/` (`common/` utilities, `pyspacer/` training + inference,
`training/` strategies, `metrics/`), with `scripts/` as CLI drivers and `tests/`
mirroring the package.

Two dependency lanes are an architectural invariant: **`[inference]`** is
deliberately minimal so serving images stay light, while **`[training]`** is a
superset. Trained models ship as a portable, pickle-free **TorchScript head +
`model.json` manifest** (not a sklearn pickle); `scikit-learn` is pinned in
lockstep across both extras because calibration semantics can shift between
releases.

For the full architecture, conventions, training-pipeline flow, settings, and
testing patterns, see **[CLAUDE.md](CLAUDE.md)** — it is the source of truth and
is kept current.
