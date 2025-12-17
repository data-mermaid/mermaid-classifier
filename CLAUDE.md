# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python package for experimenting with PySpacer-based classifiers for coral reef image classification. It includes MERMAID-relevant utilities that aren't specific to any particular classifier type.

**Stack:** Python 3.10+, PySpacer, MLflow, boto3, DuckDB, pydantic-settings

**Documentation:** See [docs](docs) directory for detailed usage explanations

## Installation

This project uses a **flat package layout** (not src-based) and is installed as a Python package.

### Standard Installation

```bash
# Utilities only
pip install https://github.com/data-mermaid/mermaid-classifier.git

# Utilities + PySpacer-based classification
pip install https://github.com/data-mermaid/mermaid-classifier.git[pyspacer]

# With JupyterLab support
pip install https://github.com/data-mermaid/mermaid-classifier.git[pyspacer,jupyterlab]

# From a specific branch
pip install "mermaid-classifier[pyspacer,jupyterlab] @ git+https://github.com/data-mermaid/mermaid-classifier.git@my-branch-name"
```

**Note:** To update, add `-U` after `install`. If the version number hasn't been bumped, you may need to `pip uninstall mermaid-classifier` first.

### Developer Installation

For local development, use an editable install:

```bash
# Clone the repo, then:
pip install -e /path/to/repo
pip install -e /path/to/repo[pyspacer,jupyterlab]  # with optional dependencies
```

### SageMaker Notes

- After shutting down and restarting a SageMaker space, re-run pip installations
- Run pip install from a Terminal tab
- Ignore the ERROR message about pip's dependency resolver at the end (relates to SageMaker-preinstalled packages)

## Configuration

Configuration is done via environment variables or a `.env` file in your working directory.

### Required Environment Variables

See `pyspacer_example/.env` for a full example. Key variables:

- `MLFLOW_TRACKING_SERVER` - URI of MLflow tracking server (ARN for SageMaker or localhost URL)
- `WEIGHTS_LOCATION` - Location of feature-extractor weights (local path or S3 URI)
- `SPACER_EXTRACTORS_CACHE_DIR` - Filesystem directory for caching extractor weights downloaded from S3/URL
- `TRAINING_INPUTS_PERCENT_MISSING_ALLOWED` - Max percent of missing feature vectors before training aborts (default: 0)
- `SPACER_AWS_ANONYMOUS` - Set to `True` to access AWS without credentials (for public S3 files)
- `MLFLOW_HTTP_REQUEST_MAX_RETRIES` - Number of retries for MLflow connection issues (default 7, recommend 2)
- `MLFLOW_DEFAULT_EXPERIMENT_NAME` - Default experiment name for MLflow logging

**Note:** Variable names are case-sensitive on Linux/Mac. Use UPPER_CASE in environment variables or `.env` files.

## Package Structure

```
mermaid_classifier/
├── common/                        # General MERMAID utilities
│   ├── benthic_attributes.py      # Benthic attribute and growth form libraries
│   └── plots.py                   # Plotting utilities
├── pyspacer/                      # PySpacer-based classification
│   ├── __init__.py                # Initializes env vars for PySpacer/MLflow
│   ├── settings.py                # Pydantic settings (reads .env)
│   ├── train.py                   # Classifier training
│   ├── annotation.py              # Classification and annotation viewing
│   └── utils.py                   # Utilities (logging, MLflow connection)
```

**v1/** - Legacy MERMAID classifier v1 work not yet incorporated into current version

## Architecture

### Configuration System

Uses `pydantic-settings` for configuration management:
- Settings defined in `mermaid_classifier.pyspacer.settings.Settings`
- Reads from `.env` in current working directory (NOT from installed package location)
- Variable names are lower_case in Python, UPPER_CASE in environment
- `set_env_vars_for_packages()` propagates settings to PySpacer and MLflow as OS env vars

**Important:** The `pyspacer` module's `__init__.py` automatically calls `set_env_vars_for_packages()` on import, so PySpacer and MLflow are ready to use after importing from `mermaid_classifier.pyspacer`.

### MLflow Integration

Training and classification integrate with MLflow for experiment tracking:
- **Local development:** Run `mlflow ui --port 8080` in a terminal with the environment activated, visit `http://localhost:8080`
- **SageMaker:** Start MLflow tracking server via SageMaker Studio > MLflow (takes ~20 minutes)

See `docs/mlflow.md` for detailed setup instructions.

### PySpacer Workflow

1. **Training:** Use `mermaid_classifier.pyspacer.train` to train classifiers on feature vectors and annotations from S3
2. **Classification:** Use `mermaid_classifier.pyspacer.annotation` to classify images and view annotations
3. Feature vectors are cached locally using `SPACER_EXTRACTORS_CACHE_DIR`

See documentation in `docs/pyspacer/train.md` and `docs/pyspacer/annotation.md` for usage.

### JupyterLab Support

- Install with `[jupyterlab]` extra for interactive matplotlib (ipympl)
- **After installing ipympl, hard-refresh the browser tab** for pan/zoom/save controls on plots
- See full example in `pyspacer_example/example.ipynb`

## Development Environment

Can be developed locally or on AWS SageMaker.

**SageMaker advantages:**
- Secure access to private S3 files via Space execution role
- Web-based IDE with real-time collaboration
- Shared MLflow tracking servers
- Many Python packages pre-installed

**Local advantages:**
- No AWS session expiration
- More IDE choices (not just VSCode/JupyterLab)
- Local MLflow tracking server with low startup cost
- Easier to customize and persist installed packages

## Important Notes

- This is a **Python package**, not a Django/web application
- No Makefile or fabfile - all commands are Python-based
- Uses hatchling as build backend (see `pyproject.toml`)
- PySpacer is installed from a specific git branch with AWS profile support: `pyspacer@git+https://github.com/coralnet/pyspacer.git@aws-profile-support-etc`
- No test framework is currently configured in this repository
- Dependencies are split: base (`boto3` only) and optional (`pyspacer`, `jupyterlab`)
