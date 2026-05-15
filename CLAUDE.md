# mermaid-classifier

PySpacer-based coral reef image classifier training library with MERMAID API integration.

## Files

| File | What | When to read |
| ---- | ---- | ------------ |
| `README.md` | Architecture, training pipeline overview, installation, configuration reference, testing patterns | Understanding the system, onboarding, setting up environment, reviewing design decisions |
| `pyproject.toml` | Package metadata, dependencies, optional extras (`pyspacer`, `jupyterlab`) | Adding dependencies, changing install extras, checking Python version requirement |

## Subdirectories

| Directory | What | When to read |
| --------- | ---- | ------------ |
| `mermaid_classifier/` | Core Python package: shared utilities and PySpacer pipeline | Modifying classifier logic, adding features, debugging training |
| `tests/` | Unit tests (unittest, not pytest) | Running tests, adding test coverage, debugging test failures |
| `scripts/` | CLI entry points: train, evaluate, generate HTML report | Running training jobs, evaluating models, generating reports |
| `docs/` | User-facing usage documentation for training and annotation | Understanding usage workflows, onboarding data scientists |
| `docker/` | Container definitions: `training/` (SageMaker TrainingJob image) and `feature_extraction/` (SageMaker Processing Job image for distributed feature extraction) | Building either container image |
| `pyspacer_example/` | Example notebook, sample `.env`, and sample data for a minimal training run | Learning usage patterns, reproducing a demo run |
| `v1/` | Legacy v1 classifier code (not incorporated into current version) | Reviewing historical implementation only |

## Build

```bash
pip install -e .[pyspacer]  # Install with training deps (local dev editable install)
```

## Test

```bash
cd tests && python -m unittest                                                      # Run all tests
cd tests && python -m unittest pyspacer.test_train.ReadCoralNetDataTest.test_method # Single test
```
