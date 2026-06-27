# mermaid-classifier docs

Start here: **[../README.md](../README.md)** for install + overview, and
**[workflow.md](workflow.md)** for the end-to-end script workflow. Architecture
and conventions live in **[../CLAUDE.md](../CLAUDE.md)**.

## Getting started
- [workflow.md](workflow.md) — the end-to-end script workflow (config → features → train → report → release) and when to use local vs SageMaker.

## Setup
- [mlflow.md](mlflow.md) — set up an MLflow tracking server (local or SageMaker).
- [mermaid_sagemaker.md](mermaid_sagemaker.md) — access SageMaker for the MERMAID team (IDEs, kernels, gotchas).

## Runbooks
- [pyspacer/train.md](pyspacer/train.md) — train a PySpacer classifier: the YAML training config, data sources, label rollup/filtering, subsampling/weighting, and the programmatic API.
- [pyspacer/annotation.md](pyspacer/annotation.md) — run classification and view annotations on images.
- [feature_extraction_at_scale.md](feature_extraction_at_scale.md) — feature extraction via parallel SageMaker Processing Jobs.
- [training_at_scale.md](training_at_scale.md) — training via a SageMaker TrainingJob.

## Research (archived findings)
- [research/hidden-layer-experiments.md](research/hidden-layer-experiments.md) — architecture search: hidden-layer sizes, learning rates, epochs.
- [research/balancing-experiments.md](research/balancing-experiments.md) — label-balancing experiments (sample weighting, subsampling).
