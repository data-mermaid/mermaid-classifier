# docs/

User-facing documentation for data scientists using mermaid-classifier.

## Files

| File | What | When to read |
| ---- | ---- | ------------ |
| `README.md` | Table of contents linking to all documentation | Finding the right doc for a task |
| `mlflow.md` | MLflow setup and usage guide | Setting up MLflow tracking server, interpreting experiment results |
| `mermaid_sagemaker.md` | SageMaker-specific setup and usage for the MERMAID team | Setting up in AWS SageMaker, configuring spaces and execution roles |
| `feature_extraction_at_scale.md` | Runbook for parallel feature extraction via SageMaker Processing Jobs: ECR setup, IAM role, launcher usage, smoke test, resume semantics | Backfilling features for many CoralNet sources at once, scaling up `build_feature_bucket.py` |
| `training_at_scale.md` | One-time AWS setup runbook + launch recipes + cost-tuning table for the SageMaker training launcher | Running training as a SageMaker TrainingJob, debugging failed runs |

## Subdirectories

| Directory | What | When to read |
| --------- | ---- | ------------ |
| `pyspacer/` | PySpacer-specific docs: training and annotation workflows | Learning the training workflow, understanding annotation/classification usage |
