"""Run a classifier training job locally.

This is the production training recipe: it builds DatasetOptions /
TrainingOptions / MLflowOptions and runs MLflowTrainingRunner against the
configured CoralNet + MERMAID data, logging the model and metrics to MLflow.

Run from the repo root with AWS SSO access to the training buckets:

    uv run python scripts/classifier_train.py

It reads AWS credentials from the `wcs` SSO profile and configuration from the
`.env` in the cwd (see `pyspacer_example/.env`). For the SageMaker equivalent,
see scripts/launch_training.py and docs/training_at_scale.md. For the script
ordering overall, see docs/workflow.md.
"""

import os

import boto3

# Resolve AWS SSO credentials before importing mermaid_classifier,
# whose Settings() object is created at import time and reads env vars then.
os.environ["AWS_PROFILE"] = "wcs"
session = boto3.Session()
credentials = session.get_credentials()
if credentials:
    creds = credentials.get_frozen_credentials()
    os.environ["SPACER_AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["SPACER_AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    if creds.token:
        os.environ["SPACER_AWS_SESSION_TOKEN"] = creds.token

# Normalize Settings -> SPACER_*/MLFLOW_* env vars now (this used to be an import
# side effect of mermaid_classifier.pyspacer). MLflowTrainingRunner.__init__ also
# calls this; it is idempotent.
from mermaid_classifier.pyspacer.settings import set_env_vars_for_packages

set_env_vars_for_packages()

from mermaid_classifier.pyspacer.options import (
    DatasetOptions,
    MLflowOptions,
    TrainingOptions,
)
from mermaid_classifier.pyspacer.runner import MLflowTrainingRunner
from mermaid_classifier.training.sample_weighting import SampleWeightingOptions
from mermaid_classifier.training.subsample import SubsampleOptions

# Production recipe, validated on the 20-source / 80-class
# tiela77_top100_min1k dataset (~1.77M annotations after rollup).
# See docs/research/hidden-layer-experiments.md (architecture / training budget)
# and docs/research/balancing-experiments.md (label balancing) for the underlying
# experiments and observed metric tradeoffs.

# Full-dataset annotation count after rollup + included-labels filter,
# measured 2026-04-30. Used as the budget for `balanced` subsampling so
# that classes smaller than the implied per-class target (~22K) are kept
# in full while the few dominant classes are capped.
FULL_DATA_TOTAL = 1_770_000


if __name__ == "__main__":
    runner = MLflowTrainingRunner(
        dataset_options=DatasetOptions(
            # Specifying False here means you're only training on CoralNet sources.
            include_mermaid=True,
            coralnet_sources_csv="../sagemaker/configs/tiela77_top108_hierarchy/sources.csv",
            label_rollup_spec_csv="../sagemaker/configs/tiela77_top108_hierarchy/rollups.csv",
            included_labels_csv="../sagemaker/configs/tiela77_top108_hierarchy/included_labels.csv",
            # Local-dev alternative for fast smoke runs (10 sources):
            # coralnet_sources_csv='../sagemaker/sources/CoralNetSourcesFirst10.csv',
            drop_growthforms=False,
            # Class-balanced subsampling. At full-data scale most of the
            # 80 classes have fewer rows than total/num_classes, so the
            # allocator mostly just caps the dominant classes (Turf
            # algae, Sand, Porites, Bare substrate) while keeping the
            # rest in full. ``min_per_class=200`` floors rare classes so
            # they aren't dropped entirely. Realized subsample on the
            # tiela77 dataset is ~457K rows.
            subsample=SubsampleOptions(
                strategy="balanced",
                total_annotations=FULL_DATA_TOTAL,
                min_per_class=200,
            ),
            # Sample weighting via the effective-number-of-samples
            # formulation (the sole strategy after the balancing sweep).
            # Pass None to disable weighting entirely.
            weighting=SampleWeightingOptions(
                weight_ratio_cap=5000.0,  # bound max:min ratio of weights
            ),
        ),
        training_options=TrainingOptions(
            # The MLP head architecture and learning rate are fixed at the
            # production values inside MermaidTrainer (see
            # docs/research/hidden-layer-experiments.md). ``epochs=40`` is a
            # generous upper bound; ``early_stopping_patience=3`` against
            # ``epoch/val_loss`` lets each run find its own minimum
            # (typically epoch 14-29 on this data).
            epochs=40,
            early_stopping_patience=3,
        ),
        mlflow_options=MLflowOptions(
            experiment_name="pyspacer-beta-test",
            model_name="GregTest",
            # Logs all input annotations to MLflow (in addition to the
            # always-logged validation split). Also possible to just log a subset.
            # extra_annotations_to_log='all',
        ),
    )
    runner.run()
