import os
import boto3

# Resolve AWS SSO credentials before importing mermaid_classifier,
# whose Settings() object is created at import time and reads env vars then.
os.environ['AWS_PROFILE'] = 'wcs'
session = boto3.Session()
credentials = session.get_credentials()
if credentials:
    creds = credentials.get_frozen_credentials()
    os.environ['SPACER_AWS_ACCESS_KEY_ID'] = creds.access_key
    os.environ['SPACER_AWS_SECRET_ACCESS_KEY'] = creds.secret_key
    if creds.token:
        os.environ['SPACER_AWS_SESSION_TOKEN'] = creds.token

from mermaid_classifier.pyspacer.train import (
        DatasetOptions, MLflowOptions, MLflowTrainingRunner, TrainingOptions)
from mermaid_classifier.training.sample_weighting import (
        SampleWeightingOptions)
from mermaid_classifier.training.subsample import SubsampleOptions


# Production recipe, validated on the 20-source / 80-class
# tiela77_top100_min1k dataset (~1.77M annotations after rollup).
# See docs/hidden-layer-experiments.md (architecture / training budget)
# and docs/balancing-experiments.md (label balancing) for the underlying
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
            coralnet_sources_csv='../sagemaker/configs/tiela77_top108_hierarchy/sources.csv',
            label_rollup_spec_csv='../sagemaker/configs/tiela77_top108_hierarchy/rollups.csv',
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
                strategy='balanced',
                total_annotations=FULL_DATA_TOTAL,
                min_per_class=200,
            ),
            # Sample weighting. Anchor of the balancing sweep; remained
            # the default after subsampling was added on top of it.
            # Other registered strategies: tree_balanced_ba_flat_gf
            # (currently unsafe -- see docs/balancing-experiments.md
            # finding #5), leaf_inverse, decomposed. Pass None to
            # disable weighting entirely.
            weighting=SampleWeightingOptions(
                strategy='effective_number',
                alpha=0.5,
                weight_ratio_cap=5000.0,  # bound max:min ratio of weights
            ),
        ),
        training_options=TrainingOptions(
            # MLP head architecture and learning rate from the
            # hidden-layer experiments. ``epochs=40`` is a generous
            # upper bound; ``early_stopping_patience=3`` against
            # ``epoch/val_loss`` lets each run find its own minimum
            # (typically epoch 14-29 on this data).
            hidden_layer_sizes=(500, 300, 100),
            learning_rate_init=1e-4,
            epochs=40,
            early_stopping_patience=3,
        ),
        mlflow_options=MLflowOptions(
            experiment_name="pyspacer-beta-test",
            model_name='GregTest',
            # Logs all input annotations to MLflow (in addition to the
            # always-logged validation split). Also possible to just log a subset.
            #extra_annotations_to_log='all',
        ),
    )
    runner.run()
