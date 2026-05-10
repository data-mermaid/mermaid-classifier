"""
Autoresearch experiment entry point.

FROZEN: DatasetOptions data fields (sources, rollups, labels, splits),
        MLflowOptions, AWS credential setup.
MODIFIABLE: Everything else — TrainingOptions, subsample/weighting
            config, runner subclass, component wiring.
"""
import os
import sys

import boto3

# Resolve AWS SSO credentials before importing mermaid_classifier,
# whose Settings() object is created at import time and reads env vars then.
os.environ['AWS_PROFILE'] = 'wcs'
session = boto3.Session()
credentials = session.get_credentials()
if credentials:
    creds = credentials.get_frozen_credentials()
    # Settings reads AWS_KEY_ID/AWS_SECRET/AWS_SESSION_TOKEN;
    # set_env_vars_for_packages() then copies them to SPACER_AWS_*.
    os.environ['AWS_KEY_ID'] = creds.access_key
    os.environ['AWS_SECRET'] = creds.secret_key
    if creds.token:
        os.environ['AWS_SESSION_TOKEN'] = creds.token

# Add the experiment directory to sys.path so local imports work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mermaid_classifier.pyspacer.train import (
    DatasetOptions, MLflowOptions, MLflowTrainingRunner, TrainingOptions)
from mermaid_classifier.training.sample_weighting import (
    SampleWeightingOptions)
from mermaid_classifier.training.subsample import SubsampleOptions

from trainer import ExperimentTrainer


# ── FROZEN: data identity ────────────────────────────────────
# DO NOT MODIFY these fields. They define what data enters training,
# how labels are mapped, and how the data is split. Changing these
# invalidates all experiment comparisons.
FROZEN_DATA = dict(
    include_mermaid=False,
    coralnet_sources_csv='../sagemaker/configs/tiela77_top100_min1k/sources.csv',
    label_rollup_spec_csv='../sagemaker/configs/tiela77_top100_min1k/rollups.csv',
    included_labels_csv="../sagemaker/configs/tiela77_top100_min1k/included_labels.csv",
    drop_growthforms=False,
    ref_val_ratios=(0.1, 0.1),
)

MLFLOW_OPTIONS = MLflowOptions(
    experiment_name="autoresearch",
    model_name='AutoResearch',
)
# ── END FROZEN ─────────────────────────────────────────────


# ── MODIFIABLE: subsampling, weighting, training ────────────────────
# The agent may change any values below.

FULL_DATA_TOTAL = 1_770_000

SUBSAMPLE = SubsampleOptions(
    strategy='balanced',
    total_annotations=FULL_DATA_TOTAL,
    min_per_class=200,
)

WEIGHTING = SampleWeightingOptions(
    strategy='effective_number',
    alpha=0.5,
    weight_ratio_cap=5000.0,
)

# Hypothesis: prior run with lr=1e-4 stopped at epoch 12 with
# training_loss still decreasing (last=6.30, min=6.18) and val_loss
# only barely plateauing (min 2.231 at epoch 9, 2.232 at epoch 12).
# Bumping LR 3x lets the model take larger steps and reach a deeper
# minimum before the patience=3 early-stopping budget runs out.
TRAINING_OPTIONS = TrainingOptions(
    hidden_layer_sizes=(500, 300, 100),
    learning_rate_init=3e-4,
    epochs=60,
    early_stopping_patience=3,
)


class ExperimentRunner(MLflowTrainingRunner):
    """Overrides trainer construction to use experiment's custom trainer."""

    def _create_trainer(self, batch_size, class_weight):
        return ExperimentTrainer(
            batch_size=batch_size,
            on_epoch_end=self._on_epoch_end,
            class_weight=class_weight,
            hidden_layer_sizes=self.training_options.hidden_layer_sizes,
            learning_rate_init=self.training_options.learning_rate_init,
            early_stopping_patience=(
                self.training_options.early_stopping_patience),
            random_state=self.training_options.random_state,
        )


if __name__ == "__main__":
    dataset_options = DatasetOptions(
        **FROZEN_DATA,
        subsample=SUBSAMPLE,
        weighting=WEIGHTING,
    )
    runner = ExperimentRunner(
        dataset_options=dataset_options,
        training_options=TRAINING_OPTIONS,
        mlflow_options=MLFLOW_OPTIONS,
    )
    runner.run()
