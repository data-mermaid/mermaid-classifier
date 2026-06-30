"""
Dataclass option containers and the Sites enum for the training pipeline.

- Sites: enum for data sources (CORALNET, MERMAID).
- Artifacts: namespace for training artifacts.
- DatasetOptions: options controlling which data goes into training.
- TrainingOptions: options controlling the training process.
- MLflowOptions: options for MLflow experiment tracking.
"""

import dataclasses
import enum

import pandas as pd

from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.training.sample_weighting import SampleWeightingOptions
from mermaid_classifier.training.subsample import SubsampleOptions


class Sites(enum.Enum):
    CORALNET = "coralnet"
    MERMAID = "mermaid"


class Artifacts:
    """
    Namespace to make it easier to track artifacts that we're
    logging later.
    """

    ba_counts: pd.DataFrame
    bagf_counts: pd.DataFrame
    coralnet_label_mapping: pd.DataFrame
    coralnet_project_stats: pd.DataFrame
    mermaid_project_stats: pd.DataFrame
    profiled_sections: list[dict[str, object]]
    train_summary_stats: dict[str, object]
    unmapped_labels: pd.DataFrame


@dataclasses.dataclass
class DatasetOptions:
    """
    include_mermaid

    Whether to include MERMAID annotations or not. False can be useful for
    any troubleshooting which is CoralNet specific.

    coralnet_manifest_uri

    S3 URI (or local path) of the CoralNet manifest parquet; None disables
    CoralNet.

    drop_growthforms

    If True, discard all growth forms from the training data.
    This is applied *before* rollups.
    This can make it easier to define certain large rollup operations,
    such as rolling everything up to top level categories, since your
    CSV spec would only need row per BA instead of one row per BA+GF combo.

    label_rollup_spec_csv

    Local filepath of a CSV file, specifying what MERMAID BA+GF combos to
    roll up to what other BA+GF combos. For example, roll up
    Acropora::Branching to Hard coral::Branching; or roll up
    Porites::Massive to Porites (without growth form specified).
    - If this file isn't specified, then nothing gets rolled up.
    - Recognized columns:
      from_ba_id -- MERMAID benthic attribute ID (a UUID) of a combo to
        roll up from.
      from_gf_id -- MERMAID growth form ID (a UUID) of a combo to
        roll up from.
      to_ba_id -- MERMAID benthic attribute ID (a UUID) of a combo to
        roll up to.
      to_gf_id -- MERMAID growth form ID (a UUID) of a combo to
        roll up to.
      Other informational columns can also be present and will be ignored.

    included_labels_csv
    excluded_labels_csv

    Local filepath of a CSV file, specifying either the MERMAID benthic
    attribute + growth form combos to accept into the training data
    (excluding all others), or the ones to leave out from the training
    data (including all others).
    - Specify at most one of these files, not both. If neither file is
      specified, then all MERMAID benthic attribute + growth form combos
      are accepted.
    - Mapping from CoralNet label IDs to MERMAID BAGFs, and rolling up
      BAGFs, will be done before applying these inclusions/exclusions.
    - Recognized columns:
      ba_id -- A MERMAID benthic attribute ID (a UUID).
      gf_id -- A MERMAID growth form ID (a UUID).
      Other informational columns can also be present and will be ignored.
    - This is applied *after* rollups.

    ref_val_ratios

    Determines the ratios of training annotations that will go into
    the train, ref (reference), and val (validation) sets.
    This is a tuple of two floats. For example, specifying (0.05, 0.1)
    means 5% into ref, 10% into val, and the rest (85%) into train.
    PySpacer has an explanation of the three sets here:
    https://github.com/coralnet/pyspacer?tab=readme-ov-file#train_classifier

    subsample

    Optional per-class subsampling applied after rollup + included-labels
    filter and before the train/ref/val split. Replaces the old
    ``annotation_limit`` knob, which was non-deterministic under DuckDB's
    parallel scans (LIMIT without ORDER BY) and so produced different
    subsets across processes.

    Today's strategies:
      * SubsampleOptions(strategy='stratified', total_annotations=N)
        -- proportional subsample preserving class distribution.
      * SubsampleOptions(strategy='balanced', total_annotations=N)
        -- equalize counts per class (capped at available rows).
    See ``mermaid_classifier.training.subsample`` for the full list and
    instructions on adding new strategies.
    """

    include_mermaid: bool = True
    coralnet_manifest_uri: str | None = None
    drop_growthforms: bool = False
    label_rollup_spec_csv: str | None = None
    included_labels_csv: str | None = None
    excluded_labels_csv: str | None = None
    ref_val_ratios: tuple[float, float] = (0.1, 0.1)
    # Optional per-class subsampling. None means use all annotations.
    # See mermaid_classifier.training.subsample for available strategies
    # and how to add new ones (e.g. effective-number, log-balanced).
    subsample: SubsampleOptions | None = None
    # Optional sample-weighting layer applied to TorchMLP cross-entropy
    # loss. None means no weighting (vanilla CE). See
    # mermaid_classifier.training.sample_weighting for available
    # strategies.
    weighting: SampleWeightingOptions | None = None


@dataclasses.dataclass
class TrainingOptions:
    """
    epochs

    Number of pyspacer training epochs to run. Acts as the upper bound:
    if ``early_stopping_patience`` is set and triggered, training stops
    earlier and the best-val_loss classifier is restored.

    The MLP architecture and learning rate are fixed at the production
    values baked into ``MermaidTrainer`` (``hidden_layer_sizes=(500,
    300, 100)`` @ ``learning_rate_init=1e-4``; see
    docs/research/hidden-layer-experiments.md), so they are not configurable here.

    early_stopping_patience

    If set to a positive int, training stops once ``epoch/val_loss`` has
    not improved for this many consecutive epochs, AND the classifier
    is restored to its state at the epoch with the lowest val_loss
    (rather than the latest epoch's state). None disables early
    stopping; the loop runs for the full ``epochs`` budget.

    Recommended starting value: 3. Lower (1-2) is too jumpy given the
    natural epoch-to-epoch noise in val_loss; higher (>5) wastes wall
    time on epochs that are unlikely to recover.
    """

    epochs: int = 10
    early_stopping_patience: int | None = None


@dataclasses.dataclass
class MLflowOptions:
    """
    experiment_name

    Name of the MLflow experiment in which to register the training run. If not
    given, then it's taken from the MLFLOW_DEFAULT_EXPERIMENT_NAME setting.

    model_name

    Name of this MLflow experiment run's model. If not given, then a model
    name will be constructed based on the experiment parameters.
    The run name is based on this too.
    This name gets truncated at 50 characters to avoid a potential crash when
    logging the model.

    extra_annotations_to_log

    In addition to the validation-split annotations (which are always logged
    as the `annotations_val` artifact so others can independently re-evaluate
    the model), optionally log a further set of training annotations as an
    MLflow artifact, in tabular form. One table row per point-annotation.
    This can serve as a sanity check or debugging aid, but the artifact can
    get quite large.
    Supported formats:
    'all': log all annotations
    's123': log annotations from CoralNet source of ID 123
    'i456': log annotations from CoralNet image of ID 456
    <not specified>: log nothing extra
    """

    experiment_name: str | None = settings.mlflow_default_experiment_name
    model_name: str | None = None
    extra_annotations_to_log: str | None = None
