"""
Configuration dataclasses for the PySpacer training pipeline.
"""
import dataclasses
import enum

import pandas as pd

from mermaid_classifier.pyspacer.settings import settings


class Sites(enum.Enum):
    CORALNET = 'coralnet'
    MERMAID = 'mermaid'


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
    profiled_sections: list[dict]
    train_summary_stats: dict
    unmapped_labels: pd.DataFrame


@dataclasses.dataclass
class DatasetOptions:
    """
    include_mermaid

    Whether to include MERMAID annotations or not. False can be useful for
    any troubleshooting which is CoralNet specific.

    coralnet_sources_csv

    Local filepath of a CSV file, specifying CoralNet sources to include in
    the training data.
    Recognized columns:
      id -- CoralNet source ID number.
      Other informational columns can also be present and will be ignored.
    If not specified, no CoralNet sources are included (so only MERMAID
    projects go into training).

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

    annotation_limit

    If specified, only get up to this many annotations for training. This can
    help with testing since the runtime is correlated to number of annotations.
    """
    include_mermaid: bool = True
    coralnet_sources_csv: str = None
    drop_growthforms: bool = False
    label_rollup_spec_csv: str = None
    included_labels_csv: str = None
    excluded_labels_csv: str = None
    ref_val_ratios: tuple[float, float] = (0.1, 0.1)
    annotation_limit: int | None = None


@dataclasses.dataclass
class TrainingOptions:
    """
    epochs: Number of training epochs.
    class_balancing: Inverse-frequency class weighting via
        CrossEntropyLoss(weight=...).
    device: 'auto' (CUDA if available, else CPU), 'cpu', 'cuda', 'cuda:0', etc.
    optimizer: 'adam', 'adamw', or 'sgd'.
    learning_rate: None = auto-select by dataset size (1e-4 for >=50K, else 1e-3).
    weight_decay: L2 regularization strength.
    hidden_layer_sizes: None = auto-select by dataset size.
    minibatch_size: Effective gradient-update batch size. With gradient
        accumulation, multiple smaller IO batches are accumulated to reach
        this size before an optimizer step.
    io_batch_size: Number of annotations loaded from disk per IO read.
        None = auto-calculate from available memory at runtime.
        Decoupled from minibatch_size to allow background prefetching of
        small chunks while the GPU trains.
    """
    epochs: int = 10
    class_balancing: bool = False
    device: str = 'auto'
    optimizer: str = 'adam'
    learning_rate: float | None = None
    weight_decay: float = 1e-4
    hidden_layer_sizes: tuple[int, ...] | None = None
    minibatch_size: int = 512
    io_batch_size: int | None = None


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

    annotations_to_log

    If specified, log training annotations as an MLflow artifact, in tabular
    form. One table row per point-annotation. This can serve as a sanity
    check, but the artifact can get quite large.
    Supported formats:
    'all': log all annotations
    's123': log annotations from CoralNet source of ID 123
    'i456': log annotations from CoralNet image of ID 456
    <not specified>: log nothing
    """
    experiment_name: str | None = settings.mlflow_default_experiment_name
    model_name: str | None = None
    annotations_to_log: str | None = None
