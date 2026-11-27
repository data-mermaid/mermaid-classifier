"""
Dataclass option containers and the Sites enum for the training pipeline.

- Sites: enum for data sources (CORALNET, MERMAID).
- Artifacts: namespace for training artifacts.
- PinnedDataset: one pinned dataset version and its feature set.
- DatasetOptions: options controlling which data goes into training.
- TrainingOptions: options controlling the training process.
- MLflowOptions: options for MLflow experiment tracking.
- training_data_lineage, lineage_params, groups_lineage, mapping_lineage: the pins and
  content hashes a run records in model.json and as MLflow params.
"""

import dataclasses
import enum
import hashlib
from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
from segdata_spec.manifest import FeatureSetPin, Pin

from mermaid_classifier.common.holdout import CORALNET_SITE, MERMAID_SITE
from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.training.sample_weighting import SampleWeightingOptions
from mermaid_classifier.training.subsample import SubsampleOptions


class Sites(enum.Enum):
    CORALNET = CORALNET_SITE
    MERMAID = MERMAID_SITE


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


LabelKind = Literal["coralnet", "mermaid"]


@dataclasses.dataclass(frozen=True)
class PinnedDataset:
    """
    One dataset version a run trains on.

    pin -- the dataset version (name, version, manifest sha256).
    features -- the feature set read for that version.
    labels -- how the points' label_id maps to MERMAID BA+GF: 'coralnet'
      label IDs go through CoralNetMermaidMapping; 'mermaid' label IDs
      carry their BA id and growth form name in labels.parquet.
    groups_csv -- local path of a file naming one dataset group per line
      to train on; None trains on every group.
    """

    pin: Pin
    features: FeatureSetPin
    labels: LabelKind
    groups_csv: str | None = None


@dataclasses.dataclass
class DatasetOptions:
    """
    datasets

    The pinned dataset versions whose points train the model, each with
    the feature set read for it (see PinnedDataset).

    taxonomy

    Pin of the mermaid-benthic taxonomy version that resolves the growth
    form names of every `labels="mermaid"` dataset. Required when any
    dataset has those labels.

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

    excluded_images_csv

    Local filepath of a CSV file listing image IDs to withhold from
    training data entirely, regardless of label. For example, this
    keeps a frozen evaluation probe's held-out images out of the
    training data those models see.
    - If this file isn't specified, then no images are excluded on
      this basis.
    - Recognized columns:
      image_id -- An image ID, matching the annotations table's
        image_id column.
      Other informational columns can also be present and will be ignored.
    - Applied independently of the label rollups and inclusion/exclusion
      filters above: a listed image is removed regardless of its BA+GF.

    mermaid_project_lookup_uri

    S3 URI (or local path) of a parquet mapping MERMAID image IDs to
    projects, with columns image_id, project_id, project_name, site_id.
    When set, MERMAID rows take that project_id (so a MERMAID project is
    a group, like a CoralNet source); an image absent from the lookup gets
    project_id 'unknown'. When unset, every MERMAID row has project_id 'all'.
    - Too many unmatched images (more than
      MERMAID_PROJECT_LOOKUP_PERCENT_UNMATCHED_ALLOWED percent, default 0,
      independent of TRAINING_INPUTS_PERCENT_MISSING_ALLOWED) aborts the run, as
      does an image that maps to more than one project.

    holdout_groups_csv
    holdout_fold

    Local filepath of a CSV assigning groups to the held-out test set or
    to cross-validation folds. A group is a (site, project_id) pair.
    - If this file isn't specified, nothing is held out.
    - Recognized columns:
      site -- 'coralnet' or 'mermaid'.
      group_id -- The CoralNet source ID, or the MERMAID project ID.
      fold -- 'test', or a non-negative integer fold number.
    - holdout_fold is 'test' (the default) or one of the fold numbers in
      the CSV. Rows of groups in that fold are tagged is_holdout. When it
      is a fold number, rows of the 'test' groups are tagged is_excluded
      and dropped, so a CV run neither trains nor evaluates on them.
    - Listing a MERMAID group requires mermaid_project_lookup_uri.

    holdout_max_per_class

    Caps each class's holdout rows (keeping whole images first, in seeded
    md5 order) to bound the feature downloads for evaluation. None keeps
    every holdout row. Has no effect when nothing is held out.
    The cap is per class, so an image can lose only some of its points:
    holdout accuracy and the CV metrics are measured on a per-class-capped
    mix, not the natural class mix, and the holdout cover metrics are
    skipped whenever the cap trims a class. The seeded md5 order is seeded
    by subsample.seed (0 with no subsample block), the same seed that
    drives the image split.

    ref_val_ratios

    Determines the ratios of training annotations that will go into
    the train, ref (reference), and val (validation) sets.
    This is a tuple of two floats. For example, specifying (0.05, 0.1)
    means 5% into ref, 10% into val, and the rest (85%) into train.
    PySpacer has an explanation of the three sets here:
    https://github.com/coralnet/pyspacer?tab=readme-ov-file#train_classifier

    split_mode

    How the non-holdout annotations are split into train/ref/val.
    - 'points' (the default): PySpacer's point-level stratified split, so
      one image's points can land in different sets. PySpacer reads its
      5,000-point ref cap (TRAINING_BATCH_LABEL_COUNT) at import, so ref
      is the smaller of ref_val_ratios[0] x N and 5,000 points.
    - 'images': each image goes whole to one set, stratified by class
      (see mermaid_classifier.pyspacer.splitting). Classes in fewer than
      3 images are dropped first. ref_val_ratios apply with no ref cap.
      Seeded by subsample.seed, or 0 without a subsample.
    Holdout rows never enter train/ref/val in either mode.

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

    feature_extractor_weights

    URI of the EfficientNet weights that produced this run's feature
    vectors. Every feature set's manifest records its extractor, and each
    must agree with this declaration.
    """

    datasets: list[PinnedDataset] = dataclasses.field(default_factory=list)
    taxonomy: Pin | None = None
    # Which EfficientNet produced the .featurevector files this run trains
    # on. A head is fitted to feature vectors, never to pixels, so this is
    # the run's statement of what its inputs mean. TrainingDataset requires
    # it, since a dataclass cannot put a mandatory field after defaulted ones.
    feature_extractor_weights: str | None = None
    drop_growthforms: bool = False
    label_rollup_spec_csv: str | None = None
    included_labels_csv: str | None = None
    excluded_labels_csv: str | None = None
    excluded_images_csv: str | None = None
    mermaid_project_lookup_uri: str | None = None
    holdout_groups_csv: str | None = None
    holdout_fold: str = "test"
    holdout_max_per_class: int | None = 3000
    ref_val_ratios: tuple[float, float] = (0.1, 0.1)
    split_mode: Literal["points", "images"] = "points"
    # Optional per-class subsampling. None means use all annotations.
    # See mermaid_classifier.training.subsample for available strategies
    # and how to add new ones (e.g. effective-number, log-balanced).
    subsample: SubsampleOptions | None = None
    # Optional sample-weighting layer applied to TorchMLP cross-entropy
    # loss. None means no weighting (vanilla CE). See
    # mermaid_classifier.training.sample_weighting for available
    # strategies.
    weighting: SampleWeightingOptions | None = None


def groups_lineage(groups: frozenset[str] | None) -> dict[str, Any] | None:
    """Identify a group selection by the sha256 of its sorted, newline-joined groups.

    None (every group of the version) stays None.
    """
    if groups is None:
        return None
    digest = hashlib.sha256("\n".join(sorted(groups)).encode()).hexdigest()
    return {"sha256": digest, "n": len(groups)}


def mapping_lineage(mapping: pd.DataFrame) -> dict[str, Any]:
    """Identify a label mapping by the sha256 of its canonical CSV.

    Columns are put in sorted order and rows sorted by every column, so neither
    order changes the hash.
    """
    columns: list[str] = sorted(str(column) for column in mapping.columns)
    canonical = mapping.reindex(columns=columns).sort_values(by=columns)
    csv_bytes = canonical.to_csv(index=False, lineterminator="\n").encode()
    return {"sha256": hashlib.sha256(csv_bytes).hexdigest(), "n": len(mapping)}


def training_data_lineage(
    options: DatasetOptions,
    groups: Mapping[str, frozenset[str] | None],
    coralnet_mapping: pd.DataFrame | None = None,
    *,
    lookup_sha256: str | None = None,
    holdout_sha256: str | None = None,
    holdout_n: int = 0,
) -> dict[str, Any]:
    """The dataset, group-selection, feature-set and taxonomy pins a run trains on.

    `groups` maps each dataset name to its resolved group selection.
    `coralnet_mapping` is the label mapping the run used; it is recorded only
    when a dataset has `labels == "coralnet"`.
    `lookup_sha256` and `holdout_sha256` are the sha256 of the lookup and holdout
    CSV bytes the run read (`holdout_n` is the CSV's group count); each block is
    recorded only when its file was read.
    """
    taxonomy = options.taxonomy
    lineage: dict[str, Any] = {
        "datasets": [
            {
                "name": source.pin.name,
                "version": source.pin.version,
                "manifest_sha256": source.pin.manifest_sha256,
                "labels": source.labels,
                "features": {
                    "family": source.features.family,
                    "manifest_sha256": source.features.manifest_sha256,
                },
                "groups": groups_lineage(groups[source.pin.name]),
            }
            for source in options.datasets
        ],
        "taxonomy": None
        if taxonomy is None
        else {
            "name": taxonomy.name,
            "version": taxonomy.version,
            "manifest_sha256": taxonomy.manifest_sha256,
        },
    }
    if coralnet_mapping is not None and any(
        source.labels == "coralnet" for source in options.datasets
    ):
        lineage["coralnet_label_mapping"] = mapping_lineage(coralnet_mapping)
    lookup_uri = options.mermaid_project_lookup_uri
    if lookup_uri and lookup_sha256 is not None:
        lineage["mermaid_project_lookup"] = {"uri": lookup_uri, "sha256": lookup_sha256}
    if options.holdout_groups_csv and holdout_sha256 is not None:
        lineage["holdout_groups"] = {
            "sha256": holdout_sha256,
            "n": holdout_n,
            "fold": options.holdout_fold,
        }
    return lineage


def lineage_params(lineage: dict[str, Any]) -> dict[str, str]:
    """Flatten `training_data_lineage` into MLflow params, one per pinned value."""
    params: dict[str, str] = {}
    for entry in lineage["datasets"]:
        prefix = f"dataset/{entry['name']}"
        params[f"{prefix}/version"] = entry["version"]
        params[f"{prefix}/manifest_sha256"] = entry["manifest_sha256"]
        params[f"{prefix}/labels"] = entry["labels"]
        params[f"{prefix}/features/family"] = entry["features"]["family"]
        params[f"{prefix}/features/manifest_sha256"] = entry["features"]["manifest_sha256"]
        selection = entry["groups"]
        params[f"{prefix}/groups_sha256"] = "all" if selection is None else selection["sha256"]
        params[f"{prefix}/groups_n"] = "all" if selection is None else str(selection["n"])
    taxonomy = lineage["taxonomy"]
    if taxonomy is not None:
        params[f"taxonomy/{taxonomy['name']}/version"] = taxonomy["version"]
        params[f"taxonomy/{taxonomy['name']}/manifest_sha256"] = taxonomy["manifest_sha256"]
    mapping = lineage.get("coralnet_label_mapping")
    if mapping is not None:
        params["coralnet_label_mapping/sha256"] = mapping["sha256"]
        params["coralnet_label_mapping/n"] = str(mapping["n"])
    lookup = lineage.get("mermaid_project_lookup")
    if lookup is not None:
        params["mermaid_project_lookup/uri"] = lookup["uri"]
        params["mermaid_project_lookup/sha256"] = lookup["sha256"]
    holdout = lineage.get("holdout_groups")
    if holdout is not None:
        params["holdout_groups/sha256"] = holdout["sha256"]
        params["holdout_groups/n"] = str(holdout["n"])
        params["holdout_groups/fold"] = holdout["fold"]
    return params


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
