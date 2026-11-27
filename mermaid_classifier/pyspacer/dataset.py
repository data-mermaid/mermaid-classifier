"""
TrainingDataset: loads, transforms, and prepares annotation data for training.

Syncs pinned segdata-spec dataset versions and their feature sets, maps each
version's label IDs to MERMAID BA+GF, applies rollup/filter specs, checks
feature coverage, and produces pyspacer-ready train/ref/val splits.
"""

import dataclasses
import functools
import hashlib
import io
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Self, TextIO

import boto3
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError
from segdata_spec import DATASET_BUCKET
from segdata_spec.manifest import FeatureSetManifest
from segdata_spec.read import (
    FeatureSet,
    Taxonomy,
    VersionTables,
    open_feature_set,
    open_tables,
    open_taxonomy,
)
from segdata_spec.sync import FeatureSyncResult, sync_feature_set, sync_tables
from spacer.data_classes import Annotation, DataLocation, ImageLabels, LabelId
from spacer.messages import TrainingTaskLabels
from spacer.task_utils import SplitMode, preprocess_labels
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from mermaid_classifier.common.benthic_attributes import (
    CoralNetMermaidMapping,
    combine_ba_gf,
    get_benthic_attribute_library,
    get_growth_form_library,
)
from mermaid_classifier.common.csv_utils import CsvSpec
from mermaid_classifier.common.duckdb_utils import (
    duckdb_add_column,
    duckdb_grouped_rows,
    duckdb_temp_table_name,
    duckdb_transform_column,
)
from mermaid_classifier.common.s3_utils import is_s3_uri, parse_s3_uri
from mermaid_classifier.pyspacer._pipeline_utils import section_profiling
from mermaid_classifier.pyspacer.extraction import extractor_spec_from_weights
from mermaid_classifier.pyspacer.inference import ExtractorSpec
from mermaid_classifier.pyspacer.label_specs import (
    HOLDOUT_TEST_FOLD,
    HoldoutGroups,
    ImageExclusionFilter,
    LabelFilter,
    LabelRollupSpec,
)
from mermaid_classifier.pyspacer.options import (
    Artifacts,
    DatasetOptions,
    LabelKind,
    PinnedDataset,
    Sites,
)
from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.pyspacer.splitting import MIN_IMAGES_PER_CLASS, split_by_image
from mermaid_classifier.pyspacer.utils import logging_config_for_script
from mermaid_classifier.training.subsample import (
    SubsampleOptions,
    compute_per_class_targets,
    water_fill,
)

logger = logging_config_for_script("train")

# Transient network failures of a feature download. botocore wraps most stream
# failures in BotoCoreError subclasses, but a raw urllib3 error can escape a body
# read. segdata-spec's integrity errors are not in this tuple and are never retried.
TRANSIENT_SYNC_ERRORS = (BotoCoreError, ProtocolError, ReadTimeoutError)
SYNC_ATTEMPTS = 3
SYNC_RETRY_DELAY_SECONDS = 1.0


def _load_csv_spec[T: CsvSpec](path: str | None, factory: Callable[[TextIO], T]) -> T:
    """A `CsvSpec` read from `path`, or the same type built from an empty CSV.

    An unset path is the caller's "accept everything" default, not a file to
    complain about missing.
    """
    if path:
        with open(path) as csv_f:
            return factory(csv_f)
    return factory(StringIO(""))


# pyarrow ships no type stubs, so its tables are typed as Any.
_Table = Any

# The manifest source type of a dataset whose label IDs are CoralNet label IDs.
CORALNET_SOURCE_TYPE = "coralnet-scrape"

# label_id -> MERMAID BA+GF for one dataset version; a null BA is an unmapped label.
_LABEL_TABLE_SCHEMA = pa.schema(
    [
        pa.field("label_id", pa.int32()),
        pa.field("benthic_attribute_id", pa.string()),
        pa.field("growth_form_id", pa.string()),
    ]
)


def make_s3_client(max_workers: int) -> Any:
    """A boto3 S3 client with a pooled connection per download worker."""
    config = Config(
        max_pool_connections=max_workers,
        region_name=settings.aws_region,
        retries={"mode": "standard", "max_attempts": 10},
    )
    if settings.aws_anonymous == "True":
        config = config.merge(Config(signature_version=UNSIGNED))
    return boto3.client(
        "s3",
        config=config,
        aws_access_key_id=settings.aws_key_id,
        aws_secret_access_key=settings.aws_secret,
        aws_session_token=settings.aws_session_token,
    )


def read_groups_csv(path: str) -> frozenset[str]:
    """The dataset groups `path` names, one per line; blank lines are skipped."""
    groups = frozenset(line.strip() for line in Path(path).read_text().splitlines() if line.strip())
    if not groups:
        raise ValueError(f"{path} names no dataset group")
    return groups


def resolve_groups(options: DatasetOptions) -> dict[str, frozenset[str] | None]:
    """Each dataset's group selection from its groups_csv; None selects every group."""
    return {
        source.pin.name: None if source.groups_csv is None else read_groups_csv(source.groups_csv)
        for source in options.datasets
    }


def select_groups(images: _Table, groups: frozenset[str] | None, dataset: str) -> _Table:
    """The `images` rows in `groups`, or every row for None."""
    if groups is None:
        return images
    absent = groups - set(images.column("group").to_pylist())
    if absent:
        raise ValueError(f"{dataset}: groups not in the dataset version: {sorted(absent)}")
    return images.filter(
        pa.array([group in groups for group in images.column("group").to_pylist()])
    )


def check_labels_kind(labels: _Table, kind: LabelKind, dataset: str, source_type: str) -> None:
    """Refuse a `labels:` kind that labels.parquet or the manifest's source contradicts.

    MERMAID labels carry a benthic attribute id on every row; CoralNet labels
    carry none, since CoralNetMermaidMapping supplies them, and come only from
    a CoralNet scrape.
    """
    if kind == "coralnet" and source_type != CORALNET_SOURCE_TYPE:
        raise ValueError(
            f"{dataset} is declared `labels: coralnet`, but its manifest's source type is"
            f" {source_type!r}, not {CORALNET_SOURCE_TYPE!r}"
        )
    n_mapped = labels.num_rows - labels.column("mermaid_benthic_attribute_id").null_count
    if kind == "coralnet" and n_mapped:
        raise ValueError(
            f"{dataset} is declared `labels: coralnet`, but {n_mapped} of its"
            " labels.parquet rows carry a MERMAID benthic attribute id"
        )
    if kind == "mermaid" and n_mapped < labels.num_rows:
        raise ValueError(
            f"{dataset} is declared `labels: mermaid`, but {labels.num_rows - n_mapped} of its"
            " labels.parquet rows carry no MERMAID benthic attribute id"
        )


def coralnet_label_table(labels: _Table, mapping: CoralNetMermaidMapping) -> _Table:
    """Map each CoralNet label ID in `labels` to its MERMAID BA+GF."""
    rows: list[dict[str, Any]] = []
    for label_id in labels.column("label_id").to_pylist():
        key = str(label_id)
        entry = mapping.mapping.get(key)
        rows.append(
            {
                "label_id": label_id,
                "benthic_attribute_id": None if entry is None else entry.benthic_attribute_id,
                "growth_form_id": None if entry is None else entry.growth_form_id,
            }
        )
    return pa.Table.from_pylist(rows, schema=_LABEL_TABLE_SCHEMA)


def mermaid_label_table(labels: _Table, growth_forms: _Table, dataset: str) -> _Table:
    """Map each label to its recorded BA and to the id of its growth form name.

    An empty or null growth form name is no growth form, stored as ''.
    """
    gf_ids = dict(
        zip(
            growth_forms.column("name").to_pylist(),
            growth_forms.column("id").to_pylist(),
            strict=True,
        )
    )
    names: list[str | None] = labels.column("mermaid_growth_form").to_pylist()
    unknown = sorted({name for name in names if name and name not in gf_ids})
    if unknown:
        raise ValueError(f"{dataset}: growth forms absent from the pinned taxonomy: {unknown}")
    return pa.table(
        {
            "label_id": labels.column("label_id"),
            "benthic_attribute_id": labels.column("mermaid_benthic_attribute_id"),
            "growth_form_id": [gf_ids[name] if name else "" for name in names],
        },
        schema=_LABEL_TABLE_SCHEMA,
    )


def project_id(group: str, site: Sites) -> str:
    """A CoralNet group `s<id>` is source `<id>`; MERMAID images share one project."""
    return group.removeprefix("s") if site is Sites.CORALNET else "all"


def dataset_annotations(
    points: _Table, images: _Table, label_table: _Table, *, site: Sites, dataset: str
) -> _Table:
    """One annotation row per point of an image in `images`, with its mapped BA+GF."""
    _check_label_ids_present(points, label_table, dataset)
    projects = pa.table(
        {
            "image_id": images.column("image_id"),
            "project_id": [project_id(group, site) for group in images.column("group").to_pylist()],
        }
    )
    joined = (
        points.select(["image_id", "row", "col", "label_id"])
        .join(projects, "image_id", join_type="inner")
        .join(label_table, "label_id", join_type="left outer")
    )
    n = joined.num_rows
    return (
        joined.set_column(
            joined.schema.get_field_index("label_id"),
            "label_id",
            pc.cast(joined.column("label_id"), pa.string()),
        )
        .append_column("site", pa.array([site.value] * n, pa.string()))
        .append_column("dataset", pa.array([dataset] * n, pa.string()))
        .append_column("bucket", pa.array([DATASET_BUCKET] * n, pa.string()))
    )


def _check_label_ids_present(points: _Table, label_table: _Table, dataset: str) -> None:
    """Refuse points whose label_id has no row in the dataset's label table."""
    known = set(label_table.column("label_id").to_pylist())
    missing = sorted(set(points.column("label_id").to_pylist()) - known)
    if missing:
        raise ValueError(
            f"{dataset}: {len(missing)} point label ids are absent from labels.parquet,"
            f" e.g. {missing[:5]}"
        )


def _sync_with_retries(dataset: str, sync: Callable[[], FeatureSyncResult]) -> FeatureSyncResult:
    """Run `sync`, retrying transient network errors; the last failure propagates.

    A retry resumes from the verified files already in the shared cache dir.
    """
    for attempt in range(1, SYNC_ATTEMPTS + 1):
        try:
            return sync()
        except TRANSIENT_SYNC_ERRORS as error:
            if attempt == SYNC_ATTEMPTS:
                raise
            logger.warning(
                "Feature sync for %s failed (attempt %d of %d): %r; retrying",
                dataset,
                attempt,
                SYNC_ATTEMPTS,
                error,
            )
            time.sleep(SYNC_RETRY_DELAY_SECONDS)
    raise AssertionError("unreachable")


def feature_set_extractor(manifest: FeatureSetManifest) -> ExtractorSpec:
    """The ExtractorSpec a feature set's manifest records."""
    fields = ExtractorSpec.__dataclass_fields__
    return ExtractorSpec.from_dict({name: manifest.extractor.get(name) for name in fields})


def _read_holdout_groups(path: str | None) -> tuple[HoldoutGroups, str | None, bytes | None]:
    """The holdout spec at `path`, the sha256 of the bytes it was parsed from, and those bytes.

    An unset path is an empty spec with no hash and no bytes.
    """
    if not path:
        return HoldoutGroups(StringIO("")), None, None
    data = Path(path).read_bytes()
    # TextIOWrapper decodes as open() does, so the spec matches a direct read.
    return (
        HoldoutGroups(io.TextIOWrapper(io.BytesIO(data))),
        hashlib.sha256(data).hexdigest(),
        data,
    )


def _holdout_groups_unselected(
    holdout_groups: HoldoutGroups,
    options: DatasetOptions,
    groups: Mapping[str, frozenset[str] | None],
) -> dict[str, str]:
    """The CoralNet holdout groups no coralnet dataset reads, as `s<id>` name -> fold.

    A group is unselected only when every coralnet dataset has a groups_csv and
    none of them lists it; a dataset without a groups_csv reads every group.
    MERMAID holdout groups are projects, which no groups_csv selects among.
    """
    selections = [
        groups[source.pin.name] for source in options.datasets if source.labels == "coralnet"
    ]
    return {
        f"s{group_id}": fold
        for (site, group_id), fold in sorted(holdout_groups.group_folds.items())
        if site == Sites.CORALNET.value
        and bool(selections)
        and all(
            selection is not None and f"s{group_id}" not in selection for selection in selections
        )
    }


def _seeded_image_order(seed: int) -> str:
    """A SQL ORDER BY key that clusters rows by image in a seed-shuffled order.

    md5 is the same in every DuckDB version, which hash() is not.
    """
    return f"md5('{int(seed)}' || dataset || '/' || image_id)"


@dataclasses.dataclass(frozen=True)
class _SubsampleAllocation:
    """How many rows each partition of the subsample keeps, and how they are ranked.

    group_available and group_quotas are keyed (ba, gf, site, project_id) and are
    non-empty only for a per-group (source_spread) allocation.
    """

    allocation_df: pd.DataFrame
    partition_by: str
    order_by: str
    join_using: str
    group_quotas: dict[tuple[str, str, str, str], int]
    group_available: dict[tuple[str, str, str, str], int]


def _ordered_allocation(targets: Mapping[tuple[str, str], int]) -> _SubsampleAllocation:
    """Per-class targets, taking each class's first rows in (site, project, image, point) order."""
    allocation_df = pd.DataFrame(
        [
            {
                "benthic_attribute_id": ba,
                "growth_form_id": gf,
                "target_n": int(n),
            }
            for (ba, gf), n in targets.items()
        ]
    )
    return _SubsampleAllocation(
        allocation_df=allocation_df,
        partition_by="benthic_attribute_id, growth_form_id",
        # (site, project_id, image_id, row, col) is unique per annotation, so
        # the row order is total across processes.
        order_by="site, project_id, image_id, row, col",
        join_using="benthic_attribute_id, growth_form_id",
        group_quotas={},
        group_available={},
    )


class TrainingDataset:
    # Set by _dedupe_points; merged into train_summary_stats.
    _dedupe_stats: dict[str, int] | None = None
    # Set by _apply_subsample when it selects rows; the runner logs nothing while None.
    _subsample_audit_df: pd.DataFrame | None = None
    _subsample_realized_total: int | None = None
    # Set only by source_spread (per class x group); None otherwise.
    _subsample_group_audit_df: pd.DataFrame | None = None
    # Set by _cap_holdout_rows when there are holdout rows to cap.
    _holdout_audit_df: pd.DataFrame | None = None
    # Set by _log_split_drops; classes the train/ref/val split removed.
    _split_dropped_bagfs: int = 0

    def __init__(self, options: DatasetOptions):
        self._setup_annotation_reading(options)
        # Maps a cached feature vector's local path (used as the key of
        # the filesystem DataLocations in self.labels) back to its
        # (bucket, feature_vector) S3 location. add_training_set_names() needs
        # this to match the labels to the annotations table.
        self._feature_path_to_s3_location: dict[str, tuple[str, str]] = {}
        # The synced feature set of each dataset, by dataset name.
        self._feature_sets: dict[str, FeatureSet] = {}

        # Set once the feature sources are resolved; runner.py seals it
        # into the exported manifest.
        self.extractor_spec: ExtractorSpec | None = None

        self.feature_loc_to_source: dict[DataLocation, tuple[str, str]] = {}

        # Evaluation-only rows; filled by prep_annotations_for_pyspacer.
        self.holdout_labels = ImageLabels()

        client = make_s3_client(settings.download_max_workers)

        # Test-fold rows are tagged and dropped here, before any feature file
        # is fetched for them.
        self.load_filtered_annotations(client)

        # Each set's manifest and index only: a run whose feature sets
        # disagree with its declared extractor downloads no feature file.
        with self.section_profiling("Resolving feature-extractor provenance"):
            indexes = self.fetch_feature_indexes(client)
            self.extractor_spec = self.resolve_extractor_spec(
                {name: result.manifest for name, result in indexes.items()}
            )

        with self.section_profiling("Detecting missing feature vectors"):
            self.handle_missing_feature_vectors(
                {name: result.index for name, result in indexes.items()}
            )

        if options.subsample is not None:
            # Runs after the missing-feature-vector drop above, so per-class
            # targets are computed from rows that can actually reach training.
            with self.section_profiling("Per-class subsampling"):
                self._apply_subsample(options.subsample)

        # Before the sync below, so capped-out holdout rows are never fetched.
        if options.holdout_max_per_class is not None:
            seed = options.subsample.seed if options.subsample is not None else 0
            with self.section_profiling("Capping holdout rows per class"):
                self._cap_holdout_rows(options.holdout_max_per_class, seed)

        with self.section_profiling("Syncing feature vectors"):
            self.sync_feature_vectors(client)

        self.labels = self.prep_annotations_for_pyspacer()

        with self.section_profiling("Tag DuckDB rows with training set"):
            self.add_training_set_names()

        self.set_train_summary_stats()

    @classmethod
    def filtered_annotations_only(cls, options: DatasetOptions) -> Self:
        """
        Build only the filtered `annotations` table (read, rollup, filter,
        dedupe, holdout-tag), with no feature index fetch, feature download
        or extractor resolution. The returned dataset is not trainable.
        """
        dataset = cls.__new__(cls)
        dataset._setup_annotation_reading(options)  # pyright: ignore[reportPrivateUsage]  # classmethod initialising its own instance
        dataset.load_filtered_annotations(make_s3_client(settings.download_max_workers))
        return dataset

    def _setup_annotation_reading(self, options: DatasetOptions) -> None:
        """Set the state `load_filtered_annotations` needs: specs, cache dir, DuckDB."""
        self.options = options
        # Resolved once: the lineage a run records must describe the selection
        # read here, even if a groups file changes afterwards.
        self.groups = resolve_groups(options)
        self.artifacts = Artifacts()
        self.profiled_sections = []
        if settings.feature_cache_dir:
            os.makedirs(settings.feature_cache_dir, exist_ok=True)
            self._feature_dir = settings.feature_cache_dir
            self._feature_temp_dir = None
        else:
            self._feature_temp_dir = tempfile.TemporaryDirectory(prefix="mermaid_features_")
            self._feature_dir = self._feature_temp_dir.name

        self.coralnet_source_ids: list[str] = []

        self.rollup_spec = _load_csv_spec(options.label_rollup_spec_csv, LabelRollupSpec)

        if options.included_labels_csv and options.excluded_labels_csv:
            raise ValueError("Specify one of included labels or excluded labels, but not both.")

        # An empty exclusion set when neither is given, so every label
        # is accepted.
        self.label_filter = _load_csv_spec(
            options.included_labels_csv or options.excluded_labels_csv,
            functools.partial(LabelFilter, inclusion=bool(options.included_labels_csv)),
        )

        self.image_exclusion_filter = _load_csv_spec(
            options.excluded_images_csv, ImageExclusionFilter
        )

        # The hash is of the bytes parsed here, so lineage describes what this run read.
        self.holdout_groups, self.holdout_groups_sha256, self.holdout_groups_bytes = (
            _read_holdout_groups(options.holdout_groups_csv)
        )
        self.mermaid_project_lookup_sha256: str | None = None
        # Fail on a bad holdout spec before any data is read.
        self.holdout_groups.validate_holdout_fold(options.holdout_fold)
        self._check_holdout_groups_can_match()

        self._duck_conn = None

    def _check_holdout_groups_can_match(self) -> None:
        """Raise for a listed holdout group no row of this run can belong to."""
        # Without the lookup every MERMAID row is project 'all', so no
        # listed MERMAID project can match.
        mermaid_groups = sorted(
            group_id
            for site, group_id in self.holdout_groups.group_folds
            if site == Sites.MERMAID.value
        )
        has_mermaid = any(source.labels == "mermaid" for source in self.options.datasets)
        if has_mermaid and mermaid_groups and not self.options.mermaid_project_lookup_uri:
            raise ValueError(
                f"holdout_groups_csv lists MERMAID group(s) {mermaid_groups}, but"
                f" mermaid_project_lookup_uri is unset, so no MERMAID row has a project."
            )
        unselected = _holdout_groups_unselected(self.holdout_groups, self.options, self.groups)
        # Groups in other CV folds are training data in this run, so their absence only warns.
        active_folds = {HOLDOUT_TEST_FOLD, self.options.holdout_fold}
        active = sorted(name for name, fold in unselected.items() if fold in active_folds)
        inactive = sorted(name for name, fold in unselected.items() if fold not in active_folds)
        if active:
            raise ValueError(
                f"holdout_groups_csv lists group(s) {active} in the test fold or holdout_fold"
                f" {self.options.holdout_fold!r} that the coralnet dataset's groups_csv does"
                f" not select, so no row of them is read. Add them to the groups_csv or"
                f" remove them from the holdout CSV."
            )
        if inactive:
            logger.warning(
                "holdout_groups_csv lists group(s) %s in other folds that the coralnet"
                " dataset's groups_csv does not select, so no row of them is read.",
                inactive,
            )

    def load_filtered_annotations(self, client: Any) -> None:
        """
        Populate the `annotations` table: read the pinned datasets' tables,
        roll up and filter labels, drop excluded images, dedupe points, tag
        holdout groups (is_holdout / is_excluded), and delete excluded rows.

        Fetches no feature index or feature file.
        """
        with self.section_profiling("Reading dataset versions"):
            self.read_datasets(client)

        # Now this should have annotations populated.
        if not self.duckdb_annotations_table_exists():
            raise ValueError("No annotations in the pinned datasets, even before label filtering.")

        # Lets holdout tagging tell a mistyped group from one filtering emptied.
        groups_read = self._groups_read() if not self.holdout_groups.is_empty() else set()

        with self.section_profiling("Rollups and filtering"):
            self._roll_up_and_filter()

        with self.section_profiling("Deduplicating points"):
            self._dedupe_points()

        with self.section_profiling("Tagging holdout groups"):
            self.holdout_groups.tag_in_duckdb(
                duck_conn=self.duck_conn,
                duck_table_name="annotations",
                holdout_fold=self.options.holdout_fold,
                groups_read=groups_read,
            )

        if self.options.holdout_groups_csv:
            # Test groups leave a CV run entirely: neither trained on nor
            # evaluated. Holdout rows stay, tagged, for later steps.
            self.duck_conn.execute("DELETE FROM annotations WHERE is_excluded")

    def _groups_read(self) -> set[tuple[str, str]]:
        """The (site, project_id) groups in the annotations table."""
        return {
            (str(site), str(project_id))
            for site, project_id in self.duck_conn.execute(
                "SELECT DISTINCT site, project_id FROM annotations"
            ).fetchall()
        }

    def _roll_up_and_filter(self) -> None:
        """Clear growth forms if asked, roll up, filter labels and exclude images, logging counts."""

        def _annotations_stats() -> tuple[int, int]:
            # (annotation rows, unique images)
            return self.duck_conn.execute(  # pyright: ignore[reportReturnType]  # duckdb fetchone returns tuple[Any, ...] | None but we know it's always (int, int)
                "SELECT COUNT(*), COUNT(DISTINCT (dataset, image_id)) FROM annotations"
            ).fetchone()

        ann_before, img_before = _annotations_stats()
        logger.info(
            "Before rollups/filtering: %s annotations, %s unique images",
            f"{ann_before:,}",
            f"{img_before:,}",
        )

        if self.options.drop_growthforms:
            # Clear all annotations' growth forms.
            duckdb_transform_column(
                duck_conn=self.duck_conn,
                duck_table_name="annotations",
                column_name="growth_form_id",
                transform_func=lambda x: "",
            )

        # Roll up BAGFs.
        self.rollup_spec.roll_up_in_duckdb(
            duck_conn=self.duck_conn,
            duck_table_name="annotations",
        )
        ann_after_rollup, img_after_rollup = _annotations_stats()
        logger.info(
            "After rollups: %s annotations (-%s), %s unique images (-%s)",
            f"{ann_after_rollup:,}",
            f"{ann_before - ann_after_rollup:,}",
            f"{img_after_rollup:,}",
            f"{img_before - img_after_rollup:,}",
        )

        # Filter out BAGFs we don't want.
        self.label_filter.filter_in_duckdb(
            duck_conn=self.duck_conn,
            duck_table_name="annotations",
        )
        ann_after_filter, img_after_filter = _annotations_stats()
        logger.info(
            "After included-labels filter: %s annotations (-%s), %s unique images (-%s)",
            f"{ann_after_filter:,}",
            f"{ann_after_rollup - ann_after_filter:,}",
            f"{img_after_filter:,}",
            f"{img_after_rollup - img_after_filter:,}",
        )

        # Remove any images excluded wholesale (e.g. a frozen
        # evaluation probe's held-out set), regardless of label.
        self.image_exclusion_filter.filter_in_duckdb(
            duck_conn=self.duck_conn,
            duck_table_name="annotations",
        )
        ann_after_exclusion, img_after_exclusion = _annotations_stats()
        logger.info(
            "After image exclusion: %s annotations (-%s), %s unique images (-%s)",
            f"{ann_after_exclusion:,}",
            f"{ann_after_filter - ann_after_exclusion:,}",
            f"{img_after_exclusion:,}",
            f"{img_after_filter - img_after_exclusion:,}",
        )
        logger.info(
            "Rollups+filter+exclusion retained %.1f%% of annotations, %.1f%% of unique images",
            100.0 * ann_after_exclusion / max(ann_before, 1),
            100.0 * img_after_exclusion / max(img_before, 1),
        )

    def _dedupe_points(self) -> None:
        """
        Make `annotations` hold at most one row per point, keyed by
        (dataset, image_id, row, col).

        Runs after rollup and filtering, so conflicts are judged on the
        final BA+GF. A point whose rows carry more than one distinct
        (benthic_attribute_id, growth_form_id) is dropped entirely. Any
        other point keeps one row: the first when ordered by every column,
        so the choice does not depend on scan order.
        """
        point_key = "dataset, image_id, row, col"
        order_by = ", ".join(
            f'"{name}"'
            for (name,) in self.duck_conn.execute(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_name = 'annotations' ORDER BY ordinal_position"
            ).fetchall()
        )

        rows_before = self.duck_conn.execute("SELECT count(*) FROM annotations").fetchall()[0][0]
        conflict_points, conflict_rows = self.duck_conn.execute(
            f"SELECT count(*), COALESCE(sum(n), 0) FROM ("
            f" SELECT count(*) AS n FROM annotations GROUP BY {point_key}"
            f" HAVING count(DISTINCT (benthic_attribute_id, growth_form_id)) > 1)"
        ).fetchall()[0]

        self.duck_conn.execute(
            f"CREATE OR REPLACE TABLE annotations AS"
            f" WITH conflicts AS ("
            f"  SELECT {point_key} FROM annotations GROUP BY {point_key}"
            f"  HAVING count(DISTINCT (benthic_attribute_id, growth_form_id)) > 1),"
            f" numbered AS ("
            f"  SELECT *, ROW_NUMBER() OVER ("
            f"   PARTITION BY {point_key} ORDER BY {order_by}) AS _rn"
            f"  FROM annotations)"
            f" SELECT n.* EXCLUDE (_rn) FROM numbered n"
            f" ANTI JOIN conflicts c USING ({point_key})"
            f" WHERE n._rn = 1"
        )

        rows_after = self.duck_conn.execute("SELECT count(*) FROM annotations").fetchall()[0][0]
        self._dedupe_stats = {
            "annotations_duplicate_rows_removed": int(rows_before - rows_after - conflict_rows),
            "points_label_conflicts_dropped": int(conflict_points),
        }
        logger.info(
            "Point dedupe: %s annotations -> %s; %s duplicate row(s) removed,"
            " %s point(s) dropped for conflicting labels",
            f"{rows_before:,}",
            f"{rows_after:,}",
            f"{self._dedupe_stats['annotations_duplicate_rows_removed']:,}",
            f"{conflict_points:,}",
        )

    def _has_holdout_column(self) -> bool:
        """Whether the annotations table carries an ``is_holdout`` column.

        Production always has it: load_filtered_annotations tags every row via
        HoldoutGroups.tag_in_duckdb, even with an empty spec. The probe exists for
        callers that build ``annotations`` directly, e.g.
        tests/training/test_subsample/test_subsample_in_duckdb.py and
        tests/pyspacer/test_training_dataset_pipeline.py.
        """
        columns = self.duck_conn.execute("DESCRIBE annotations").fetchall()
        return any(c[0] == "is_holdout" for c in columns)

    def _apply_subsample(self, opts: SubsampleOptions) -> None:
        """Deterministic per-class subsampling of the annotations table.

        Runs after rollup + included-labels filter, before the
        train/ref/val split. Only rows with ``is_holdout = FALSE`` take
        part: per-class targets come from their counts, and holdout rows
        pass through untouched (``_cap_holdout_rows`` bounds those).
        Resets the subsample audits first, so an empty table leaves them None.

        Pipeline (all in DuckDB, in-process):

          1. Read per-class counts of the non-holdout annotations,
             keyed by ``(benthic_attribute_id, growth_form_id)``. Both
             columns are post-rollup at this point.
          2. Call ``compute_per_class_targets(opts, counts)`` to get
             per-class target row counts. Strategy is dispatched by
             ``opts.strategy`` (today: 'stratified' or 'balanced'; new
             strategies plug in via ``training/subsample/registry.py``).
          3. Build the allocation by ``opts.within_class``:
             * ``'ordered'``: a deterministic ROW_NUMBER() partitioned
               by class and ordered by ``(site, project_id, image_id,
               row, col)``, keeping ``_rn <= target``.
             * ``'source_spread'``: split each class's target across its
               groups ``(site, project_id)`` with ``water_fill``, then
               keep ``_rn <= quota`` of a ROW_NUMBER() partitioned by
               class and group and ordered by a seeded image md5, so
               whole images are taken first.
             Both orderings are total, so the subsample is identical
             across processes and DuckDB thread counts.
          4. Replace the ``annotations`` table with the surviving rows
             and stash the realized per-class counts (and, for
             source_spread, per-(class, group) counts) for the MLflow audit.
        """
        self._subsample_audit_df = None
        self._subsample_realized_total = None
        self._subsample_group_audit_df = None
        has_holdout = self._has_holdout_column()
        # Holdout rows are set aside from the subsample, not counted in it.
        candidates = (
            "(SELECT * FROM annotations WHERE NOT is_holdout)" if has_holdout else ("annotations")
        )

        # Step 1: per-class counts (deterministic ORDER BY for stable
        # iteration order downstream).
        rows = self.duck_conn.execute(
            "SELECT benthic_attribute_id, growth_form_id, COUNT(*)"
            f" FROM {candidates}"
            " GROUP BY benthic_attribute_id, growth_form_id"
            " ORDER BY benthic_attribute_id, growth_form_id"
        ).fetchall()
        class_counts: dict[tuple[str, str], int] = {(ba, gf): n for ba, gf, n in rows}

        if not class_counts:
            logger.warning("Subsampling skipped: annotations table is empty.")
            return

        # Step 2: dispatch to the strategy registry.
        targets = compute_per_class_targets(opts, class_counts)

        # Step 3: the allocation, joined against a deterministic ROW_NUMBER.
        if opts.within_class == "source_spread":
            allocation = self._source_spread_allocation(candidates, targets, opts.seed)
        else:
            allocation = _ordered_allocation(targets)
        self._select_allocated_rows(allocation, candidates, has_holdout)

        # Step 4: the after-the-fact audit trail, confirming in MLflow that
        # two parallel runs saw the same data.
        per_class_audit = self._per_class_subsample_audit(candidates, class_counts, targets)
        self._subsample_audit_df = per_class_audit
        if allocation.group_available:
            self._subsample_group_audit_df = self._per_group_subsample_audit(candidates, allocation)

        realized_total = int(per_class_audit["realized_n"].sum())
        self._subsample_realized_total = realized_total
        logger.info(
            f"Subsample applied: strategy={opts.strategy!r},"
            f" within_class={opts.within_class!r},"
            f" classes={len(class_counts)},"
            f" target_total={opts.total_annotations},"
            f" realized_total={realized_total}"
        )

    def _source_spread_allocation(
        self, candidates: str, targets: Mapping[tuple[str, str], int], seed: int
    ) -> _SubsampleAllocation:
        """Each class's target split across its (site, project_id) groups by ``water_fill``."""
        group_rows = self.duck_conn.execute(
            "SELECT benthic_attribute_id, growth_form_id, site, project_id, COUNT(*)"
            f" FROM {candidates}"
            " GROUP BY ALL"
            " ORDER BY ALL"
        ).fetchall()
        group_available: dict[tuple[str, str, str, str], int] = {}
        per_class_groups: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
        for ba, gf, site, project_id, n in group_rows:
            group_available[(ba, gf, site, project_id)] = n
            per_class_groups.setdefault((ba, gf), {})[(site, project_id)] = n
        group_quotas: dict[tuple[str, str, str, str], int] = {}
        for (ba, gf), groups in per_class_groups.items():
            for (site, project_id), quota in water_fill(groups, targets.get((ba, gf), 0)).items():
                group_quotas[(ba, gf, site, project_id)] = quota
        allocation_df = pd.DataFrame(
            [
                {
                    "benthic_attribute_id": ba,
                    "growth_form_id": gf,
                    "site": site,
                    "project_id": project_id,
                    "target_n": int(quota),
                }
                for (ba, gf, site, project_id), quota in group_quotas.items()
            ]
        )
        return _SubsampleAllocation(
            allocation_df=allocation_df,
            partition_by="benthic_attribute_id, growth_form_id, site, project_id",
            # The seeded md5 clusters a group's rows by image; the identity breaks digest ties.
            order_by=f"{_seeded_image_order(seed)}, dataset, image_id, row, col",
            join_using="benthic_attribute_id, growth_form_id, site, project_id",
            group_quotas=group_quotas,
            group_available=group_available,
        )

    def _select_allocated_rows(
        self, allocation: _SubsampleAllocation, candidates: str, has_holdout: bool
    ) -> None:
        """Replace ``annotations`` with each partition's first ``target_n`` candidate rows,
        plus every holdout row."""
        self.duck_conn.register("_subsample_targets", allocation.allocation_df)
        try:
            holdout_union = " UNION ALL BY NAME SELECT * FROM annotations WHERE is_holdout"
            self.duck_conn.execute(
                "CREATE OR REPLACE TABLE annotations AS"
                " WITH numbered AS ("
                "   SELECT *,"
                f"          ROW_NUMBER() OVER ("
                f"              PARTITION BY {allocation.partition_by}"
                f"              ORDER BY {allocation.order_by}"
                "          ) AS _rn"
                f"   FROM {candidates}"
                " )"
                " SELECT n.* EXCLUDE (_rn)"
                " FROM numbered n"
                f" JOIN _subsample_targets t USING ({allocation.join_using})"
                " WHERE n._rn <= t.target_n" + (holdout_union if has_holdout else "")
            )
        finally:
            self.duck_conn.unregister("_subsample_targets")

    def _per_class_subsample_audit(
        self,
        candidates: str,
        class_counts: Mapping[tuple[str, str], int],
        targets: Mapping[tuple[str, str], int],
    ) -> pd.DataFrame:
        """Each class's pre-subsample count, target and realized count."""
        realized_rows = self.duck_conn.execute(
            "SELECT benthic_attribute_id, growth_form_id, COUNT(*) AS n"
            f" FROM {candidates}"
            " GROUP BY benthic_attribute_id, growth_form_id"
            " ORDER BY benthic_attribute_id, growth_form_id"
        ).fetchall()
        realized_by_cls = {(ba, gf): n for ba, gf, n in realized_rows}
        return pd.DataFrame(
            [
                {
                    "benthic_attribute_id": ba,
                    "growth_form_id": gf,
                    "pre_count": class_counts[(ba, gf)],
                    "target_n": targets.get((ba, gf), 0),
                    "realized_n": realized_by_cls.get((ba, gf), 0),
                }
                for (ba, gf) in sorted(class_counts)
            ]
        )

    def _per_group_subsample_audit(
        self, candidates: str, allocation: _SubsampleAllocation
    ) -> pd.DataFrame:
        """Each (class, group)'s available rows, quota and realized count."""
        realized_group_rows = self.duck_conn.execute(
            "SELECT benthic_attribute_id, growth_form_id, site, project_id, COUNT(*)"
            f" FROM {candidates}"
            " GROUP BY ALL"
        ).fetchall()
        realized_by_group = {(ba, gf, site, pid): n for ba, gf, site, pid, n in realized_group_rows}
        return pd.DataFrame(
            [
                {
                    "benthic_attribute_id": ba,
                    "growth_form_id": gf,
                    "site": site,
                    "project_id": project_id,
                    "available": allocation.group_available[(ba, gf, site, project_id)],
                    "quota": allocation.group_quotas[(ba, gf, site, project_id)],
                    "realized": realized_by_group.get((ba, gf, site, project_id), 0),
                }
                for (ba, gf, site, project_id) in sorted(allocation.group_available)
            ]
        )

    def _cap_holdout_rows(self, max_per_class: int, seed: int = 0) -> None:
        """Cap each class's holdout rows at ``max_per_class``.

        Keeps whole images first: within a class (not per group), rows are
        ordered by a seeded md5 of the dataset and image id, so the survivors
        cluster by image. Non-holdout rows are untouched. A no-op, leaving
        ``_holdout_audit_df`` None, when there are no holdout rows.
        """
        self._holdout_audit_df = None
        available_rows = self.duck_conn.execute(
            "SELECT benthic_attribute_id, growth_form_id, COUNT(*)"
            " FROM annotations WHERE is_holdout"
            " GROUP BY benthic_attribute_id, growth_form_id"
            " ORDER BY benthic_attribute_id, growth_form_id"
        ).fetchall()
        if not available_rows:
            return

        self.duck_conn.execute(
            "CREATE OR REPLACE TABLE annotations AS"
            " WITH numbered AS ("
            "   SELECT *,"
            "          ROW_NUMBER() OVER ("
            "              PARTITION BY benthic_attribute_id, growth_form_id"
            f"              ORDER BY {_seeded_image_order(seed)},"
            "                       dataset, image_id, site, project_id, row, col"
            "          ) AS _rn"
            "   FROM annotations WHERE is_holdout"
            " )"
            " SELECT * EXCLUDE (_rn) FROM numbered"
            f" WHERE _rn <= {int(max_per_class)}"
            " UNION ALL BY NAME SELECT * FROM annotations WHERE NOT is_holdout"
        )

        realized_by_cls = {
            (ba, gf): n
            for ba, gf, n in self.duck_conn.execute(
                "SELECT benthic_attribute_id, growth_form_id, COUNT(*)"
                " FROM annotations WHERE is_holdout"
                " GROUP BY benthic_attribute_id, growth_form_id"
            ).fetchall()
        }
        self._holdout_audit_df = pd.DataFrame(
            [
                {
                    "benthic_attribute_id": ba,
                    "growth_form_id": gf,
                    "available": n,
                    "realized": realized_by_cls.get((ba, gf), 0),
                }
                for ba, gf, n in available_rows
            ]
        )
        logger.info(
            "Holdout cap applied: max_per_class=%s, holdout rows %s -> %s",
            max_per_class,
            f"{sum(n for _, _, n in available_rows):,}",
            f"{sum(realized_by_cls.values()):,}",
        )

    def cleanup(self):
        """Clean up temporary feature vector files."""
        if self._feature_temp_dir is not None:
            self._feature_temp_dir.cleanup()

    @contextmanager
    def section_profiling(self, section_name: str):
        with section_profiling(self.profiled_sections, section_name):
            yield

    def read_datasets(self, client: Any) -> None:
        """Sync every pinned version's tables and load their points as annotations."""
        cache_dir = Path(self._feature_dir)
        pins = {source.pin.name: source.pin for source in self.options.datasets}
        sync_tables(client, pins, cache_dir, bucket=DATASET_BUCKET, taxonomy=self.options.taxonomy)
        taxonomy = (
            None
            if self.options.taxonomy is None
            else open_taxonomy(cache_dir, DATASET_BUCKET, self.options.taxonomy)
        )
        coralnet_mapping = CoralNetMermaidMapping()
        for source in self.options.datasets:
            tables = open_tables(cache_dir, DATASET_BUCKET, source.pin)
            self._append_annotations(
                self._source_annotations(source, tables, taxonomy, coralnet_mapping)
            )

        lookup_uri = self.options.mermaid_project_lookup_uri
        if lookup_uri and any(source.labels == "mermaid" for source in self.options.datasets):
            lookup_path = self._local_lookup_path(client, lookup_uri)
            self.mermaid_project_lookup_sha256 = hashlib.sha256(
                Path(lookup_path).read_bytes()
            ).hexdigest()
            self._apply_mermaid_project_lookup(lookup_path)

        # Project-level stats before any filtering, unmapped labels included.
        self.artifacts.coralnet_project_stats = self.compute_project_stats(
            site=Sites.CORALNET.value
        )
        self.artifacts.mermaid_project_stats = self.compute_project_stats(site=Sites.MERMAID.value)

        if any(source.labels == "coralnet" for source in self.options.datasets):
            self.artifacts.coralnet_label_mapping = coralnet_mapping.get_dataframe()
            self._drop_unmapped_labels()
            self.coralnet_source_ids = [
                str(r[0])
                for r in self.duck_conn.execute(
                    "SELECT DISTINCT project_id FROM annotations WHERE site = ?"
                    " ORDER BY CAST(project_id AS INTEGER)",
                    [Sites.CORALNET.value],
                ).fetchall()
            ]

    def _local_lookup_path(self, client: Any, lookup_uri: str) -> str:
        """`lookup_uri` itself when local; an S3 lookup is fetched into the cache dir first."""
        if not is_s3_uri(lookup_uri):
            return lookup_uri
        bucket, key = parse_s3_uri(lookup_uri)
        local_path = Path(self._feature_dir) / "mermaid_project_lookup" / bucket / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(client.get_object(Bucket=bucket, Key=key)["Body"].read())
        return str(local_path)

    def _apply_mermaid_project_lookup(self, lookup_path: str) -> None:
        """Give each MERMAID row its image's project from the lookup; 'unknown' if absent."""
        self._check_project_lookup_is_one_to_one(lookup_path)
        # One project per image (checked above), so the join cannot fan out
        # rows; a lookup repeating an image/project pair is fine.
        self.duck_conn.execute(
            "CREATE OR REPLACE TABLE annotations AS"
            " SELECT a.* REPLACE ("
            f"  CASE WHEN a.site = '{Sites.MERMAID.value}'"
            "   THEN COALESCE(l.project_id, 'unknown') ELSE a.project_id END AS project_id)"
            " FROM annotations a"
            " LEFT JOIN ("
            "  SELECT CAST(image_id AS VARCHAR) AS image_id,"
            "   min(CAST(project_id AS VARCHAR)) AS project_id"
            "  FROM read_parquet(?) GROUP BY 1) l"
            f" ON a.site = '{Sites.MERMAID.value}' AND a.image_id = l.image_id",
            [lookup_path],
        )
        self._check_unmatched_mermaid_images()

    def _check_project_lookup_is_one_to_one(self, lookup_uri: str) -> None:
        """Raise if the lookup maps any image to more than one project."""
        ambiguous = self.duck_conn.execute(
            "SELECT CAST(image_id AS VARCHAR) FROM read_parquet(?)"
            " GROUP BY 1 HAVING count(DISTINCT project_id) > 1"
            " ORDER BY 1 LIMIT 3",
            [lookup_uri],
        ).fetchall()
        if ambiguous:
            raise ValueError(
                f"MERMAID project lookup {lookup_uri} maps an image to more than one project,"
                f" such as: {[r[0] for r in ambiguous]}"
            )

    def _check_unmatched_mermaid_images(self) -> None:
        """Log MERMAID images the lookup left as 'unknown'; raise if too many."""
        unmatched, total = self.duck_conn.execute(
            "SELECT count(DISTINCT image_id) FILTER (WHERE project_id = 'unknown'),"
            " count(DISTINCT image_id)"
            f" FROM annotations WHERE site = '{Sites.MERMAID.value}'"
        ).fetchall()[0]
        logger.info(
            "MERMAID project lookup left %s of %s image(s) unmatched (project_id 'unknown').",
            f"{unmatched:,}",
            f"{total:,}",
        )
        allowed = settings.mermaid_project_lookup_percent_unmatched_allowed
        if unmatched > total * allowed / 100:
            raise RuntimeError(
                f"Too many MERMAID images are missing from the project lookup"
                f" ({unmatched} of {total}). Regenerate the lookup so it covers"
                f" these images, or raise the tolerance with the"
                f" MERMAID_PROJECT_LOOKUP_PERCENT_UNMATCHED_ALLOWED setting."
            )

    def _source_annotations(
        self,
        source: PinnedDataset,
        tables: VersionTables,
        taxonomy: Taxonomy | None,
        coralnet_mapping: CoralNetMermaidMapping,
    ) -> _Table:
        name = source.pin.name
        if tables.points is None:
            raise ValueError(f"{name} is not a points dataset, so it has no points to train on")
        check_labels_kind(tables.labels, source.labels, name, tables.manifest.source.type)
        if source.labels == "coralnet":
            label_table = coralnet_label_table(tables.labels, coralnet_mapping)
        elif taxonomy is None:
            raise ValueError(f"{name} has `labels: mermaid`, which needs a taxonomy pin")
        else:
            label_table = mermaid_label_table(tables.labels, taxonomy.growth_forms, name)
        return dataset_annotations(
            tables.points,
            select_groups(tables.images, self.groups[name], name),
            label_table,
            site=Sites(source.labels),
            dataset=name,
        )

    def _append_annotations(self, table: _Table) -> None:
        if self.duckdb_annotations_table_exists():
            # BY NAME: the column order of the existing table need not match.
            query_start = "INSERT INTO annotations BY NAME"
        else:
            query_start = "CREATE TABLE annotations AS"
        self.duck_conn.register("_dataset_annotations", table)
        try:
            self.duck_conn.execute(f"{query_start} SELECT * FROM _dataset_annotations")
        finally:
            self.duck_conn.unregister("_dataset_annotations")

    def _drop_unmapped_labels(self) -> None:
        """Record, then drop, the CoralNet points whose label maps to no MERMAID BA."""
        self.artifacts.unmapped_labels = self.duck_conn.execute(
            "SELECT"
            " label_id,"
            " count(*) AS num_annotations,"
            " count(DISTINCT project_id) AS num_projects,"
            " FROM annotations"
            " WHERE benthic_attribute_id IS NULL"
            " GROUP BY label_id"
            " ORDER BY num_annotations DESC"
        ).fetch_df()
        self.duck_conn.execute("DELETE FROM annotations WHERE benthic_attribute_id IS NULL")

    def duckdb_annotations_table_exists(self) -> bool:
        # https://duckdb.org/docs/stable/sql/meta/information_schema
        table_query_result = self.duck_conn.execute(
            "SELECT * FROM information_schema.tables WHERE table_name = 'annotations'"
        ).fetchall()

        # Result should be [] if doesn't exist, which 'bools' to False.
        return bool(table_query_result)

    def fetch_feature_indexes(self, client: Any) -> dict[str, FeatureSyncResult]:
        """Each dataset's verified feature-set manifest and index, by dataset name.

        Selecting no image fetches no feature file.
        """
        return {
            source.pin.name: sync_feature_set(
                client,
                source.features,
                Path(self._feature_dir),
                bucket=DATASET_BUCKET,
                image_ids=[],
                max_workers=settings.download_max_workers,
            )
            for source in self.options.datasets
        }

    def resolve_extractor_spec(self, manifests: Mapping[str, FeatureSetManifest]) -> ExtractorSpec:
        """One extractor for the whole run, or refuse to train.

        A head fitted across two feature spaces is wrong in a way no later
        gate detects: the artifact loads, the parity gate passes, and every
        score is drawn from a distribution the head never saw. So a feature
        set whose extractor differs from `dataset.feature_extractor_weights`
        stops the run here.
        """
        if not self.options.feature_extractor_weights:
            raise ValueError(
                "This run declares no dataset.feature_extractor_weights, so nothing"
                " checks which extractor produced its feature sets. Set that field in"
                " the training config."
            )
        declared = extractor_spec_from_weights(self.options.feature_extractor_weights)
        found = {
            f"{name} feature set {manifest.family}": feature_set_extractor(manifest)
            for name, manifest in manifests.items()
        }
        differing = {
            source: spec for source, spec in found.items() if spec.identity() != declared.identity()
        }
        if differing:
            detail = "\n".join(
                f"  {source}: {spec.describe()}" for source, spec in differing.items()
            )
            raise ValueError(
                "Feature sets record a different extractor from"
                f" dataset.feature_extractor_weights ({declared.describe()}), so"
                f" their vectors are not what this run declares:\n{detail}"
            )
        # A producer's own record names where the weights were used.
        spec = next(iter(found.values()), declared)
        logger.info("Feature extractor for this run: %s", spec.describe())
        return spec

    def handle_missing_feature_vectors(self, indexes: Mapping[str, _Table]) -> None:
        """Drop annotations of images their dataset's feature index lacks.

        Images absent from the index count against
        TRAINING_INPUTS_PERCENT_MISSING_ALLOWED, as a share of the distinct
        images in the annotations table; above it the run stops.
        """
        names = [name for name, index in indexes.items() for _ in range(index.num_rows)]
        indexed = pd.DataFrame(  # noqa: F841  # pyright: ignore[reportUnusedVariable]  # referenced by name in DuckDB SQL via Python-scope scanning
            {
                "dataset": names,
                "image_id": [
                    i for index in indexes.values() for i in index.column("image_id").to_pylist()
                ],
                "feature_vector": [
                    k for index in indexes.values() for k in index.column("key").to_pylist()
                ],
            }
        )

        with (
            duckdb_temp_table_name(self.duck_conn) as indexed_table_name,
            duckdb_temp_table_name(self.duck_conn) as missing_table_name,
        ):
            self.duck_conn.execute(
                f"CREATE TEMP TABLE {indexed_table_name} AS"
                " SELECT dataset, image_id, feature_vector FROM indexed"
            )

            in_annotations_count = self.duck_conn.execute(
                "SELECT COUNT(DISTINCT (dataset, image_id)) FROM annotations"
            ).fetchall()[0][0]

            self.duck_conn.execute(
                f"CREATE TABLE {missing_table_name} AS"
                f" SELECT DISTINCT a.dataset, a.image_id FROM annotations a"
                f" LEFT JOIN {indexed_table_name} f USING (dataset, image_id)"
                f" WHERE f.feature_vector IS NULL"
            )
            missing_count = self.duck_conn.execute(
                f"SELECT COUNT(*) FROM {missing_table_name}"
            ).fetchall()[0][0]
            missing_examples = [
                tup[0]
                for tup in self.duck_conn.execute(
                    f"SELECT dataset || '/' || image_id FROM {missing_table_name}"
                    " ORDER BY dataset, image_id LIMIT 3"
                ).fetchall()
            ]

            # Abort if too many are missing (check before mutating).
            examples_str = "\n".join(missing_examples)
            missing_threshold = (
                in_annotations_count * settings.training_inputs_percent_missing_allowed / 100
            )
            if missing_count > missing_threshold:
                raise RuntimeError(
                    f"Too many images have no feature vector"
                    f" ({missing_count}), such as:"
                    f"\n{examples_str}"
                    f"\nYou can configure the tolerance for missing"
                    f" feature vectors with the"
                    f" TRAINING_INPUTS_PERCENT_MISSING_ALLOWED setting."
                )

            # Keep the annotations whose image is indexed, with its feature key.
            self.duck_conn.execute(
                f"CREATE OR REPLACE TABLE annotations AS"
                f" SELECT a.*, f.feature_vector FROM annotations a"
                f" JOIN {indexed_table_name} f USING (dataset, image_id)"
            )

        if missing_count > 0:
            logger.warning(
                f"Skipping {missing_count} image(s) because"
                f" their dataset's feature set does not index them."
                f" Example(s):"
                f"\n{examples_str}"
            )

    def sync_feature_vectors(self, client: Any) -> None:
        """Download and verify the feature file of every image still in the annotations."""
        cache_dir = Path(self._feature_dir)
        for source in self.options.datasets:
            image_ids = [
                row[0]
                for row in self.duck_conn.execute(
                    "SELECT DISTINCT image_id FROM annotations WHERE dataset = ? ORDER BY image_id",
                    [source.pin.name],
                ).fetchall()
            ]
            logger.info("Syncing features for %s: %d images", source.pin.name, len(image_ids))
            start = time.monotonic()
            _sync_with_retries(
                source.pin.name,
                functools.partial(
                    sync_feature_set,
                    client,
                    source.features,
                    cache_dir,
                    bucket=DATASET_BUCKET,
                    image_ids=image_ids,
                    max_workers=settings.download_max_workers,
                ),
            )
            logger.info(
                "Synced features for %s in %.1f s", source.pin.name, time.monotonic() - start
            )
            self._feature_sets[source.pin.name] = open_feature_set(
                cache_dir, DATASET_BUCKET, source.features
            )

    def prep_annotations_for_pyspacer(self) -> TrainingTaskLabels:
        """
        Build PySpacer labels from the synced feature files and split them
        into train/ref/val by `options.split_mode`.

        Holdout rows (is_holdout) take no part in the split: they become
        `self.holdout_labels`, which is empty when there are none.
        """
        has_holdout = self._has_holdout_column()
        with self.section_profiling("Building PySpacer labels"):
            annotations_by_image = duckdb_grouped_rows(
                duck_conn=self.duck_conn,
                duck_table_name="annotations",
                grouping_column_names=["bucket", "feature_vector"],
                extra_order_column_names=["row", "col"],
            )
            labels_data = ImageLabels()
            holdout_data = ImageLabels()

            for rows in annotations_by_image:
                # Here, in one loop iteration, we're given all the
                # annotation rows for a single image.
                first_row = rows[0]
                feature_set = self._feature_sets[str(first_row["dataset"])]
                local_path = str(feature_set.path(str(first_row["image_id"])))
                # add_training_set_names maps the cached path back to the
                # annotations table's (bucket, feature_vector).
                self._feature_path_to_s3_location[local_path] = (
                    str(first_row["bucket"]),
                    str(first_row["feature_vector"]),
                )

                image_annotations: list[Annotation] = []
                holdout_annotations: list[Annotation] = []
                for row in rows:
                    annotation = (
                        int(row["row"]),
                        int(row["col"]),
                        combine_ba_gf(str(row["benthic_attribute_id"]), str(row["growth_form_id"])),
                    )
                    if has_holdout and bool(row["is_holdout"]):
                        holdout_annotations.append(annotation)
                    else:
                        image_annotations.append(annotation)

                feature_loc = DataLocation(storage_type="filesystem", key=local_path)
                if image_annotations:
                    labels_data.add_image(feature_loc, image_annotations)
                if holdout_annotations:
                    holdout_data.add_image(feature_loc, holdout_annotations)
                self.feature_loc_to_source[feature_loc] = (
                    str(first_row["site"]),
                    str(first_row["project_id"]),
                )

            self.holdout_labels = holdout_data

        with self.section_profiling(f"Splitting train/ref/val ({self.options.split_mode})"):
            split = self._split_labels(labels_data)
        self._log_split_drops(labels_data, split)
        return split

    def _split_labels(self, labels_data: ImageLabels) -> TrainingTaskLabels:
        """Split `labels_data` into train/ref/val by `options.split_mode`."""
        if self.options.split_mode == "images":
            subsample = self.options.subsample
            return preprocess_labels(
                split_by_image(
                    labels_data,
                    self.options.ref_val_ratios,
                    seed=subsample.seed if subsample is not None else 0,
                    # The cache dir differs per run; the S3 location does not.
                    stable_key=lambda loc: "/".join(self._feature_path_to_s3_location[loc.key]),
                )
            )
        return preprocess_labels(
            labels_data,
            split_ratios=self.options.ref_val_ratios,
            split_mode=SplitMode.POINTS_STRATIFIED,
        )

    def _log_split_drops(self, before: ImageLabels, after: TrainingTaskLabels) -> None:
        """
        Log by name each class the split removed, and count them.

        PySpacer drops a class missing from train or ref from all three sets.
        A class in MIN_IMAGES_PER_CLASS or more images could have reached
        every set, so its drop is a warning; a rarer class's drop is info.
        """
        kept = after.train.classes_set | after.ref.classes_set | after.val.classes_set
        images_per_class: Counter[LabelId] = Counter()
        for loc in before.keys():  # noqa: SIM118 -- ImageLabels is not a dict
            images_per_class.update({label for _, _, label in before[loc]})
        dropped = sorted(set(images_per_class) - kept, key=str)
        self._split_dropped_bagfs = len(dropped)

        def describe(labels: list[LabelId]) -> str:
            return ", ".join(
                f"{label}: {images_per_class[label]}, {before.label_count_per_class[label]}"
                for label in labels
            )

        common = [label for label in dropped if images_per_class[label] >= MIN_IMAGES_PER_CLASS]
        rare = [label for label in dropped if images_per_class[label] < MIN_IMAGES_PER_CLASS]
        if common:
            logger.warning(
                "Split drops %d class(es) in %d or more images from train, ref and val"
                " (class: images, points): %s",
                len(common),
                MIN_IMAGES_PER_CLASS,
                describe(common),
            )
        if rare:
            logger.info(
                "Split drops %d class(es) in fewer than %d images from train, ref and val"
                " (class: images, points): %s",
                len(rare),
                MIN_IMAGES_PER_CLASS,
                describe(rare),
            )

    @property
    def duck_conn(self):
        """
        Return this TrainingDataset's DuckDB connection, made once.
        """
        if self._duck_conn is None:
            self._duck_conn = duckdb.connect()
        return self._duck_conn

    def compute_project_stats(
        self, site: str | None = None, has_training_sets: bool = False
    ) -> pd.DataFrame:
        where_clause = "" if site is None else f"WHERE site = '{site}'"

        counts_sql = " count(DISTINCT image_id) AS num_images, count(*) AS num_annotations"
        if has_training_sets:
            counts_sql += (
                ","
                " countif(training_set = 'train') AS train,"
                " countif(training_set = 'ref') AS ref,"
                " countif(training_set = 'val') AS val,"
                " countif(training_set = 'holdout') AS holdout,"
                " countif(training_set IS NULL) AS dropped"
            )

        result = self.duck_conn.execute(
            f"SELECT site, project_id,"
            f" {counts_sql}"
            f" FROM annotations"
            f" {where_clause}"
            f" GROUP BY site, project_id"
            # MERMAID, then CoralNet; because MERMAID's currently just
            # one row (no projects distinction).
            f" ORDER BY site DESC, project_id"
        )
        return result.fetch_df()

    def add_training_set_names(self):
        """
        Match up DuckDB annotations with train/ref/val/holdout.
        This will add a training_set column to the annotations table.
        """
        # A duplicated key would fan each label row out in the join below.
        duplicate_keys = self.duck_conn.execute(
            "SELECT count(*) FROM (SELECT 1 FROM annotations"
            " GROUP BY bucket, feature_vector, row, col HAVING count(*) > 1)"
        ).fetchall()[0][0]
        if duplicate_keys:
            raise ValueError(
                f"annotations has {duplicate_keys} (bucket, feature_vector, row, col) key(s)"
                f" that are not unique, so training sets cannot be matched to rows."
            )

        training_sets = [
            ("train", self.labels.train),
            ("ref", self.labels.ref),
            ("val", self.labels.val),
            ("holdout", self.holdout_labels),
        ]
        # Higher means fewer DuckDB operations; lower might reduce
        # memory usage.
        batch_size = 50000

        with duckdb_temp_table_name(self.duck_conn) as temp_table_name:
            # The columns here besides training_set are the ones that should
            # uniquely identify a particular annotation.
            self.duck_conn.execute(
                f"CREATE TABLE {temp_table_name}"
                f" (bucket VARCHAR,"
                f"  feature_vector VARCHAR,"
                f"  row VARCHAR,"
                f"  col VARCHAR,"
                f"  training_set VARCHAR)"
            )

            for set_name, training_set in training_sets:
                values_batch: list[tuple[str, str, int, int, str]] = []

                for feature_loc, row, col in self.generate_training_set_annotations(training_set):
                    # feature_loc is a filesystem DataLocation whose key is a
                    # local download path; map it back to the original S3
                    # (bucket, feature_vector) so it matches the annotations
                    # table's join columns.
                    bucket, feature_vector = self._feature_path_to_s3_location[feature_loc.key]
                    tup = (
                        bucket,
                        feature_vector,
                        row,
                        col,
                        set_name,
                    )
                    values_batch.append(tup)

                    if len(values_batch) > batch_size:
                        self._add_tuples_to_table(temp_table_name, values_batch)
                        values_batch = []

                if len(values_batch) > 0:
                    # Last batch
                    self._add_tuples_to_table(temp_table_name, values_batch)

            # Join annotations with the temp table to add the training_set info.
            #
            # LEFT OUTER JOIN ensures that annotations with no training_set
            # match just get NULL for that column. A regular JOIN would instead
            # drop those annotations rows entirely.
            self.duck_conn.execute(
                f"CREATE OR REPLACE TABLE annotations AS"
                f" SELECT *"
                f" FROM annotations"
                f"  LEFT OUTER JOIN {temp_table_name}"
                f"  USING (bucket, feature_vector, row, col)"
            )

    def _add_tuples_to_table(
        self, table_name: str, tuples: list[tuple[str, str, int, int, str]]
    ) -> None:
        df = pd.DataFrame.from_records(tuples)  # noqa: F841  # pyright: ignore[reportUnusedVariable]  # referenced by name in DuckDB SQL via Python-scope scanning
        self.duck_conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")

    @staticmethod
    def generate_training_set_annotations(training_set: ImageLabels):
        for feature_loc in training_set.keys():  # noqa: SIM118 — ImageLabels.keys() is not a plain dict; __iter__ differs
            image_annotations = training_set[feature_loc]
            for row, col, _ in image_annotations:
                yield feature_loc, row, col

    def set_train_summary_stats(self):
        # Counts per BA.
        self.duck_conn.execute(
            "CREATE TABLE ba_counts AS"
            " SELECT"
            " benthic_attribute_id,"
            " count(DISTINCT project_id) AS num_projects,"
            " count(*) AS num_annotations,"
            " countif(training_set = 'train') AS train,"
            " countif(training_set = 'ref') AS ref,"
            " countif(training_set = 'val') AS val,"
            " countif(training_set = 'holdout') AS holdout,"
            " countif(training_set IS NULL) AS dropped"
            " FROM annotations GROUP BY benthic_attribute_id"
        )
        ba_library = get_benthic_attribute_library()
        gf_library_inst = get_growth_form_library()

        def _ba_id_to_name(ba_id: str | None) -> str | None:
            return ba_library.id_to_name(ba_id) if ba_id is not None else None

        def _gf_id_to_name(gf_id: str | None) -> str | None:
            return gf_library_inst.id_to_name(gf_id) if gf_id is not None else None

        # Add BA names alongside the IDs for readability.
        duckdb_add_column(
            duck_conn=self.duck_conn,
            duck_table_name="ba_counts",
            base_column_name="benthic_attribute_id",
            new_column_name="benthic_attribute_name",
            base_to_new_func=_ba_id_to_name,
        )
        # Sort by total annotation count, and reorder columns
        # while we're at it.
        self.artifacts.ba_counts = self.duck_conn.execute(
            "SELECT"
            " benthic_attribute_name,"
            " num_projects,"
            " num_annotations,"
            " train,"
            " ref,"
            " val,"
            " holdout,"
            " dropped,"
            " benthic_attribute_id"
            " FROM ba_counts"
            " ORDER BY num_annotations DESC"
        ).fetch_df()

        # Counts per BAGF.
        self.duck_conn.execute(
            "CREATE TABLE bagf_counts AS"
            " SELECT"
            " benthic_attribute_id,"
            " growth_form_id,"
            " count(DISTINCT project_id) AS num_projects,"
            " count(*) AS num_annotations,"
            " countif(training_set = 'train') AS train,"
            " countif(training_set = 'ref') AS ref,"
            " countif(training_set = 'val') AS val,"
            " countif(training_set = 'holdout') AS holdout,"
            " countif(training_set IS NULL) AS dropped"
            " FROM annotations"
            " GROUP BY benthic_attribute_id, growth_form_id"
        )
        # Add BA names for readability.
        duckdb_add_column(
            duck_conn=self.duck_conn,
            duck_table_name="bagf_counts",
            base_column_name="benthic_attribute_id",
            new_column_name="benthic_attribute_name",
            base_to_new_func=_ba_id_to_name,
        )
        # Add GF names for readability.
        duckdb_add_column(
            duck_conn=self.duck_conn,
            duck_table_name="bagf_counts",
            base_column_name="growth_form_id",
            new_column_name="growth_form_name",
            base_to_new_func=_gf_id_to_name,
        )
        # Sort by annotation count, and reorder columns
        # while we're at it.
        self.artifacts.bagf_counts = self.duck_conn.execute(
            "SELECT"
            " benthic_attribute_name,"
            " growth_form_name,"
            " num_projects,"
            " num_annotations,"
            " train,"
            " ref,"
            " val,"
            " holdout,"
            " dropped,"
            " benthic_attribute_id,"
            " growth_form_id"
            " FROM bagf_counts"
            " ORDER BY num_annotations DESC"
        ).fetch_df()

        # Overall counts.
        counts = self.duck_conn.execute(
            "SELECT count(*), count(DISTINCT image_id) FROM annotations"
        ).fetchall()[0]
        total_annotations = counts[0]
        num_of_images = counts[1]
        num_of_bas = self.artifacts.ba_counts.shape[0]
        num_of_bagfs = self.artifacts.bagf_counts.shape[0]

        # The split drops BAGFs too rare to reach train, ref and val. Holdout
        # rows take no part in the split, so they count on neither side:
        # a dropped class keeping holdout rows is still dropped.
        split_candidates = "WHERE NOT is_holdout" if self._has_holdout_column() else ""
        split_counts, kept_counts = (
            self.duck_conn.execute(
                "SELECT"
                " count(*),"
                " count(DISTINCT benthic_attribute_id),"
                " count(DISTINCT (benthic_attribute_id, growth_form_id))"
                f" FROM annotations {where}"
            ).fetchall()[0]
            for where in (split_candidates, "WHERE training_set IN ('train', 'ref', 'val')")
        )
        annotations_dropped = split_counts[0] - kept_counts[0]
        bas_dropped = split_counts[1] - kept_counts[1]
        bagfs_dropped = split_counts[2] - kept_counts[2]

        self.artifacts.train_summary_stats = {
            "annotations": total_annotations,
            "annotations_train": self.labels.train.label_count,
            "annotations_ref": self.labels.ref.label_count,
            "annotations_val": self.labels.val.label_count,
            "annotations_holdout": self.holdout_labels.label_count,
            "annotations_dropped": annotations_dropped,
            "images": num_of_images,
            "bas": num_of_bas,
            "bas_dropped": bas_dropped,
            "bagfs": num_of_bagfs,
            "bagfs_dropped": bagfs_dropped,
            "bagfs_dropped_by_split": self._split_dropped_bagfs,
            **(self._dedupe_stats or {}),
        }

    def describe_train_summary_stats(self):
        return (
            "{annotations} annotations"
            " ({annotations_train} train,"
            " {annotations_ref} ref,"
            " {annotations_val} val,"
            " {annotations_holdout} holdout,"
            " {annotations_dropped} dropped during stratification) from"
            " {images} images."
            " Representation: {bas} BAs and"
            " {bagfs} BA-GF combos"
            " (dropped: {bas_dropped} BAs, {bagfs_dropped} BA-GFs).".format(
                **self.artifacts.train_summary_stats
            )
        )

    def get_annotations(self, log_spec: str):
        if log_spec == "all":
            query = "SELECT * EXCLUDE (is_holdout, is_excluded) FROM annotations"
        elif match := re.fullmatch(r"s(\d+)", log_spec):
            cn_source_id = match.groups()[0]
            query = (
                f"SELECT * EXCLUDE (is_holdout, is_excluded) FROM annotations"
                f" WHERE site = '{Sites.CORALNET.value}'"
                f" AND project_id = '{cn_source_id}'"
            )
        elif match := re.fullmatch(r"i(\d+)", log_spec):
            cn_image_id = match.groups()[0]
            query = (
                f"SELECT * EXCLUDE (is_holdout, is_excluded) FROM annotations"
                f" WHERE site = '{Sites.CORALNET.value}'"
                f" AND image_id = '{cn_image_id}'"
            )
        else:
            raise ValueError(f"Unsupported annotations log spec: {log_spec}")

        return self.duck_conn.execute(query).fetch_df()
