"""
TrainingDataset: loads, transforms, and prepares annotation data for training.

Reads CoralNet manifest parquet and MERMAID Parquet via DuckDB,
maps CoralNet label IDs to MERMAID BA+GF, applies rollup/filter specs,
validates feature vector availability, and produces pyspacer-ready
train/ref/val splits.
"""

import functools
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from io import StringIO
from typing import Self, TextIO

import duckdb
import pandas as pd
from s3fs.core import S3FileSystem
from spacer.data_classes import Annotation, DataLocation, ImageLabels, LabelId
from spacer.messages import TrainingTaskLabels
from spacer.task_utils import SplitMode, preprocess_labels

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
from mermaid_classifier.common.s3_utils import EMPTY_SHA256, download_features_parallel
from mermaid_classifier.pyspacer._pipeline_utils import section_profiling
from mermaid_classifier.pyspacer.extraction import (
    SIDECAR_FILENAME,
    extractor_spec_from_weights,
)
from mermaid_classifier.pyspacer.inference import ExtractorSpec
from mermaid_classifier.pyspacer.label_specs import (
    HoldoutGroups,
    ImageExclusionFilter,
    LabelFilter,
    LabelRollupSpec,
)
from mermaid_classifier.pyspacer.options import (
    Artifacts,
    DatasetOptions,
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


def _load_csv_spec[T: CsvSpec](path: str | None, factory: Callable[[TextIO], T]) -> T:
    """A `CsvSpec` read from `path`, or the same type built from an empty CSV.

    An unset path is the caller's "accept everything" default, not a file to
    complain about missing.
    """
    if path:
        with open(path) as csv_f:
            return factory(csv_f)
    return factory(StringIO(""))


class TrainingDataset:
    # Set by _dedupe_points; merged into train_summary_stats.
    _dedupe_stats: dict[str, int] | None = None
    # Set by _log_split_drops; classes the train/ref/val split removed.
    _split_dropped_bagfs: int = 0
    # Only __init__ creates a temp feature dir; cleanup() tolerates its absence.
    _feature_temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def __init__(self, options: DatasetOptions):
        self._setup_annotation_reading(options)
        # Populated by _apply_subsample if subsampling is enabled; stays
        # None otherwise. The runner's MLflow logging path checks for
        # None so the no-subsample path stays artifact-free.
        self._subsample_audit_df: pd.DataFrame | None = None
        self._subsample_realized_total: int | None = None
        # Set only by source_spread (per class x group) and by an applied
        # holdout cap; None otherwise.
        self._subsample_group_audit_df: pd.DataFrame | None = None
        self._holdout_audit_df: pd.DataFrame | None = None
        if settings.feature_cache_dir:
            os.makedirs(settings.feature_cache_dir, exist_ok=True)
            self._feature_dir = settings.feature_cache_dir
            self._feature_temp_dir = None
        else:
            self._feature_temp_dir = tempfile.TemporaryDirectory(prefix="mermaid_features_")
            self._feature_dir = self._feature_temp_dir.name
        # Maps a downloaded feature vector's local path (used as the key of
        # the filesystem DataLocations in self.labels) back to its original
        # (bucket, feature_vector) S3 location. add_training_set_names() needs
        # this to match the labels to the annotations table.
        self._feature_path_to_s3_location: dict[str, tuple[str, str]] = {}

        # Set once the feature sources are resolved; runner.py seals it
        # into the exported manifest.
        self.extractor_spec: ExtractorSpec | None = None

        self.feature_loc_to_source: dict[DataLocation, tuple[str, str]] = {}

        # Evaluation-only rows; filled by prep_annotations_for_pyspacer.
        self.holdout_labels = ImageLabels()

        self.load_filtered_annotations()

        if self.options.holdout_groups_csv:
            # Test groups leave a CV run entirely: neither trained on nor
            # evaluated. Holdout rows stay, tagged, for later steps.
            self.duck_conn.execute("DELETE FROM annotations WHERE is_excluded")

        if options.include_mermaid or options.coralnet_manifest_uri:
            feature_prefixes = self.feature_source_prefixes()

            # Before the listing below, which is the slow part: a run whose
            # feature sources disagree about their extractor is not worth
            # enumerating.
            with self.section_profiling("Resolving feature-extractor provenance"):
                self.extractor_spec = self.resolve_extractor_spec(feature_prefixes)

            # We'll check the annotation data's feature paths against the
            # feature vectors that are actually present in S3, for every
            # site that has annotations.

            with self.section_profiling("Detecting missing feature vectors"):
                # First, list the feature paths present in S3 for each site
                # (this can take a while). The listings return `bucket/key`
                # strings, which match the `feature_full` we build per row.
                present: set[str] = set()
                for prefix in feature_prefixes:
                    present |= set(self.s3.find(path=prefix))
                # Check against annotation data.
                self.handle_missing_feature_vectors(present)

        if options.subsample is not None:
            # Runs after the missing-feature-vector drop above, so per-class
            # targets are computed from rows that can actually reach training.
            with self.section_profiling("Per-class subsampling"):
                self._apply_subsample(options.subsample)

        if options.holdout_max_per_class is not None:
            seed = options.subsample.seed if options.subsample is not None else 0
            with self.section_profiling("Capping holdout rows per class"):
                self._cap_holdout_rows(options.holdout_max_per_class, seed)

        self.labels = self.prep_annotations_for_pyspacer()

        with self.section_profiling("Tag DuckDB rows with training set"):
            self.add_training_set_names()

        self.set_train_summary_stats()

    @classmethod
    def filtered_annotations_only(cls, options: DatasetOptions) -> Self:
        """
        Build only the filtered `annotations` table (read, rollup, filter,
        dedupe, holdout-tag), with no feature-vector listing, download or
        extractor resolution. The returned dataset is not trainable.
        """
        dataset = cls.__new__(cls)
        dataset._setup_annotation_reading(options)  # pyright: ignore[reportPrivateUsage]  # classmethod initialising its own instance
        dataset.load_filtered_annotations()
        return dataset

    def _setup_annotation_reading(self, options: DatasetOptions) -> None:
        """Set the state `load_filtered_annotations` needs: specs, S3, DuckDB."""
        self.options = options
        self.artifacts = Artifacts()
        self.profiled_sections = []

        # CoralNet data is defined by a manifest parquet (None disables it).
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

        self.holdout_groups = _load_csv_spec(options.holdout_groups_csv, HoldoutGroups)
        # Fail on a bad fold before any data is read.
        self.holdout_groups.validate_holdout_fold(options.holdout_fold)
        # Without the lookup every MERMAID row is project 'all', so no
        # listed MERMAID project can match.
        mermaid_groups = sorted(
            group_id
            for site, group_id in self.holdout_groups.group_folds
            if site == Sites.MERMAID.value
        )
        if options.include_mermaid and mermaid_groups and not options.mermaid_project_lookup_uri:
            raise ValueError(
                f"holdout_groups_csv lists MERMAID group(s) {mermaid_groups}, but"
                f" mermaid_project_lookup_uri is unset, so no MERMAID row has a project."
            )

        # https://s3fs.readthedocs.io/en/latest/api.html#s3fs.core.S3FileSystem
        self.s3 = S3FileSystem(
            anon=settings.aws_anonymous == "True",
            key=settings.aws_key_id,
            secret=settings.aws_secret,
            token=settings.aws_session_token,
        )
        self._duck_conn = None

    def load_filtered_annotations(self) -> None:
        """
        Populate the `annotations` table: read the CoralNet and MERMAID
        annotations, roll up and filter labels, drop excluded images,
        dedupe points, and tag holdout groups (is_holdout / is_excluded).

        Needs no feature-vector listing or download.
        """
        if self.options.coralnet_manifest_uri:
            with self.section_profiling("Reading CoralNet annotations"):
                self.read_coralnet_manifest()
        else:
            # An empty dataframe
            self.artifacts.coralnet_project_stats = pd.DataFrame()

        if self.options.include_mermaid:
            with self.section_profiling("Reading MERMAID annotations"):
                self.read_mermaid_data()
        else:
            self.artifacts.mermaid_project_stats = pd.DataFrame()

        # Now this should have annotations populated.
        if not self.duckdb_annotations_table_exists():
            raise ValueError(
                "No annotations from CoralNet or MERMAID, even before label filtering."
            )

        def _annotations_stats() -> tuple[int, int]:
            # (annotation rows, distinct feature_vector i.e. unique images)
            return self.duck_conn.execute(  # pyright: ignore[reportReturnType]  # duckdb fetchone returns tuple[Any, ...] | None but we know it's always (int, int)
                "SELECT COUNT(*), COUNT(DISTINCT feature_vector) FROM annotations"
            ).fetchone()

        # Lets holdout tagging tell a mistyped group from one filtering emptied.
        groups_read: set[tuple[str, str]] = set()
        if not self.holdout_groups.is_empty():
            groups_read = {
                (str(site), str(project_id))
                for site, project_id in self.duck_conn.execute(
                    "SELECT DISTINCT site, project_id FROM annotations"
                ).fetchall()
            }

        with self.section_profiling("Rollups and filtering"):
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

        with self.section_profiling("Deduplicating points"):
            self._dedupe_points()

        with self.section_profiling("Tagging holdout groups"):
            self.holdout_groups.tag_in_duckdb(
                duck_conn=self.duck_conn,
                duck_table_name="annotations",
                holdout_fold=self.options.holdout_fold,
                groups_read=groups_read,
            )

    def _dedupe_points(self) -> None:
        """
        Make `annotations` hold at most one row per point, keyed by
        (bucket, feature_vector, row, col).

        Runs after rollup and filtering, so conflicts are judged on the
        final BA+GF. A point whose rows carry more than one distinct
        (benthic_attribute_id, growth_form_id) is dropped entirely. Any
        other point keeps one row: the first when ordered by every column,
        so the choice does not depend on scan order.
        """
        point_key = "bucket, feature_vector, row, col"
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
        columns = self.duck_conn.execute("DESCRIBE annotations").fetchall()
        return any(c[0] == "is_holdout" for c in columns)

    def _apply_subsample(self, opts: SubsampleOptions) -> None:
        """Deterministic per-class subsampling of the annotations table.

        Replaces the old non-deterministic ``annotation_limit`` block.
        Runs after rollup + included-labels filter, before the
        train/ref/val split. Only rows with ``is_holdout = FALSE`` take
        part: per-class targets come from their counts, and holdout rows
        pass through untouched (``_cap_holdout_rows`` bounds those).

        Pipeline (all in DuckDB, in-process):

          1. Read per-class counts of the non-holdout annotations,
             keyed by ``(benthic_attribute_id, growth_form_id)``. Both
             columns are post-rollup at this point.
          2. Call ``compute_per_class_targets(opts, counts)`` to get
             per-class target row counts. Strategy is dispatched by
             ``opts.strategy`` (today: 'stratified' or 'balanced'; new
             strategies plug in via ``training/subsample/registry.py``).
          3. Select rows per class, by ``opts.within_class``:
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
             and stash realized per-class counts for the MLflow audit.

        Extension hooks:
          * Stratify at a coarser level by changing the PARTITION BY
            columns and surfacing a ``stratification_level`` field on
            ``SubsampleOptions``.
          * Bootstrap oversampling for rare classes would be a
            UNION-ALL pass over the surviving rows; gate on a new
            ``opts.oversample`` field.
        """
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

        # Step 3: register the allocation as a DuckDB DataFrame and JOIN it
        # against a deterministic ROW_NUMBER. We pass the dataframe via
        # the connection's variable namespace (``df`` -> SQL view).
        group_available: dict[tuple[str, str, str, str], int] = {}
        group_quotas: dict[tuple[str, str, str, str], int] = {}
        if opts.within_class == "source_spread":
            group_rows = self.duck_conn.execute(
                "SELECT benthic_attribute_id, growth_form_id, site, project_id, COUNT(*)"
                f" FROM {candidates}"
                " GROUP BY ALL"
                " ORDER BY ALL"
            ).fetchall()
            per_class_groups: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
            for ba, gf, site, project_id, n in group_rows:
                group_available[(ba, gf, site, project_id)] = n
                per_class_groups.setdefault((ba, gf), {})[(site, project_id)] = n
            for (ba, gf), groups in per_class_groups.items():
                for (site, project_id), quota in water_fill(
                    groups, targets.get((ba, gf), 0)
                ).items():
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
            partition_by = "benthic_attribute_id, growth_form_id, site, project_id"
            # md5 of the seed-salted feature vector clusters a group's rows
            # by image, and is the same in every DuckDB version, which
            # hash() is not; feature_vector breaks digest ties.
            order_by = f"md5('{int(opts.seed)}' || feature_vector), feature_vector, row, col"
            join_using = "benthic_attribute_id, growth_form_id, site, project_id"
        else:
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
            partition_by = "benthic_attribute_id, growth_form_id"
            # The (site, project_id, image_id, row, col) tuple is unique
            # per annotation in this schema (CoralNet and MERMAID both
            # produce point-level rows keyed this way), giving a fully
            # deterministic row order across processes.
            order_by = "site, project_id, image_id, row, col"
            join_using = "benthic_attribute_id, growth_form_id"

        self.duck_conn.register("_subsample_targets", allocation_df)
        try:
            holdout_union = " UNION ALL BY NAME SELECT * FROM annotations WHERE is_holdout"
            self.duck_conn.execute(
                "CREATE OR REPLACE TABLE annotations AS"
                " WITH numbered AS ("
                "   SELECT *,"
                f"          ROW_NUMBER() OVER ("
                f"              PARTITION BY {partition_by}"
                f"              ORDER BY {order_by}"
                "          ) AS _rn"
                f"   FROM {candidates}"
                " )"
                " SELECT n.* EXCLUDE (_rn)"
                " FROM numbered n"
                f" JOIN _subsample_targets t USING ({join_using})"
                " WHERE n._rn <= t.target_n" + (holdout_union if has_holdout else "")
            )
        finally:
            self.duck_conn.unregister("_subsample_targets")

        # Step 4: log realized per-class counts. This is the after-the-
        # fact audit trail: lets us confirm in MLflow that two parallel
        # runs really did see the same data.
        realized_rows = self.duck_conn.execute(
            "SELECT benthic_attribute_id, growth_form_id, COUNT(*) AS n"
            f" FROM {candidates}"
            " GROUP BY benthic_attribute_id, growth_form_id"
            " ORDER BY benthic_attribute_id, growth_form_id"
        ).fetchall()
        realized_by_cls = {(ba, gf): n for ba, gf, n in realized_rows}

        per_class_audit = pd.DataFrame(
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
        # Stash on self; logged to MLflow by the runner alongside the
        # other dataset artifacts.
        self._subsample_audit_df = per_class_audit

        if opts.within_class == "source_spread":
            realized_group_rows = self.duck_conn.execute(
                "SELECT benthic_attribute_id, growth_form_id, site, project_id, COUNT(*)"
                f" FROM {candidates}"
                " GROUP BY ALL"
            ).fetchall()
            realized_by_group = {
                (ba, gf, site, pid): n for ba, gf, site, pid, n in realized_group_rows
            }
            self._subsample_group_audit_df = pd.DataFrame(
                [
                    {
                        "benthic_attribute_id": ba,
                        "growth_form_id": gf,
                        "site": site,
                        "project_id": project_id,
                        "available": group_available[(ba, gf, site, project_id)],
                        "quota": group_quotas[(ba, gf, site, project_id)],
                        "realized": realized_by_group.get((ba, gf, site, project_id), 0),
                    }
                    for (ba, gf, site, project_id) in sorted(group_available)
                ]
            )

        realized_total = int(per_class_audit["realized_n"].sum())
        self._subsample_realized_total = realized_total
        logger.info(
            f"Subsample applied: strategy={opts.strategy!r},"
            f" within_class={opts.within_class!r},"
            f" classes={len(class_counts)},"
            f" target_total={opts.total_annotations},"
            f" realized_total={realized_total}"
        )

    def _cap_holdout_rows(self, max_per_class: int, seed: int = 0) -> None:
        """Cap each class's holdout rows at ``max_per_class``.

        Keeps whole images first: within a class (not per group), rows are
        ordered by a seeded md5 of the feature vector, so the survivors
        cluster by image. Non-holdout rows are untouched. A no-op, leaving
        ``_holdout_audit_df`` None, when there are no holdout rows.
        """
        self._holdout_audit_df = None
        if not self._has_holdout_column():
            return

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
            f"              ORDER BY md5('{int(seed)}' || feature_vector),"
            "                       feature_vector, site, project_id, image_id, row, col"
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

    def read_mermaid_data(self):
        parquet_path = settings.mermaid_annotations_parquet_pattern.format(
            mermaid_train_data_bucket=settings.mermaid_train_data_bucket,
        )

        if self.duckdb_annotations_table_exists():
            # INSERT INTO ... BY NAME ... means that we don't have to match
            # the column order of the existing table.
            # https://duckdb.org/docs/stable/sql/statements/insert#insert-into--by-name
            query_start = "INSERT INTO annotations BY NAME"
        else:
            # Didn't read any CoralNet data, so we have to create
            # the annotations table.
            query_start = "CREATE TABLE annotations AS"
        lookup_uri = self.options.mermaid_project_lookup_uri
        if lookup_uri:
            self._check_project_lookup_is_one_to_one(lookup_uri)
            # One project per image (checked above), so the join cannot
            # fan out rows; a lookup repeating an image/project pair is fine.
            project_sql = "COALESCE(l.project_id, 'unknown')"
            from_sql = (
                f"read_parquet('{parquet_path}') m"
                f" LEFT JOIN ("
                f"  SELECT CAST(image_id AS VARCHAR) AS image_id,"
                f"   min(CAST(project_id AS VARCHAR)) AS project_id"
                f"  FROM read_parquet(?) GROUP BY 1) l"
                f" ON CAST(m.image_id AS VARCHAR) = l.image_id"
            )
            params = [lookup_uri]
        else:
            project_sql = "'all'"
            from_sql = f"read_parquet('{parquet_path}') m"
            params = []
        self.duck_conn.execute(
            query_start + f" SELECT"
            f"  CAST(m.image_id AS VARCHAR) AS image_id, m.row, m.col,"
            f"  m.benthic_attribute_id,"
            f"  COALESCE(m.growth_form_id, '') AS growth_form_id,"
            f" '{Sites.MERMAID.value}' AS site,"
            f" '{settings.mermaid_train_data_bucket}' AS bucket,"
            f" {project_sql} AS project_id,"
            f"  concat('mermaid/', m.image_id, '_featurevector')"
            f"   AS feature_vector"
            f" FROM {from_sql}",
            params,
        )
        if lookup_uri:
            self._check_unmatched_mermaid_images()

        # Get project-level stats before applying any further filters.
        self.artifacts.mermaid_project_stats = self.compute_project_stats(
            site=Sites.MERMAID.value, has_training_sets=False
        )

        # growth_form_id at this point is a real UUID, '' (for absent growth form),
        # or the legacy literal string 'None'. Normalize 'None' to ''.
        def transform_func(gf_id: str | None) -> str | None:
            if gf_id == "None":
                return ""
            return gf_id

        duckdb_transform_column(
            duck_conn=self.duck_conn,
            duck_table_name="annotations",
            column_name="growth_form_id",
            transform_func=transform_func,
        )

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

    def read_coralnet_manifest(self):
        if self.duckdb_annotations_table_exists():
            # Since CoralNet's data is not formatted to the open data
            # bucket's specs yet, it's easier to read in CN data first,
            # then transform the table to open data format, then read
            # in MERMAID data.
            #
            # As opposed to reading MERMAID first, transforming to CN
            # format, reading in CN, and transforming back to open data
            # format.
            raise RuntimeError(
                "Due to format technicalities, CoralNet data must be read in before MERMAID data."
            )

        manifest_uri = self.options.coralnet_manifest_uri

        # Read the manifest parquet into the `annotations` table, normalizing
        # to the open data bucket column layout (row, col, image_id, label_id,
        # site, bucket, project_id, feature_vector). The manifest is a
        # per-annotation-point dataset; label mapping, rollups, and filtering
        # happen downstream below, exactly as for the old per-source CSV path.
        try:
            self.duck_conn.execute(
                "CREATE TABLE annotations AS"
                " SELECT"
                "  row,"
                "  col,"
                "  CAST(image_id AS VARCHAR) AS image_id,"
                "  CAST(coralnet_id AS VARCHAR) AS label_id,"
                f"  '{Sites.CORALNET.value}' AS site,"
                f"  '{settings.coralnet_train_data_bucket}' AS bucket,"
                "  CAST(source_id AS VARCHAR) AS project_id,"
                # e.g. s123/features/i456.featurevector
                "  's' || CAST(source_id AS VARCHAR) || '/features/i' || CAST(image_id AS VARCHAR)"
                "   || '.featurevector' AS feature_vector"
                " FROM read_parquet(?)"
                # Defense-in-depth: the inner join in build_manifest_relation
                # already excludes null image_id, but guard here in case the
                # parquet was produced by another path.
                " WHERE image_id IS NOT NULL AND image_id <> ''",
                [manifest_uri],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read CoralNet manifest parquet at '{manifest_uri}'. "
                f"It must be readable and contain columns: "
                f"source_id, image_id, row, col, coralnet_id. "
                f"Underlying error: {exc}"
            ) from exc

        self.coralnet_source_ids = [
            str(r[0])
            for r in self.duck_conn.execute(
                "SELECT DISTINCT project_id FROM annotations ORDER BY CAST(project_id AS INTEGER)"
            ).fetchall()
        ]

        # Get project-level stats before applying any further filters.
        self.artifacts.coralnet_project_stats = self.compute_project_stats(
            site=Sites.CORALNET.value, has_training_sets=False
        )

        label_mapping = CoralNetMermaidMapping()
        self.artifacts.coralnet_label_mapping = label_mapping.get_dataframe()

        # Add BAs and GFs to the DuckDB table, using
        # our mapping from CoralNet label IDs to MERMAID BAs/GFs.
        def label_to_ba(label: str | None) -> str | None:
            if label is None or label not in label_mapping:
                return None
            entry = label_mapping[label]
            return entry.benthic_attribute_id

        duckdb_add_column(
            duck_conn=self.duck_conn,
            duck_table_name="annotations",
            base_column_name="label_id",
            new_column_name="benthic_attribute_id",
            base_to_new_func=label_to_ba,
        )

        def label_to_gf(label: str | None) -> str | None:
            if label is None or label not in label_mapping:
                return None
            entry = label_mapping[label]
            return entry.growth_form_id

        duckdb_add_column(
            duck_conn=self.duck_conn,
            duck_table_name="annotations",
            base_column_name="label_id",
            new_column_name="growth_form_id",
            base_to_new_func=label_to_gf,
        )

        # The BA IS NULL rows are the ones that aren't mapped to
        # anything in MERMAID. Save stats on these rows.
        #
        # TODO: It could be nice to include label names in this artifact.
        #  That would require reading in some data which maps
        #  CoralNet label IDs to names.
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

        # Then filter out those rows before moving on.
        self.duck_conn.execute("DELETE FROM annotations WHERE benthic_attribute_id IS NULL")

    def duckdb_annotations_table_exists(self) -> bool:
        # https://duckdb.org/docs/stable/sql/meta/information_schema
        table_query_result = self.duck_conn.execute(
            "SELECT * FROM information_schema.tables WHERE table_name = 'annotations'"
        ).fetchall()

        # Result should be [] if doesn't exist, which 'bools' to False.
        return bool(table_query_result)

    def feature_source_prefixes(self) -> list[str]:
        """The S3 prefixes this run's `.featurevector` files are read from."""
        prefixes: list[str] = []
        if self.options.include_mermaid:
            prefixes.append(f"s3://{settings.mermaid_train_data_bucket}/mermaid/")
        if self.options.coralnet_manifest_uri:
            prefixes.append(f"s3://{settings.coralnet_train_data_bucket}/")
        return prefixes

    def read_extractor_sidecar(self, prefix: str) -> ExtractorSpec | None:
        """The spec the job that wrote `prefix` left there, if it left one.

        None for a prefix we read but do not produce -- CoralNet's public
        bucket -- which the run's own declaration has to cover instead.
        """
        key = prefix + SIDECAR_FILENAME
        if not self.s3.exists(key):
            return None
        return ExtractorSpec.from_dict(json.loads(self.s3.cat_file(key)))

    def resolve_extractor_spec(self, prefixes: list[str]) -> ExtractorSpec:
        """One extractor for the whole run, or refuse to train.

        A head fitted across two feature spaces is wrong in a way no later
        gate detects: the artifact loads, the parity gate passes, and every
        score is drawn from a distribution the head never saw. So sources
        that disagree stop the run here.

        `dataset.feature_extractor_weights` is checked against every sidecar
        rather than merely filling gaps, which is what catches a config
        naming weights the bucket was not built with.
        """
        declared = (
            extractor_spec_from_weights(self.options.feature_extractor_weights)
            if self.options.feature_extractor_weights
            else None
        )

        found: dict[str, ExtractorSpec] = {}
        for prefix in prefixes:
            sidecar = self.read_extractor_sidecar(prefix)
            if sidecar is not None:
                if sidecar.weights_sha256 == EMPTY_SHA256:
                    raise ValueError(
                        f"{prefix}{SIDECAR_FILENAME} predates the #112 fix: its"
                        " weights_sha256 is the hash of an empty file, so the hash is"
                        " unrecorded. The feature vectors are not necessarily wrong."
                        " Confirm which weights produced them, then rewrite the sidecar"
                        " with the correct weights_sha256."
                    )
                found[prefix] = sidecar
            elif declared is not None:
                found[prefix] = declared
            else:
                raise ValueError(
                    f"{prefix} carries no {SIDECAR_FILENAME}, and this run declares no"
                    " dataset.feature_extractor_weights, so nothing records which"
                    " extractor produced its feature vectors. Set that field in the"
                    " training config."
                )
        if declared is not None:
            found["dataset.feature_extractor_weights"] = declared

        if len({spec.identity() for spec in found.values()}) > 1:
            detail = "\n".join(f"  {src}: {spec.describe()}" for src, spec in found.items())
            raise ValueError(
                "Feature sources disagree about which extractor produced them, so"
                f" their vectors are not comparable:\n{detail}"
            )

        # Prefer a producer's own record over the run's declaration: both
        # name the same bytes, but the sidecar names where they were used.
        spec = next(
            (found[prefix] for prefix in prefixes if prefix in found),
            declared,
        )
        assert spec is not None  # every prefix either resolved or raised above
        logger.info("Feature extractor for this run: %s", spec.describe())
        return spec

    def handle_missing_feature_vectors(self, present_feature_paths: set[str]) -> None:
        # Check every site's annotation feature paths against the set of
        # feature vectors actually present in S3. Annotations whose feature
        # vector is absent are dropped (and the run aborts if too many are
        # missing). `present_feature_paths` holds `bucket/key` strings, the
        # same shape as the `feature_full` we build per annotation row.

        # Build the annotation data's full feature paths, in DuckDB.
        self.duck_conn.execute(
            "CREATE OR REPLACE TABLE annotations AS"
            " SELECT *,"
            "  bucket || '/' || feature_vector AS feature_full"
            " FROM annotations"
        )

        with (
            duckdb_temp_table_name(self.duck_conn) as s3_features_table_name,
            duckdb_temp_table_name(self.duck_conn) as missing_features_table_name,
        ):
            # Get the S3 feature paths into another table.
            s3_paths_df = pd.DataFrame({"feature_full": list(present_feature_paths)})  # noqa: F841  # pyright: ignore[reportUnusedVariable]  # referenced by name in DuckDB SQL via Python-scope scanning
            self.duck_conn.execute(
                f"CREATE TEMP TABLE {s3_features_table_name} AS SELECT * FROM s3_paths_df"
            )

            in_annotations_count = self.duck_conn.execute(
                "SELECT COUNT(DISTINCT feature_full) FROM annotations"
            ).fetchall()[0][0]

            # Get the annotation feature paths that are missing from S3,
            # into another table.
            self.duck_conn.execute(
                f"CREATE TABLE {missing_features_table_name} AS"
                f" SELECT DISTINCT a.feature_full FROM annotations a"
                f" LEFT JOIN {s3_features_table_name} s USING (feature_full)"
                # Annotations whose features are not found in S3.
                f" WHERE s.feature_full IS NULL"
            )

            missing_count = self.duck_conn.execute(
                f"SELECT COUNT(*) FROM {missing_features_table_name}"
            ).fetchall()[0][0]

            result_tuples = self.duck_conn.execute(
                f"SELECT feature_full FROM {missing_features_table_name} LIMIT 3"
            ).fetchall()
            missing_examples = [tup[0] for tup in result_tuples]

            # Abort if too many are missing (check before mutating).
            examples_str = "\n".join(missing_examples)
            missing_threshold = (
                in_annotations_count * settings.training_inputs_percent_missing_allowed / 100
            )
            if missing_count > missing_threshold:
                raise RuntimeError(
                    f"Too many feature vectors are missing"
                    f" ({missing_count}), such as:"
                    f"\n{examples_str}"
                    f"\nYou can configure the tolerance for missing"
                    f" feature vectors with the"
                    f" TRAINING_INPUTS_PERCENT_MISSING_ALLOWED setting."
                )

            # Filter out missing feature vectors from annotations table.
            self.duck_conn.execute(
                f"CREATE OR REPLACE TABLE annotations AS"
                f" SELECT a.* FROM annotations a"
                f" LEFT JOIN {s3_features_table_name} s USING (feature_full)"
                # Keep annotations whose features are found in S3.
                f" WHERE s.feature_full IS NOT NULL"
            )

        # Don't need the feature_full column anymore.
        self.duck_conn.execute("ALTER TABLE annotations DROP feature_full")

        # Log a warning if any are missing.
        if missing_count > 0:
            logger.warning(
                f"Skipping {missing_count} feature vector(s) because"
                f" the files aren't in S3."
                f" Example(s):"
                f"\n{examples_str}"
            )

    def prep_annotations_for_pyspacer(self) -> TrainingTaskLabels:
        """
        Download the feature vectors and split the annotations into
        train/ref/val by `options.split_mode`.

        Holdout rows (is_holdout) take no part in the split: they become
        `self.holdout_labels`, which is empty when there are none.
        """
        has_holdout = self._has_holdout_column()
        with self.section_profiling("Collecting feature paths"):
            annotations_by_image = duckdb_grouped_rows(
                duck_conn=self.duck_conn,
                duck_table_name="annotations",
                grouping_column_names=["bucket", "feature_vector"],
                extra_order_column_names=["row", "col"],
            )

            # First pass: collect annotations and unique S3 keys.
            s3_keys: dict[tuple[str, str], str] = {}
            tmp_root = self._feature_dir
            # (bucket, key, site, project_id, split annotations, holdout annotations)
            image_data: list[tuple[str, str, str, str, list[Annotation], list[Annotation]]] = []

            for rows in annotations_by_image:
                # Here, in one loop iteration, we're given all the
                # annotation rows for a single image.
                first_row = rows[0]
                bucket = str(first_row["bucket"])
                feature_bucket_path = str(first_row["feature_vector"])
                site = str(first_row["site"])
                project_id = str(first_row["project_id"])

                image_annotations: list[Annotation] = []
                holdout_annotations: list[Annotation] = []

                # One annotation per row.
                for row in rows:
                    bagf = combine_ba_gf(
                        str(row["benthic_attribute_id"]), str(row["growth_form_id"])
                    )

                    annotation = (
                        int(row["row"]),
                        int(row["col"]),
                        bagf,
                    )
                    if has_holdout and bool(row["is_holdout"]):
                        holdout_annotations.append(annotation)
                    else:
                        image_annotations.append(annotation)

                s3_key = (bucket, feature_bucket_path)
                if s3_key not in s3_keys:
                    local_path = os.path.join(tmp_root, bucket, feature_bucket_path)
                    s3_keys[s3_key] = local_path
                    # Remember how to get back from the local download path
                    # to the original S3 location, for add_training_set_names.
                    self._feature_path_to_s3_location[local_path] = s3_key

                image_data.append(
                    (
                        bucket,
                        feature_bucket_path,
                        site,
                        project_id,
                        image_annotations,
                        holdout_annotations,
                    )
                )

        # Parallel download of all feature vectors from S3.
        with self.section_profiling("Downloading feature vectors"):
            failed_keys = download_features_parallel(
                s3_keys, max_workers=settings.download_max_workers
            )

        if failed_keys:
            # The downloader's per-key warning lands on a different logger
            # than train.log; naming the keys here reaches a post-mortem.
            logger.warning(
                f"{len(failed_keys)} feature vector download(s) failed: {sorted(failed_keys)}"
            )

        with self.section_profiling("Building PySpacer labels"):
            # Build ImageLabels with filesystem DataLocations.
            labels_data = ImageLabels()
            holdout_data = ImageLabels()

            for (
                bucket,
                feature_bucket_path,
                site,
                project_id,
                image_annotations,
                holdout_annotations,
            ) in image_data:
                # Skip images whose feature file failed to download.
                if (bucket, feature_bucket_path) in failed_keys:
                    continue

                local_path = s3_keys[(bucket, feature_bucket_path)]

                feature_loc = DataLocation(
                    storage_type="filesystem",
                    key=local_path,
                )
                if image_annotations:
                    labels_data.add_image(feature_loc, image_annotations)
                if holdout_annotations:
                    holdout_data.add_image(feature_loc, holdout_annotations)
                self.feature_loc_to_source[feature_loc] = (site, project_id)

            self.holdout_labels = holdout_data

        with self.section_profiling(f"Splitting train/ref/val ({self.options.split_mode})"):
            if self.options.split_mode == "images":
                subsample = self.options.subsample
                split = preprocess_labels(
                    split_by_image(
                        labels_data,
                        self.options.ref_val_ratios,
                        seed=subsample.seed if subsample is not None else 0,
                        # The download dir differs per run; the S3 location does not.
                        stable_key=lambda loc: "/".join(self._feature_path_to_s3_location[loc.key]),
                    )
                )
            else:
                split = preprocess_labels(
                    labels_data,
                    split_ratios=self.options.ref_val_ratios,
                    split_mode=SplitMode.POINTS_STRATIFIED,
                )
        self._log_split_drops(labels_data, split)
        return split

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
        Return a DuckDB connection which includes the ability to read
        files from S3.

        Each TrainingDataset instance only establishes such a connection
        once.

        Limitation: each TrainingDataset can only read from a single S3
        region.
        """
        if self._duck_conn is None:
            self._duck_conn = duckdb.connect()

            # Load the DuckDB extension which allows reading remote files
            # from S3.
            # https://duckdb.org/docs/stable/core_extensions/httpfs/overview
            try:
                self._duck_conn.load_extension("httpfs")
            except duckdb.IOException:
                # Extension not installed yet.
                self._duck_conn.install_extension("httpfs")
                self._duck_conn.load_extension("httpfs")

            # Configure region and auth, if present.
            # https://duckdb.org/docs/stable/core_extensions/httpfs/s3api
            #
            # Beware not to use the deprecated S3 API, which has syntax like
            # `SET s3_region = 'us-east-1'`:
            # https://duckdb.org/docs/stable/core_extensions/httpfs/s3api_legacy_authentication

            if settings.aws_anonymous == "False":
                if settings.aws_key_id:
                    # Manual provision of a key.
                    query = (
                        f"CREATE OR REPLACE SECRET secret ("
                        f" TYPE s3,"
                        f" PROVIDER config,"
                        f" KEY_ID '{settings.aws_key_id}',"
                        f" SECRET '{settings.aws_secret}',"
                    )
                    if settings.aws_session_token:
                        query += f" SESSION_TOKEN '{settings.aws_session_token}',"
                else:
                    # The credential_chain provider allows automatically
                    # fetching AWS credentials, like through the IMDS.
                    query = "CREATE OR REPLACE SECRET secret ( TYPE s3, PROVIDER credential_chain,"
                query += f" REGION '{settings.aws_region}')"

                self._duck_conn.execute(query)

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
        Match up DuckDB annotations with train/ref/val.
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
