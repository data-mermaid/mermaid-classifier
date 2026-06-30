"""
TrainingDataset: loads, transforms, and prepares annotation data for training.

Reads CoralNet per-source CSVs from S3 and MERMAID Parquet via DuckDB,
maps CoralNet label IDs to MERMAID BA+GF, applies rollup/filter specs,
validates feature vector availability, and produces pyspacer-ready
train/ref/val splits.
"""

import os
import re
import tempfile
from contextlib import contextmanager
from io import StringIO

import duckdb
import pandas as pd
from s3fs.core import S3FileSystem
from spacer.data_classes import DataLocation, ImageLabels
from spacer.task_utils import SplitMode, preprocess_labels

from mermaid_classifier.common.benthic_attributes import (
    CoralNetMermaidMapping,
    combine_ba_gf,
    get_benthic_attribute_library,
    get_growth_form_library,
)
from mermaid_classifier.common.duckdb_utils import (
    duckdb_add_column,
    duckdb_grouped_rows,
    duckdb_temp_table_name,
    duckdb_transform_column,
)
from mermaid_classifier.pyspacer._pipeline_utils import (
    download_features_parallel,
    section_profiling,
)
from mermaid_classifier.pyspacer.label_specs import (
    LabelFilter,
    LabelRollupSpec,
)
from mermaid_classifier.pyspacer.options import (
    Artifacts,
    DatasetOptions,
    Sites,
)
from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.pyspacer.utils import logging_config_for_script
from mermaid_classifier.training.subsample import (
    SubsampleOptions,
    compute_per_class_targets,
)

logger = logging_config_for_script("train")


class TrainingDataset:
    def __init__(self, options: DatasetOptions):

        self.options = options
        self.artifacts = Artifacts()
        self.profiled_sections = []
        # Populated by _apply_subsample if subsampling is enabled; stays
        # None otherwise. The runner's MLflow logging path checks for
        # None so the no-subsample path stays artifact-free.
        self._subsample_audit_df: pd.DataFrame | None = None
        self._subsample_realized_total: int | None = None
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

        # CoralNet data is defined by a manifest parquet (None disables it).
        self.coralnet_source_ids: list[str] = []

        if options.label_rollup_spec_csv:
            with open(options.label_rollup_spec_csv) as csv_f:
                self.rollup_spec = LabelRollupSpec(csv_f)
        else:
            # Empty rollup-targets set, meaning nothing gets rolled up.
            self.rollup_spec = LabelRollupSpec(StringIO(""))

        if options.included_labels_csv and options.excluded_labels_csv:
            raise ValueError("Specify one of included labels or excluded labels, but not both.")

        if options.included_labels_csv:
            with open(options.included_labels_csv) as csv_f:
                self.label_filter = LabelFilter(csv_f, inclusion=True)
        elif options.excluded_labels_csv:
            with open(options.excluded_labels_csv) as csv_f:
                self.label_filter = LabelFilter(csv_f, inclusion=False)
        else:
            # No inclusion or exclusion set specified means we accept
            # all labels.
            # In other words, an empty exclusion set.
            self.label_filter = LabelFilter(StringIO(""), inclusion=False)

        # https://s3fs.readthedocs.io/en/latest/api.html#s3fs.core.S3FileSystem
        self.s3 = S3FileSystem(
            anon=settings.aws_anonymous == "True",
            key=settings.aws_key_id,
            secret=settings.aws_secret,
            token=settings.aws_session_token,
        )
        self._duck_conn = None

        self.feature_loc_to_source: dict[DataLocation, tuple[str, str]] = {}

        if self.options.coralnet_manifest_uri:
            with self.section_profiling("Reading CoralNet annotations"):
                self.read_coralnet_manifest()
        else:
            # An empty dataframe
            self.artifacts.coralnet_project_stats = pd.DataFrame()

        if options.include_mermaid:
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

        with self.section_profiling("Rollups and filtering"):
            ann_before, img_before = _annotations_stats()
            logger.info(
                "Before rollups/filtering: %s annotations, %s unique images",
                f"{ann_before:,}",
                f"{img_before:,}",
            )

            if options.drop_growthforms:
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
            logger.info(
                "Rollups+filter retained %.1f%% of annotations, %.1f%% of unique images",
                100.0 * ann_after_filter / max(ann_before, 1),
                100.0 * img_after_filter / max(img_before, 1),
            )

        if options.subsample is not None:
            with self.section_profiling("Per-class subsampling"):
                self._apply_subsample(options.subsample)

        if options.include_mermaid or options.coralnet_manifest_uri:
            # We'll check the annotation data's feature paths against the
            # feature vectors that are actually present in S3, for every
            # site that has annotations.

            with self.section_profiling("Detecting missing feature vectors"):
                # First, list the feature paths present in S3 for each site
                # (this can take a while). The listings return `bucket/key`
                # strings, which match the `feature_full` we build per row.
                present: set[str] = set()
                if options.include_mermaid:
                    mermaid_bucket = settings.mermaid_train_data_bucket
                    present |= set(self.s3.find(path=f"s3://{mermaid_bucket}/mermaid/"))
                if options.coralnet_manifest_uri:
                    coralnet_bucket = settings.coralnet_train_data_bucket
                    present |= set(self.s3.find(path=f"s3://{coralnet_bucket}/"))
                # Check against annotation data.
                self.handle_missing_feature_vectors(present)

        self.labels = self.prep_annotations_for_pyspacer()

        with self.section_profiling("Tag DuckDB rows with training set"):
            self.add_training_set_names()

        self.set_train_summary_stats()

    def _apply_subsample(self, opts: SubsampleOptions) -> None:
        """Deterministic per-class subsampling of the annotations table.

        Replaces the old non-deterministic ``annotation_limit`` block.
        Runs after rollup + included-labels filter, before the
        train/ref/val split.

        Pipeline (all in DuckDB, in-process):

          1. Read per-class counts of the current annotations table,
             keyed by ``(benthic_attribute_id, growth_form_id)``. Both
             columns are post-rollup at this point.
          2. Call ``compute_per_class_targets(opts, counts)`` to get
             per-class target row counts. Strategy is dispatched by
             ``opts.strategy`` (today: 'stratified' or 'balanced'; new
             strategies plug in via ``training/subsample/registry.py``).
          3. Materialize the target dict as a temp table and JOIN it
             against a deterministic ROW_NUMBER() partitioned by class
             and ordered by ``(site, project_id, image_id, row, col)``
             -- the per-annotation primary key in this schema. That
             ordering makes the subsample identical across processes
             and across DuckDB thread counts.
          4. Replace the ``annotations`` table with the surviving rows
             and log realized per-class counts to MLflow as a CSV
             artifact for after-the-fact auditing.

        Extension hooks:
          * Stratify at a coarser level by changing the PARTITION BY
            columns and surfacing a ``stratification_level`` field on
            ``SubsampleOptions``.
          * Bootstrap oversampling for rare classes would be a
            UNION-ALL pass over the surviving rows; gate on a new
            ``opts.oversample`` field.
        """
        # Step 1: per-class counts (deterministic ORDER BY for stable
        # iteration order downstream).
        rows = self.duck_conn.execute(
            "SELECT benthic_attribute_id, growth_form_id, COUNT(*)"
            " FROM annotations"
            " GROUP BY benthic_attribute_id, growth_form_id"
            " ORDER BY benthic_attribute_id, growth_form_id"
        ).fetchall()
        class_counts: dict[tuple[str, str], int] = {(ba, gf): n for ba, gf, n in rows}

        if not class_counts:
            logger.warning("Subsampling skipped: annotations table is empty.")
            return

        # Step 2: dispatch to the strategy registry.
        targets = compute_per_class_targets(opts, class_counts)

        # Step 3: register targets as a DuckDB DataFrame and JOIN it
        # against a deterministic ROW_NUMBER. We pass the dataframe via
        # the connection's variable namespace (``df`` -> SQL view).
        targets_df = pd.DataFrame(
            [
                {
                    "benthic_attribute_id": ba,
                    "growth_form_id": gf,
                    "target_n": int(n),
                }
                for (ba, gf), n in targets.items()
            ]
        )
        self.duck_conn.register("_subsample_targets", targets_df)
        try:
            # The (site, project_id, image_id, row, col) tuple is
            # unique per annotation in this schema (CoralNet and
            # MERMAID both produce point-level rows keyed this way),
            # giving a fully deterministic row order across processes.
            self.duck_conn.execute(
                "CREATE OR REPLACE TABLE annotations AS"
                " WITH numbered AS ("
                "   SELECT *,"
                "          ROW_NUMBER() OVER ("
                "              PARTITION BY benthic_attribute_id,"
                "                           growth_form_id"
                "              ORDER BY site, project_id, image_id,"
                "                       row, col"
                "          ) AS _rn"
                "   FROM annotations"
                " )"
                " SELECT n.* EXCLUDE (_rn)"
                " FROM numbered n"
                " JOIN _subsample_targets t USING ("
                "     benthic_attribute_id, growth_form_id"
                " )"
                " WHERE n._rn <= t.target_n"
            )
        finally:
            self.duck_conn.unregister("_subsample_targets")

        # Step 4: log realized per-class counts. This is the after-the-
        # fact audit trail: lets us confirm in MLflow that two parallel
        # runs really did see the same data.
        realized_rows = self.duck_conn.execute(
            "SELECT benthic_attribute_id, growth_form_id, COUNT(*) AS n"
            " FROM annotations"
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
        realized_total = int(per_class_audit["realized_n"].sum())
        self._subsample_realized_total = realized_total
        logger.info(
            f"Subsample applied: strategy={opts.strategy!r},"
            f" classes={len(class_counts)},"
            f" target_total={opts.total_annotations},"
            f" realized_total={realized_total}"
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
        self.duck_conn.execute(
            query_start + f" SELECT"
            f"  image_id, row, col,"
            f"  benthic_attribute_id, growth_form_id,"
            f" '{Sites.MERMAID.value}' AS site,"
            f" '{settings.mermaid_train_data_bucket}' AS bucket,"
            f" 'all' AS project_id,"
            f"  concat('mermaid/', image_id, '_featurevector')"
            f"   AS feature_vector"
            f" FROM read_parquet('{parquet_path}')"
        )

        # Get project-level stats before applying any further filters.
        self.artifacts.mermaid_project_stats = self.compute_project_stats(
            site=Sites.MERMAID.value, has_training_sets=False
        )

        # For growth forms, we get '' from the CoralNet-MERMAID
        # mapping, but the string 'None' from the MERMAID annotations
        # parquet.
        # Normalize the latter to ''.
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

    def prep_annotations_for_pyspacer(self):

        with self.section_profiling("Collecting feature paths"):
            annotations_by_image = duckdb_grouped_rows(
                duck_conn=self.duck_conn,
                duck_table_name="annotations",
                grouping_column_names=["bucket", "feature_vector"],
            )

            # First pass: collect annotations and unique S3 keys.
            s3_keys: dict[tuple[str, str], str] = {}
            tmp_root = self._feature_dir
            # image_data: list of (bucket, key, annotations)
            image_data = []

            for rows in annotations_by_image:
                # Here, in one loop iteration, we're given all the
                # annotation rows for a single image.
                first_row = rows[0]
                bucket = str(first_row["bucket"])
                feature_bucket_path = str(first_row["feature_vector"])
                site = str(first_row["site"])
                project_id = str(first_row["project_id"])

                image_annotations = []

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
                    image_annotations.append(annotation)

                s3_key = (bucket, feature_bucket_path)
                if s3_key not in s3_keys:
                    local_path = os.path.join(tmp_root, bucket, feature_bucket_path)
                    s3_keys[s3_key] = local_path
                    # Remember how to get back from the local download path
                    # to the original S3 location, for add_training_set_names.
                    self._feature_path_to_s3_location[local_path] = s3_key

                image_data.append(
                    (bucket, feature_bucket_path, site, project_id, image_annotations)
                )

        # Parallel download of all feature vectors from S3.
        with self.section_profiling("Downloading feature vectors"):
            failed_keys = download_features_parallel(
                s3_keys, max_workers=settings.download_max_workers
            )

        if failed_keys:
            logger.warning(f"{len(failed_keys)} feature vector download(s) failed.")

        with self.section_profiling("Building PySpacer labels"):
            # Build ImageLabels with filesystem DataLocations.
            labels_data = ImageLabels()

            for bucket, feature_bucket_path, site, project_id, image_annotations in image_data:
                # Skip images whose feature file failed to download.
                if (bucket, feature_bucket_path) in failed_keys:
                    continue

                local_path = s3_keys[(bucket, feature_bucket_path)]

                feature_loc = DataLocation(
                    storage_type="filesystem",
                    key=local_path,
                )
                labels_data.add_image(feature_loc, image_annotations)
                self.feature_loc_to_source[feature_loc] = (site, project_id)

            return preprocess_labels(
                labels_data,
                split_ratios=self.options.ref_val_ratios,
                split_mode=SplitMode.POINTS_STRATIFIED,
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
        training_sets = [
            ("train", self.labels.train),
            ("ref", self.labels.ref),
            ("val", self.labels.val),
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

        # Stratified splitting will drop annotations of BAGFs which are rare
        # enough to not reach the 1 annotation threshold for ref/val.
        # Here we count what has been dropped.
        counts = self.duck_conn.execute(
            "SELECT"
            " count(*),"
            " count(DISTINCT benthic_attribute_id),"
            " count(DISTINCT (benthic_attribute_id, growth_form_id))"
            " FROM annotations"
            " WHERE training_set IS NOT NULL"
        ).fetchall()[0]
        non_dropped_annotations = counts[0]
        non_dropped_bas = counts[1]
        non_dropped_bagfs = counts[2]
        annotations_dropped = total_annotations - non_dropped_annotations
        bas_dropped = num_of_bas - non_dropped_bas
        bagfs_dropped = num_of_bagfs - non_dropped_bagfs

        self.artifacts.train_summary_stats = {
            "annotations": total_annotations,
            "annotations_train": self.labels.train.label_count,
            "annotations_ref": self.labels.ref.label_count,
            "annotations_val": self.labels.val.label_count,
            "annotations_dropped": annotations_dropped,
            "images": num_of_images,
            "bas": num_of_bas,
            "bas_dropped": bas_dropped,
            "bagfs": num_of_bagfs,
            "bagfs_dropped": bagfs_dropped,
        }

    def describe_train_summary_stats(self):
        return (
            "{annotations} annotations"
            " ({annotations_train} train,"
            " {annotations_ref} ref,"
            " {annotations_val} val,"
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
            query = "SELECT * FROM annotations"
        elif match := re.fullmatch(r"s(\d+)", log_spec):
            cn_source_id = match.groups()[0]
            query = (
                f"SELECT * FROM annotations"
                f" WHERE site = '{Sites.CORALNET.value}'"
                f" AND project_id = '{cn_source_id}'"
            )
        elif match := re.fullmatch(r"i(\d+)", log_spec):
            cn_image_id = match.groups()[0]
            query = (
                f"SELECT * FROM annotations"
                f" WHERE site = '{Sites.CORALNET.value}'"
                f" AND image_id = '{cn_image_id}'"
            )
        else:
            raise ValueError(f"Unsupported annotations log spec: {log_spec}")

        return self.duck_conn.execute(query).fetch_df()
