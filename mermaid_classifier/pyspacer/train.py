"""
Train a classifier using feature vectors and annotations
provided on S3.
"""
from contextlib import contextmanager
import dataclasses
from datetime import datetime, timedelta
import enum
from io import StringIO
import math
from pathlib import Path
import re
import time
import typing

import duckdb
import matplotlib.pyplot as plt
try:
    import mlflow
    MLFLOW_IMPORT_ERROR = None
except ImportError as err:
    MLFLOW_IMPORT_ERROR = err
import numpy as np
import pandas as pd
import psutil
from s3fs.core import S3FileSystem
import sklearn
from spacer.data_classes import DataLocation, ImageLabels, ValResults
from spacer.messages import TrainClassifierMsg
from spacer.storage import load_classifier
from spacer.tasks import train_classifier
from spacer.task_utils import preprocess_labels, SplitMode

from mermaid_classifier.common.benthic_attributes import (
    BAGF_SEP,
    BenthicAttributeLibrary,
    combine_ba_gf,
    CoralNetMermaidMapping,
    GrowthFormLibrary,
    split_ba_gf,
)
from mermaid_classifier.common.csv_utils import ColumnSpec, CsvSpec
from mermaid_classifier.common.duckdb_utils import (
    duckdb_add_column,
    duckdb_filter_on_column,
    duckdb_grouped_rows,
    duckdb_replace_column,
    duckdb_temp_table_name,
    duckdb_transform_column,
)
from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.pyspacer.utils import (
    logging_config_for_script, mlflow_connect)


logger = logging_config_for_script('train')


ba_library = BenthicAttributeLibrary()
gf_library = GrowthFormLibrary()


class Sites(enum.Enum):
    CORALNET = 'coralnet'
    MERMAID = 'mermaid'


@contextmanager
def section_profiling(profiled_sections: list[dict], section_name: str):
    """
    Performance-profile a wrapped section of code and save the stats
    (time, memory) as part of the passed structure.
    """
    approx_start_date = datetime.now()
    # This is more accurate, but doesn't have time-of-day info.
    start_time = time.perf_counter()

    yield

    seconds_elapsed = time.perf_counter() - start_time
    section_profile = dict(
        # Name for this section of code.
        name=section_name,
        # Number of seconds.
        seconds=format(seconds_elapsed, '.1f'),
        # Hours, minutes, seconds, ns.
        hms=str(timedelta(seconds=seconds_elapsed)),
        # Date and time, to see if the sections we've chosen skip any
        # substantial time blocks that we should also be monitoring.
        approx_start=approx_start_date.strftime('%b %d %H:%M:%S'),
        memory_usage_at_end=f'{psutil.virtual_memory().percent}%',
    )
    profiled_sections.append(section_profile)

    logger.debug(
        f"{section_name} -"
        f" Elapsed time = {section_profile['hms']},"
        f" Memory usage at end = {section_profile['memory_usage_at_end']}"
    )


class LabelFilter(CsvSpec):
    """
    A CSV-defined spec which says what benthic attribute + growth form
    combos to include in, or exclude from, training data.
    """
    column_specs = [
        ColumnSpec(name='ba_id', allow_blank=False),
        ColumnSpec(name='gf_id'),
    ]

    def __init__(self, csv_file: typing.TextIO, inclusion: bool = True):
        self.bagf_set: set[tuple[str, str]] = set()

        super().__init__(csv_file=csv_file)

        self.inclusion = inclusion

    def per_row_init_action(self, row):
        # Ensure absent values are just '', not '' or None.
        self.bagf_set.add((row['ba_id'], row.get('gf_id') or ''))

    def accepts_bagf(self, bagf_id: str):
        ba_id, gf_id = split_ba_gf(bagf_id)

        if self.inclusion:
            return (ba_id, gf_id) in self.bagf_set
        else:
            return (ba_id, gf_id) not in self.bagf_set

    def filter_in_duckdb(
        self,
        duck_conn: duckdb.DuckDBPyConnection,
        duck_table_name: str,
        ba_id_column_name: str = 'benthic_attribute_id',
        gf_id_column_name: str = 'growth_form_id',
    ):
        """
        Filter down the rows in the given DuckDB table, based on the
        benthic attribute ID and growth form ID columns, and this
        instance's filter rules.
        """
        # Concatenate BA+GF so that we can define the filter as a
        # single-column operation.
        # https://duckdb.org/docs/stable/sql/functions/text#concat_wsseparator-string-
        duck_conn.execute(
            f"CREATE OR REPLACE TABLE {duck_table_name} AS"
            f" SELECT"
            f"  *,"
            f"  concat_ws("
            f"   '{BAGF_SEP}', {ba_id_column_name}, {gf_id_column_name})"
            f"   AS bagf_id"
            f" FROM {duck_table_name}"
        )

        # Filter.
        duckdb_filter_on_column(
            duck_conn=duck_conn,
            duck_table_name=duck_table_name,
            column_name='bagf_id',
            inclusion_func=self.accepts_bagf,
        )

        # Don't need the combined BAGF column anymore.
        duck_conn.execute(
            f"ALTER TABLE {duck_table_name} DROP bagf_id"
        )


class LabelRollupSpec(CsvSpec):
    """
    A CSV-defined spec which says what BA+GF combos to roll up to
    what other BA+GF combos.
    """
    column_specs = [
        ColumnSpec(name='from_ba_id', allow_blank=False),
        ColumnSpec(name='from_gf_id'),
        ColumnSpec(name='to_ba_id', allow_blank=False),
        ColumnSpec(name='to_gf_id'),
    ]

    def __init__(self, *args, **kwargs):
        self.lookup = dict()

        super().__init__(*args, **kwargs)

    def per_row_init_action(self, row):
        # Ensure absent values are just '', not '' or None.
        key = (row['from_ba_id'], row.get('from_gf_id') or '')
        value = (row['to_ba_id'], row.get('to_gf_id') or '')
        self.lookup[key] = value

    def roll_up(self, bagf_id: str) -> str:
        ba_id, gf_id = split_ba_gf(bagf_id)

        if (ba_id, gf_id) in self.lookup:
            new_ba_id, new_gf_id = self.lookup[(ba_id, gf_id)]
            return combine_ba_gf(new_ba_id, new_gf_id)
        # If this BAGF is not in the rollup spec, then we leave the
        # BAGF as is.
        return bagf_id

    def roll_up_in_duckdb(
        self,
        duck_conn: duckdb.DuckDBPyConnection,
        duck_table_name: str,
        ba_id_column_name: str = 'benthic_attribute_id',
        gf_id_column_name: str = 'growth_form_id',
    ):
        """
        Roll up the BA IDs and GF IDs in the given DuckDB table,
        based on this instance's rollup rules.
        """
        # Concatenate BA+GF so that we can define the rollup as a
        # single-column transform.
        # https://duckdb.org/docs/stable/sql/functions/text#concat_wsseparator-string-
        # If there's no GF, then the result is the BA plus separator.
        duck_conn.execute(
            f"CREATE OR REPLACE TABLE {duck_table_name} AS"
            f" SELECT"
            f"  *,"
            f"  concat_ws("
            f"   '{BAGF_SEP}', {ba_id_column_name}, {gf_id_column_name})"
            f"   AS bagf_id"
            f" FROM {duck_table_name}"
        )

        # Apply the rollup.
        duckdb_transform_column(
            duck_conn=duck_conn,
            duck_table_name=duck_table_name,
            column_name='bagf_id',
            transform_func=self.roll_up,
        )

        # Propagate rolled-up BAGF back to the split BA-GF fields.
        # https://duckdb.org/docs/stable/sql/functions/text#split_partstring-separator-index
        duck_conn.execute(
            f"CREATE OR REPLACE TABLE {duck_table_name} AS"
            f" SELECT"
            f"  *,"
            f"  bagf_id.split_part('{BAGF_SEP}', 1)"
            f"   AS rollup_{ba_id_column_name},"
            f"  bagf_id.split_part('{BAGF_SEP}', 2)"
            f"   AS rollup_{gf_id_column_name}"
            f" FROM {duck_table_name}"
        )
        duckdb_replace_column(
            duck_conn=duck_conn,
            duck_table_name=duck_table_name,
            column_name=ba_id_column_name,
            new_values_column_name=f"rollup_{ba_id_column_name}",
        )
        duckdb_replace_column(
            duck_conn=duck_conn,
            duck_table_name=duck_table_name,
            column_name=gf_id_column_name,
            new_values_column_name=f"rollup_{gf_id_column_name}",
        )

        # Don't need the combined BAGF column anymore.
        duck_conn.execute(
            f"ALTER TABLE {duck_table_name} DROP bagf_id"
        )


class CNSourceFilter(CsvSpec):

    column_specs = [
        ColumnSpec(name='id', allow_blank=False),
    ]

    source_id_list: list[str]

    def __init__(self, csv_file: typing.TextIO):
        """
        Initialize using a CSV file that specifies a set
        of CoralNet sources.
        """
        self.source_id_list = []

        super().__init__(csv_file=csv_file)

    def per_row_init_action(self, row):
        self.source_id_list.append(row['id'])

    def is_empty(self):
        return len(self.source_id_list) == 0


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

    drop_growthforms

    If True, discard all growth forms from the training data. Technically
    redundant with the rollup spec, but this is a very simple-to-specify
    option which can be useful.

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
    label_rollup_spec_csv: str = None
    included_labels_csv: str = None
    excluded_labels_csv: str = None
    drop_growthforms: bool = False
    ref_val_ratios: tuple[float, float] = (0.1, 0.1)
    annotation_limit: int | None = None


@dataclasses.dataclass
class TrainingOptions:
    """
    epochs

    Number of pyspacer training epochs to run.
    """
    epochs: int = 10


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


class TrainingDataset:

    def __init__(self, options: DatasetOptions):

        self.options = options
        self.artifacts = Artifacts()
        self.profiled_sections = []

        if options.coralnet_sources_csv:
            with open(options.coralnet_sources_csv) as csv_f:
                self.cn_source_filter = CNSourceFilter(csv_f)
        else:
            # Empty set of CoralNet sources.
            self.cn_source_filter = CNSourceFilter(StringIO(''))

        if options.label_rollup_spec_csv:
            with open(options.label_rollup_spec_csv) as csv_f:
                self.rollup_spec = LabelRollupSpec(csv_f)
        else:
            # Empty rollup-targets set, meaning nothing gets rolled up.
            self.rollup_spec = LabelRollupSpec(StringIO(''))

        if options.included_labels_csv and options.excluded_labels_csv:
            raise ValueError(
                "Specify one of included labels or"
                " excluded labels, but not both.")

        if options.included_labels_csv:
            with open(options.included_labels_csv) as csv_f:
                self.label_filter = LabelFilter(
                    csv_f, inclusion=True)
        elif options.excluded_labels_csv:
            with open(options.excluded_labels_csv) as csv_f:
                self.label_filter = LabelFilter(
                    csv_f, inclusion=False)
        else:
            # No inclusion or exclusion set specified means we accept
            # all labels.
            # In other words, an empty exclusion set.
            self.label_filter = LabelFilter(
                StringIO(''), inclusion=False)

        # https://s3fs.readthedocs.io/en/latest/api.html#s3fs.core.S3FileSystem
        self.s3 = S3FileSystem(
            anon=settings.aws_anonymous,
            key=settings.aws_key_id,
            secret=settings.aws_secret,
            token=settings.aws_session_token,
        )
        self._duck_conn = None

        if not self.cn_source_filter.is_empty():
            with self.section_profiling("Reading CoralNet annotations"):
                self.read_coralnet_data()
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
                "No annotations from CoralNet or MERMAID, even before"
                " label filtering.")

        with self.section_profiling("Rollups and filtering"):

            # Roll up BAGFs.
            self.rollup_spec.roll_up_in_duckdb(
                duck_conn=self.duck_conn,
                duck_table_name='annotations',
            )

            if options.drop_growthforms:
                # Clear all annotations' growth forms.
                duckdb_transform_column(
                    duck_conn=self.duck_conn,
                    duck_table_name='annotations',
                    column_name='growth_form_id',
                    transform_func=lambda x: '',
                )

            # Filter out BAGFs we don't want.
            self.label_filter.filter_in_duckdb(
                duck_conn=self.duck_conn,
                duck_table_name='annotations',
            )

        if options.annotation_limit:

            # See how many annotations we have.
            count_up_to_limit = self.duck_conn.execute(
                f"SELECT count(*)"
                f" FROM annotations"
                f" LIMIT {options.annotation_limit + 1}"
            ).fetchall()[0][0]

            # If we're over limit, log that fact.
            if count_up_to_limit > options.annotation_limit:
                logger.debug(
                    f"Truncating the train data to"
                    f" {options.annotation_limit} annotations."
                )

            # Determine how to balance out the annotations between
            # sites.
            # Ideally, each site contributes half the limit; but if one
            # site's total is less than half the limit, we grab more from
            # the other site.
            cn_count_up_to_limit = self.duck_conn.execute(
                f"SELECT count(*)"
                f" FROM annotations"
                f" WHERE site = '{Sites.CORALNET.value}'"
                f" LIMIT {options.annotation_limit + 1}"
            ).fetchall()[0][0]
            mm_count_up_to_limit = self.duck_conn.execute(
                f"SELECT count(*)"
                f" FROM annotations"
                f" WHERE site = '{Sites.MERMAID.value}'"
                f" LIMIT {options.annotation_limit + 1}"
            ).fetchall()[0][0]
            half_limit = math.ceil(options.annotation_limit / 2)
            if cn_count_up_to_limit <= half_limit:
                cn_count = cn_count_up_to_limit
                mm_count = options.annotation_limit - cn_count
            else:
                mm_count = min(mm_count_up_to_limit, half_limit)
                cn_count = options.annotation_limit - mm_count

            # Get the appropriate number of annotations from each site.
            # https://duckdb.org/docs/stable/sql/query_syntax/setops#union
            self.duck_conn.execute(
                f"CREATE OR REPLACE TABLE annotations AS ("
                f" (SELECT * FROM annotations"
                f"  WHERE site = '{Sites.CORALNET.value}'"
                f"  LIMIT {cn_count})"
                f" UNION ALL"
                f" (SELECT * FROM annotations"
                f"  WHERE site = '{Sites.MERMAID.value}'"
                f"  LIMIT {mm_count})"
                f")"
            )

        if options.include_mermaid:
            # We'll check the annotation data's image IDs against the
            # feature vectors that are actually present in S3.
            # TODO: Also check CoralNet bucket/folder for missing features.

            with self.section_profiling("Detecting missing feature vectors"):
                # First, get the paths present in S3 (this can take a while).
                mermaid_bucket = settings.mermaid_train_data_bucket
                mermaid_full_paths_in_s3 = set(
                    self.s3.find(
                        path=f's3://{mermaid_bucket}/mermaid/'
                    )
                )
                # Check against annotation data.
                self.handle_missing_feature_vectors(mermaid_full_paths_in_s3)

        with self.section_profiling("Prep annotations for PySpacer"):
            self.labels = self.prep_annotations_for_pyspacer()
            self.add_training_set_names()

        self.set_train_summary_stats()

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
            query_start = f"INSERT INTO annotations BY NAME"
        else:
            # Didn't read any CoralNet data, so we have to create
            # the annotations table.
            query_start = f"CREATE TABLE annotations AS"
        self.duck_conn.execute(
            query_start +
            f" SELECT"
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
            site=Sites.MERMAID.value, has_training_sets=False)

        # For growth forms, we get '' from the CoralNet-MERMAID
        # mapping, but the string 'None' from the MERMAID annotations
        # parquet.
        # Normalize the latter to ''.
        def transform_func(gf_id):
            if gf_id == 'None':
                return ''
            return gf_id
        duckdb_transform_column(
            duck_conn=self.duck_conn,
            duck_table_name='annotations',
            column_name='growth_form_id',
            transform_func=transform_func,
        )

    def read_coralnet_data(self):
        annotations_uri_list = []

        for source_id in self.cn_source_filter.source_id_list:

            # One row per point annotation,
            # with columns including Image ID, Row, Column, Label ID
            annotations_uri = settings.coralnet_annotations_csv_pattern.format(
                coralnet_train_data_bucket=settings.coralnet_train_data_bucket,
                source_id=source_id,
            )
            annotations_uri_list.append(annotations_uri)

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
                "Due to format technicalities, CoralNet data must be read"
                " in before MERMAID data.")

        # Read each selected source's annotations-CSV into a single
        # DuckDB table.
        #
        # `CREATE TABLE ... AS SELECT ...` is from:
        # https://duckdb.org/docs/stable/data/csv/overview
        #
        # Passing a list of CSV files into read_csv() is from:
        # https://duckdb.org/docs/stable/data/multiple_files/overview
        #
        # CSV options:
        # https://duckdb.org/docs/stable/data/csv/overview#parameters
        self.duck_conn.execute(
            f"CREATE TABLE annotations AS"
            f" SELECT *"
            f" FROM read_csv({annotations_uri_list}, filename = true)"
        )

        # 1) Normalize the column names with the open data bucket parquet
        #  format, and
        # 2) Create some new columns derived from the existing ones, based
        #  on how the CoralNet data is organized. We want to do this while
        #  still in DuckDB format, because we'd like to have data-proc
        #  steps after a certain point to not have site-specific cases.
        #
        # CREATE OR REPLACE TABLE:
        # https://duckdb.org/2024/10/11/duckdb-tricks-part-2#repeated-data-transformation-steps
        # regexp_extract and || (concat):
        # https://duckdb.org/docs/stable/sql/functions/text
        # Also potentially helpful:
        # https://betterstack.com/community/guides/scaling-python/duckdb-python/
        self.duck_conn.execute(
            f"CREATE OR REPLACE TABLE annotations AS"
            f" SELECT"
            # Fix the case of this column name to match open data bucket
            # format.
            f' "Row" AS row,'
            # Make this column name match open data bucket format, and use
            # quotes to avoid keyword clashing. Be sure to wrap in double
            # quotes. Single quotes would get it interpreted as
            # "give every row this constant string value".
            f' "Column" AS col,'
            # Later we find it easier to assume these are text, not integers.
            # And quotes help again, this time since the name has spaces in it.
            f' CAST("Image ID" AS VARCHAR) AS image_id,'
            f' CAST("Label ID" AS VARCHAR) AS label_id,'
            # Here we DO want to give every row this constant string value.
            f" '{Sites.CORALNET.value}' AS site,"
            f" '{settings.coralnet_train_data_bucket}' AS bucket,"
            # Extract the source ID, and name it 'project_id' as a
            # site-inspecific term.
            rf" filename.regexp_extract('/s(\d+)/', 1) AS project_id,"
            # e.g. s123/features/i456.featurevector
            f" 's' || project_id || '/features/i' || image_id"
            f"  || '.featurevector' AS feature_vector"
            f" FROM annotations"
        )

        # Get project-level stats before applying any further filters.
        self.artifacts.coralnet_project_stats = self.compute_project_stats(
            site=Sites.CORALNET.value, has_training_sets=False)

        label_mapping = CoralNetMermaidMapping()
        self.artifacts.coralnet_label_mapping = label_mapping.get_dataframe()

        # Add BAs and GFs to the DuckDB table, using
        # our mapping from CoralNet label IDs to MERMAID BAs/GFs.
        def label_to_ba(label):
            if label not in label_mapping:
                return None
            entry = label_mapping[label]
            return entry.benthic_attribute_id
        duckdb_add_column(
            duck_conn=self.duck_conn,
            duck_table_name='annotations',
            base_column_name='label_id',
            new_column_name='benthic_attribute_id',
            base_to_new_func=label_to_ba,
        )
        def label_to_gf(label):
            if label not in label_mapping:
                return None
            entry = label_mapping[label]
            return entry.growth_form_id
        duckdb_add_column(
            duck_conn=self.duck_conn,
            duck_table_name='annotations',
            base_column_name='label_id',
            new_column_name='growth_form_id',
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
        self.duck_conn.execute(
            "DELETE FROM annotations WHERE benthic_attribute_id IS NULL"
        )

    def duckdb_annotations_table_exists(self) -> bool:
        # https://duckdb.org/docs/stable/sql/meta/information_schema
        table_query_result = self.duck_conn.execute(
            f"SELECT * FROM information_schema.tables"
            f" WHERE table_name = 'annotations'"
        ).fetchall()

        # Result should be [] if doesn't exist, which 'bools' to False.
        return bool(table_query_result)

    def handle_missing_feature_vectors(self, mermaid_full_paths_in_s3: set):

        # Build the annotation data's full feature paths, in DuckDB.
        self.duck_conn.execute(
            f"CREATE OR REPLACE TABLE annotations AS"
            f" SELECT *,"
            f"  bucket || '/' || feature_vector AS feature_full"
            f" FROM annotations"
        )

        with (
            duckdb_temp_table_name(self.duck_conn)
            as anno_features_table_name,
            duckdb_temp_table_name(self.duck_conn)
            as s3_features_table_name,
            duckdb_temp_table_name(self.duck_conn)
            as missing_features_table_name,
        ):

            # Get the annotation data's unique feature paths into a table.
            self.duck_conn.execute(
                f"CREATE TABLE {anno_features_table_name} AS"
                f" SELECT DISTINCT feature_full FROM annotations"
                f" WHERE site = '{Sites.MERMAID.value}'"
            )

            # Get the S3 feature paths into another table.
            s3_paths_df = pd.DataFrame(
                {'feature_full': list(mermaid_full_paths_in_s3)})
            self.duck_conn.execute(
                f"CREATE TEMP TABLE {s3_features_table_name}"
                f" AS SELECT * FROM s3_paths_df"
            )

            in_annotations_count = self.duck_conn.execute(
                f"SELECT COUNT(DISTINCT feature_full) FROM annotations"
                f" WHERE site = '{Sites.MERMAID.value}'"
            ).fetchall()[0][0]

            # Get the annotation feature paths that are missing from S3,
            # into another table.
            self.duck_conn.execute(
                f"CREATE TABLE {missing_features_table_name} AS"
                f" SELECT DISTINCT a.feature_full FROM annotations a"
                f" LEFT JOIN {s3_features_table_name} s USING (feature_full)"
                # MERMAID annotations whose features are not found in S3.
                f" WHERE a.site = '{Sites.MERMAID.value}'"
                f"  AND s.feature_full IS NULL"
            )

            missing_count = self.duck_conn.execute(
                f"SELECT COUNT(*) FROM {missing_features_table_name}"
            ).fetchall()[0][0]

            result_tuples = self.duck_conn.execute(
                f"SELECT feature_full FROM {missing_features_table_name}"
                f" LIMIT 3"
            ).fetchall()
            missing_examples = [tup[0] for tup in result_tuples]

            # Filter out missing feature vectors from annotations table.
            self.duck_conn.execute(
                f"CREATE OR REPLACE TABLE annotations AS"
                f" SELECT a.* FROM annotations a"
                f" LEFT JOIN {s3_features_table_name} s USING (feature_full)"
                # Keep all non-MERMAID annotations (since we don't yet detect
                # which of those have features present).
                f" WHERE a.site != '{Sites.MERMAID.value}'"
                # Keep MERMAID annotations whose features are found in S3.
                f"  OR s.feature_full IS NOT NULL"
            )

        # Don't need the feature_full column anymore.
        self.duck_conn.execute(
            f"ALTER TABLE annotations DROP feature_full"
        )

        # Abort if too many are missing.
        examples_str = "\n".join(missing_examples)
        missing_threshold = (
            in_annotations_count
            * settings.training_inputs_percent_missing_allowed / 100
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

        # Log a warning if any are missing.
        if missing_count > 0:
            logger.warning(
                f"Skipping {missing_count} feature vector(s) because"
                f" the files aren't in S3."
                f" Example(s):"
                f"\n{examples_str}")

    def prep_annotations_for_pyspacer(self):

        annotations_by_image = duckdb_grouped_rows(
            duck_conn=self.duck_conn,
            duck_table_name='annotations',
            grouping_column_names=['bucket', 'feature_vector'],
        )

        labels_data = ImageLabels()

        for rows in annotations_by_image:

            # Here, in one loop iteration, we're given all the
            # annotation rows for a single image.
            first_row = rows[0]
            bucket = first_row['bucket']
            feature_bucket_path = first_row['feature_vector']

            image_annotations = []

            # One annotation per row.
            for row in rows:

                bagf = combine_ba_gf(
                    row['benthic_attribute_id'], row['growth_form_id'])

                annotation = (
                    int(row['row']),
                    int(row['col']),
                    bagf,
                )
                image_annotations.append(annotation)

            feature_loc = DataLocation(
                storage_type='s3',
                bucket_name=bucket,
                key=feature_bucket_path,
            )
            labels_data.add_image(feature_loc, image_annotations)

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
                self._duck_conn.load_extension('httpfs')
            except duckdb.IOException:
                # Extension not installed yet.
                self._duck_conn.install_extension('httpfs')
                self._duck_conn.load_extension('httpfs')

            # Configure region and auth, if present.
            # https://duckdb.org/docs/stable/core_extensions/httpfs/s3api
            #
            # Beware not to use the deprecated S3 API, which has syntax like
            # `SET s3_region = 'us-east-1'`:
            # https://duckdb.org/docs/stable/core_extensions/httpfs/s3api_legacy_authentication

            if settings.aws_anonymous == 'False':
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
                        query += (
                            f" SESSION_TOKEN '{settings.aws_session_token}',"
                        )
                else:
                    # The credential_chain provider allows automatically
                    # fetching AWS credentials, like through the IMDS.
                    query = (
                        f"CREATE OR REPLACE SECRET secret ("
                        f" TYPE s3,"
                        f" PROVIDER credential_chain,"
                    )
                query += f" REGION '{settings.aws_region}')"

                self._duck_conn.execute(query)

        return self._duck_conn

    def compute_project_stats(self, site=None, has_training_sets=False):
        if site is None:
            where_clause = ""
        else:
            where_clause = f"WHERE site = '{site}'"

        counts_sql = (
            " count(DISTINCT image_id) AS num_images,"
            " count(*) AS num_annotations"
        )
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
            ('train', self.labels.train),
            ('ref', self.labels.ref),
            ('val', self.labels.val),
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
                values_batch: list[tuple] = []

                for feature_loc, row, col in (
                    self.generate_training_set_annotations(training_set)
                ):
                    tup = (
                        feature_loc.bucket_name,
                        feature_loc.key,
                        row,
                        col,
                        set_name,
                    )
                    values_batch.append(tup)

                    if len(values_batch) > batch_size:
                        self._add_tuples_to_table(
                            temp_table_name, values_batch)
                        values_batch = []

                if len(values_batch) > 0:
                    # Last batch
                    self._add_tuples_to_table(
                        temp_table_name, values_batch)

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

    def _add_tuples_to_table(self, table_name, tuples: list[tuple]):
        df = pd.DataFrame.from_records(tuples)
        self.duck_conn.execute(
            f"INSERT INTO {table_name} SELECT * FROM df"
        )

    @staticmethod
    def generate_training_set_annotations(training_set: ImageLabels):
        for feature_loc in training_set.keys():
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
        # Add BA names alongside the IDs for readability.
        duckdb_add_column(
            duck_conn=self.duck_conn,
            duck_table_name='ba_counts',
            base_column_name='benthic_attribute_id',
            new_column_name='benthic_attribute_name',
            base_to_new_func=ba_library.id_to_name,
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
            duck_table_name='bagf_counts',
            base_column_name='benthic_attribute_id',
            new_column_name='benthic_attribute_name',
            base_to_new_func=ba_library.id_to_name,
        )
        # Add GF names for readability.
        duckdb_add_column(
            duck_conn=self.duck_conn,
            duck_table_name='bagf_counts',
            base_column_name='growth_form_id',
            new_column_name='growth_form_name',
            base_to_new_func=gf_library.id_to_name,
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
            "SELECT"
            " count(*),"
            " count(DISTINCT image_id)"
            " FROM annotations"
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

        self.artifacts.train_summary_stats = dict(
            annotations=total_annotations,
            annotations_train=self.labels.train.label_count,
            annotations_ref=self.labels.ref.label_count,
            annotations_val=self.labels.val.label_count,
            annotations_dropped=annotations_dropped,
            images=num_of_images,
            bas=num_of_bas,
            bas_dropped=bas_dropped,
            bagfs=num_of_bagfs,
            bagfs_dropped=bagfs_dropped,
        )

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
                **self.artifacts.train_summary_stats)
        )

    def get_annotations(self, log_spec: str):

        if log_spec == 'all':
            query = "SELECT * FROM annotations"
        elif match := re.fullmatch(r's(\d+)', log_spec):
            cn_source_id = match.groups()[0]
            query = (
                f"SELECT * FROM annotations"
                f" WHERE site = '{Sites.CORALNET.value}'"
                f" AND project_id = '{cn_source_id}'"
            )
        elif match := re.fullmatch(r'i(\d+)', log_spec):
            cn_image_id = match.groups()[0]
            query = (
                f"SELECT * FROM annotations"
                f" WHERE site = '{Sites.CORALNET.value}'"
                f" AND image_id = '{cn_image_id}'"
            )
        else:
            raise ValueError(
                f"Unsupported annotations log spec: {log_spec}")

        return self.duck_conn.execute(query).fetch_df()


class TrainingRunner:
    """
    Base runner class.

    This class can be used as-is for training, although it won't save
    any results. Still, it doesn't have any MLflow dependency, so it
    can be used to make testing easier when running a
    tracking server feels onerous.
    It could also be extended to support tracking software other than
    MLflow.
    """
    dataset: TrainingDataset = None
    profiled_sections: list[dict]

    def __init__(
        self,
        dataset_options: DatasetOptions = None,
        training_options: TrainingOptions = None,
    ):
        self.dataset_options = dataset_options or DatasetOptions()
        self.training_options = training_options or TrainingOptions()

    def run(self, run_name: str | None = None):
        if run_name is None:
            run_name = self.current_time_str()
        logger.info(f"Run: {run_name}")

        self.dataset = TrainingDataset(self.dataset_options)

        # The dataset's profiled sections are done. The runner will add the
        # remaining profiled sections.
        self.profiled_sections = self.dataset.profiled_sections.copy()

        # Log dataset artifacts now, so they can be inspected during
        # training.
        with self.section_profiling("Logging dataset artifacts"):
            self.log_dataset_artifacts()

        logger.info("Proceeding to train with:")
        logger.info(self.dataset.describe_train_summary_stats())

        # Not sure about saving these anywhere other than memory
        # for now.
        model_loc = DataLocation('memory', key='classifier.pkl')
        valresult_loc = DataLocation('memory', key='valresult.json')

        train_msg = TrainClassifierMsg(
            job_token=f'experiment_run_{run_name}',
            trainer_name='minibatch',
            nbr_epochs=self.training_options.epochs,
            clf_type='MLP',
            labels=self.dataset.labels,
            previous_model_locs=[],
            model_loc=model_loc,
            valresult_loc=valresult_loc,
            feature_cache_dir=TrainClassifierMsg.FeatureCache.AUTO,
        )

        with self.section_profiling("PySpacer training call"):
            return_msg = train_classifier(train_msg)

        logger.info(
            f"Train time (from return msg): {return_msg.runtime:.1f} s")

        logger.info(
            f"New model's accuracy: {self.format_accuracy(return_msg.acc)}")

        ref_accs_str = ", ".join(
            [self.format_accuracy(acc) for acc in return_msg.ref_accs])
        logger.debug(
            f"Accuracy progression during training epochs: {ref_accs_str}")

        return return_msg, model_loc, valresult_loc

    def log_dataset_artifacts(self):
        """
        This base runner doesn't have anywhere to log artifacts to.
        Subclasses should override as appropriate.
        """
        pass

    @contextmanager
    def section_profiling(self, section_name: str):
        with section_profiling(self.profiled_sections, section_name):
            yield

    @staticmethod
    def current_time_str():
        current_time = datetime.now()
        return current_time.strftime('%Y%m%dT%H%M%S')

    @staticmethod
    def format_accuracy(accuracy):
        return f'{100*accuracy:.1f}%'


class MLflowTrainingRunner(TrainingRunner):

    def __init__(
        self,
        *args,
        mlflow_options: MLflowOptions = None,
        **kwargs
    ):
        if MLFLOW_IMPORT_ERROR:
            # MLflow couldn't be imported.
            raise MLFLOW_IMPORT_ERROR

        time_taken = mlflow_connect()
        logger.info(f"Time to connect to MLflow tracking: {time_taken}")

        super().__init__(*args, **kwargs)
        self.mlflow_options = mlflow_options or MLflowOptions()

    def run(self, run_name=None):

        model_name = self._get_model_name()
        if run_name is None:
            run_name = f'{model_name}-{self.current_time_str()}'

        logger.info(f"Experiment: {self.mlflow_options.experiment_name}")
        mlflow.set_experiment(self.mlflow_options.experiment_name)

        with mlflow.start_run(run_name=run_name):

            training_options_to_log = dict(epochs=self.training_options.epochs)

            mlflow.log_params(training_options_to_log)

            # Here's the actual training and data prep.
            return_msg, model_loc, valresult_loc = super().run(
                run_name=run_name)

            profiles_df = pd.DataFrame(self.profiled_sections)
            mlflow.log_table(profiles_df, 'profiled_sections.json')

            # Note that log_metric() only takes numeric values.
            accuracy_pct = return_msg.acc * 100
            mlflow.log_metric("accuracy", accuracy_pct)

            # ref_accs is probably the only other part of return_msg to save.
            ref_accs_dict = dict()
            for epoch_number, acc in enumerate(return_msg.ref_accs, 1):
                ref_accs_dict[epoch_number] = self.format_accuracy(acc)
            mlflow.log_dict(ref_accs_dict, 'epoch_ref_accuracies.yaml')

            val_results = ValResults.load(valresult_loc)
            self.log_confusion_matrix(
                val_results=val_results,
                normalize=False,
                filestem='confusion_matrix/frequencies',
            )
            self.log_confusion_matrix(
                val_results=val_results,
                normalize=True,
                filestem='confusion_matrix/percents',
            )

            # Save and register the trained model.
            signature = mlflow.models.infer_signature(
                params=training_options_to_log)
            model_info = mlflow.sklearn.log_model(
                sk_model=load_classifier(model_loc),
                registered_model_name=model_name,
                signature=signature,
            )

        logger.info(f"Model ID: {model_info.model_id}")

        return return_msg, model_loc

    def _get_model_name(self):
        """
        Model name for MLflow logging purposes.

        Only alphanumeric chars and hyphens are allowed in MLflow model names.
        So we'll use hyphens as the 'outer' word separator, and CamelCaps as
        the 'inner' one.
        """
        if self.mlflow_options.model_name is not None:

            model_name = self.mlflow_options.model_name

        else:

            if self.dataset_options.included_labels_csv:
                as_path = Path(self.dataset_options.included_labels_csv)
                model_name = f'Include{self.alphanumeric_only_str(as_path.stem)}'
            elif self.dataset_options.excluded_labels_csv:
                as_path = Path(self.dataset_options.excluded_labels_csv)
                model_name = f'Exclude{self.alphanumeric_only_str(as_path.stem)}'
            else:
                model_name = 'AllLabels'

            if self.dataset_options.label_rollup_spec_csv:
                as_path = Path(self.dataset_options.label_rollup_spec_csv)
                model_name += f'-Rollup{self.alphanumeric_only_str(as_path.stem)}'

            if self.dataset_options.coralnet_sources_csv:
                as_path = Path(self.dataset_options.coralnet_sources_csv)
                model_name += f'-{self.alphanumeric_only_str(as_path.stem)}'

            if limit := self.dataset_options.annotation_limit:
                model_name += f'-AnnoLimit{limit}'

        # There's a 62 character limit for the 'model package group name'
        # which is built from the model name. For example, it could be the
        # model name with a suffix of -c78374. So we'll make the model name
        # under 62 minus 7 characters with some leeway, to be safe. If we
        # exceed the limit, then logging the model fails.
        return model_name[:50]

    @staticmethod
    def alphanumeric_only_str(s: str):
        """
        Return a version of s which has the non-alphanumeric chars removed.
        """
        return ''.join([char for char in s if char.isalnum()])

    def log_dataset_artifacts(self):
        """
        Log various options and stats for the training dataset.
        """
        assert self.dataset is not None

        artifacts = self.dataset.artifacts

        mlflow.log_table(
            self.dataset.cn_source_filter.csv_dataframe,
            'coralnet_sources_included.json')

        if self.dataset.label_filter.inclusion:
            table_filename = 'labels_included.json'
        else:
            table_filename = 'labels_excluded.json'
        mlflow.log_table(
            self.dataset.label_filter.csv_dataframe, table_filename)

        mlflow.log_table(
            self.dataset.rollup_spec.csv_dataframe, 'rollup_spec.json')

        # Number of images and annotations from each CN source and from
        # MERMAID.
        # First, before filtering (this is what's present in S3).
        # https://pandas.pydata.org/docs/reference/api/pandas.concat.html
        mlflow.log_table(
            pd.concat([
                artifacts.mermaid_project_stats,
                artifacts.coralnet_project_stats,
            ]),
            'project_stats_raw.json')
        # And here, after filtering (this is what training actually gets).
        mlflow.log_table(
            self.dataset.compute_project_stats(has_training_sets=True),
            'project_stats_train_data.json')

        mlflow.log_dict(
            artifacts.train_summary_stats, 'train_summary.yaml')

        mlflow.log_table(artifacts.ba_counts, 'ba_counts.json')
        mlflow.log_table(artifacts.bagf_counts, 'bagf_counts.json')

        if not self.dataset.cn_source_filter.is_empty():
            # These only apply if CoralNet data is included.
            mlflow.log_table(
                artifacts.coralnet_label_mapping,
                'coralnet_label_mapping.json')
            mlflow.log_table(
                artifacts.unmapped_labels,
                'unmapped_labels.json')

        # Log other options given to the training process.
        other_options = dict(
            drop_growthforms=self.dataset_options.drop_growthforms,
            annotation_limit=self.dataset_options.annotation_limit,
        )
        mlflow.log_dict(other_options, 'other_options.yaml')

        # Log annotations, if specified.
        if self.mlflow_options.annotations_to_log is not None:
            log_spec = self.mlflow_options.annotations_to_log.lower()
            df = self.dataset.get_annotations(log_spec)

            mlflow.log_table(df, f'annotations_{log_spec}.json')

    def log_confusion_matrix(
        self, val_results: ValResults, normalize: bool, filestem: str
    ):
        """
        Make a confusion matrix out of the training evaluation results.
        """
        matrix = sklearn.metrics.confusion_matrix(
            y_true=val_results.gt,
            y_pred=val_results.est,
            labels=range(len(val_results.classes)),
            # 'true': values between 0 and 1 for each cell.
            # None: Each cell has a frequency.
            normalize='true' if normalize else None,
        )

        if normalize:
            # 0-to-1 values -> integer percents.
            matrix = np.int64(np.floor(matrix * 100))

        # Sort by frequency.
        bagf_ids_in_freq_order = []
        # This artifact already has BA-GF combos sorted by frequency
        # in the whole dataset (which should be pretty much the same
        # order as frequency in val, due to stratification); highest
        # frequency first.
        for _, row in self.dataset.artifacts.bagf_counts.iterrows():
            bagf_id = combine_ba_gf(
                row['benthic_attribute_id'], row['growth_form_id'])
            if bagf_id not in val_results.classes:
                # This BA-GF combo must have gotten dropped entirely due
                # to not enough annotations.
                continue
            bagf_ids_in_freq_order.append(bagf_id)
        # For each ID in the frequency order, give it 0 if it appears 1st
        # in val_results.classes, 1 if it appears 2nd, 2 if 3rd, etc.
        class_indexes_of_freq_ranking = [
            val_results.classes.index(bagf_id)
            for bagf_id in bagf_ids_in_freq_order]
        # Order columns by frequency.
        matrix = matrix[:, class_indexes_of_freq_ranking]
        # Order rows by frequency.
        matrix = matrix[class_indexes_of_freq_ranking, :]

        bagf_names = [
            ba_library.bagf_id_to_name(bagf_id, gf_library)
            for bagf_id in bagf_ids_in_freq_order
        ]

        # Log the confusion matrix as a table.

        # To dataframe, labeling each column with a BA-GF combo.
        df = pd.DataFrame(data=matrix, columns=bagf_names)
        # Add column to label each row with a BA-GF combo.
        df.insert(loc=0, column='-', value=bagf_names)
        mlflow.log_table(df, filestem + '.json')

        # Log the confusion matrix as a figure.

        # Create square figure, with size scaled to number of labels
        num_labels = len(bagf_names)
        fig_size = max(12, num_labels * 0.6)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # Matplotlib visualization of the confusion matrix.
        display = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=bagf_names)
        display.plot(
            ax=ax,
            cmap='Blues',
            # Prevent "100" displaying as "1e+02".
            values_format='d',
            # A color legend feels unnecessary here.
            colorbar=False,
        )

        # Move x-axis labels to the top.
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        # Rotate x-axis tick labels to prevent their texts from overlapping.
        label_font_size = max(8, min(12, 150 / num_labels))
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha='left',
            rotation_mode='anchor',
            fontsize=label_font_size,
        )
        # Match y-axis labels' font size with the x axis.
        plt.setp(
            ax.get_yticklabels(),
            fontsize=label_font_size,
        )

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Log as figure
        mlflow.log_figure(fig, filestem + '.png')

        # Close figure to free memory
        plt.close(fig)
