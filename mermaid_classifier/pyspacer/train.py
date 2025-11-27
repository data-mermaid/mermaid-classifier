"""
Train a classifier using feature vectors and annotations
provided on S3.
"""
from datetime import datetime
import enum
from io import StringIO
import itertools
import math
from pathlib import Path
import typing

import duckdb
try:
    import mlflow
    from mlflow.models import infer_signature
    MLFLOW_IMPORT_ERROR = None
except ImportError as err:
    MLFLOW_IMPORT_ERROR = err
import pandas as pd
import psutil
from s3fs.core import S3FileSystem
from spacer.data_classes import DataLocation, ImageLabels
from spacer.messages import TrainClassifierMsg
from spacer.storage import load_classifier
from spacer.tasks import train_classifier
from spacer.task_utils import preprocess_labels, SplitMode

from mermaid_classifier.common.benthic_attributes import (
    BenthicAttributeLibrary, CoralNetMermaidMapping, GrowthFormLibrary)
from mermaid_classifier.common.csv_utils import CsvSpec
from mermaid_classifier.common.duckdb_utils import (
    duckdb_add_column,
    duckdb_filter_on_column,
    duckdb_grouped_rows,
    duckdb_transform_column,
)
from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.pyspacer.utils import (
    logging_config_for_script, mlflow_connect)


logger = logging_config_for_script('train')


class Sites(enum.Enum):
    CORALNET = 'coralnet'
    MERMAID = 'mermaid'


def log_memory_usage(message):
    memory_usage = psutil.virtual_memory()
    logger.debug(f"{message} - Memory usage: {memory_usage.percent}%")


def format_accuracy(accuracy):
    return f'{100*accuracy:.1f}%'


def alphanumeric_only_str(s: str):
    """
    Return a version of s which has the non-alphanumeric chars removed.
    """
    return ''.join([char for char in s if char.isalnum()])


class BenthicAttrSet(CsvSpec):

    target_column_candidates = ['id', 'name']

    _ba_library = None

    @classmethod
    def get_library(cls):
        # Lazy-load the BA library, and save it at the class
        # level so we only load it once at most.
        if cls._ba_library is None:
            cls._ba_library = BenthicAttributeLibrary()
        return cls._ba_library

    def __init__(self, csv_file: typing.TextIO):
        """
        Initialize using a local CSV file that specifies a set
        of MERMAID benthic attributes.
        """
        self.ba_set = set()

        super().__init__(csv_file=csv_file)

    def per_item_init_action(self, target_column, target_value):

        # Regardless of what the CSV input has,
        # sticking with IDs from here on out will make life easier.
        if target_column == 'id':
            ba_id = target_value
        else:
            # 'name'
            ba_id = self.get_library().by_name[target_value]['id']
        self.ba_set.add(ba_id)


class BenthicAttrFilter(BenthicAttrSet):

    def __init__(self, *args, inclusion: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.inclusion = inclusion

    def accepts_ba(self, ba_id):
        if self.inclusion:
            return ba_id in self.ba_set
        else:
            return ba_id not in self.ba_set

    def filter_in_duckdb(
        self,
        duck_conn: 'DuckDBPyConnection',
        duck_table_name: str,
        ba_id_column_name: str = 'benthic_attribute_id',
    ):
        """
        Filter down the rows in the given DuckDB table, based on the
        benthic attribute ID column and this instance's filter rules.
        """
        duckdb_filter_on_column(
            duck_conn=duck_conn,
            duck_table_name=duck_table_name,
            column_name=ba_id_column_name,
            inclusion_func=self.accepts_ba,
        )


class BenthicAttrRollupSpec(BenthicAttrSet):

    def __init__(self, *args, **kwargs):
        """
        The CSV file should list rollup targets. For each target, any label
        that is a descendant of a target gets rolled up to that target.
        """
        super().__init__(*args, **kwargs)
        self.rollup_targets = self.ba_set

    def roll_up(self, ba_id):
        # Roll up to the earliest ancestor which is a rollup target.
        for ancestor_id in self.get_library().get_ancestor_ids(ba_id):
            if ancestor_id in self.rollup_targets:
                return ancestor_id
        # Or don't roll up if there's no such ancestor.
        return ba_id

    def roll_up_in_duckdb(
        self,
        duck_conn: 'DuckDBPyConnection',
        duck_table_name: str,
        ba_id_column_name: str = 'benthic_attribute_id',
    ):
        """
        Roll up the benthic attribute IDs in the given DuckDB table,
        based on this instance's rollup rules.
        """
        duckdb_transform_column(
            duck_conn, duck_table_name, ba_id_column_name, self.roll_up)


class CNSourceFilter(CsvSpec):

    target_column_candidates = ['id']

    source_id_list: list[str]

    def __init__(self, csv_file: typing.TextIO):
        """
        Initialize using a CSV file that specifies a set
        of CoralNet sources.
        """
        self.source_id_list = []

        super().__init__(csv_file=csv_file)

    def per_item_init_action(self, _target_column, target_value):
        self.source_id_list.append(target_value)

    def is_empty(self):
        return len(self.source_id_list) == 0


class Artifacts:
    """
    Namespace to make it easier to track artifacts that we're
    logging to MLflow later.
    """
    ba_counts: pd.DataFrame
    bagf_counts: pd.DataFrame
    coralnet_label_mapping: pd.DataFrame
    coralnet_project_stats: pd.DataFrame
    mermaid_project_stats: pd.DataFrame
    train_summary_stats: dict
    unmapped_labels: pd.DataFrame


class TrainingDataset:

    def __init__(
        self,
        include_mermaid: bool = True,
        coralnet_sources_csv: str = None,
        included_benthicattrs_csv: str = None,
        excluded_benthicattrs_csv: str = None,
        benthicattr_rollup_targets_csv: str = None,
        drop_growthforms: bool = False,
        annotation_limit: int | None = None,
    ):
        self.artifacts = Artifacts()

        if coralnet_sources_csv:
            with open(coralnet_sources_csv) as csv_f:
                self.cn_source_filter = CNSourceFilter(csv_f)
        else:
            # Empty set of CoralNet sources.
            self.cn_source_filter = CNSourceFilter(StringIO(''))

        if included_benthicattrs_csv and excluded_benthicattrs_csv:
            raise ValueError(
                "Specify one of included benthic attrs or"
                " excluded benthic attrs, but not both.")

        if included_benthicattrs_csv:
            with open(included_benthicattrs_csv) as csv_f:
                self.benthicattr_filter = BenthicAttrFilter(
                    csv_f, inclusion=True)
        elif excluded_benthicattrs_csv:
            with open(excluded_benthicattrs_csv) as csv_f:
                self.benthicattr_filter = BenthicAttrFilter(
                    csv_f, inclusion=False)
        else:
            # No inclusion or exclusion set specified means we accept
            # all labels.
            # In other words, an empty exclusion set.
            self.benthicattr_filter = BenthicAttrFilter(
                StringIO(''), inclusion=False)

        if benthicattr_rollup_targets_csv:
            with open(benthicattr_rollup_targets_csv) as csv_f:
                self.rollup_spec = BenthicAttrRollupSpec(csv_f)
        else:
            # Empty rollup-targets set, meaning nothing gets rolled up.
            self.rollup_spec = BenthicAttrRollupSpec(StringIO(''))

        self.drop_growthforms = drop_growthforms
        self.annotation_limit = annotation_limit

        # https://s3fs.readthedocs.io/en/latest/api.html#s3fs.core.S3FileSystem
        self.s3 = S3FileSystem(
            anon=settings.aws_anonymous,
            key=settings.aws_key_id,
            secret=settings.aws_secret,
            token=settings.aws_session_token,
        )
        self._duck_conn = None

        self.data = dict()
        mermaid_full_paths_in_s3 = set()

        # This'll get filled in on-demand later. That way, we can abort
        # reading in data at any time without worrying about whether
        # this will get set.
        self._labels = None

        if not self.cn_source_filter.is_empty():
            self.read_coralnet_data()
        else:
            # An empty dataframe
            self.artifacts.coralnet_project_stats = pd.DataFrame()

        if include_mermaid:
            self.read_mermaid_data()

            # When we iterate through the annotation data's images, we'll
            # check the image IDs against the feature vectors that are
            # actually present in S3.
            # Note: this line can take a while to run.
            mermaid_full_paths_in_s3 = set(
                self.s3.find(
                    path=f's3://{settings.mermaid_train_data_bucket}/mermaid/'
                )
            )
        else:
            self.artifacts.mermaid_project_stats = pd.DataFrame()

        # Now this should have annotations populated.
        if not self.duckdb_annotations_table_exists():
            raise ValueError(
                "No annotations from CoralNet or MERMAID, even before"
                " label filtering.")

        self.handle_missing_feature_vectors(mermaid_full_paths_in_s3)

        # Roll up benthic attributes.
        self.rollup_spec.roll_up_in_duckdb(
            duck_conn=self.duck_conn,
            duck_table_name='annotations',
        )

        if self.drop_growthforms:
            # Null out all annotations' growth forms.
            duckdb_transform_column(
                duck_conn=self.duck_conn,
                duck_table_name='annotations',
                column_name='growth_form_id',
                transform_func=lambda x: None,
            )

        # Filter out benthic attributes we don't want.
        self.benthicattr_filter.filter_in_duckdb(
            duck_conn=self.duck_conn,
            duck_table_name='annotations',
        )

        if self.annotation_limit:

            # See how many annotations we have.
            count_up_to_limit = self.duck_conn.execute(
                f"SELECT count(*)"
                f" FROM annotations"
                f" LIMIT {self.annotation_limit + 1}"
            ).fetchall()[0][0]

            # If we're over limit, log that fact.
            if count_up_to_limit > self.annotation_limit:
                logger.debug(
                    f"Truncating the train data to"
                    f" {self.annotation_limit} annotations."
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
                f" LIMIT {self.annotation_limit + 1}"
            ).fetchall()[0][0]
            mm_count_up_to_limit = self.duck_conn.execute(
                f"SELECT count(*)"
                f" FROM annotations"
                f" WHERE site = '{Sites.MERMAID.value}'"
                f" LIMIT {self.annotation_limit + 1}"
            ).fetchall()[0][0]
            half_limit = math.ceil(self.annotation_limit / 2)
            if cn_count_up_to_limit <= half_limit:
                cn_count = cn_count_up_to_limit
                mm_count = self.annotation_limit - cn_count
            else:
                mm_count = min(mm_count_up_to_limit, half_limit)
                cn_count = self.annotation_limit - mm_count

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

        annotations_by_image = duckdb_grouped_rows(
            duck_conn=self.duck_conn,
            duck_table_name='annotations',
            grouping_column_names=['bucket', 'feature_vector'],
        )

        for rows in annotations_by_image:

            # Here, in one loop iteration, we're given all the
            # annotation rows for a single image.
            first_row = rows[0]
            bucket = first_row['bucket']
            feature_bucket_path = first_row['feature_vector']

            image_annotations = []

            # One annotation per row.
            for row in rows:

                benthic_attribute_id = row['benthic_attribute_id']

                if row['growth_form_id'] is None:
                    # Either we've chosen not to get growth forms, or
                    # this annotation has no growth form.
                    bagf = benthic_attribute_id
                else:
                    # Include growth form.
                    # MERMAID API uses :: as the BA-GF separator.
                    bagf = '::'.join([
                        benthic_attribute_id, row['growth_form_id']])

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
            self.data[feature_loc] = image_annotations

        self.set_train_summary_stats()

    def read_mermaid_data(self):

        mermaid_annotations_s3_uri = (
            f's3://{settings.mermaid_train_data_bucket}/mermaid/'
            f'mermaid_confirmed_annotations.parquet'
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
            f" FROM read_parquet('{mermaid_annotations_s3_uri}')"
        )

        # Get project-level stats before applying any further filters.
        self.artifacts.mermaid_project_stats = self.compute_project_stats(
            site=Sites.MERMAID.value)

        # For growth forms, we get NULL/None from the CoralNet-MERMAID
        # mapping, but the string 'None' from the MERMAID annotations
        # parquet.
        # Normalize the latter to NULL/None.
        def transform_func(gf_id):
            if gf_id is None or gf_id == 'None':
                return None
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

            source_folder_uri = (
                f's3://{settings.coralnet_train_data_bucket}/s{source_id}'
            )

            # One row per point annotation,
            # with columns including Image ID, Row, Column, Label ID
            annotations_uri = f'{source_folder_uri}/annotations.csv'
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
            site=Sites.CORALNET.value)

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

    def handle_missing_feature_vectors(self, mermaid_full_paths_in_s3):

        # Build full feature paths in DuckDB.
        self.duck_conn.execute(
            f"CREATE OR REPLACE TABLE annotations AS"
            f" SELECT *,"
            f"  bucket || '/' || feature_vector AS feature_full"
            f" FROM annotations"
        )

        # Detect missing feature vectors.
        # TODO: Also check CoralNet bucket/folder for missing features.
        result = self.duck_conn.execute(
            f"SELECT DISTINCT feature_full FROM annotations"
            f" WHERE site = '{Sites.MERMAID.value}'"
        ).fetchall()
        mermaid_full_paths_in_annos = [tup[0] for tup in result]
        missing_feature_paths = (
            set(mermaid_full_paths_in_annos) - mermaid_full_paths_in_s3)

        # Filter out missing feature vectors from annotations table.
        duckdb_filter_on_column(
            duck_conn=self.duck_conn,
            duck_table_name='annotations',
            column_name='feature_full',
            inclusion_func=lambda p: p not in missing_feature_paths,
        )

        # Don't need the feature_full column anymore.
        self.duck_conn.execute(
            f"ALTER TABLE annotations DROP feature_full"
        )

        # Abort if too many are missing.
        missing_count = len(missing_feature_paths)
        examples_iterator = itertools.islice(missing_feature_paths, 3)
        examples_str = "\n".join(list(examples_iterator))
        missing_threshold = (
            len(self.data)
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
        if missing_feature_paths:
            logger.warning(
                f"Skipping {missing_count} feature vector(s) because"
                f" the files aren't in S3."
                f" Example(s):"
                f"\n{examples_str}")

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
            query = (
                f"CREATE OR REPLACE SECRET secret ("
                f" TYPE s3,"
                f" PROVIDER config,"
            )
            if settings.aws_key_id:
                query += f" KEY_ID '{settings.aws_key_id}',"
                query += f" SECRET '{settings.aws_secret}',"
            if settings.aws_session_token:
                query += f" SESSION_TOKEN '{settings.aws_session_token}',"
            query += f" REGION '{settings.aws_region}')"

            self._duck_conn.execute(query)

        return self._duck_conn

    @property
    def labels(self):
        if self._labels is None:
            self._labels = preprocess_labels(
                ImageLabels(self.data),
                # 10% ref, 10% val, 80% train.
                split_ratios=(0.1, 0.1),
                split_mode=SplitMode.POINTS_STRATIFIED,
            )
        return self._labels

    source_image_stats_df: pd.DataFrame

    def compute_project_stats(self, site=None):
        if site is None:
            where_clause = ""
        else:
            where_clause = f"WHERE site = '{site}'"

        result = self.duck_conn.execute(
            f"SELECT site, project_id,"
            f" count(DISTINCT image_id) AS num_images,"
            f" count(*) AS num_annotations"
            f" FROM annotations"
            f" {where_clause}"
            f" GROUP BY site, project_id"
            # MERMAID, then CoralNet; because MERMAID's currently just
            # one row (no projects distinction).
            f" ORDER BY site DESC, project_id"
        )
        return result.fetch_df()

    def set_train_summary_stats(self):

        # Annotation count per BA, sorted by most common first.
        self.duck_conn.execute(
            "CREATE TABLE ba_counts AS"
            " SELECT"
            " benthic_attribute_id,"
            " count(*) AS num_annotations,"
            " count(DISTINCT project_id) AS num_projects"
            " FROM annotations GROUP BY benthic_attribute_id"
        )
        # Add BA names alongside the IDs for readability.
        duckdb_add_column(
            duck_conn=self.duck_conn,
            duck_table_name='ba_counts',
            base_column_name='benthic_attribute_id',
            new_column_name='benthic_attribute_name',
            base_to_new_func=BenthicAttrSet.get_library().id_to_name,
        )
        # Sort by annotation count, and reorder columns
        # while we're at it.
        self.artifacts.ba_counts = self.duck_conn.execute(
            "SELECT"
            " benthic_attribute_name,"
            " num_annotations,"
            " num_projects,"
            " benthic_attribute_id"
            " FROM ba_counts"
            " ORDER BY num_annotations DESC"
        ).fetch_df()

        # Annotation count per BAGF, most common first.
        # We have to watch out for NULL growth form IDs since:
        # - NULL values are ignored in most aggregate functions, like count().
        #   https://duckdb.org/docs/stable/sql/data_types/nulls#null-and-aggregate-functions
        # - Joining on a column with NULL values could result in those rows
        #   being lost.
        # So we use coalesce() to temporarily use '0' as placeholders for
        # NULL.
        # https://duckdb.org/docs/stable/sql/data_types/nulls#null-and-functions
        self.duck_conn.execute(
            "CREATE TABLE bagf_counts AS"
            " SELECT"
            " benthic_attribute_id,"
            " coalesce(growth_form_id, '0') as growth_form_id,"
            " count(*) AS num_annotations,"
            " count(DISTINCT project_id) AS num_projects"
            " FROM annotations"
            " GROUP BY benthic_attribute_id, growth_form_id"
        )
        # Add BA names for readability.
        duckdb_add_column(
            duck_conn=self.duck_conn,
            duck_table_name='bagf_counts',
            base_column_name='benthic_attribute_id',
            new_column_name='benthic_attribute_name',
            base_to_new_func=BenthicAttrSet.get_library().id_to_name,
        )
        # Add GF names for readability.
        gf_library = GrowthFormLibrary()
        duckdb_add_column(
            duck_conn=self.duck_conn,
            duck_table_name='bagf_counts',
            base_column_name='growth_form_id',
            new_column_name='growth_form_name',
            base_to_new_func=gf_library.by_id.get,
        )
        # Transform '0' back to NULL, now that we're done with ops that
        # would trip on NULL.
        duckdb_transform_column(
            duck_conn=self.duck_conn,
            duck_table_name='bagf_counts',
            column_name='growth_form_id',
            transform_func=lambda x: None if x == '0' else x,
        )
        # Sort by annotation count, and reorder columns
        # while we're at it.
        self.artifacts.bagf_counts = self.duck_conn.execute(
            "SELECT"
            " benthic_attribute_name,"
            " growth_form_name,"
            " num_annotations,"
            " num_projects,"
            " benthic_attribute_id,"
            " growth_form_id"
            " FROM bagf_counts"
            " ORDER BY num_annotations DESC"
        ).fetch_df()

        self.artifacts.train_summary_stats = dict(
            total_annotations=self.labels.label_count,
            train_annotations=self.labels.train.label_count,
            ref_annotations=self.labels.ref.label_count,
            val_annotations=self.labels.val.label_count,
            num_of_images=len(self.data),
            num_of_benthic_attributes=self.artifacts.ba_counts.shape[0],
            num_of_ba_gf_combinations=self.artifacts.bagf_counts.shape[0],
        )

    def describe_train_summary_stats(self):
        return (
            "{total_annotations} annotations"
            " ({train_annotations} train,"
            " {ref_annotations} ref,"
            " {val_annotations} val) from"
            " {num_of_images} images."
            " {num_of_benthic_attributes} BAs and"
            " {num_of_ba_gf_combinations} BA-GF combos"
            " are represented here.".format(
                **self.artifacts.train_summary_stats)
        )

    def log_mlflow_artifacts(self):
        """
        Log various options and stats for the training dataset.
        """
        mlflow.log_table(
            self.cn_source_filter.csv_dataframe,
            'coralnet_sources_included.json')

        if self.benthicattr_filter.inclusion:
            table_filename = 'benthic_attrs_included.json'
        else:
            table_filename = 'benthic_attrs_excluded.json'
        mlflow.log_table(
            self.benthicattr_filter.csv_dataframe, table_filename)

        mlflow.log_table(
            self.rollup_spec.csv_dataframe, 'rollup_spec.json')

        # Number of images and annotations from each CN source and from
        # MERMAID.
        # First, before filtering (this is what's present in S3).
        # https://pandas.pydata.org/docs/reference/api/pandas.concat.html
        mlflow.log_table(
            pd.concat([
                self.artifacts.mermaid_project_stats,
                self.artifacts.coralnet_project_stats,
            ]),
            'project_stats_raw.json')
        # And here, after filtering (this is what training actually gets).
        mlflow.log_table(
            self.compute_project_stats(),
            'project_stats_train_data.json')

        mlflow.log_dict(
            self.artifacts.train_summary_stats, 'train_summary.yaml')

        mlflow.log_table(self.artifacts.ba_counts, 'ba_counts.json')
        mlflow.log_table(self.artifacts.bagf_counts, 'bagf_counts.json')

        if not self.cn_source_filter.is_empty():
            # These only apply if CoralNet data is included.
            mlflow.log_table(
                self.artifacts.coralnet_label_mapping,
                'coralnet_label_mapping.json')
            mlflow.log_table(
                self.artifacts.unmapped_labels,
                'unmapped_labels.json')

        # Log other options given to the training process.
        other_options = dict(
            drop_growthforms=self.drop_growthforms,
            annotation_limit=self.annotation_limit,
        )
        mlflow.log_dict(other_options, 'other_options.yaml')


def run_training(
    include_mermaid: bool = True,
    coralnet_sources_csv: str = None,
    included_benthicattrs_csv: str = None,
    excluded_benthicattrs_csv: str = None,
    benthicattr_rollup_targets_csv: str = None,
    drop_growthforms: bool = False,
    annotation_limit: int | None = None,
    epochs: int = 10,
    experiment_name: str | None = None,
    model_name: str | None = None,
    disable_mlflow: bool = False,
):
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

    included_benthicattrs_csv
    excluded_benthicattrs_csv

    Local filepath of a CSV file, specifying either the MERMAID benthic
    attributes to accept from the training data (excluding all others), or
    the ones to leave out from the training data (including all others).
    - Specify at most one of these files, not both. If neither file is
      specified, then all MERMAID benthic attributes are accepted.
    - Mapping from CoralNet label IDs to MERMAID BA IDs will be done before
      applying these inclusions/exclusions.
    - Growth forms are ignored here.
    - Recognized columns:
      id -- A MERMAID benthic attribute ID (a UUID).
      name -- Human-readable name of a MERMAID benthic attribute. If an id
        column is present, name is ignored. Else, the name column is used
        instead. Either id or name must be present.
      Other informational columns can also be present and will be ignored.

    benthicattr_rollup_targets_csv

    Local filepath of a CSV file, specifying the MERMAID benthic attributes
    to roll up to. So if this includes Acroporidae for example, then all
    descendants of Acroporidae (Acropora, Acropora arabensis, Alveopora...)
    get rolled up to Acroporidae for training.
    - If this file isn't specified, then nothing gets rolled up.
    - Growth forms are ignored here.
    - Recognized columns and their semantics are the same as
      included_benthicattrs_csv.

    drop_growthforms

    If True, discard growth forms from the training data. Basically another
    dimension of rolling up labels.

    annotation_limit

    If specified, only get up to this many annotations for training. This can
    help with testing since the runtime is correlated to number of annotations.

    epochs

    Number of training epochs to run.

    experiment_name

    Name of the MLflow experiment in which to register the training run. If not
    given, then it's taken from the MLFLOW_EXPERIMENT_NAME env var.

    model_name

    Name of this MLflow experiment run's model. If not given, then a model name will be
    constructed based on the experiment parameters.
    The run name is based on this too.

    disable_mlflow

    If True, don't connect to or log to a MLflow tracking server. This can
    make testing easier when running a tracking server feels onerous.

    TODO: Be able to specify an example image (or set of them?) whose
    training annotations are logged along with the model. Since logging all
    training annotations might be too much, but looking at a small sample
    is a good sanity check.
    """
    experiment_name = (
        experiment_name or settings.mlflow_default_experiment_name)

    training_dataset = TrainingDataset(
        include_mermaid=include_mermaid,
        coralnet_sources_csv=coralnet_sources_csv,
        included_benthicattrs_csv=included_benthicattrs_csv,
        excluded_benthicattrs_csv=excluded_benthicattrs_csv,
        benthicattr_rollup_targets_csv=benthicattr_rollup_targets_csv,
        drop_growthforms=drop_growthforms,
        annotation_limit=annotation_limit,
    )

    # Other prep before training.

    log_memory_usage('Memory usage after creating labels')
    logger.info("Proceeding to train with:")
    logger.info(training_dataset.describe_train_summary_stats())

    current_time = datetime.now()
    time_str = current_time.strftime('%Y%m%dT%H%M%S')

    experiment_params = dict(epochs=epochs)

    # Only alphanumeric chars and hyphens are allowed in MLflow model names.
    # So we'll use hyphens as the 'outer' word separator, and CamelCaps as
    # the 'inner' one.

    if model_name is None:

        if included_benthicattrs_csv:
            as_path = Path(included_benthicattrs_csv)
            model_name = f'Include{alphanumeric_only_str(as_path.stem)}'
        elif excluded_benthicattrs_csv:
            as_path = Path(excluded_benthicattrs_csv)
            model_name = f'Exclude{alphanumeric_only_str(as_path.stem)}'
        else:
            model_name = 'AllLabels'

        if benthicattr_rollup_targets_csv:
            as_path = Path(benthicattr_rollup_targets_csv)
            model_name += f'-Rollup{alphanumeric_only_str(as_path.stem)}'

        if coralnet_sources_csv:
            as_path = Path(coralnet_sources_csv)
            model_name += f'-{alphanumeric_only_str(as_path.stem)}'

        if annotation_limit:
            model_name += f'-AnnoLimit{annotation_limit}'

    run_name = f'{model_name}-{time_str}'

    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Run: {run_name}")

    # Just store the model in memory for now, since it'll be saved out to
    # MLflow later anyway.
    model_loc = DataLocation('memory', key='classifier.pkl')
    # Not sure about saving this as an artifact yet. If we do, then
    # use a location type other than 'memory'.
    valresult_loc = DataLocation('memory', key='valresult.json')

    train_msg = TrainClassifierMsg(
        job_token=f'experiment_run_{run_name}',
        trainer_name='minibatch',
        nbr_epochs=experiment_params['epochs'],
        clf_type='MLP',
        labels=training_dataset.labels,
        previous_model_locs=[],
        model_loc=model_loc,
        valresult_loc=valresult_loc,
        feature_cache_dir=TrainClassifierMsg.FeatureCache.AUTO,
    )

    log_memory_usage("Memory usage before training")

    if disable_mlflow:

        return_msg = train_classifier(train_msg)

    elif MLFLOW_IMPORT_ERROR:

        # Options say to use MLflow, but it couldn't be imported.
        raise MLFLOW_IMPORT_ERROR

    else:

        time_taken = mlflow_connect()
        logger.info(f"Time to connect to MLflow tracking: {time_taken}")

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):

            # Log dataset artifacts first, so they can be inspected during
            # training.
            training_dataset.log_mlflow_artifacts()

            return_msg = train_classifier(train_msg)

            logger.info(
                f"Train time (from return msg): {return_msg.runtime:.1f} s")
            log_memory_usage("Memory usage after training")

            mlflow.log_params(experiment_params)

            # Note that log_metric() only takes numeric values.
            accuracy_pct = return_msg.acc * 100
            mlflow.log_metric("accuracy", accuracy_pct)

            # ref_accs is probably the only other part of return_msg to save.
            ref_accs_dict = dict()
            for epoch_number, acc in enumerate(return_msg.ref_accs, 1):
                ref_accs_dict[epoch_number] = format_accuracy(acc)
            mlflow.log_dict(ref_accs_dict, 'epoch_ref_accuracies.yaml')

            # Save and register the trained model.
            signature = infer_signature(params=experiment_params)
            model_info = mlflow.sklearn.log_model(
                sk_model=load_classifier(model_loc),
                registered_model_name=model_name,
                signature=signature,
            )

            logger.info(f"Model ID: {model_info.model_id}")

    logger.info(
        f"New model's accuracy: {format_accuracy(return_msg.acc)}")

    ref_accs_str = ", ".join(
        [format_accuracy(acc) for acc in return_msg.ref_accs])
    logger.debug(
        f"Accuracy progression during training epochs: {ref_accs_str}")
