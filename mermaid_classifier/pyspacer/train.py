"""
Train a classifier using feature vectors and annotations
provided on S3.
"""
from collections import Counter
from dataclasses import make_dataclass
from datetime import datetime
import enum
from io import StringIO
from pathlib import Path
import typing

import duckdb
try:
    import mlflow
    from mlflow.models import infer_signature
    MLFLOW_IMPORT_ERROR = None
except ImportError as err:
    MLFLOW_IMPORT_ERROR = err
# MLflow requires pandas.
# Might as well also use it for reading CSV.
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


def csv_to_dataframe(csv_file: typing.TextIO):
    """
    CSV file to pandas dataframe. Use this for non-mathematical data
    such as strings or integers that just represent IDs.

    Throws pd.errors.EmptyDataError if there's no CSV data.
    """
    return pd.read_csv(
        csv_file,
        # Don't convert blank cells to NaN. It seems easier to
        # just check if row.get('...') is truthy, than to
        # check if it's NaN, when we're not dealing with mathematical
        # data in the first place.
        keep_default_na=False,
    )


class BenthicAttrSet:

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

        try:
            self.csv_dataframe = csv_to_dataframe(csv_file)
        except pd.errors.EmptyDataError:
            # It just errors if there's no CSV data, so we manually
            # create an empty dataframe in this case.
            # We also short-circuit to prevent any other odd cases.
            self.csv_dataframe = pd.DataFrame()
            return

        csv_filename = getattr(csv_file, 'name', "<File-like obj>")

        if 'id' in self.csv_dataframe.columns:
            target_column = 'id'
        elif 'name' in self.csv_dataframe.columns:
            target_column = 'name'
        else:
            raise ValueError(
                f"{csv_filename}:"
                f"doesn't have `id` or `name` column")

        for index, row in self.csv_dataframe.iterrows():
            target_value = row.get(target_column)
            if not target_value:
                raise ValueError(
                    f"{csv_filename}:"
                    f"{target_column} not found in row {index + 1}")

            # Sticking with IDs from here on out will make life easier.
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


class CNSourceFilter:

    source_id_list: list[str]

    def __init__(self, csv_file: typing.TextIO):
        """
        Initialize using a CSV file that specifies a set
        of CoralNet sources.
        """
        csv_dataframe = csv_to_dataframe(csv_file)
        self.source_id_list = []

        csv_filename = getattr(csv_file, 'name', "<File-like obj>")

        if 'id' not in csv_dataframe.columns:
            raise ValueError(
                f"{csv_filename}:"
                f" doesn't have `id` column")

        for index, row in csv_dataframe.iterrows():
            source_id = row.get('id')
            if not source_id:
                raise ValueError(
                    f"{csv_filename}:"
                    f" id not found in row {index + 1}")
            self.source_id_list.append(source_id)

        if len(self.source_id_list) == 0:
            raise ValueError("No sources specified")


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
        cn_source_filter = None
        if coralnet_sources_csv:
            with open(coralnet_sources_csv) as csv_f:
                cn_source_filter = CNSourceFilter(csv_f)

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
        self.annos_by_ba = Counter()
        self.annos_by_bagf = Counter()
        self.all_annos_count = 0
        missing_features = []
        mermaid_full_paths_in_s3 = set()

        # This'll get filled in on-demand later. That way, we can abort
        # reading in data at any time without worrying about whether
        # this will get set.
        self._labels = None

        self.coralnet_mermaid_label_mapping = CoralNetMermaidMapping()

        if coralnet_sources_csv:
            self.read_coralnet_data(cn_source_filter)

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

        # Now this should have annotations populated.
        if not self.duckdb_annotations_table_exists():
            raise ValueError(
                "No annotations from CoralNet or MERMAID, even before"
                " label filtering.")

        annotations_duckdb = self.duck_conn.execute(
            f"SELECT * FROM annotations")

        # TODO: Group by site AND image, just in case image IDs might
        #  overlap between sites (though that's not anticipated with
        #  CN/MM).
        annotations_by_image = self.grouped_annotation_rows(
            annotations_duckdb, 'image_id')

        for rows in annotations_by_image:

            # Here, in one loop iteration, we're given all the
            # annotation rows for a single image.
            first_row = rows[0]
            site = first_row['site']
            bucket = first_row['bucket']
            image_id = first_row['image_id']
            feature_bucket_path = first_row['feature_vector']

            image_annotations = []
            image_annos_by_ba = Counter()
            image_annos_by_bagf = Counter()

            # One annotation per row.
            for row in rows:

                if not self.benthicattr_filter.accepts_ba(
                    row['benthic_attribute_id']
                ):
                    # This BA is being filtered out of the training data.
                    continue

                benthic_attribute_id = self.rollup_spec.roll_up(
                    row['benthic_attribute_id'])

                # TODO: None from CoralNet-MERMAID mapping, 'None' from
                #  MERMAID annotations parquet. Not sure what should change
                #  to improve consistency.
                if (
                    self.drop_growthforms
                    or row['growth_form_id'] is None
                    or row['growth_form_id'] == 'None'
                ):
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
                image_annos_by_ba[benthic_attribute_id] += 1
                image_annos_by_bagf[bagf] += 1

            impending_annotation_count = (
                self.all_annos_count + len(image_annotations))
            if (
                self.annotation_limit
                and
                impending_annotation_count > self.annotation_limit
            ):
                logger.debug(
                    f"Currently have {self.all_annos_count} annotations."
                    f" Stopping because the {len(image_annotations)}"
                    f" annotations from the next image ({image_id})"
                    f" would put us over the limit of {self.annotation_limit}."
                )
                return

            # TODO: Also check CoralNet bucket/folder for missing features.
            if site == Sites.MERMAID.value:
                feature_full_path = f'{bucket}/{feature_bucket_path}'
                if feature_full_path not in mermaid_full_paths_in_s3:
                    logger.warning(
                        f"Skipping feature vector because couldn't find"
                        f" the file in S3: {feature_full_path}")
                    missing_features.append(feature_full_path)
                    continue

            feature_loc = DataLocation(
                storage_type='s3',
                bucket_name=bucket,
                key=feature_bucket_path,
            )
            self.data[feature_loc] = image_annotations
            self.all_annos_count += len(image_annotations)
            self.annos_by_ba += image_annos_by_ba
            self.annos_by_bagf += image_annos_by_bagf

        missing_threshold = (
            len(self.data)
            * settings.training_inputs_percent_missing_allowed / 100
        )
        if len(missing_features) > missing_threshold:
            raise RuntimeError(
                f"Too many feature vectors are missing"
                f" ({len(missing_features)}), such as:"
                f"\n{'\n'.join(missing_features[:3])}"
                f"\nYou can configure the tolerance for missing"
                f" feature vectors with the"
                f" TRAINING_INPUTS_PERCENT_MISSING_ALLOWED setting."
            )

    def read_mermaid_data(self):

        mermaid_annotations_s3_uri = (
            f's3://{settings.mermaid_train_data_bucket}/mermaid/'
            f'mermaid_confirmed_annotations.parquet'
        )

        if self.duckdb_annotations_table_exists():
            # INSERT INTO ... BY NAME ... means that we don't have to match
            # the column order of the existing table.
            # https://duckdb.org/docs/stable/sql/statements/insert#insert-into--by-name
            self.duck_conn.execute(
                f"INSERT INTO annotations BY NAME"
                f" SELECT"
                f"  image_id, row, col,"
                f"  benthic_attribute_id, growth_form_id,"
                f" 'MERMAID' AS site,"
                f" '{settings.mermaid_train_data_bucket}' AS bucket,"
                f"  NULL AS project_id,"
                f"  concat('mermaid/', image_id, '_featurevector')"
                f"   AS feature_vector"
                f" FROM read_parquet('{mermaid_annotations_s3_uri}')"
            )
        else:
            # Didn't read any CoralNet data, so we have to create
            # the annotations table.
            self.duck_conn.execute(
                f"CREATE TABLE annotations AS"
                f" SELECT *"
                f" FROM read_parquet('{mermaid_annotations_s3_uri}')"
            )

    def read_coralnet_data(self, cn_source_filter):
        annotations_uri_list = []

        for source_id in cn_source_filter.source_id_list:

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

        # Get all the unique CN labels in the dataset, and their
        # frequencies.
        cn_label_frequencies = dict(
            self.duck_conn.execute(
                "SELECT label_id, COUNT(*)"
                " FROM annotations GROUP BY label_id"
            ).fetchall()
        )

        # Add BAs and GFs to the DuckDB database, using
        # our mapping from CoralNet label IDs to MERMAID BAs/GFs.
        self.coralnet_mermaid_label_mapping.add_bagf_in_duckdb(
            cn_label_ids=list(cn_label_frequencies.keys()),
            duck_conn=self.duck_conn,
            duck_table_name='annotations',
        )

    def duckdb_annotations_table_exists(self) -> bool:
        # https://duckdb.org/docs/stable/sql/meta/information_schema
        table_query_result = self.duck_conn.execute(
            f"SELECT * FROM information_schema.tables"
            f" WHERE table_name = 'annotations'"
        ).fetchall()

        # Result should be [] if doesn't exist, which 'bools' to False.
        return bool(table_query_result)

    def annotation_rows(
        self,
        annotations_duckdb: 'DuckDBPyRelation',
    ) -> typing.Generator['pandas.core.series.Series', None, None]:
        """
        Reads from a DuckDB relation (chunkifying to avoid memory issues),
        and generates pandas dataframe rows.
        """
        # Fetch in chunks, not all at once, to avoid running out of
        # memory.
        # An example chunk size is 2048 rows x 12 columns.
        #
        # fetchmany() is similar, but returns lists of tuples instead.
        # A dataframe provides dict-like access which is a bit nicer.
        while True:
            dataframe = annotations_duckdb.fetch_df_chunk()
            if dataframe.shape[0] == 0:
                # Empty dataframe; no more chunks left.
                break

            # Some ways to inspect the pandas dataframe in a debugger:
            # dataframe
            # dataframe.columns    # Column names
            # dataframe.iloc[0]    # First row
            # set([row['growth_form_name']
            #     for _index, row in dataframe.iterrows()])
            # [(row['row'], row['col']) for _index, row in dataframe.iterrows()
            #  if row['image_id'] == '0032dba6-8357-42e2-bace-988f99032286']

            for _index, row in dataframe.iterrows():
                # With this generator behavior, the chunkifying detail
                # is invisible to the caller.
                yield row

    def grouped_annotation_rows(
        self,
        annotations_duckdb: 'DuckDBPyRelation',
        grouping_column: str,
    ) -> typing.Generator['pandas.core.series.Series', None, None]:

        grouping_value = None
        group_rows = []

        for row in self.annotation_rows(annotations_duckdb):
            if grouping_value != row[grouping_column]:
                if grouping_value:
                    # End of group.
                    yield group_rows

                # Start of group.
                grouping_value = row[grouping_column]
                group_rows = []

            group_rows.append(row)

        # End of last group.
        yield group_rows

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

    def get_stats(self):
        return dict(
            total_annotations=self.labels.label_count,
            train_annotations=self.labels.train.label_count,
            ref_annotations=self.labels.ref.label_count,
            val_annotations=self.labels.val.label_count,
            num_of_images=len(self.data),
            num_of_benthic_attributes=len(self.annos_by_ba),
            num_of_ba_gf_combinations=len(self.annos_by_bagf),
        )

    def describe_stats(self):
        return (
            "Proceeding to train with {total_annotations}"
            " annotations ({train_annotations} train,"
            " {ref_annotations} ref,"
            " {val_annotations} val) from"
            " {num_of_images} images."
            " {num_of_benthic_attributes} BAs and"
            " {num_of_ba_gf_combinations} BA-GF combos"
            " are represented here.".format(**self.get_stats())
        )

    def log_mlflow_artifacts(self):

        if self.benthicattr_filter.inclusion:
            table_filename = 'included_benthicattrs.json'
        else:
            table_filename = 'excluded_benthicattrs.json'
        mlflow.log_table(
            self.benthicattr_filter.csv_dataframe, table_filename)

        mlflow.log_table(
            self.rollup_spec.csv_dataframe, 'rollup_spec.json')

        # Not sure if logging the entirety of the inputs is worth it
        # (since it's a lot), but we can at least log the sizes of the
        # inputs.
        # They don't seem like 'metrics', nor are they strictly 'params'
        # or 'inputs', nor are they 'outputs'... so we just log as a dict.
        mlflow.log_dict(self.get_stats(), 'input_stats.yaml')

        # Log annotation count per BA and per BAGF,
        # each sorted by most common first.
        #
        # For each, we construct a list of dataclass instances,
        # which is a format that dataframes support.
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
        CountLogEntry = make_dataclass(
            'CountLogEntry',
            [('id', str), ('name', str), ('count', int)])
        by_ba_with_names = []
        for ba_id, count in self.annos_by_ba.most_common():
            ba_name = BenthicAttrSet.get_library().id_to_name(ba_id)
            by_ba_with_names.append(CountLogEntry(
                id=ba_id,
                name=ba_name,
                count=count,
            ))
        by_bagf_with_names = []
        gf_library = GrowthFormLibrary()
        for bagf_id, count in self.annos_by_bagf.most_common():
            bagf_name = BenthicAttrSet.get_library().bagf_id_to_name(
                bagf_id, gf_library)
            by_bagf_with_names.append(CountLogEntry(
                id=bagf_id,
                name=bagf_name,
                count=count,
            ))

        mlflow.log_table(pd.DataFrame(by_ba_with_names), 'ba_counts.json')
        mlflow.log_table(pd.DataFrame(by_bagf_with_names), 'bagf_counts.json')

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
    logger.info(training_dataset.describe_stats())

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

            return_msg = train_classifier(train_msg)

            logger.info(
                f"Train time (from return msg): {return_msg.runtime:.1f} s")
            log_memory_usage("Memory usage after training")

            mlflow.log_params(experiment_params)

            # Note that log_metric() only takes numeric values.
            accuracy_pct = return_msg.acc * 100
            mlflow.log_metric("accuracy", accuracy_pct)

            training_dataset.log_mlflow_artifacts()

            # ref_accs is probably the only other part of return_msg to save.
            ref_accs_dict = dict()
            for epoch_number, acc in enumerate(return_msg.ref_accs, 1):
                ref_accs_dict[epoch_number] = format_accuracy(acc)
            mlflow.log_dict(ref_accs_dict, 'epoch_ref_accuracies.yaml')

            # TODO: MLflow artifact with image counts from MERMAID and
            #  from each CN source.

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
