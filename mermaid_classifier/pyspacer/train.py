"""
Train a classifier using feature vectors and annotations
provided on S3.
"""
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path
import typing
from urllib.parse import urlparse

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
from spacer.data_classes import ImageLabels
from spacer.messages import DataLocation, TrainClassifierMsg
from spacer.storage import load_classifier
from spacer.tasks import train_classifier
from spacer.task_utils import preprocess_labels, SplitMode

from mermaid_classifier.common.benthic_attributes import (
    BenthicAttributeLibrary)
from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.pyspacer.utils import (
    logging_config_for_script, mlflow_connect)


logger = logging_config_for_script('train')


AWS_REGION = 'us-east-1'
TRAINING_BUCKET = 'coral-reef-training'


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
            self.csv_dataframe = pd.read_csv(
                csv_file,
                # Don't convert blank cells to NaN. It seems easier to
                # just check if row.get('...') is truthy, than to
                # check if it's NaN, when we're not dealing with numeric
                # data in the first place.
                keep_default_na=False,
            )
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


class TrainingDataset:

    data: dict
    annos_by_ba: Counter
    annos_by_bagf: Counter

    def __init__(
        self,
        included_benthicattrs_csv: str = None,
        excluded_benthicattrs_csv: str = None,
        benthicattr_rollup_targets_csv: str = None,
        drop_growthforms: bool = False,
        annotation_limit: int | None = None,
    ):

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

        # TODO: This is just MERMAID so far; also get CoralNet

        # As we iterate through annotations, we'll check the image IDs
        # against the feature vectors that are actually present in S3.
        s3 = S3FileSystem(anon=settings.spacer_aws_anonymous)
        mermaid_full_paths_in_s3 = set(
            s3.find(path=f's3://{TRAINING_BUCKET}/mermaid/'))
        missing_features = []

        mermaid_annotations_s3_uri = (
            f's3://{TRAINING_BUCKET}/mermaid/'
            f'mermaid_confirmed_annotations.parquet'
        )

        # TODO: Option to filter by CN source (and MERMAID equiv.?).
        #   Like a CSV with columns for site (CN or MERMAID)
        #   and source/project ID.

        annotations_by_image = self.grouped_data_rows_from_parquet(
            mermaid_annotations_s3_uri, 'image_id')

        self.data = dict()
        self.annos_by_ba = Counter()
        self.annos_by_bagf = Counter()
        all_annos_count = 0

        # This'll get filled in on-demand later. That way, we can abort
        # reading in data at any time without worrying about whether
        # this will get set.
        self._labels = None

        for rows in annotations_by_image:

            # Here, in one loop iteration, we're given all the
            # annotation rows for a single image.
            image_id = rows[0]['image_id']
            feature_bucket_path = f'mermaid/{image_id}_featurevector'
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

                if row['growth_form_id'] != 'None':
                    # MERMAID API uses :: as the BA-GF separator.
                    bagf = '::'.join([
                        benthic_attribute_id, row['growth_form_id']])
                else:
                    bagf = benthic_attribute_id

                annotation = (
                    int(row['row']),
                    int(row['col']),
                    bagf,
                )
                image_annotations.append(annotation)
                image_annos_by_ba[benthic_attribute_id] += 1
                image_annos_by_bagf[bagf] += 1

            if (
                annotation_limit
                and
                all_annos_count + len(image_annotations) > annotation_limit
            ):
                # Adding this image's annotations would put us
                # over the annotation limit.
                return

            feature_full_path = f'{TRAINING_BUCKET}/{feature_bucket_path}'
            if feature_full_path not in mermaid_full_paths_in_s3:
                logger.warning(
                    f"Skipping feature vector because couldn't find"
                    f" the file in S3: {feature_bucket_path}")
                missing_features.append(feature_bucket_path)
                continue

            self.data[feature_bucket_path] = image_annotations
            all_annos_count += len(image_annotations)
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

    @staticmethod
    def data_rows_from_parquet(
        parquet_uri: str,
    ) -> typing.Generator['pandas.core.series.Series', None, None]:
        """
        Reads from a parquet file (chunkifying to avoid memory issues),
        and generates pandas dataframe rows.
        """
        duck_conn = duckdb.connect()

        try:
            duck_conn.load_extension('httpfs')
        except duckdb.IOException:
            # Extension not installed yet.
            duck_conn.install_extension('httpfs')
            duck_conn.load_extension('httpfs')

        if urlparse(parquet_uri).scheme == 's3':
            duck_conn.execute(f"SET s3_region='{AWS_REGION}'")
            # TODO: anon=True equiv. for DuckDB? or is it even needed?

        # TODO: Support getting specific rows rather than just *
        #  to reduce the amount of data that has to be read in.
        duck_rel = duck_conn.execute(
            f"SELECT * FROM read_parquet('{parquet_uri}')")

        # Fetch in chunks, not all at once, to avoid running out of
        # memory.
        # An example chunk size is 2048 rows x 12 columns.
        #
        # fetchmany() is similar, but returns lists of tuples instead.
        # A dataframe provides dict-like access which is a bit nicer.
        while True:
            dataframe = duck_rel.fetch_df_chunk()
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

    @classmethod
    def grouped_data_rows_from_parquet(
        cls,
        parquet_uri: str,
        grouping_column: str,
    ) -> typing.Generator['pandas.core.series.Series', None, None]:

        grouping_value = None
        group_rows = []

        for row in cls.data_rows_from_parquet(parquet_uri):
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

        # Not sure if logging the entirely of the inputs is worth it
        # (since it's a lot), but we can at least log the sizes of the
        # inputs.
        # They don't seem like 'metrics', nor are they strictly 'params'
        # or 'inputs', nor are they 'outputs'... so we just log as a dict.
        mlflow.log_dict(self.get_stats(), 'input_stats.yaml')

        # Log annotation count per BA and per BAGF.
        mlflow.log_dict(self.annos_by_ba, 'ba_counts.yaml')
        mlflow.log_dict(self.annos_by_bagf, 'bagf_counts.yaml')


def run_training(
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
    included_benthicattrs_csv
    excluded_benthicattrs_csv

    Local filepath of a CSV file, specifying either the MERMAID benthic
    attributes to accept from the training data (excluding all others), or
    the ones to leave out from the training data (including all others).
    (TODO: Also support S3 path for potentially easier bookkeeping/sharing)
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
    (TODO: Implement)

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
    # Not sure about saving this yet. It could at least be used to
    # generate and log a confusion matrix.
    valresult_loc = DataLocation('memory', key='valresult.json')

    train_msg = TrainClassifierMsg(
        job_token=f'experiment_run_{run_name}',
        trainer_name='minibatch',
        nbr_epochs=experiment_params['epochs'],
        clf_type='MLP',
        labels=training_dataset.labels,
        features_loc=DataLocation(
            's3', bucket_name=TRAINING_BUCKET, key=''),
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

        mlflow_connect()
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

            # TODO: Log confusion matrix?
            # TODO: Log valresult, or is it too big?

            # TODO: Ensure one of the env / requirements artifacts includes
            # the pyspacer version.

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
