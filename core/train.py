"""
Train a classifier using feature vectors and annotations
provided on S3.
"""
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path
import typing

try:
    import mlflow
    from mlflow.models import infer_signature
    MLFLOW_IMPORT_ERROR = None
except ImportError as err:
    MLFLOW_IMPORT_ERROR = err
# MLflow requires pandas.
# Might as well also use it for reading Parquet and CSV.
import pandas as pd
import psutil
from spacer.data_classes import ImageLabels
from spacer.messages import DataLocation, TrainClassifierMsg
from spacer.storage import load_classifier
from spacer.tasks import train_classifier
from spacer.task_utils import preprocess_labels, SplitMode

from benthic_attributes import BenthicAttrHierarchy
from utils import logging_config_for_script, mlflow_connect, Settings


logger = logging_config_for_script('train')


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

    _full_hierarchy = None

    @classmethod
    def get_hierarchy(cls):
        # Lazy-load the full BA hierarchy, and save it at the class
        # level so we only load it once at most.
        if cls._full_hierarchy is None:
            cls._full_hierarchy = BenthicAttrHierarchy()
        return cls._full_hierarchy

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
                ba_id = self.get_hierarchy().by_name[target_value]['id']
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
        for ancestor_id in self.get_hierarchy().get_ancestor_ids(ba_id):
            if ancestor_id in self.rollup_targets:
                return ancestor_id
        # Or don't roll up if there's no such ancestor.
        return ba_id


def run_training(
    included_benthicattrs_csv: str = None,
    excluded_benthicattrs_csv: str = None,
    benthicattr_rollup_targets_csv: str = None,
    drop_growthforms: bool = False,
    epochs: int = 10,
    experiment_name: str | None = None,
    model_name: str | None = None,
    disable_mlflow: bool = False,
    annotation_limit: int | None = None,
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

    annotation_limit

    If specified, only get up to this many annotations for training. This can
    help with testing since the runtime is correlated to number of annotations.

    TODO: Be able to specify an example image (or set of them?) whose
    training annotations are logged along with the model. Since logging all
    training annotations might be too much, but looking at a small sample
    is a good sanity check.
    """
    settings = Settings()
    experiment_name = (
        experiment_name or settings.MLFLOW_DEFAULT_EXPERIMENT_NAME)

    if included_benthicattrs_csv and excluded_benthicattrs_csv:
        raise ValueError(
            "Specify one of included benthic attrs or"
            " excluded benthic attrs, but not both.")

    if included_benthicattrs_csv:
        with open(included_benthicattrs_csv) as csv_f:
            benthicattr_filter = BenthicAttrFilter(csv_f, inclusion=True)
    elif excluded_benthicattrs_csv:
        with open(excluded_benthicattrs_csv) as csv_f:
            benthicattr_filter = BenthicAttrFilter(csv_f, inclusion=False)
    else:
        # No inclusion or exclusion set specified means we accept all labels.
        # In other words, an empty exclusion set.
        benthicattr_filter = BenthicAttrFilter(
            StringIO(''), inclusion=False)

    if benthicattr_rollup_targets_csv:
        with open(benthicattr_rollup_targets_csv) as csv_f:
            rollup_spec = BenthicAttrRollupSpec(csv_f)
    else:
        # Empty rollup-targets set, meaning nothing gets rolled up.
        rollup_spec = BenthicAttrRollupSpec(StringIO(''))

    # Accessing annotations the same way as
    # https://github.com/data-mermaid/image-classification-open-data
    #
    # TODO: This is just MERMAID so far; also get CoralNet
    dataframe = pd.read_parquet(
        f's3://{TRAINING_BUCKET}/mermaid/mermaid_confirmed_annotations.parquet',
        storage_options={"anon": True},
    )
    # Some ways to inspect the dataframe in a debugger
    # (may not be the best ways):
    # dataframe
    # dataframe.columns    # Column names
    # dataframe.iloc[0]    # First row
    # set([row['growth_form_name'] for _index, row in dataframe.iterrows()])
    # [(row['row'], row['col']) for _index, row in dataframe.iterrows()
    #  if row['image_id'] == '0032dba6-8357-42e2-bace-988f99032286']

    image_id = None
    labels_data = dict()
    ba_counts = Counter()
    bagf_counts = Counter()

    # TODO: Option to filter by CN source (and MERMAID equiv.?).
    #   Like a CSV with columns for site (CN or MERMAID)
    #   and source/project ID.
    # TODO: iterrows() likely won't work in terms of memory for very
    #   large DFs. But what's the alternative if we really want all
    #   rows' data?

    for index, row in dataframe.iterrows():
        if image_id != row['image_id']:
            # End of previous image's annotations, and start of
            # another image's annotations.
            image_id = row['image_id']
            feature_filepath = f'mermaid/{image_id}_featurevector'

            labels_data[feature_filepath] = []

        if not benthicattr_filter.accepts_ba(row['benthic_attribute_id']):
            # This BA is being filtered out of the training data.
            continue

        if annotation_limit and index >= annotation_limit:
            break

        benthic_attribute_id = rollup_spec.roll_up(row['benthic_attribute_id'])

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
        labels_data[feature_filepath].append(annotation)

        ba_counts[benthic_attribute_id] += 1
        bagf_counts[bagf] += 1

    labels = preprocess_labels(
        ImageLabels(labels_data),
        # 10% ref, 10% val, 80% train.
        split_ratios=(0.1, 0.1),
        split_mode=SplitMode.POINTS_STRATIFIED,
    )

    log_memory_usage('Memory usage after creating labels')
    input_stats = dict(
        total_annotations=labels.label_count,
        train_annotations=labels.train.label_count,
        ref_annotations=labels.ref.label_count,
        val_annotations=labels.val.label_count,
        num_of_images=len(labels_data),
        num_of_benthic_attributes=len(ba_counts),
        num_of_ba_gf_combinations=len(bagf_counts),
    )
    logger.info(
        "Proceeding to train with {total_annotations}"
        " annotations ({train_annotations} train,"
        " {ref_annotations} ref,"
        " {val_annotations} val) from"
        " {num_of_images} images."
        " {num_of_benthic_attributes} BAs and"
        " {num_of_ba_gf_combinations} BA-GF combos"
        " are represented here.".format(**input_stats)
    )

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
        labels=labels,
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

            if benthicattr_filter.inclusion:
                table_filename = 'included_benthicattrs.json'
            else:
                table_filename = 'excluded_benthicattrs.json'
            mlflow.log_table(
                benthicattr_filter.csv_dataframe, table_filename)

            mlflow.log_table(
                rollup_spec.csv_dataframe, 'rollup_spec.json')

            # Not sure if logging the entirely of the inputs is worth it
            # (since it's a lot), but we can at least log the sizes of the
            # inputs.
            # They don't seem like 'metrics', nor are they strictly 'params'
            # or 'inputs', nor are they 'outputs'... so we just log as a dict.
            mlflow.log_dict(input_stats, 'input_stats.yaml')

            # Log annotation count per BA and per BAGF.
            mlflow.log_dict(ba_counts, 'ba_counts.yaml')
            mlflow.log_dict(bagf_counts, 'bagf_counts.yaml')

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
