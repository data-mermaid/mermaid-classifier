"""
Train a classifier using feature vectors and annotations
provided on S3.
"""
import dataclasses
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path

try:
    import mlflow
    MLFLOW_IMPORT_ERROR = None
except ImportError as err:
    MLFLOW_IMPORT_ERROR = err
import pandas as pd
import psutil
from mlflow.tracking.fluent import run_id_to_system_metrics_monitor
from mermaid_classifier.pyspacer.swap_monitor import SwapMonitor
from spacer.data_classes import DataLocation, ValResults
from spacer.messages import TrainClassifierMsg
from spacer.storage import load_classifier

from spacer.tasks import train_classifier as spacer_train_classifier

from mermaid_classifier.pyspacer.config import (
    DatasetOptions,
    MLflowOptions,
    TrainingOptions,
)
from mermaid_classifier.pyspacer.dataset import (
    TrainingDataset,
    ba_library,
    gf_library,
    section_profiling,
)
from mermaid_classifier.pyspacer.settings import training_batch_size
from mermaid_classifier.pyspacer.trainer import MermaidTrainer
from mermaid_classifier.pyspacer.metrics import (
    MetricsContext, MetricsCoordinator,
)
from mermaid_classifier.pyspacer.metrics._logging import (
    log_dataframe as _log_dataframe,
)
from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.pyspacer.utils import (
    logging_config_for_script, mlflow_connect)


logger = logging_config_for_script('train')


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

        try:
            self.dataset = TrainingDataset(self.dataset_options)

            # The dataset's profiled sections are done. The runner will
            # add the remaining profiled sections.
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

            if self.training_options.io_batch_size is not None:
                io_batch_size = self.training_options.io_batch_size
                logger.info(
                    f"IO batch size: {io_batch_size}"
                    f" (from TrainingOptions)")
            elif settings.spacer_batch_size is not None:
                io_batch_size = int(settings.spacer_batch_size)
                logger.info(
                    f"IO batch size: {io_batch_size}"
                    f" (from SPACER_BATCH_SIZE)")
            else:
                num_classes = len(self.dataset.labels.ref.classes_set)
                io_batch_size, available_gb = training_batch_size(
                    num_classes=num_classes)
                logger.info(
                    f"IO batch size: {io_batch_size}"
                    f" (auto, based on {available_gb:.1f} GB"
                    f" available memory, {num_classes} classes)")

            trainer = MermaidTrainer(
                io_batch_size=io_batch_size,
                minibatch_size=self.training_options.minibatch_size,
                on_epoch_end=self._on_epoch_end,
                class_balancing=self.training_options.class_balancing,
                device=self.training_options.device,
                optimizer=self.training_options.optimizer,
                learning_rate=self.training_options.learning_rate,
                weight_decay=self.training_options.weight_decay,
                hidden_layer_sizes=self.training_options.hidden_layer_sizes,
                feature_cache_dir=self.dataset._feature_dir,
            )

            train_msg = TrainClassifierMsg(
                job_token=f'experiment_run_{run_name}',
                trainer=trainer,
                nbr_epochs=self.training_options.epochs,
                clf_type='MLP',
                labels=self.dataset.labels,
                previous_model_locs=[],
                model_loc=model_loc,
                valresult_loc=valresult_loc,
                feature_cache_dir=TrainClassifierMsg.FeatureCache.DISABLED,
            )

            with self.section_profiling("PySpacer training call"):
                return_msg = spacer_train_classifier(train_msg)

            logger.info(
                f"Train time (from return msg):"
                f" {return_msg.runtime:.1f} s")

            logger.info(
                f"New model's accuracy:"
                f" {self.format_metric(return_msg.acc)}")

            ref_accs_str = ", ".join(
                [str(self.format_metric(acc))
                 for acc in return_msg.ref_accs])
            logger.debug(
                f"Accuracy progression during training epochs:"
                f" {ref_accs_str}")

            return return_msg, model_loc, valresult_loc
        finally:
            if self.dataset is not None:
                self.dataset.cleanup()

    def _on_epoch_end(self, metrics: dict):
        """Called after each training epoch. Override for logging."""
        pass

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
    def format_metric(metric: float):
        # The input may be a numpy float, which may not serialize well
        # in artifacts. So, convert to a regular float, using float().
        return round(float(metric), 3)


def _flatten_dataclass_for_logging(instance, prefix=''):
    """Convert a dataclass to a flat dict for mlflow.log_params().
    Path-like fields (name contains 'csv' or 'path') log as basename only.
    Tuples become strings, None becomes empty string."""
    result = {}
    for field in dataclasses.fields(instance):
        key = f"{prefix}{field.name}" if prefix else field.name
        value = getattr(instance, field.name)
        if value is None:
            value = ''
        elif isinstance(value, (tuple, list)):
            value = str(value)
        elif isinstance(value, str) and (
                'csv' in field.name or 'path' in field.name):
            value = os.path.basename(value) if value else ''
        result[key] = value
    return result


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
        mlflow.enable_system_metrics_logging()
        mlflow.set_experiment(self.mlflow_options.experiment_name)

        with mlflow.start_run(run_name=run_name):

            # Add swap monitoring to MLflow's system metrics polling loop.
            run_id = mlflow.active_run().info.run_id
            if run_id in run_id_to_system_metrics_monitor:
                run_id_to_system_metrics_monitor[run_id].monitors.append(SwapMonitor())

            training_params = _flatten_dataclass_for_logging(
                self.training_options, prefix='training.')
            mlflow.log_params(training_params)

            dataset_params = _flatten_dataclass_for_logging(
                self.dataset_options, prefix='dataset.')
            mlflow.log_params(dataset_params)

            self.log_system_specs()

            # Here's the actual training and data prep.
            return_msg, model_loc, valresult_loc = super().run(
                run_name=run_name)

            profiles_df = pd.DataFrame(self.profiled_sections)
            self.log_dataframe(profiles_df, 'profiled_sections')

            val_results = ValResults.load(valresult_loc)
            mlflow.log_dict(val_results.serialize(), 'valresult.json')

            clf = load_classifier(model_loc)

            ctx = MetricsContext(
                val_results=val_results,
                ba_library=ba_library,
                gf_library=gf_library,
                format_func=self.format_metric,
                dataset=self.dataset,
                clf=clf,
            )

            coordinator = MetricsCoordinator(
                ctx, duck_conn=self.dataset.duck_conn)
            coordinator.compute_and_log_all()

            # Accuracy and ref_accs come from pyspacer's return_msg,
            # not our metrics module.
            mlflow.log_metric(
                'accuracy', self.format_metric(return_msg.acc))
            ref_accs_dict = {
                epoch: self.format_metric(acc)
                for epoch, acc in enumerate(return_msg.ref_accs, 1)
            }
            mlflow.log_dict(ref_accs_dict, 'epoch_ref_accuracies.yaml')

            # Save and register the trained model.
            signature = mlflow.models.infer_signature(
                params=training_params)
            model_info = mlflow.sklearn.log_model(
                sk_model=clf,
                registered_model_name=model_name,
                signature=signature,
            )

        logger.info(f"Model ID: {model_info.model_id}")

        return return_msg, model_loc

    def _on_epoch_end(self, metrics: dict):
        """Log per-epoch metrics to MLflow.

        Logs step-based metrics that appear as live charts in the MLflow UI
        during training.
        """
        step = metrics["epoch"]
        mlflow.log_metric(
            "epoch/ref_accuracy", metrics["ref_accuracy"], step=step)
        if metrics["training_loss"] is not None:
            mlflow.log_metric(
                "epoch/training_loss", metrics["training_loss"], step=step)
        mlflow.log_metric(
            "epoch/cumulative_seconds",
            metrics["cumulative_seconds"], step=step)

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

    @staticmethod
    def _existing_ancestor(path: str) -> str:
        """Walk up from path until an existing directory is found."""
        p = path
        while not os.path.exists(p):
            parent = os.path.dirname(p)
            if parent == p:
                return '/'
            p = parent
        return p

    def log_system_specs(self):
        mlflow.log_dict(
            dict(
                total_ram_gb=psutil.virtual_memory().total / 10**9,
                free_storage_gb=psutil.disk_usage(
                    self._existing_ancestor(
                        settings.feature_cache_dir or '/')
                ).free / 10**9,
            ),
            'system_specs.yaml',
        )

    def log_dataset_artifacts(self):
        """
        Log various options and stats for the training dataset.
        """
        assert self.dataset is not None

        artifacts = self.dataset.artifacts

        mlflow.log_text(
            self.dataset.cn_source_filter.csv_text,
            'coralnet_sources_included.csv')

        if self.dataset.label_filter.inclusion:
            csv_filename = 'labels_included.csv'
        else:
            csv_filename = 'labels_excluded.csv'
        mlflow.log_text(
            self.dataset.label_filter.csv_text, csv_filename)

        mlflow.log_text(
            self.dataset.rollup_spec.csv_text, 'rollup_spec.csv')

        # Number of images and annotations from each CN source and from
        # MERMAID.
        # First, before filtering (this is what's present in S3).
        # https://pandas.pydata.org/docs/reference/api/pandas.concat.html
        self.log_dataframe(
            pd.concat([
                artifacts.mermaid_project_stats,
                artifacts.coralnet_project_stats,
            ]),
            'project_stats_raw')
        # And here, after filtering (this is what training actually gets).
        self.log_dataframe(
            self.dataset.compute_project_stats(has_training_sets=True),
            'project_stats_train_data')

        mlflow.log_dict(
            artifacts.train_summary_stats, 'train_summary.yaml')

        self.log_dataframe(artifacts.ba_counts, 'ba_counts')
        self.log_dataframe(artifacts.bagf_counts, 'bagf_counts')

        if not self.dataset.cn_source_filter.is_empty():
            # These only apply if CoralNet data is included.
            self.log_dataframe(
                artifacts.coralnet_label_mapping,
                'coralnet_label_mapping')
            self.log_dataframe(
                artifacts.unmapped_labels,
                'unmapped_labels')

        # Log annotations, if specified.
        if self.mlflow_options.annotations_to_log is not None:
            log_spec = self.mlflow_options.annotations_to_log.lower()
            df = self.dataset.get_annotations(log_spec)

            self.log_dataframe(df, f'annotations_{log_spec}')

        # Always log the validation split annotations so others can
        # independently re-evaluate the model.
        val_annotations_df = self.dataset.duck_conn.execute(
            "SELECT * FROM annotations WHERE training_set = 'val'"
        ).fetch_df()
        self.log_dataframe(val_annotations_df, 'annotations_val')

    def log_dataframe(self, df, filestem):
        """
        MLflow's log_table() saves a .json instead of .csv, which means
        the resulting artifact file cannot be inspected readily with an
        external program such as Excel / LibreOffice Calc.
        To save a .csv, we use log_text() instead, with this helper function.
        """
        _log_dataframe(self.dataset.duck_conn, df, filestem)
