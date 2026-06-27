"""
Train a classifier using feature vectors and annotations
provided on S3.
"""

import os
import tempfile
import typing
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import psutil
from mlflow.tracking.fluent import run_id_to_system_metrics_monitor
from spacer.task_utils import preprocess_labels

from mermaid_classifier.common.benthic_attributes import (
    get_benthic_attribute_library,
    get_growth_form_library,
    split_ba_gf,
)
from mermaid_classifier.pyspacer._pipeline_utils import section_profiling
from mermaid_classifier.pyspacer.dataset import TrainingDataset
from mermaid_classifier.pyspacer.inference import (
    export_artifact,
    load_predictor,
)
from mermaid_classifier.pyspacer.metrics import (
    MetricsContext,
    MetricsCoordinator,
)
from mermaid_classifier.pyspacer.metrics._logging import (
    log_dataframe as _log_dataframe,
)
from mermaid_classifier.pyspacer.mlflow_model import log_artifact_model
from mermaid_classifier.pyspacer.options import (
    DatasetOptions,
    MLflowOptions,
    TrainingOptions,
)
from mermaid_classifier.pyspacer.settings import (
    set_env_vars_for_packages,
    settings,
    training_batch_size,
)
from mermaid_classifier.pyspacer.swap_monitor import SwapMonitor
from mermaid_classifier.pyspacer.trainer import MermaidTrainer
from mermaid_classifier.pyspacer.utils import logging_config_for_script, mlflow_connect
from mermaid_classifier.training.sample_weighting import compute_class_weights

logger = logging_config_for_script("train")


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

    dataset: TrainingDataset | None = None
    profiled_sections: list[dict[str, object]]

    def __init__(
        self,
        dataset_options: DatasetOptions | None = None,
        training_options: TrainingOptions | None = None,
    ):
        # Normalize Settings -> SPACER_*/MLFLOW_* env vars before any PySpacer
        # or MLflow work. This used to run as an import side effect of
        # mermaid_classifier.pyspacer; it now runs here (the programmatic entry
        # point for all training) and in the scripts/ mains. Idempotent.
        set_env_vars_for_packages()
        self.dataset_options = dataset_options or DatasetOptions()
        self.training_options = training_options or TrainingOptions()

    def run(self, run_name: str | None = None, cleanup_dataset: bool = True):
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

            num_classes = len(self.dataset.labels.ref.classes_set)

            if settings.spacer_batch_size is not None:
                batch_size = settings.spacer_batch_size
                logger.info(f"Batch size: {batch_size} (from SPACER_BATCH_SIZE)")
            else:
                batch_size, available_gb = training_batch_size(num_classes=num_classes)
                logger.info(
                    f"Batch size: {batch_size}"
                    f" (auto, based on {available_gb:.1f} GB"
                    f" available memory, {num_classes} classes)"
                )

            class_weight, weighting_log = self._compute_class_weights(
                self.dataset.labels,
            )
            self._weighting_log = weighting_log

            trainer = MermaidTrainer(
                batch_size=batch_size,
                on_epoch_end=self._on_epoch_end,
                class_weight=class_weight,
                early_stopping_patience=(self.training_options.early_stopping_patience),
            )

            # Train directly via the trainer — no pickle round-trip. pyspacer's
            # train_classifier task only adds label preprocessing + a pickle
            # store on top of this call; we keep the preprocessing and drop the
            # store. previous_model_locs was always [], so pc_models is [].
            labels = preprocess_labels(self.dataset.labels)
            with self.section_profiling("PySpacer training call"):
                clf_calibrated, val_results, return_msg = trainer(
                    labels, self.training_options.epochs, []
                )

            logger.info(f"Train time (from return msg): {return_msg.runtime:.1f} s")

            logger.info(f"New model's accuracy: {self.format_metric(return_msg.acc)}")

            ref_accs_str = ", ".join([str(self.format_metric(acc)) for acc in return_msg.ref_accs])
            logger.debug(f"Accuracy progression during training epochs: {ref_accs_str}")

            return return_msg, clf_calibrated, val_results
        finally:
            # When cleanup_dataset is False, the caller (MLflowTrainingRunner
            # .run) keeps the dataset's downloaded val features alive past this
            # method so it can eval the exported artifact against them, and is
            # responsible for cleanup afterward.
            if cleanup_dataset and self.dataset is not None:
                self.dataset.cleanup()

    def _on_epoch_end(self, metrics: dict[str, object]) -> None:
        """Called after each training epoch. Override for logging."""
        pass

    def _compute_class_weights(
        self,
        labels: typing.Any,
    ) -> tuple[dict[str, float] | None, dict[str, object]]:
        """Compute per-class loss weights from training-set class counts,
        using the configured ``DatasetOptions.weighting`` strategy.

        Returns ``(class_weight_or_none, weighting_log)``. ``class_weight``
        is a dict mapping class label -> weight, or None if weighting is
        disabled. ``weighting_log`` is a dict capturing per-class actions
        and summary stats for MLflow (consumed by MLflowTrainingRunner;
        the base TrainingRunner discards it).
        """
        opts = self.dataset_options.weighting
        if opts is None or not opts.enabled:
            return None, {"enabled": False}

        # Per-class training counts come straight from the ImageLabels
        # Counter — no extra disk/network IO.
        class_counts = dict(labels.train.label_count_per_class)

        weights = compute_class_weights(
            class_counts=class_counts,
            options=opts,
        )

        # Build the per-class log table (count + weight). The per-class
        # rare/keep decision now lives in the label-transforms plan
        # artifact, not here — this table only reports loss-weight
        # spread for the kept (post-transform) class set.
        rows = []
        for cls, count in class_counts.items():
            rows.append(
                {
                    "bagf_id": cls,
                    "count": int(count),
                    "weight": float(weights.get(cls, 0.0)),
                }
            )
        per_class_df = pd.DataFrame(rows)

        # Summary stats over the weights produced for the kept class set.
        weight_series = per_class_df["weight"]
        if len(weight_series) > 0 and weight_series.max() > 0:
            summary = {
                "weight_mean": float(weight_series.mean()),
                "weight_median": float(weight_series.median()),
                "weight_p5": float(weight_series.quantile(0.05)),
                "weight_p95": float(weight_series.quantile(0.95)),
                "weight_max_min_ratio": float(
                    weight_series.max() / max(weight_series.min(), 1e-12)
                ),
                "n_classes": int(len(per_class_df)),
            }
        else:
            summary = {
                "weight_mean": 0.0,
                "weight_median": 0.0,
                "weight_p5": 0.0,
                "weight_p95": 0.0,
                "weight_max_min_ratio": 0.0,
                "n_classes": int(len(per_class_df)),
            }

        return weights, {
            "enabled": True,
            "options": opts,
            "per_class_df": per_class_df,
            "summary": summary,
        }

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
        return current_time.strftime("%Y%m%dT%H%M%S")

    @staticmethod
    def format_metric(metric: float):
        # The input may be a numpy float, which may not serialize well
        # in artifacts. So, convert to a regular float, using float().
        return round(float(metric), 3)


class MLflowTrainingRunner(TrainingRunner):
    def __init__(
        self, *args: typing.Any, mlflow_options: MLflowOptions | None = None, **kwargs: typing.Any
    ) -> None:
        # Normalize Settings -> SPACER_*/MLFLOW_* env vars *before* the first
        # mlflow_connect() below, which needs MLFLOW_HTTP_REQUEST_MAX_RETRIES to
        # be set (otherwise a failed initial connection retries far more times
        # than configured). This used to be set as an import side effect of
        # mermaid_classifier.pyspacer; super().__init__() also calls it, but
        # that runs after mlflow_connect(). Idempotent, so the double call is
        # harmless.
        set_env_vars_for_packages()

        time_taken = mlflow_connect()
        logger.info(f"Time to connect to MLflow tracking: {time_taken}")

        super().__init__(*args, **kwargs)
        self.mlflow_options = mlflow_options or MLflowOptions()

    def run(self, run_name: str | None = None, cleanup_dataset: bool = True) -> typing.Any:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]  # intentionally narrows cleanup_dataset; MLflow runner always manages cleanup itself

        model_name = self._get_model_name()
        if run_name is None:
            run_name = f"{model_name}-{self.current_time_str()}"

        logger.info(f"Experiment: {self.mlflow_options.experiment_name}")
        mlflow.enable_system_metrics_logging()
        mlflow.set_experiment(self.mlflow_options.experiment_name)

        return_msg: typing.Any = None
        model_info: typing.Any = None

        with mlflow.start_run(run_name=run_name):
            # Add swap monitoring to MLflow's system metrics polling loop.
            run_id = mlflow.active_run().info.run_id  # pyright: ignore[reportOptionalMemberAccess]  # active_run() is non-None inside start_run() context
            if run_id in run_id_to_system_metrics_monitor:
                run_id_to_system_metrics_monitor[run_id].monitors.append(SwapMonitor())

            training_options_to_log = {
                "epochs": self.training_options.epochs,
                "early_stopping_patience": (
                    self.training_options.early_stopping_patience
                    if self.training_options.early_stopping_patience is not None
                    else ""
                ),
            }

            mlflow.log_params(training_options_to_log)

            dataset_options_to_log = {
                "include_mermaid": self.dataset_options.include_mermaid,
                "coralnet_sources_csv": os.path.basename(
                    self.dataset_options.coralnet_sources_csv or ""
                ),
                "drop_growthforms": self.dataset_options.drop_growthforms,
                "label_rollup_spec_csv": os.path.basename(
                    self.dataset_options.label_rollup_spec_csv or ""
                ),
                "excluded_labels_csv": os.path.basename(
                    self.dataset_options.excluded_labels_csv or ""
                ),
                "included_labels_csv": os.path.basename(
                    self.dataset_options.included_labels_csv or ""
                ),
                "ref_val_ratios": str(self.dataset_options.ref_val_ratios),
            }
            mlflow.log_params(dataset_options_to_log)

            # Subsample params: logged before training so they show up
            # alongside dataset/training params even if the training
            # run later fails. Per-class realized counts are logged as
            # an artifact after dataset prep (see _log_subsample_audit).
            if self.dataset_options.subsample is not None:
                mlflow.log_params(self.dataset_options.subsample.to_log_dict())
            else:
                mlflow.log_params({"subsample/enabled": False})

            # Sample-weighting params: logged before training so they
            # show up alongside dataset/training params even if the
            # training run later fails.
            if self.dataset_options.weighting is not None:
                mlflow.log_params(self.dataset_options.weighting.to_log_dict())
            else:
                mlflow.log_params({"weighting/enabled": False})

            self.log_system_specs()

            # Here's the actual training and data prep. cleanup_dataset=False
            # keeps the downloaded val features alive so we can eval the
            # exported artifact against them below; the finally cleans up.
            try:
                return_msg, clf_calibrated, val_results = super().run(
                    run_name=run_name, cleanup_dataset=False
                )
                assert self.dataset is not None  # set by base TrainingRunner.run()

                # Weighting artifacts/metrics are stashed by the base run()
                # via _compute_class_weights. Log them now.
                self._log_weighting_artifacts()
                self._log_subsample_audit()

                profiles_df = pd.DataFrame(self.profiled_sections)
                self.log_dataframe(profiles_df, "profiled_sections")

                # val_results now comes from the trainer in memory (no reload).
                mlflow.log_dict(val_results.serialize(), "valresult.json")

                # Eval-the-artifact: export the deployable TorchScript artifact
                # and evaluate THAT, so the logged metrics reflect what actually
                # ships. The parity reference batch is the first val batch (real
                # features, loaded the same way the coordinator loads val data).
                ref_batch = next(iter(self.dataset.labels.val.load_data_in_batches()), None)
                if ref_batch is None:
                    raise RuntimeError(
                        "Val split yielded no feature batch; refusing to export"
                        " an unverified artifact."
                    )
                # load_data_in_batches yields zip(*pairs); unpacking gives a
                # (features, labels) pair of tuples, same as the metrics
                # coordinator consumes. We only need the feature vectors.
                ref_features, _ref_labels = ref_batch

                ba_library = get_benthic_attribute_library()
                gf_library = get_growth_form_library()

                with tempfile.TemporaryDirectory() as artifact_dir:
                    artifact_dir = Path(artifact_dir)
                    # Parity-gated export (ParityError if max|Δ| > 1e-6).
                    model_pt, _manifest, _max_diff = export_artifact(
                        clf_calibrated,
                        artifact_dir,
                        reference_features=ref_features,
                        config={"patch_size": 224},
                    )
                    model_json = artifact_dir / "model.json"
                    # ManifestError on schema/class-count/input_dim mismatch.
                    predictor = load_predictor(model_pt, model_json)

                    ctx = MetricsContext(
                        val_results=val_results,
                        ba_library=ba_library,
                        gf_library=gf_library,
                        format_func=self.format_metric,
                        dataset=self.dataset,
                        clf=predictor,
                    )

                    coordinator = MetricsCoordinator(ctx, duck_conn=self.dataset.duck_conn)
                    coordinator.compute_and_log_all()

                    # Accuracy and ref_accs come from pyspacer's return_msg, not
                    # our metrics module (training-progress, not artifact-based).
                    mlflow.log_metric("accuracy", self.format_metric(return_msg.acc))
                    ref_accs_dict = {
                        epoch: self.format_metric(acc)
                        for epoch, acc in enumerate(return_msg.ref_accs, 1)
                    }
                    mlflow.log_dict(ref_accs_dict, "epoch_ref_accuracies.yaml")  # pyright: ignore[reportArgumentType]  # mlflow log_dict expects dict[str, Any] but int keys are valid in YAML

                    # Store the deployable artifact (model.pt + model.json) as
                    # the registered model via the pyfunc shim — one loader
                    # everywhere.
                    signature = mlflow.models.infer_signature(params=training_options_to_log)  # pyright: ignore[reportPrivateImportUsage]  # mlflow.models is a submodule, not formally re-exported
                    model_info = log_artifact_model(
                        model_pt,
                        model_json,
                        registered_model_name=model_name,
                        signature=signature,
                    )
            finally:
                if getattr(self, "dataset", None) is not None:
                    self.dataset.cleanup()  # pyright: ignore[reportOptionalMemberAccess]  # getattr guard above ensures non-None

        logger.info(f"Model ID: {model_info.model_id}")

        return return_msg, model_info

    def _on_epoch_end(self, metrics: dict[str, Any]) -> None:
        """Log per-epoch metrics to MLflow.

        Logs step-based metrics that appear as live charts in the MLflow UI
        during training. The val_loss + val_accuracy pair is the
        canonical overfitting detector: when training_loss continues to
        drop while val_loss begins to rise, the model is starting to
        memorize the training set rather than generalize.

        On the final epoch (whether the run hit the epoch budget or
        triggered early stopping) the trainer adds summary fields
        (`final_epoch`, `early_stopped`, `best_val_epoch`,
        `best_val_loss`); we log them as flat scalar metrics so they're
        easy to query post-hoc.
        """
        step = int(metrics["epoch"])  # type: ignore[arg-type]
        mlflow.log_metric("epoch/ref_accuracy", float(metrics["ref_accuracy"]), step=step)  # type: ignore[arg-type]
        if metrics.get("val_accuracy") is not None:
            mlflow.log_metric("epoch/val_accuracy", float(metrics["val_accuracy"]), step=step)  # type: ignore[arg-type]
        if metrics.get("val_loss") is not None:
            mlflow.log_metric("epoch/val_loss", float(metrics["val_loss"]), step=step)  # type: ignore[arg-type]
        if metrics["training_loss"] is not None:
            mlflow.log_metric("epoch/training_loss", float(metrics["training_loss"]), step=step)  # type: ignore[arg-type]
        mlflow.log_metric(
            "epoch/cumulative_seconds", float(metrics["cumulative_seconds"]), step=step
        )  # type: ignore[arg-type]

        # One-shot early-stopping summary (only present on the final
        # epoch). Logged as scalar metrics with no step so they appear
        # alongside the other run-level numbers.
        if metrics.get("final_epoch") is not None:
            mlflow.log_metric("early_stop/final_epoch", float(metrics["final_epoch"]), step=0)  # type: ignore[arg-type]
            mlflow.log_metric(
                "early_stop/triggered", float(bool(metrics.get("early_stopped"))), step=0
            )
            if metrics.get("best_val_epoch") is not None:
                mlflow.log_metric(
                    "early_stop/best_val_epoch",
                    float(metrics["best_val_epoch"]),
                    step=0,  # type: ignore[arg-type]
                )
            if metrics.get("best_val_loss") is not None:
                mlflow.log_metric(
                    "early_stop/best_val_loss",
                    float(metrics["best_val_loss"]),
                    step=0,  # type: ignore[arg-type]
                )

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
                model_name = f"Include{self.alphanumeric_only_str(as_path.stem)}"
            elif self.dataset_options.excluded_labels_csv:
                as_path = Path(self.dataset_options.excluded_labels_csv)
                model_name = f"Exclude{self.alphanumeric_only_str(as_path.stem)}"
            else:
                model_name = "AllLabels"

            if self.dataset_options.label_rollup_spec_csv:
                as_path = Path(self.dataset_options.label_rollup_spec_csv)
                model_name += f"-Rollup{self.alphanumeric_only_str(as_path.stem)}"

            if self.dataset_options.coralnet_sources_csv:
                as_path = Path(self.dataset_options.coralnet_sources_csv)
                model_name += f"-{self.alphanumeric_only_str(as_path.stem)}"

            if (subsample := self.dataset_options.subsample) is not None:
                # e.g. -SubStratified400000 or -SubBalanced1770000
                model_name += f"-Sub{subsample.strategy.capitalize()}{subsample.total_annotations}"

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
        return "".join([char for char in s if char.isalnum()])

    def _log_weighting_artifacts(self) -> None:
        """Log per-class weight artifacts and summary metrics gathered by
        the base ``TrainingRunner._compute_class_weights`` call.

        Logs nothing if weighting was disabled. Decorates the per-class
        DataFrame with human-readable BA and GF names from the same
        libraries the rest of the runner uses for taxonomy resolution.
        """
        weighting_log = getattr(self, "_weighting_log", None)
        if not weighting_log or not weighting_log.get("enabled"):
            return

        df = weighting_log["per_class_df"].copy()

        # Decorate with BA/GF names. Defensive: if any label fails to
        # parse, leave the name columns blank rather than failing the run.
        ba_library = get_benthic_attribute_library()
        gf_library = get_growth_form_library()

        def _decorate(row: pd.Series) -> pd.Series:  # type: ignore[type-arg]
            try:
                ba_id, gf_id = split_ba_gf(str(row["bagf_id"]))
                ba_name = ba_library.id_to_name(ba_id) if ba_id else ""
                gf_name = gf_library.id_to_name(gf_id) if gf_id else ""
            except Exception:
                ba_id, gf_id, ba_name, gf_name = "", "", "", ""
            return pd.Series(
                {
                    "ba_id": ba_id,
                    "gf_id": gf_id,
                    "ba_name": ba_name,
                    "gf_name": gf_name,
                }
            )

        decorated = df.apply(_decorate, axis=1)
        df = pd.concat([df, decorated], axis=1)
        df = df[  # type: ignore[index]
            [
                "bagf_id",
                "ba_id",
                "ba_name",
                "gf_id",
                "gf_name",
                "count",
                "weight",
            ]
        ].sort_values("weight", ascending=False)  # pyright: ignore[reportCallIssue]  # pandas DataFrame.sort_values overload issue with column-subscript result

        self.log_dataframe(df, "weighting/per_class_weights")

        summary = weighting_log["summary"]
        for k, v in summary.items():
            mlflow.log_metric(f"weighting/{k}", float(v), step=0)

    def _log_subsample_audit(self) -> None:
        """Log the per-class audit CSV and a single realized-total metric
        produced by ``TrainingDataset._apply_subsample``.

        Logs nothing if subsampling was disabled. The CSV is the
        after-the-fact proof that two parallel runs trained on the same
        rows: identical strategy + identical inputs must produce
        identical (target_n, realized_n) per class. If you ever extend
        this with a new strategy (e.g. effective_number) or a coarser
        ``stratification_level``, the same audit covers it without
        changes here.
        """
        df = getattr(self.dataset, "_subsample_audit_df", None)
        if df is None:
            return
        # Decorate with human-readable names so the CSV is self-contained.
        ba_library = get_benthic_attribute_library()
        gf_library = get_growth_form_library()

        def _decorate(row: pd.Series) -> pd.Series:  # type: ignore[type-arg]
            try:
                _ba_id = str(row["benthic_attribute_id"])
                _gf_id = str(row["growth_form_id"])
                ba_name = ba_library.id_to_name(_ba_id) if _ba_id else ""
                gf_name = gf_library.id_to_name(_gf_id) if _gf_id else ""
            except Exception:
                ba_name, gf_name = "", ""
            return pd.Series({"ba_name": ba_name, "gf_name": gf_name})

        if not df.empty:
            decorated = df.apply(_decorate, axis=1)
            out = pd.concat([df, decorated], axis=1)
            out = out[  # type: ignore[index]
                [
                    "benthic_attribute_id",
                    "ba_name",
                    "growth_form_id",
                    "gf_name",
                    "pre_count",
                    "target_n",
                    "realized_n",
                ]
            ].sort_values("realized_n", ascending=False)  # pyright: ignore[reportCallIssue]  # pandas DataFrame.sort_values overload issue with column-subscript result
            self.log_dataframe(out, "subsample/per_class_counts")

        realized = getattr(self.dataset, "_subsample_realized_total", None)
        if realized is not None:
            mlflow.log_metric("subsample/realized_total", float(realized), step=0)

    @staticmethod
    def _existing_ancestor(path: str) -> str:
        """Walk up from path until an existing directory is found."""
        p = path
        while not os.path.exists(p):
            parent = os.path.dirname(p)
            if parent == p:
                return "/"
            p = parent
        return p

    def log_system_specs(self):
        mlflow.log_dict(
            {
                "total_ram_gb": psutil.virtual_memory().total / 10**9,
                "free_storage_gb": psutil.disk_usage(
                    self._existing_ancestor(settings.feature_cache_dir or "/")
                ).free
                / 10**9,
            },
            "system_specs.yaml",
        )

    def log_dataset_artifacts(self):
        """
        Log various options and stats for the training dataset.
        """
        assert self.dataset is not None

        artifacts = self.dataset.artifacts

        mlflow.log_text(self.dataset.cn_source_filter.csv_text, "coralnet_sources_included.csv")

        if self.dataset.label_filter.inclusion:
            csv_filename = "labels_included.csv"
        else:
            csv_filename = "labels_excluded.csv"
        mlflow.log_text(self.dataset.label_filter.csv_text, csv_filename)

        mlflow.log_text(self.dataset.rollup_spec.csv_text, "rollup_spec.csv")

        # Number of images and annotations from each CN source and from
        # MERMAID.
        # First, before filtering (this is what's present in S3).
        # https://pandas.pydata.org/docs/reference/api/pandas.concat.html
        self.log_dataframe(
            pd.concat(
                [
                    artifacts.mermaid_project_stats,
                    artifacts.coralnet_project_stats,
                ]
            ),
            "project_stats_raw",
        )
        # And here, after filtering (this is what training actually gets).
        self.log_dataframe(
            self.dataset.compute_project_stats(has_training_sets=True), "project_stats_train_data"
        )

        mlflow.log_dict(artifacts.train_summary_stats, "train_summary.yaml")

        self.log_dataframe(artifacts.ba_counts, "ba_counts")
        self.log_dataframe(artifacts.bagf_counts, "bagf_counts")

        if not self.dataset.cn_source_filter.is_empty():
            # These only apply if CoralNet data is included.
            self.log_dataframe(artifacts.coralnet_label_mapping, "coralnet_label_mapping")
            self.log_dataframe(artifacts.unmapped_labels, "unmapped_labels")

        # Log extra annotations, if specified.
        if self.mlflow_options.extra_annotations_to_log is not None:
            log_spec = self.mlflow_options.extra_annotations_to_log.lower()
            df = self.dataset.get_annotations(log_spec)

            self.log_dataframe(df, f"annotations_{log_spec}")

        # Always log the validation split annotations so others can
        # independently re-evaluate the model.
        val_annotations_df = self.dataset.duck_conn.execute(
            "SELECT * FROM annotations WHERE training_set = 'val'"
        ).fetch_df()
        self.log_dataframe(val_annotations_df, "annotations_val")

    def log_dataframe(self, df: pd.DataFrame, filestem: str) -> None:
        """
        MLflow's log_table() saves a .json instead of .csv, which means
        the resulting artifact file cannot be inspected readily with an
        external program such as Excel / LibreOffice Calc.
        To save a .csv, we use log_text() instead, with this helper function.
        """
        assert self.dataset is not None
        _log_dataframe(self.dataset.duck_conn, df, filestem)
