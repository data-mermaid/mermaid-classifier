"""Orchestrates all metric groups and handles MLflow logging."""

import logging
import sys

import duckdb
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from mermaid_classifier.pyspacer.metrics._context import (
    MetricsContext,
    MetricsContextError,
)
from mermaid_classifier.pyspacer.metrics._logging import log_dataframe
from mermaid_classifier.pyspacer.metrics._results import MetricGroupResult
from mermaid_classifier.pyspacer.metrics._taxonomy_helpers import (
    build_ba_paths,
    build_ba_to_top,
)

# The compute_* names below are used indirectly: _get_metric_groups() resolves
# them from this module's namespace at call time (sys.modules[__name__]) so
# that test mocks targeting coordinator.<func_name> are honoured.
# mock.patch replaces names in the module __dict__; the sys.modules lookup sees
# those replacements, injecting failures into the correct call site.
from mermaid_classifier.pyspacer.metrics.calibration import (
    compute_calibration,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from mermaid_classifier.pyspacer.metrics.classification import (
    compute_balanced_accuracy_mcc,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    compute_confusion_matrices,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    compute_precision_recall_f1,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from mermaid_classifier.pyspacer.metrics.cover import (
    compute_cover,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from mermaid_classifier.pyspacer.metrics.per_source import (
    compute_per_source,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from mermaid_classifier.pyspacer.metrics.probability import (
    compute_probability,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from mermaid_classifier.pyspacer.metrics.ranking import (
    compute_ranking,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from mermaid_classifier.pyspacer.metrics.registry import METRIC_GROUPS, MetricGroupFunc
from mermaid_classifier.pyspacer.metrics.taxonomic import (
    compute_taxonomic,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)

logger = logging.getLogger(__name__)


class MetricsCoordinator:
    """Run all applicable metric groups and log results to MLflow."""

    def __init__(
        self,
        ctx: MetricsContext,
        duck_conn: duckdb.DuckDBPyConnection,
    ):
        self.ctx = ctx
        self.duck_conn = duck_conn

    def compute_and_log_all(self):
        """Run all applicable metric groups and log to MLflow."""
        try:
            self.ctx.validate()
        except MetricsContextError:
            logger.warning("Metrics skipped: context validation failed", exc_info=True)
            return

        # val_results.classes is list[LabelId] (int|str); MERMAID always uses str.
        classes_str: list[str] = self.ctx.val_results.classes  # pyright: ignore[reportAssignmentType]
        self.ctx.ba_to_top = build_ba_to_top(classes_str, self.ctx.ba_library)
        self.ctx.ba_paths = build_ba_paths(classes_str, self.ctx.ba_library)

        if self.ctx.clf is not None and self.ctx.dataset is not None:
            self._precompute_probabilities()

        for name, func in self._get_metric_groups():
            try:
                result = func(self.ctx)
                self._log_result(result)
            except Exception:
                logger.warning(f"Metric group '{name}' failed", exc_info=True)

    def _precompute_probabilities(self):
        """Pre-compute the full probability matrix for val set.

        Sets ctx.val_proba and ctx.val_gt_labels. If this fails,
        probability and ranking metrics will be skipped (gated by
        val_proba being None).
        """
        try:
            assert (
                self.ctx.dataset is not None
            )  # guarded by caller: only called when dataset is not None
            all_proba = []
            all_gt = []
            for batch_x, batch_y in self.ctx.dataset.labels.val.load_data_in_batches():
                all_proba.append(self.ctx.clf.predict_proba(batch_x))
                all_gt.extend(batch_y)
            self.ctx.val_proba = np.vstack(all_proba)
            self.ctx.val_gt_labels = all_gt
        except Exception:
            logger.warning(
                "Failed to pre-compute probability matrix; "
                "probability and ranking metrics will be skipped",
                exc_info=True,
            )

    def _get_metric_groups(self) -> list[tuple[str, MetricGroupFunc]]:
        """Return ordered list of (name, func).

        Skip groups whose required inputs aren't available.

        Functions are resolved through this module's namespace at call time
        so that test mocks targeting coordinator.<func_name> are honoured.
        mock.patch replaces names in the module __dict__; sys.modules lookup
        sees those replacements.
        """
        mod = sys.modules[__name__]
        groups: list[tuple[str, MetricGroupFunc]] = []
        for spec in METRIC_GROUPS:
            if spec.requires_dataset and self.ctx.dataset is None:
                continue
            if spec.requires_val_proba and self.ctx.val_proba is None:
                continue
            func: MetricGroupFunc = getattr(mod, spec.func.__name__)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            groups.append((spec.name, func))
        return groups

    def _log_result(self, result: MetricGroupResult):
        """Log all parts of a MetricGroupResult to MLflow."""
        for scalar in result.scalars:
            # Skip NaN values: MLflow's log_model re-logs run metrics to
            # associate them with the model, and its dedup-on-conflict filter
            # can't recognize an already-logged NaN (NaN != NaN), so it retries
            # the insert and crashes on the metrics UNIQUE constraint.
            if np.isnan(scalar.value):
                logger.warning(
                    "Skipping metric %r with non-finite value %r", scalar.name, scalar.value
                )
                continue
            mlflow.log_metric(scalar.name, scalar.value)

        for df_result in result.dataframes:
            log_dataframe(self.duck_conn, df_result.df, df_result.artifact_path)

        for dict_result in result.dicts:
            mlflow.log_dict(dict_result.data, dict_result.artifact_path)

        for fig_result in result.figures:
            try:
                mlflow.log_figure(fig_result.fig, fig_result.artifact_path)
            finally:
                plt.close(fig_result.fig)
