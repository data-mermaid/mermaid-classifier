"""Characterization tests for MetricsCoordinator.compute_and_log_all().

The coordinator orchestrates all metric groups and logs results to MLflow.
These tests characterize:

1. Happy path: compute_and_log_all() completes, records ≥1 mlflow.log_metric call,
   and a known metric name (precision_macro) appears.
2. Per-group error isolation: if one metric group raises, the coordinator catches it
   and still logs metrics from other groups — this is the key invariant that
   issue #73's registry refactor must preserve.
3. Invalid context: if ctx.validate() fails, compute_and_log_all() returns early
   without raising and logs no metrics.

No clf or dataset is provided in any test, so only the always-available groups run
(confusion_matrices, precision_recall_f1, balanced_accuracy_mcc, taxonomic,
calibration). This keeps the test offline and dependency-free.
"""

import unittest
from unittest import mock

import duckdb

from mermaid_classifier.pyspacer.metrics import MetricsContext, MetricsCoordinator
from pyspacer.metrics_test_helpers import (
    MockBALibrary,
    MockGFLibrary,
    format_metric,
    make_val_results,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> MetricsContext:
    """Build a minimal valid MetricsContext with 3 classes and 7 predictions."""
    # Classes must be keys that MockBALibrary.by_id knows (A1, A2, B1)
    # combined with an empty GF string to form valid bagf IDs.
    classes = ["A1::", "A2::", "B1::"]
    gt = [0, 1, 2, 0, 1, 2, 0]
    est = [0, 1, 2, 1, 0, 2, 0]
    val_results = make_val_results(gt, est, classes)
    return MetricsContext(
        val_results=val_results,
        ba_library=MockBALibrary(),
        gf_library=MockGFLibrary(),
        format_func=format_metric,
    )


def _make_bad_ctx() -> MetricsContext:
    """Build a context whose validate() will fail (unknown class in ba_library)."""

    class _BadBALibrary:
        def bagf_id_to_name(self, bagf_id, gf_library):
            raise KeyError(f"unknown: {bagf_id}")

    classes = ["UNKNOWN_CLASS::"]
    val_results = make_val_results([0], [0], classes)
    return MetricsContext(
        val_results=val_results,
        ba_library=_BadBALibrary(),
        gf_library=MockGFLibrary(),
        format_func=format_metric,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class HappyPathTest(unittest.TestCase):
    """compute_and_log_all() completes and records metrics."""

    def setUp(self):
        self.conn = duckdb.connect()
        self.ctx = _make_ctx()

    def test_completes_without_raising(self):
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow"),
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
        ):
            coord = MetricsCoordinator(self.ctx, self.conn)
            # If this raises, the test fails.
            coord.compute_and_log_all()

    def test_at_least_one_metric_logged(self):
        """At least one mlflow.log_metric call is made on a valid context."""
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow") as mock_mlflow,
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
        ):
            coord = MetricsCoordinator(self.ctx, self.conn)
            coord.compute_and_log_all()

        self.assertGreater(len(mock_mlflow.log_metric.call_args_list), 0)

    def test_precision_macro_metric_is_logged(self):
        """A known stable metric name — precision_macro — must appear in the calls."""
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow") as mock_mlflow,
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
        ):
            coord = MetricsCoordinator(self.ctx, self.conn)
            coord.compute_and_log_all()

        metric_names = [call.args[0] for call in mock_mlflow.log_metric.call_args_list]
        self.assertIn(
            "precision_macro",
            metric_names,
            msg=f"precision_macro not found in logged metrics: {metric_names}",
        )

    def test_precision_macro_value_is_numeric(self):
        """The logged precision_macro value should be a finite float."""
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow") as mock_mlflow,
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
        ):
            coord = MetricsCoordinator(self.ctx, self.conn)
            coord.compute_and_log_all()

        # Find the precision_macro call
        for call in mock_mlflow.log_metric.call_args_list:
            if call.args[0] == "precision_macro":
                value = call.args[1]
                self.assertIsInstance(value, (int, float))
                self.assertGreater(value, 0.0)
                return
        self.fail("precision_macro not logged")


class ErrorIsolationTest(unittest.TestCase):
    """A failing metric group must not abort the remaining groups."""

    def setUp(self):
        self.conn = duckdb.connect()
        self.ctx = _make_ctx()

    def test_failed_group_does_not_raise(self):
        """If calibration raises, compute_and_log_all() still completes."""
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow"),
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
            mock.patch(
                "mermaid_classifier.pyspacer.metrics.coordinator.compute_calibration",
                side_effect=RuntimeError("injected failure"),
            ),
        ):
            coord = MetricsCoordinator(self.ctx, self.conn)
            # Must not raise even though calibration fails.
            coord.compute_and_log_all()

    def test_other_groups_still_log_after_one_fails(self):
        """Metrics from other groups are still logged when calibration fails."""
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow") as mock_mlflow,
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
            mock.patch(
                "mermaid_classifier.pyspacer.metrics.coordinator.compute_calibration",
                side_effect=RuntimeError("injected failure"),
            ),
        ):
            coord = MetricsCoordinator(self.ctx, self.conn)
            coord.compute_and_log_all()

        # precision_recall_f1 group (which is separate from calibration) must
        # still have logged precision_macro.
        metric_names = [call.args[0] for call in mock_mlflow.log_metric.call_args_list]
        self.assertIn(
            "precision_macro",
            metric_names,
            msg=(
                "precision_macro should still be logged even when calibration fails;"
                f" got metric_names={metric_names}"
            ),
        )

    def test_failed_group_logs_fewer_metrics_than_healthy_run(self):
        """A run with one failed group logs fewer metrics than a clean run."""
        # Clean run
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow") as mock_clean,
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
        ):
            MetricsCoordinator(self.ctx, self.conn).compute_and_log_all()
        clean_count = len(mock_clean.log_metric.call_args_list)

        # Run with calibration failing
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow") as mock_broken,
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
            mock.patch(
                "mermaid_classifier.pyspacer.metrics.coordinator.compute_calibration",
                side_effect=RuntimeError("injected failure"),
            ),
        ):
            MetricsCoordinator(self.ctx, self.conn).compute_and_log_all()
        broken_count = len(mock_broken.log_metric.call_args_list)

        self.assertGreater(
            clean_count,
            broken_count,
            msg=(
                f"Broken run ({broken_count}) should log fewer metrics"
                f" than clean run ({clean_count})"
            ),
        )


class InvalidContextTest(unittest.TestCase):
    """An invalid context causes compute_and_log_all to return early, logging nothing."""

    def setUp(self):
        self.conn = duckdb.connect()
        self.ctx = _make_bad_ctx()

    def test_does_not_raise_on_invalid_context(self):
        """compute_and_log_all() must not raise when context validation fails."""
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow"),
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
        ):
            coord = MetricsCoordinator(self.ctx, self.conn)
            # Must not raise.
            coord.compute_and_log_all()

    def test_no_metrics_logged_on_invalid_context(self):
        """When context is invalid, no mlflow.log_metric calls are made."""
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow") as mock_mlflow,
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
        ):
            coord = MetricsCoordinator(self.ctx, self.conn)
            coord.compute_and_log_all()

        self.assertEqual(
            len(mock_mlflow.log_metric.call_args_list),
            0,
            msg="No metrics should be logged when context validation fails",
        )


if __name__ == "__main__":
    unittest.main()
