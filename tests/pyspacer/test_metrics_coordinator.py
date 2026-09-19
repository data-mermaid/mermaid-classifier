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

import dataclasses
import unittest
from unittest import mock

import duckdb

from mermaid_classifier.pyspacer.metrics import MetricsContext, MetricsCoordinator
from mermaid_classifier.pyspacer.metrics import registry as metrics_registry
from pyspacer.metrics_test_helpers import (
    MockBALibrary,
    MockGFLibrary,
    format_metric,
    make_val_results,
)


def _registry_with_failing_group(name: str):
    """Patch the metric registry so the named group's func raises when called.

    Injects the failure at the registry seam (where the coordinator now looks
    metric groups up) rather than patching an import in the coordinator module.
    """

    def _raise(_ctx):
        raise RuntimeError("injected failure")

    known = {spec.name for spec in metrics_registry.METRIC_GROUPS}
    if name not in known:
        raise ValueError(
            f"unknown metric group {name!r}; cannot inject failure. Known groups: {sorted(known)}"
        )

    patched = [
        dataclasses.replace(spec, func=_raise) if spec.name == name else spec
        for spec in metrics_registry.METRIC_GROUPS
    ]
    return mock.patch.object(metrics_registry, "METRIC_GROUPS", patched)


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
    val_results = make_val_results([0], [0], ["UNKNOWN_CLASS::"])
    return MetricsContext(
        val_results=val_results,
        ba_library=MockBALibrary(),
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

    def test_other_groups_still_log_after_one_fails(self):
        """Metrics from other groups are still logged when calibration fails."""
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow") as mock_mlflow,
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
            _registry_with_failing_group("calibration"),
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


class InvalidContextTest(unittest.TestCase):
    """An invalid context causes compute_and_log_all to return early, logging nothing."""

    def setUp(self):
        self.conn = duckdb.connect()
        self.ctx = _make_bad_ctx()

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
