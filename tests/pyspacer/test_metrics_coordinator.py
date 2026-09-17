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

Where a test reaches a gated group it hands the context an inert stand-in for
the dataset or the classifier, so the groups that compute anything are the
always-available ones (confusion_matrices, precision_recall_f1,
balanced_accuracy_mcc, taxonomic, calibration). This keeps the test offline and
dependency-free.
"""

import dataclasses
import logging
import unittest
from unittest import mock

import duckdb

from mermaid_classifier.pyspacer.metrics import MetricsContext, MetricsCoordinator
from mermaid_classifier.pyspacer.metrics import registry as metrics_registry
from pyspacer.metrics_test_helpers import (
    MockBALibrary,
    MockClf,
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
            _registry_with_failing_group("calibration"),
        ):
            coord = MetricsCoordinator(self.ctx, self.conn)
            # Must not raise even though calibration fails.
            coord.compute_and_log_all()

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
            _registry_with_failing_group("calibration"),
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


class ApplicableGroupsTest(unittest.TestCase):
    """Each group is gated on the context inputs it actually reads."""

    def _names(self, ctx: MetricsContext) -> list[str]:
        return [spec.name for spec in metrics_registry.applicable_metric_groups(ctx)]

    def test_region_probe_is_skipped_without_a_classifier(self):
        """The probe is scored through the exported predictor; without one
        there is nothing to score it with."""
        self.assertNotIn("region_probe", self._names(_make_ctx()))

    def test_region_probe_applies_once_a_classifier_is_present(self):
        ctx = _make_ctx()
        ctx.clf = MockClf(["A1::", "A2::", "B1::"])
        self.assertIn("region_probe", self._names(ctx))

    def test_region_is_skipped_without_a_dataset(self):
        """The validation-split group reads the dataset's per-image region
        map."""
        self.assertNotIn("region", self._names(_make_ctx()))

    def test_region_applies_once_a_dataset_is_present(self):
        ctx = _make_ctx()
        ctx.dataset = object()
        self.assertIn("region", self._names(ctx))

    def test_a_classifier_alone_does_not_bring_in_the_validation_group(self):
        ctx = _make_ctx()
        ctx.clf = MockClf(["A1::", "A2::", "B1::"])
        self.assertNotIn("region", self._names(ctx))


class RegionGroupIsolationTest(unittest.TestCase):
    """A region group that raises must leave the rest of the run standing,
    and must not leave its absence unexplained."""

    def setUp(self):
        self.conn = duckdb.connect()
        self.ctx = _make_ctx()
        self.ctx.dataset = object()
        # Whether some other module's dictConfig has disabled this logger
        # must not decide whether the warning below can be observed.
        coordinator_logger = logging.getLogger("mermaid_classifier.pyspacer.metrics.coordinator")
        self.addCleanup(setattr, coordinator_logger, "disabled", coordinator_logger.disabled)
        coordinator_logger.disabled = False

    def _logged_metrics(self, failing_group: str) -> dict[str, float]:
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow") as mock_mlflow,
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
            _registry_with_failing_group(failing_group),
        ):
            MetricsCoordinator(self.ctx, self.conn).compute_and_log_all()
        return {call.args[0]: call.args[1] for call in mock_mlflow.log_metric.call_args_list}

    def test_run_survives_and_region_rates_are_absent(self):
        logged = self._logged_metrics("region")
        self.assertIn("precision_macro", logged)
        self.assertEqual(
            [],
            [name for name in logged if name.startswith("region_val/oor")],
            msg="a failed group must not publish rates",
        )

    def test_a_failed_region_group_leaves_its_status_at_zero(self):
        """The status is logged before the work that can raise, so a group
        that blew up is distinguishable in MLflow from one that never ran —
        a warning nobody can see is not an artifact."""
        self.assertEqual(self._logged_metrics("region")["region_val/scored"], 0.0)

    def test_a_failed_probe_group_leaves_its_own_status_at_zero(self):
        self.ctx.clf = MockClf(["A1::", "A2::", "B1::"])
        self.assertEqual(self._logged_metrics("region_probe")["region_probe/scored"], 0.0)

    def test_the_failed_group_is_named_in_a_warning(self):
        with (
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.mlflow"),
            mock.patch("mermaid_classifier.pyspacer.metrics.coordinator.log_dataframe"),
            _registry_with_failing_group("region"),
            self.assertLogs(
                "mermaid_classifier.pyspacer.metrics.coordinator", level="WARNING"
            ) as logs,
        ):
            MetricsCoordinator(self.ctx, self.conn).compute_and_log_all()

        self.assertTrue(
            any("region" in message for message in logs.output),
            msg=f"the failed group is not named in the warnings: {logs.output}",
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
