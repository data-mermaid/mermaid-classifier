"""
Tests for mermaid_classifier.pyspacer.trainer.

Covers two distinct features of MermaidTrainer:
  - Batched calibration produces math-identical results to the standard
    CalibratedClassifierCV(cv='prefit').fit() approach.
  - Early-stopping bookkeeping correctly snapshots, restores, and
    populates _early_stop_info under a scripted val_loss schedule.
"""

import unittest
from unittest import mock

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

from mermaid_classifier.pyspacer.trainer import MermaidTrainer


def _make_mock_labels(X, y, batch_size):
    """Create a mock ImageLabels whose load_data_in_batches yields
    chunks of the given arrays, matching the real method's return type:
    yields (list[np.ndarray], list[label]) per batch."""
    mock_labels = mock.Mock()

    def batch_generator(batch_size=batch_size):
        for i in range(0, len(X), batch_size):
            end = min(i + batch_size, len(X))
            x_batch = [X[j] for j in range(i, end)]
            y_batch = [y[j] for j in range(i, end)]
            yield x_batch, y_batch

    mock_labels.load_data_in_batches = batch_generator
    return mock_labels


class CalibrateInBatchesTest(unittest.TestCase):
    """
    Verify that MermaidTrainer._calibrate_in_batches produces identical
    calibration to CalibratedClassifierCV(cv='prefit').fit().
    """

    def _train_and_calibrate_both_ways(self, n_classes):
        """
        Helper: create synthetic data, train a classifier, calibrate
        with both the old (standard) and new (batched) approaches,
        return both calibrated classifiers and the ref data.
        """
        # Generate separable Gaussian clusters (one centroid per class)
        # rather than pure noise. With real class signal, the base classifier
        # learns sensible weights and its decision_function scores stay in
        # a moderate range, which in turn makes the Platt calibration fit
        # well-behaved. Pure-noise data produces extreme scores that make
        # the calibration objective very flat, so tiny floating-point
        # differences across compute environments can push the converged
        # sigmoid parameters around enough to fail the test.
        rng = np.random.RandomState(42)
        n_samples = 500
        n_features = 50
        n_per_class = n_samples // n_classes
        classes = [f"class_{i}" for i in range(n_classes)]

        centroids = rng.randn(n_classes, n_features).astype(np.float32) * 3.0
        X = np.vstack(
            [
                centroids[i] + rng.randn(n_per_class, n_features).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        y = np.array([cls for cls in classes for _ in range(n_per_class)])

        # Shuffle so train/ref split isn't class-stratified by half.
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]

        split = len(X) // 2
        X_train, X_ref = X[:split], X[split:]
        y_train, y_ref = y[:split], y[split:]

        # Train base classifier using partial_fit (same as production).
        # Pin random_state so weight initialization is deterministic across
        # runs. Without it the MLP seeds from NumPy's OS-entropy global RNG,
        # so the trained weights — and the cross-BLAS FP drift they produce —
        # vary every run, making the calibration-equivalence comparison flaky
        # on Linux CI.
        clf = MLPClassifier(hidden_layer_sizes=(20,), learning_rate_init=1e-3, random_state=0)
        clf.partial_fit(X_train, y_train, classes=classes)

        # Standard calibration (current approach)
        clf_standard = CalibratedClassifierCV(clf, cv="prefit")
        clf_standard.fit(X_ref, y_ref)

        # Batched calibration (new approach) via mock ImageLabels
        mock_labels = _make_mock_labels(X_ref, y_ref, batch_size=100)
        trainer = MermaidTrainer(batch_size=100)
        clf_batched = trainer._calibrate_in_batches(clf, mock_labels)

        return clf_standard, clf_batched, X_ref

    def test_mlp_calibration_equivalence(self):
        """Batched calibration matches standard, for multiclass and the binary edge case."""
        # 2 classes exercises sklearn's binary calibration path; 5 the multiclass path.
        for n_classes in (5, 2):
            with self.subTest(n_classes=n_classes):
                std, batched, X = self._train_and_calibrate_both_ways(n_classes)
                self._assert_calibration_equivalent(std, batched, X)

    def _assert_calibration_equivalent(self, clf_std, clf_batched, X_test):
        """
        Assert that both calibrated classifiers produce identical:
        1. Per-class sigmoid parameters (a_, b_)
        2. predict_proba outputs on test data
        """
        # Compare per-class sigmoid parameters.
        # Tolerances give the tests headroom across compute environments.
        # Different BLAS implementations (e.g. Apple Accelerate vs Linux
        # OpenBLAS/MKL) and the chunked-vs-single-shot matmul patterns in
        # the two calibration paths produce tiny floating-point differences,
        # which can shift the L-BFGS-B-fit sigmoid parameters slightly.
        # atol covers params near zero; rtol scales for larger magnitudes.
        for cc_std, cc_batch in zip(
            clf_std.calibrated_classifiers_,
            clf_batched.calibrated_classifiers_,
            strict=False,
        ):
            for cal_std, cal_batch in zip(cc_std.calibrators, cc_batch.calibrators, strict=False):
                np.testing.assert_allclose(
                    cal_std.a_,
                    cal_batch.a_,
                    rtol=1e-3,
                    atol=5e-3,
                    err_msg="Sigmoid parameter 'a' differs",
                )
                np.testing.assert_allclose(
                    cal_std.b_,
                    cal_batch.b_,
                    rtol=1e-3,
                    atol=5e-3,
                    err_msg="Sigmoid parameter 'b' differs",
                )

        # Compare predict_proba outputs — the actual API surface.
        # Even if sigmoid params differ at the solver-tolerance level,
        # the resulting probabilities should be very close. atol provides
        # headroom for small-probability entries (where strict rtol gets
        # very tight), and rtol catches material drift on larger values.
        # These tolerances must absorb the same cross-BLAS floating-point
        # noise as the sigmoid-parameter checks above (e.g. Linux OpenBLAS
        # produces ~1e-4 absolute / ~1.5e-3 relative drift vs Apple
        # Accelerate), so they are kept in line with that rtol rather than
        # tighter — a stricter bound here fails on Linux CI for FP-noise
        # reasons unrelated to calibration correctness.
        proba_std = clf_std.predict_proba(X_test)
        proba_batch = clf_batched.predict_proba(X_test)
        np.testing.assert_allclose(
            proba_std, proba_batch, rtol=3e-3, atol=2e-4, err_msg="predict_proba outputs differ"
        )

        # Verify PySpacer compatibility attributes
        self.assertIsInstance(clf_batched, CalibratedClassifierCV)
        self.assertEqual(clf_batched.cv, "prefit")
        self.assertTrue(hasattr(clf_batched, "calibrated_classifiers_"))
        self.assertTrue(hasattr(clf_batched, "classes_"))
        self.assertTrue(hasattr(clf_batched, "estimator"))


class EarlyStoppingConstructorTest(unittest.TestCase):
    """Validation of the early_stopping_patience constructor argument."""

    def test_patience_zero_rejected(self):
        with self.assertRaisesRegex(ValueError, "early_stopping_patience"):
            MermaidTrainer(batch_size=10, early_stopping_patience=0)

    def test_invalid_metric_rejected(self):
        with self.assertRaisesRegex(ValueError, "early_stopping_metric"):
            MermaidTrainer(batch_size=10, early_stopping_metric="val_f1")  # pyright: ignore

    def test_known_metrics_accepted(self):
        for metric in ("val_loss", "val_accuracy"):
            MermaidTrainer(batch_size=10, early_stopping_metric=metric)


class EarlyStoppingBehaviorTest(unittest.TestCase):
    """End-to-end early-stopping behavior with a scripted val_loss schedule.

    We script val_loss via a subclass that overrides
    `_calc_acc_and_log_loss_batched` to return values from a queue,
    and use a small synthetic dataset that a real TorchMLPClassifier
    can chew through quickly. Each test scripts a different val_loss
    pattern (monotone-down, V-shape, plateau-then-rise) and asserts
    the right epoch is the stopping/best epoch.
    """

    @staticmethod
    def _make_synth_labels(seed: int = 0):
        """Build mock train/ref/val ImageLabels-like objects.

        Spacer's real ImageLabels is hard to construct outside its
        normal pipeline; we mock just what MermaidTrainer.__call__
        actually touches. Attribute accesses inside f-string log lines
        are eagerly evaluated, so we set every attribute the trainer
        reads (even via debug logging) explicitly.
        """
        rng = np.random.RandomState(seed)
        n_features = 16
        classes = ["c0", "c1", "c2"]

        def _make_split(n):
            X = rng.randn(n, n_features).astype(np.float32)
            y = list(rng.choice(classes, size=n))
            return X, y

        train_X, train_y = _make_split(30)
        ref_X, ref_y = _make_split(15)
        val_X, val_y = _make_split(15)

        def _split_obj(X, y):
            obj = mock.Mock()
            obj.classes_set = set(classes)
            obj.label_count = len(y)
            # __len__ for the trainer's "Data sets" debug line.
            obj.__len__ = mock.Mock(return_value=len(y))

            def gen(batch_size=10, random_seed=None):
                for i in range(0, len(X), batch_size):
                    end = min(i + batch_size, len(X))
                    x_batch = [X[j] for j in range(i, end)]
                    y_batch = list(y[i:end])
                    yield x_batch, y_batch

            obj.load_data_in_batches = gen
            return obj

        train = _split_obj(train_X, train_y)
        ref = _split_obj(ref_X, ref_y)
        val = _split_obj(val_X, val_y)

        labels = mock.Mock()
        labels.train = train
        labels.ref = ref
        labels.val = val
        labels.label_count = train.label_count + ref.label_count + val.label_count
        labels.__len__ = mock.Mock(return_value=len(train_X))
        return labels

    @staticmethod
    def _scripted_trainer(val_loss_schedule, val_acc_schedule=None, **kwargs):
        """Return a MermaidTrainer subclass instance whose
        _calc_acc_and_log_loss_batched yields scripted val_loss
        values; val_acc is fixed at 0.5 unless val_acc_schedule is given.
        """
        schedule = list(val_loss_schedule)
        acc_schedule = None if val_acc_schedule is None else list(val_acc_schedule)

        class Scripted(MermaidTrainer):
            def __init__(self_, *a, **kw):
                super().__init__(*a, **kw)
                self_._scripted_remaining = list(schedule)

            def _calc_acc_and_log_loss_batched(self_, clf, labels, classes_list):
                if not self_._scripted_remaining:
                    raise AssertionError(
                        "val_loss schedule exhausted; trainer ran for more epochs than expected"
                    )
                loss = self_._scripted_remaining.pop(0)
                acc = 0.5 if acc_schedule is None else acc_schedule.pop(0)
                return float(acc), float(loss)

        return Scripted(batch_size=10, **kwargs)

    def _run(self, schedule, patience, n_epochs, acc_schedule=None, **trainer_kwargs):
        labels = self._make_synth_labels(seed=0)
        captured = []
        trainer = self._scripted_trainer(
            schedule,
            acc_schedule,
            on_epoch_end=lambda m: captured.append(dict(m)),
            early_stopping_patience=patience,
            **trainer_kwargs,
        )
        trainer(labels, nbr_epochs=n_epochs, pc_models=[])
        return trainer, captured

    def test_no_patience_runs_full_budget(self):
        """patience=None: trainer runs for the full epoch budget."""
        # 5 epochs, monotone-decreasing val_loss; no early stop.
        schedule = [1.0, 0.9, 0.8, 0.7, 0.6]
        trainer, captured = self._run(schedule, patience=None, n_epochs=5)
        self.assertEqual(len(captured), 5)
        info = trainer._early_stop_info
        self.assertFalse(info["enabled"])
        self.assertEqual(info["stop_reason"], "budget_exhausted")
        self.assertEqual(info["final_epoch"], 5)
        # When ES is off, no snapshot is kept.
        self.assertIsNone(info["best_val_epoch"])
        self.assertIsNone(info["best_val_loss"])

    def test_monotone_down_no_stop(self):
        """patience=2 + always-improving val_loss: no early stop."""
        schedule = [1.0, 0.9, 0.8, 0.7, 0.6]
        trainer, captured = self._run(schedule, patience=2, n_epochs=5)
        self.assertEqual(len(captured), 5)
        info = trainer._early_stop_info
        self.assertTrue(info["enabled"])
        self.assertEqual(info["stop_reason"], "budget_exhausted")
        self.assertEqual(info["final_epoch"], 5)
        # Best is the last epoch (5).
        self.assertEqual(info["best_val_epoch"], 5)
        self.assertAlmostEqual(info["best_val_loss"], 0.6)

    def test_v_shape_triggers_after_patience(self):
        """val_loss minimum at epoch 3, then rises for patience+ epochs.

        Schedule: 1.0, 0.9, 0.8 (best=ep3), 0.85, 0.9 (patience=2 -> stop @ ep5).
        """
        schedule = [1.0, 0.9, 0.8, 0.85, 0.9, 1.0, 1.1]  # extras unused
        trainer, captured = self._run(schedule, patience=2, n_epochs=10)
        info = trainer._early_stop_info
        self.assertTrue(info["enabled"])
        self.assertEqual(info["stop_reason"], "early_stopping")
        # Trained 5 epochs total: 3 to best + 2 patience = stop @ epoch 5.
        self.assertEqual(info["final_epoch"], 5)
        self.assertEqual(info["best_val_epoch"], 3)
        self.assertAlmostEqual(info["best_val_loss"], 0.8)
        # Five callback events fired (one per epoch run).
        self.assertEqual(len(captured), 5)
        # Final-epoch callback carries the summary fields.
        self.assertTrue(captured[-1]["early_stopped"])
        self.assertEqual(captured[-1]["final_epoch"], 5)
        self.assertEqual(captured[-1]["best_val_epoch"], 3)

    def test_patience_one_immediate_stop_on_first_regression(self):
        """patience=1: stops the very first epoch val_loss doesn't improve."""
        schedule = [1.0, 0.5, 0.6]  # best=ep2, regress at ep3 -> stop
        trainer, captured = self._run(schedule, patience=1, n_epochs=10)
        info = trainer._early_stop_info
        self.assertEqual(info["stop_reason"], "early_stopping")
        self.assertEqual(info["final_epoch"], 3)
        self.assertEqual(info["best_val_epoch"], 2)

    def test_summary_only_on_final_epoch(self):
        """early_stopped/best_val_epoch fields only on the last epoch."""
        schedule = [1.0, 0.9, 0.95, 0.96]  # best=ep2, stops @ep4 (patience=2)
        trainer, captured = self._run(schedule, patience=2, n_epochs=10)
        # Non-final epochs MUST NOT have the summary fields.
        for cb in captured[:-1]:
            self.assertNotIn("early_stopped", cb)
            self.assertNotIn("final_epoch", cb)
        # Final epoch DOES have them.
        self.assertIn("early_stopped", captured[-1])
        self.assertIn("final_epoch", captured[-1])
        self.assertIn("best_val_epoch", captured[-1])
        self.assertIn("best_val_loss", captured[-1])

    def test_default_metric_is_val_loss(self):
        """Without early_stopping_metric, selection follows val_loss even
        when accuracy would pick a different epoch."""
        losses = [1.0, 0.8, 0.9, 0.95]
        accs = [0.5, 0.52, 0.6, 0.7]
        trainer, _ = self._run(losses, patience=2, n_epochs=10, acc_schedule=accs)
        info = trainer._early_stop_info
        self.assertEqual(info["early_stopping_metric"], "val_loss")
        self.assertEqual(info["best_val_epoch"], 2)

    def test_val_accuracy_keeps_training_past_loss_minimum(self):
        """Loss bottoms at epoch 2 and rises while accuracy keeps rising:
        accuracy selection runs the full budget and restores the last epoch."""
        losses = [1.0, 0.8, 0.9, 1.0, 1.1, 1.2]
        accs = [0.50, 0.52, 0.54, 0.56, 0.57, 0.58]
        trainer, captured = self._run(
            losses,
            patience=2,
            n_epochs=6,
            acc_schedule=accs,
            early_stopping_metric="val_accuracy",
        )
        info = trainer._early_stop_info
        self.assertEqual(len(captured), 6)
        self.assertEqual(info["stop_reason"], "budget_exhausted")
        self.assertEqual(info["best_val_epoch"], 6)
        self.assertAlmostEqual(info["best_val_accuracy"], 0.58)
        # best_val_loss is the val_loss at the best (accuracy) epoch.
        self.assertAlmostEqual(info["best_val_loss"], 1.2)
        self.assertEqual(info["early_stopping_metric"], "val_accuracy")

    def test_val_accuracy_restores_best_accuracy_epoch(self):
        """Accuracy peaks at epoch 3; the restored snapshot is epoch 3's."""
        losses = [1.0, 0.9, 0.95, 1.0, 1.1]
        accs = [0.50, 0.55, 0.60, 0.58, 0.57]
        labels = self._make_synth_labels(seed=0)
        snapshots = []

        class Recording(MermaidTrainer):
            def _calibrate_in_batches(self_, clf, ref_labels):
                snapshots.append(clf)
                return super()._calibrate_in_batches(clf, ref_labels)

        epoch_clfs = []

        def scripted(self_, clf, lbls, classes_list):
            epoch_clfs.append(clf)
            return accs.pop(0), losses.pop(0)

        Recording._calc_acc_and_log_loss_batched = scripted  # pyright: ignore
        trainer = Recording(
            batch_size=10, early_stopping_patience=2, early_stopping_metric="val_accuracy"
        )
        trainer(labels, nbr_epochs=10, pc_models=[])
        info = trainer._early_stop_info
        self.assertEqual(info["stop_reason"], "early_stopping")
        self.assertEqual(info["final_epoch"], 5)
        self.assertEqual(info["best_val_epoch"], 3)
        self.assertAlmostEqual(info["best_val_accuracy"], 0.60)
        self.assertAlmostEqual(info["best_val_loss"], 0.95)
        # The calibrated clf is a deepcopy snapshot, not the live (epoch 5) clf.
        self.assertIsNot(snapshots[0], epoch_clfs[-1])

    def test_val_accuracy_patience_triggers_when_accuracy_stalls(self):
        accs = [0.5, 0.6, 0.6, 0.59]  # ties are not improvements
        losses = [1.0, 0.9, 0.8, 0.7]
        trainer, captured = self._run(
            losses,
            patience=2,
            n_epochs=10,
            acc_schedule=accs,
            early_stopping_metric="val_accuracy",
        )
        info = trainer._early_stop_info
        self.assertEqual(info["stop_reason"], "early_stopping")
        self.assertEqual(info["final_epoch"], 4)
        self.assertEqual(info["best_val_epoch"], 2)
        self.assertTrue(captured[-1]["early_stopped"])

    def test_callback_carries_metric_keys_on_final_epoch(self):
        accs = [0.5, 0.6, 0.55]
        losses = [1.0, 0.9, 0.95]
        _, captured = self._run(
            losses,
            patience=1,
            n_epochs=10,
            acc_schedule=accs,
            early_stopping_metric="val_accuracy",
        )
        final = captured[-1]
        self.assertEqual(final["early_stopping_metric"], "val_accuracy")
        self.assertEqual(final["best_val_epoch"], 2)
        self.assertAlmostEqual(final["best_val_accuracy"], 0.6)
        self.assertAlmostEqual(final["best_val_loss"], 0.9)
        self.assertNotIn("best_val_accuracy", captured[0])


class RunnerEarlyStopMetricLoggingTest(unittest.TestCase):
    """MLflowTrainingRunner._on_epoch_end logs the best-epoch accuracy."""

    @staticmethod
    def _logged(extra):
        from mermaid_classifier.pyspacer.runner import MLflowTrainingRunner

        metrics = {
            "epoch": 4,
            "ref_accuracy": 0.5,
            "val_accuracy": 0.6,
            "val_loss": 0.9,
            "training_loss": 0.4,
            "cumulative_seconds": 1.0,
            "final_epoch": 5,
            "early_stopped": True,
            "best_val_epoch": 3,
            "best_val_loss": 0.8,
            **extra,
        }
        with mock.patch("mermaid_classifier.pyspacer.runner.mlflow") as m_mlflow:
            MLflowTrainingRunner._on_epoch_end(mock.Mock(), metrics)
        return {c.args[0]: (c.args[1], c.kwargs) for c in m_mlflow.log_metric.call_args_list}

    def test_best_val_accuracy_logged_at_step_zero(self):
        logged = self._logged({"best_val_accuracy": 0.61})
        self.assertEqual(logged["early_stop/best_val_accuracy"], (0.61, {"step": 0}))

    def test_absent_best_val_accuracy_not_logged(self):
        self.assertNotIn("early_stop/best_val_accuracy", self._logged({}))
