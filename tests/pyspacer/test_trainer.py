"""
Tests for mermaid_classifier.pyspacer.trainer.

Covers two distinct features of MermaidTrainer:
  - Batched calibration produces math-identical results to the standard
    CalibratedClassifierCV(cv='prefit').fit() approach.
  - Early-stopping bookkeeping correctly snapshots, restores, and
    populates _early_stop_info under a scripted val_loss schedule.
"""

import ast
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

import mermaid_classifier.pyspacer.trainer as trainer_module
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
        classes = [f'class_{i}' for i in range(n_classes)]

        centroids = rng.randn(n_classes, n_features).astype(np.float32) * 3.0
        X = np.vstack([
            centroids[i] + rng.randn(n_per_class, n_features).astype(np.float32)
            for i in range(n_classes)
        ])
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
        clf = MLPClassifier(
            hidden_layer_sizes=(20,), learning_rate_init=1e-3,
            random_state=0)
        clf.partial_fit(X_train, y_train, classes=classes)

        # Standard calibration (current approach)
        clf_standard = CalibratedClassifierCV(clf, cv="prefit")
        clf_standard.fit(X_ref, y_ref)

        # Batched calibration (new approach) via mock ImageLabels
        mock_labels = _make_mock_labels(X_ref, y_ref, batch_size=100)
        trainer = MermaidTrainer(batch_size=100)
        clf_batched = trainer._calibrate_in_batches(clf, mock_labels)

        return clf_standard, clf_batched, X_ref

    def test_mlp_multiclass_calibration_equivalence(self):
        """MLPClassifier with 5 classes: batch == standard."""
        std, batched, X = self._train_and_calibrate_both_ways(5)
        self._assert_calibration_equivalent(std, batched, X)

    def test_mlp_binary_calibration_equivalence(self):
        """MLPClassifier with 2 classes (binary edge case): batch == standard."""
        std, batched, X = self._train_and_calibrate_both_ways(2)
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
        ):
            for cal_std, cal_batch in zip(
                cc_std.calibrators, cc_batch.calibrators
            ):
                np.testing.assert_allclose(
                    cal_std.a_, cal_batch.a_, rtol=1e-3, atol=5e-3,
                    err_msg="Sigmoid parameter 'a' differs")
                np.testing.assert_allclose(
                    cal_std.b_, cal_batch.b_, rtol=1e-3, atol=5e-3,
                    err_msg="Sigmoid parameter 'b' differs")

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
            proba_std, proba_batch, rtol=3e-3, atol=2e-4,
            err_msg="predict_proba outputs differ")

        # Verify PySpacer compatibility attributes
        self.assertIsInstance(clf_batched, CalibratedClassifierCV)
        self.assertEqual(clf_batched.cv, 'prefit')
        self.assertTrue(hasattr(clf_batched, 'calibrated_classifiers_'))
        self.assertTrue(hasattr(clf_batched, 'classes_'))
        self.assertTrue(hasattr(clf_batched, 'estimator'))


class EarlyStoppingConstructorTest(unittest.TestCase):
    """Validation of the early_stopping_patience constructor argument."""

    def test_default_is_none(self):
        t = MermaidTrainer(batch_size=10)
        self.assertIsNone(t.early_stopping_patience)
        self.assertIsNone(t._early_stop_info)

    def test_patience_one_accepted(self):
        t = MermaidTrainer(batch_size=10, early_stopping_patience=1)
        self.assertEqual(t.early_stopping_patience, 1)

    def test_patience_zero_rejected(self):
        with self.assertRaisesRegex(ValueError, "early_stopping_patience"):
            MermaidTrainer(batch_size=10, early_stopping_patience=0)

    def test_patience_negative_rejected(self):
        with self.assertRaisesRegex(ValueError, "early_stopping_patience"):
            MermaidTrainer(batch_size=10, early_stopping_patience=-1)


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
        classes = ['c0', 'c1', 'c2']

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
    def _scripted_trainer(val_loss_schedule, **kwargs):
        """Return a MermaidTrainer subclass instance whose
        _calc_acc_and_log_loss_batched yields scripted val_loss
        values (val_acc fixed at 0.5 -- the test only cares about loss).
        """
        schedule = list(val_loss_schedule)

        class Scripted(MermaidTrainer):
            def __init__(self_, *a, **kw):
                super().__init__(*a, **kw)
                self_._scripted_remaining = list(schedule)

            def _calc_acc_and_log_loss_batched(
                self_, clf, labels, classes_list):
                if not self_._scripted_remaining:
                    raise AssertionError(
                        "val_loss schedule exhausted; trainer ran for"
                        " more epochs than expected")
                loss = self_._scripted_remaining.pop(0)
                return 0.5, float(loss)

        return Scripted(batch_size=10, **kwargs)

    def _run(self, schedule, patience, n_epochs):
        labels = self._make_synth_labels(seed=0)
        captured = []
        trainer = self._scripted_trainer(
            schedule,
            on_epoch_end=lambda m: captured.append(dict(m)),
            early_stopping_patience=patience,
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
        self.assertFalse(info['enabled'])
        self.assertEqual(info['stop_reason'], 'budget_exhausted')
        self.assertEqual(info['final_epoch'], 5)
        # When ES is off, no snapshot is kept.
        self.assertIsNone(info['best_val_epoch'])
        self.assertIsNone(info['best_val_loss'])

    def test_monotone_down_no_stop(self):
        """patience=2 + always-improving val_loss: no early stop."""
        schedule = [1.0, 0.9, 0.8, 0.7, 0.6]
        trainer, captured = self._run(schedule, patience=2, n_epochs=5)
        self.assertEqual(len(captured), 5)
        info = trainer._early_stop_info
        self.assertTrue(info['enabled'])
        self.assertEqual(info['stop_reason'], 'budget_exhausted')
        self.assertEqual(info['final_epoch'], 5)
        # Best is the last epoch (5).
        self.assertEqual(info['best_val_epoch'], 5)
        self.assertAlmostEqual(info['best_val_loss'], 0.6)

    def test_v_shape_triggers_after_patience(self):
        """val_loss minimum at epoch 3, then rises for patience+ epochs.

        Schedule: 1.0, 0.9, 0.8 (best=ep3), 0.85, 0.9 (patience=2 -> stop @ ep5).
        """
        schedule = [1.0, 0.9, 0.8, 0.85, 0.9, 1.0, 1.1]  # extras unused
        trainer, captured = self._run(schedule, patience=2, n_epochs=10)
        info = trainer._early_stop_info
        self.assertTrue(info['enabled'])
        self.assertEqual(info['stop_reason'], 'early_stopping')
        # Trained 5 epochs total: 3 to best + 2 patience = stop @ epoch 5.
        self.assertEqual(info['final_epoch'], 5)
        self.assertEqual(info['best_val_epoch'], 3)
        self.assertAlmostEqual(info['best_val_loss'], 0.8)
        # Five callback events fired (one per epoch run).
        self.assertEqual(len(captured), 5)
        # Final-epoch callback carries the summary fields.
        self.assertTrue(captured[-1]['early_stopped'])
        self.assertEqual(captured[-1]['final_epoch'], 5)
        self.assertEqual(captured[-1]['best_val_epoch'], 3)

    def test_patience_one_immediate_stop_on_first_regression(self):
        """patience=1: stops the very first epoch val_loss doesn't improve."""
        schedule = [1.0, 0.5, 0.6]  # best=ep2, regress at ep3 -> stop
        trainer, captured = self._run(schedule, patience=1, n_epochs=10)
        info = trainer._early_stop_info
        self.assertEqual(info['stop_reason'], 'early_stopping')
        self.assertEqual(info['final_epoch'], 3)
        self.assertEqual(info['best_val_epoch'], 2)

    def test_summary_only_on_final_epoch(self):
        """early_stopped/best_val_epoch fields only on the last epoch."""
        schedule = [1.0, 0.9, 0.95, 0.96]  # best=ep2, stops @ep4 (patience=2)
        trainer, captured = self._run(schedule, patience=2, n_epochs=10)
        # Non-final epochs MUST NOT have the summary fields.
        for cb in captured[:-1]:
            self.assertNotIn('early_stopped', cb)
            self.assertNotIn('final_epoch', cb)
        # Final epoch DOES have them.
        self.assertIn('early_stopped', captured[-1])
        self.assertIn('final_epoch', captured[-1])
        self.assertIn('best_val_epoch', captured[-1])
        self.assertIn('best_val_loss', captured[-1])


class TrainerCleanupGuardTest(unittest.TestCase):
    """Guard against reintroducing the removed SGD/clf_type path or the
    no-benefit pyspacer imports into trainer.py (#58)."""

    def setUp(self):
        self.source = Path(trainer_module.__file__).read_text()

    def test_no_sgd_or_clf_type_references(self):
        for token in ('SGDClassifier', 'clf_type'):
            self.assertNotIn(
                token, self.source,
                f"{token} should not reappear in trainer.py")

    def test_no_dead_pyspacer_imports(self):
        # Walk the actual import nodes (not substrings) so equivalent
        # import styles -- e.g. `import spacer.config as config` -- can't
        # slip past the guard.
        offenders = []
        for node in ast.walk(ast.parse(self.source)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # `import spacer.config[.x]` in any aliased form
                    if (alias.name == 'spacer.config'
                            or alias.name.startswith('spacer.config.')):
                        offenders.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = {alias.name for alias in node.names}
                # `from spacer import config`
                if module == 'spacer' and 'config' in names:
                    offenders.append("from spacer import config")
                # `from spacer.config[.x] import ...`
                if (module == 'spacer.config'
                        or module.startswith('spacer.config.')):
                    offenders.append(f"from {module} import ...")
                # `from spacer.train_utils import calc_acc`
                if module == 'spacer.train_utils' and 'calc_acc' in names:
                    offenders.append(
                        "from spacer.train_utils import calc_acc")
        self.assertEqual(
            offenders, [],
            "trainer.py should not re-import spacer.config or the dead"
            f" calc_acc helper (use sklearn.metrics.accuracy_score); found:"
            f" {offenders}")
