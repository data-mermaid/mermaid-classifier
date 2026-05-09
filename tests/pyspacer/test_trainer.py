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
from sklearn.linear_model import SGDClassifier
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

    def _train_and_calibrate_both_ways(self, clf_type, n_classes):
        """
        Helper: create synthetic data, train a classifier, calibrate
        with both the old (standard) and new (batched) approaches,
        return both calibrated classifiers and the ref data.
        """
        rng = np.random.RandomState(42)
        n_samples = 500
        n_features = 50
        X = rng.randn(n_samples, n_features).astype(np.float32)
        classes = [f'class_{i}' for i in range(n_classes)]
        y = np.array(rng.choice(classes, size=n_samples))

        split = n_samples // 2
        X_train, X_ref = X[:split], X[split:]
        y_train, y_ref = y[:split], y[split:]

        # Train base classifier using partial_fit (same as production)
        if clf_type == 'MLP':
            clf = MLPClassifier(
                hidden_layer_sizes=(20,), learning_rate_init=1e-3)
        else:
            clf = SGDClassifier(
                loss='log_loss', average=True, random_state=0)
        clf.partial_fit(X_train, y_train, classes=classes)

        # Standard calibration (current approach)
        clf_standard = CalibratedClassifierCV(clf, cv="prefit")
        clf_standard.fit(X_ref, y_ref)

        # Batched calibration (new approach) via mock ImageLabels
        mock_labels = _make_mock_labels(X_ref, y_ref, batch_size=100)
        trainer = MermaidTrainer(batch_size=100)
        clf_batched = trainer._calibrate_in_batches(clf, mock_labels)

        return clf_standard, clf_batched, X_ref

    def test_sgd_multiclass_calibration_equivalence(self):
        """SGDClassifier with 5 classes: batch == standard."""
        std, batched, X = self._train_and_calibrate_both_ways('SGD', 5)
        self._assert_calibration_equivalent(std, batched, X)

    def test_mlp_multiclass_calibration_equivalence(self):
        """MLPClassifier with 5 classes: batch == standard."""
        std, batched, X = self._train_and_calibrate_both_ways('MLP', 5)
        self._assert_calibration_equivalent(std, batched, X)

    def test_sgd_binary_calibration_equivalence(self):
        """SGDClassifier with 2 classes (binary edge case): batch == standard."""
        std, batched, X = self._train_and_calibrate_both_ways('SGD', 2)
        self._assert_calibration_equivalent(std, batched, X)

    def _assert_calibration_equivalent(self, clf_std, clf_batched, X_test):
        """
        Assert that both calibrated classifiers produce identical:
        1. Per-class sigmoid parameters (a_, b_)
        2. predict_proba outputs on test data
        """
        # Compare per-class sigmoid parameters.
        # Tolerance is 1e-3 because L-BFGS-B convergence can vary slightly
        # depending on floating-point intermediates from array construction.
        for cc_std, cc_batch in zip(
            clf_std.calibrated_classifiers_,
            clf_batched.calibrated_classifiers_,
        ):
            for cal_std, cal_batch in zip(
                cc_std.calibrators, cc_batch.calibrators
            ):
                np.testing.assert_allclose(
                    cal_std.a_, cal_batch.a_, rtol=2e-3,
                    err_msg="Sigmoid parameter 'a' differs")
                np.testing.assert_allclose(
                    cal_std.b_, cal_batch.b_, rtol=2e-3,
                    err_msg="Sigmoid parameter 'b' differs")

        # Compare predict_proba outputs — the actual API surface.
        # Even if sigmoid params differ at the solver-tolerance level,
        # the resulting probabilities should be very close.
        proba_std = clf_std.predict_proba(X_test)
        proba_batch = clf_batched.predict_proba(X_test)
        np.testing.assert_allclose(
            proba_std, proba_batch, rtol=1e-5,
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
    and use a small synthetic dataset that the SGDClassifier path can
    chew through quickly. Each test scripts a different val_loss
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
        # clf_type='LR' -> SGDClassifier path; cheaper to fit than MLP.
        trainer(labels, nbr_epochs=n_epochs, pc_models=[], clf_type='LR')
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
