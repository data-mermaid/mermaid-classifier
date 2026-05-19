"""
Tests for mermaid_classifier.pyspacer.trainer.

Verifies that MermaidTrainer._calibrate_in_batches() produces mathematically
identical results to the standard CalibratedClassifierCV(cv='prefit').fit()
approach.
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
        proba_std = clf_std.predict_proba(X_test)
        proba_batch = clf_batched.predict_proba(X_test)
        np.testing.assert_allclose(
            proba_std, proba_batch, rtol=1e-4, atol=1e-5,
            err_msg="predict_proba outputs differ")

        # Verify PySpacer compatibility attributes
        self.assertIsInstance(clf_batched, CalibratedClassifierCV)
        self.assertEqual(clf_batched.cv, 'prefit')
        self.assertTrue(hasattr(clf_batched, 'calibrated_classifiers_'))
        self.assertTrue(hasattr(clf_batched, 'classes_'))
        self.assertTrue(hasattr(clf_batched, 'estimator'))
