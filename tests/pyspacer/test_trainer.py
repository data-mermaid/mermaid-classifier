"""
Tests for mermaid_classifier.pyspacer.trainer.

Verifies that MermaidTrainer._calibrate_in_batches() produces mathematically
identical results to the standard CalibratedClassifierCV(cv='prefit').fit()
approach, tests TorchMLPClassifier, and tests StreamingFeatureDataset.
"""

import dataclasses
import pickle
import unittest
from collections import Counter
from unittest import mock

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from spacer.data_classes import DataLocation, ImageFeatures

from mermaid_classifier.pyspacer.torch_classifier import (
    OPTIMIZERS,
    PrefetchDataLoader,
    StreamingFeatureDataset,
    TorchMLPClassifier,
)
from mermaid_classifier.pyspacer.trainer import MermaidTrainer
from pyspacer.npz_test_helpers import make_npz

# ---------------------------------------------------------------------------
# Mock infrastructure for ImageLabels
#
# StreamingFeatureDataset uses two different data paths:
#   1. _load_image() — accesses labels.keys(), labels._data, labels.filesystem_cache,
#      and ImageFeatures.load(). Used by the training loop.
#   2. load_data_in_batches() — used by _calc_acc and _calibrate.
#
# The mock helpers below support both paths from the same data.
# _feature_store is a module-level dict mapping DataLocation → _MockImageFeatures,
# used by _mock_load_features to stand in for ImageFeatures.load().
# ---------------------------------------------------------------------------

_feature_store: dict[DataLocation, '_MockImageFeatures'] = {}

_ANNOS_PER_IMAGE = 10


class _MockImageFeatures:
    """Fake ImageFeatures whose match_with_annotations yields (feat, label)."""

    def __init__(self, features_by_rowcol):
        self._features = features_by_rowcol
        self.valid_rowcol = True

    def match_with_annotations(self, annotations):
        for row, col, label in annotations:
            yield self._features[(row, col)], label


def _mock_load_features(loc, filesystem_cache=None):
    """Drop-in replacement for ImageFeatures.load in tests."""
    return _feature_store[loc]


def _add_parallel_loading_attrs(mock_labels, X, y):
    """Add keys(), _data, filesystem_cache to a mock ImageLabels so
    StreamingFeatureDataset._load_image works."""
    data = {}
    for img_start in range(0, len(X), _ANNOS_PER_IMAGE):
        img_end = min(img_start + _ANNOS_PER_IMAGE, len(X))
        loc = DataLocation(
            'memory', key=f'mock_{id(mock_labels)}_{img_start}')
        data[loc] = [(j, 0, y[j]) for j in range(img_start, img_end)]
        _feature_store[loc] = _MockImageFeatures(
            {(j, 0): X[j] for j in range(img_start, img_end)})
    mock_labels._data = data
    mock_labels.filesystem_cache = None
    mock_labels.keys = lambda: data.keys()


def _make_mock_labels(X, y, batch_size):
    """Create a mock ImageLabels with both load_data_in_batches and
    parallel-loading support."""
    mock_labels = mock.Mock()

    def batch_generator(batch_size=batch_size, random_seed=None):
        for i in range(0, len(X), batch_size):
            end = min(i + batch_size, len(X))
            x_batch = [X[j] for j in range(i, end)]
            y_batch = [y[j] for j in range(i, end)]
            yield x_batch, y_batch

    mock_labels.load_data_in_batches = batch_generator
    _add_parallel_loading_attrs(mock_labels, X, y)
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
        rng = np.random.RandomState(42)
        n_samples = 500
        n_features = 50
        X = rng.randn(n_samples, n_features).astype(np.float32)
        classes = [f'class_{i}' for i in range(n_classes)]
        y = np.array(rng.choice(classes, size=n_samples))

        split = n_samples // 2
        X_train, X_ref = X[:split], X[split:]
        y_train, y_ref = y[:split], y[split:]

        # Train base classifier using sklearn MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=(20,), learning_rate_init=1e-3)
        clf.partial_fit(X_train, y_train, classes=classes)

        # Standard calibration (current approach)
        clf_standard = CalibratedClassifierCV(clf, cv="prefit")
        clf_standard.fit(X_ref, y_ref)

        # Batched calibration (new approach) via mock ImageLabels
        mock_labels = _make_mock_labels(X_ref, y_ref, batch_size=100)
        trainer = MermaidTrainer(minibatch_size=100)
        batch_iter = mock_labels.load_data_in_batches(batch_size=100)
        clf_batched = trainer._calibrate(clf, batch_iter)

        return clf_standard, clf_batched, X_ref

    def test_mlp_multiclass_calibration_equivalence(self):
        """MLPClassifier with 5 classes: batch == standard."""
        std, batched, X = self._train_and_calibrate_both_ways(5)
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
                    cal_std.a_, cal_batch.a_, rtol=1e-3,
                    err_msg="Sigmoid parameter 'a' differs")
                np.testing.assert_allclose(
                    cal_std.b_, cal_batch.b_, rtol=1e-3,
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

    def test_torch_mlp_multiclass_calibration_equivalence(self):
        """TorchMLPClassifier with 5 classes: batch calibration works."""
        rng = np.random.RandomState(42)
        n_samples = 500
        n_features = 50
        X = rng.randn(n_samples, n_features).astype(np.float32)
        classes = [f'class_{i}' for i in range(5)]
        y = np.array(rng.choice(classes, size=n_samples))

        split = n_samples // 2
        X_train, X_ref = X[:split], X[split:]
        y_train, y_ref = y[:split], y[split:]

        clf = TorchMLPClassifier(
            hidden_layer_sizes=(20,), learning_rate_init=1e-3)
        clf.init_model(X_train.shape[1], classes)
        label_to_idx = {label: idx for idx, label in enumerate(classes)}
        x_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(
            [label_to_idx[l] for l in y_train], dtype=torch.long)
        clf.train_step(x_t, y_t)

        # Standard calibration
        clf_standard = CalibratedClassifierCV(clf, cv="prefit")
        clf_standard.fit(X_ref, y_ref)

        # Batched calibration
        mock_labels = _make_mock_labels(X_ref, y_ref, batch_size=100)
        trainer = MermaidTrainer(minibatch_size=100)
        batch_iter = mock_labels.load_data_in_batches(batch_size=100)
        clf_batched = trainer._calibrate(clf, batch_iter)

        self._assert_calibration_equivalent(clf_standard, clf_batched, X_ref)


class StreamingFeatureDatasetTest(unittest.TestCase):
    """Tests for StreamingFeatureDataset."""

    def setUp(self):
        _feature_store.clear()
        self._patcher = mock.patch.object(
            ImageFeatures, 'load', side_effect=_mock_load_features)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        _feature_store.clear()

    def test_batch_count_and_shapes(self):
        """Verify correct number of batches and tensor shapes."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 20).astype(np.float32)
        classes = ['a', 'b', 'c']
        y = rng.choice(classes, size=100).tolist()
        label_to_idx = {label: idx for idx, label in enumerate(classes)}

        mock_labels = _make_mock_labels(X, y, batch_size=30)
        dataset = StreamingFeatureDataset(
            mock_labels, label_to_idx, batch_size=30)

        batches = list(dataset)
        # 100 samples / 30 batch_size = 3 full + 1 remainder of 10
        self.assertEqual(len(batches), 4)

        # Full batches
        for x_batch, y_batch in batches[:3]:
            self.assertEqual(x_batch.shape, (30, 20))
            self.assertEqual(y_batch.shape, (30,))
            self.assertEqual(x_batch.dtype, torch.float32)
            self.assertEqual(y_batch.dtype, torch.long)

        # Remainder batch
        x_last, y_last = batches[-1]
        self.assertEqual(x_last.shape, (10, 20))
        self.assertEqual(y_last.shape, (10,))

    def test_label_indexing(self):
        """Verify label mapping is correct across all batches."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10).astype(np.float32)
        classes = ['a', 'b', 'c']
        y = rng.choice(classes, size=50).tolist()
        label_to_idx = {label: idx for idx, label in enumerate(classes)}

        mock_labels = _make_mock_labels(X, y, batch_size=20)
        dataset = StreamingFeatureDataset(
            mock_labels, label_to_idx, batch_size=20)

        all_labels = []
        for _, y_batch in dataset:
            all_labels.extend(y_batch.tolist())

        expected = [label_to_idx[label] for label in y]
        self.assertEqual(all_labels, expected)

    def test_exact_batch_size(self):
        """When data is exactly divisible, no remainder batch."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 10).astype(np.float32)
        y = rng.choice(['a', 'b'], size=60).tolist()
        label_to_idx = {'a': 0, 'b': 1}

        mock_labels = _make_mock_labels(X, y, batch_size=20)
        dataset = StreamingFeatureDataset(
            mock_labels, label_to_idx, batch_size=20)

        batches = list(dataset)
        self.assertEqual(len(batches), 3)
        for x_batch, _ in batches:
            self.assertEqual(x_batch.shape[0], 20)

    def test_parallel_loading_same_total_samples(self):
        """num_workers>1 produces the same total samples as sequential."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 20).astype(np.float32)
        classes = ['a', 'b', 'c']
        y = rng.choice(classes, size=100).tolist()
        label_to_idx = {label: idx for idx, label in enumerate(classes)}

        mock_labels = _make_mock_labels(X, y, batch_size=30)

        seq = StreamingFeatureDataset(
            mock_labels, label_to_idx, batch_size=30, num_workers=1)
        par = StreamingFeatureDataset(
            mock_labels, label_to_idx, batch_size=30, num_workers=2)

        seq_total = sum(x.shape[0] for x, _ in seq)
        par_total = sum(x.shape[0] for x, _ in par)
        self.assertEqual(seq_total, 100)
        self.assertEqual(par_total, 100)


class TorchMLPClassifierTest(unittest.TestCase):
    """Tests for TorchMLPClassifier."""

    def _make_data(self, n_samples=200, n_features=20, n_classes=5):
        rng = np.random.RandomState(42)
        X = rng.randn(n_samples, n_features).astype(np.float32)
        classes = [f'class_{i}' for i in range(n_classes)]
        y = rng.choice(classes, size=n_samples).tolist()
        return X, y, classes

    def _make_tensors(self, X, y, classes):
        """Convert numpy data to tensors suitable for train_step."""
        label_to_idx = {label: idx for idx, label in enumerate(classes)}
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(
            [label_to_idx[l] for l in y], dtype=torch.long)
        return x_tensor, y_tensor

    def test_init_model_sets_classes(self):
        X, y, classes = self._make_data()
        clf = TorchMLPClassifier(hidden_layer_sizes=(20,))
        self.assertIsNone(clf.classes_)

        clf.init_model(X.shape[1], classes)
        self.assertIsNotNone(clf.classes_)
        self.assertEqual(len(clf.classes_), 5)
        self.assertEqual(set(clf.classes_), set(classes))

    def test_predict_returns_valid_labels(self):
        X, y, classes = self._make_data()
        clf = TorchMLPClassifier(hidden_layer_sizes=(20,))
        clf.init_model(X.shape[1], classes)
        x_t, y_t = self._make_tensors(X, y, classes)
        clf.train_step(x_t, y_t)

        preds = clf.predict(X[:10])
        for p in preds:
            self.assertIn(p, classes)

    def test_predict_proba_shape_and_sum(self):
        X, y, classes = self._make_data()
        clf = TorchMLPClassifier(hidden_layer_sizes=(20,))
        clf.init_model(X.shape[1], classes)
        x_t, y_t = self._make_tensors(X, y, classes)
        clf.train_step(x_t, y_t)

        probs = clf.predict_proba(X[:10])
        self.assertEqual(probs.shape, (10, 5))
        np.testing.assert_allclose(
            probs.sum(axis=1), np.ones(10), atol=1e-6)
        self.assertEqual(probs.dtype, np.float64)

    def test_loss_curve_populated(self):
        X, y, classes = self._make_data()
        clf = TorchMLPClassifier(hidden_layer_sizes=(20,))
        clf.init_model(X.shape[1], classes)
        self.assertEqual(len(clf.loss_curve_), 0)

        x_t, y_t = self._make_tensors(X, y, classes)
        clf.train_step(x_t[:100], y_t[:100])
        self.assertEqual(len(clf.loss_curve_), 1)

        clf.train_step(x_t[100:], y_t[100:])
        self.assertEqual(len(clf.loss_curve_), 2)

    def test_pickle_roundtrip(self):
        X, y, classes = self._make_data()
        clf = TorchMLPClassifier(hidden_layer_sizes=(20,))
        clf.init_model(X.shape[1], classes)
        x_t, y_t = self._make_tensors(X, y, classes)
        clf.train_step(x_t, y_t)

        preds_before = clf.predict(X[:10])

        # Round-trip through pickle protocol=2 (matching PySpacer)
        data = pickle.dumps(clf, protocol=2)
        clf2 = pickle.loads(data)

        preds_after = clf2.predict(X[:10])
        np.testing.assert_array_equal(preds_before, preds_after)
        self.assertEqual(len(clf2.loss_curve_), len(clf.loss_curve_))
        np.testing.assert_array_equal(clf2.classes_, clf.classes_)

    def test_calibration_compatibility(self):
        X, y, classes = self._make_data(n_samples=300)
        clf = TorchMLPClassifier(hidden_layer_sizes=(20,))
        clf.init_model(X.shape[1], classes)
        x_t, y_t = self._make_tensors(X[:200], y[:200], classes)
        clf.train_step(x_t, y_t)

        # Wrap in CalibratedClassifierCV — the production path
        cal_clf = CalibratedClassifierCV(clf, cv='prefit')
        cal_clf.fit(X[200:], y[200:])

        probs = cal_clf.predict_proba(X[:10])
        self.assertEqual(probs.shape, (10, 5))
        preds = cal_clf.predict(X[:10])
        for p in preds:
            self.assertIn(p, classes)

    def test_no_decision_function(self):
        clf = TorchMLPClassifier(hidden_layer_sizes=(20,))
        self.assertFalse(hasattr(clf, 'decision_function'))

    def test_class_weight_balanced(self):
        """Verify class weighting improves minority class accuracy."""
        rng = np.random.RandomState(42)
        n_features = 20
        # Heavily imbalanced: 450 of 'a', 50 of 'b'
        X = rng.randn(500, n_features).astype(np.float32)
        y = ['a'] * 450 + ['b'] * 50
        classes = ['a', 'b']
        perm = rng.permutation(500)
        X = X[perm]
        y = [y[i] for i in perm]

        # Compute balanced weights
        total = 500
        n_cls = 2
        weights = {'a': total / (n_cls * 450), 'b': total / (n_cls * 50)}

        clf_weighted = TorchMLPClassifier(
            hidden_layer_sizes=(20,), learning_rate_init=1e-3,
            class_weight=weights)
        clf_unweighted = TorchMLPClassifier(
            hidden_layer_sizes=(20,), learning_rate_init=1e-3)

        clf_weighted.init_model(n_features, classes)
        clf_unweighted.init_model(n_features, classes)

        label_to_idx = {label: idx for idx, label in enumerate(classes)}
        x_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(
            [label_to_idx[l] for l in y], dtype=torch.long)

        for _ in range(5):
            clf_weighted.train_step(x_t, y_t)
            clf_unweighted.train_step(x_t, y_t)

        # Evaluate on minority class 'b' samples
        b_indices = [i for i, label in enumerate(y) if label == 'b']
        b_X = X[b_indices]

        preds_w = clf_weighted.predict(b_X)
        preds_u = clf_unweighted.predict(b_X)

        b_acc_weighted = sum(1 for p in preds_w if p == 'b') / len(preds_w)
        b_acc_unweighted = sum(1 for p in preds_u if p == 'b') / len(preds_u)

        # Weighted should do at least as well on minority class
        self.assertGreaterEqual(b_acc_weighted, b_acc_unweighted)

    def test_class_weight_none_default(self):
        clf = TorchMLPClassifier(hidden_layer_sizes=(20,))
        self.assertIsNone(clf.class_weight)

    def test_zero_count_class_weight(self):
        """A class with weight=1.0 (zero-count fallback) does not crash."""
        X, y, classes = self._make_data(n_classes=3)
        # Weight for class_2 = 1.0 (simulating zero-count)
        weights = {'class_0': 2.0, 'class_1': 0.5, 'class_2': 1.0}

        clf = TorchMLPClassifier(
            hidden_layer_sizes=(20,), class_weight=weights)
        clf.init_model(X.shape[1], classes)
        x_t, y_t = self._make_tensors(X, y, classes)
        clf.train_step(x_t, y_t)
        preds = clf.predict(X[:10])
        self.assertEqual(len(preds), 10)


def _make_mock_training_labels(X_train, y_train, X_ref, y_ref,
                               X_val, y_val, batch_size):
    """Create a mock TrainingTaskLabels with train/ref/val splits."""
    mock_labels = mock.Mock()

    def _make_image_labels_mock(X, y):
        m = mock.Mock()
        m.label_count = len(y)
        m.classes_set = set(y)
        m.label_count_per_class = Counter(y)

        def batch_gen(batch_size=batch_size, random_seed=None):
            for i in range(0, len(X), batch_size):
                end = min(i + batch_size, len(X))
                x_batch = [X[j] for j in range(i, end)]
                y_batch = [y[j] for j in range(i, end)]
                yield x_batch, y_batch

        m.load_data_in_batches = batch_gen
        m.__len__ = lambda self_: 1
        _add_parallel_loading_attrs(m, X, y)
        return m

    mock_labels.train = _make_image_labels_mock(X_train, y_train)
    mock_labels.ref = _make_image_labels_mock(X_ref, y_ref)
    mock_labels.val = _make_image_labels_mock(X_val, y_val)
    mock_labels.label_count = len(y_train) + len(y_ref) + len(y_val)
    return mock_labels


class MermaidTrainerIntegrationTest(unittest.TestCase):
    """Test MermaidTrainer with TorchMLPClassifier end-to-end."""

    def setUp(self):
        _feature_store.clear()
        self._patcher = mock.patch.object(
            ImageFeatures, 'load', side_effect=_mock_load_features)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        _feature_store.clear()

    def _make_imbalanced_data(self):
        rng = np.random.RandomState(42)
        n_features = 20
        X = rng.randn(300, n_features).astype(np.float32)
        y = np.array(['a'] * 200 + ['b'] * 60 + ['c'] * 40)
        perm = rng.permutation(300)
        X, y = X[perm], y[perm]
        return (X[:150], y[:150].tolist(),
                X[150:225], y[150:225].tolist(),
                X[225:], y[225:].tolist())

    def test_mlp_with_class_balancing(self):
        """MermaidTrainer with class_balancing=True uses TorchMLPClassifier."""
        X_tr, y_tr, X_ref, y_ref, X_val, y_val = self._make_imbalanced_data()
        labels = _make_mock_training_labels(
            X_tr, y_tr, X_ref, y_ref, X_val, y_val, batch_size=75)

        trainer = MermaidTrainer(class_balancing=True)
        clf_cal, val_results, ret_msg = trainer(
            labels, nbr_epochs=2, pc_models=[], clf_type='MLP')

        self.assertIsInstance(clf_cal, CalibratedClassifierCV)
        self.assertIsInstance(clf_cal.estimator, TorchMLPClassifier)
        self.assertIsNotNone(clf_cal.estimator.class_weight)

    def test_mlp_without_class_balancing(self):
        """MermaidTrainer default uses TorchMLPClassifier without weights."""
        X_tr, y_tr, X_ref, y_ref, X_val, y_val = self._make_imbalanced_data()
        labels = _make_mock_training_labels(
            X_tr, y_tr, X_ref, y_ref, X_val, y_val, batch_size=75)

        trainer = MermaidTrainer()
        clf_cal, val_results, ret_msg = trainer(
            labels, nbr_epochs=1, pc_models=[], clf_type='MLP')

        self.assertIsInstance(clf_cal, CalibratedClassifierCV)
        self.assertIsInstance(clf_cal.estimator, TorchMLPClassifier)
        self.assertIsNone(clf_cal.estimator.class_weight)

    def test_serialize_with_class_balancing(self):
        trainer = MermaidTrainer(class_balancing=True)
        data = trainer.serialize()
        self.assertTrue(data['class_balancing'])

    def test_serialize_without_class_balancing(self):
        trainer = MermaidTrainer()
        data = trainer.serialize()
        self.assertNotIn('class_balancing', data)

    def test_adamw_optimizer(self):
        """MermaidTrainer with optimizer='adamw' trains successfully."""
        X_tr, y_tr, X_ref, y_ref, X_val, y_val = self._make_imbalanced_data()
        labels = _make_mock_training_labels(
            X_tr, y_tr, X_ref, y_ref, X_val, y_val, batch_size=75)

        trainer = MermaidTrainer(optimizer='adamw')
        clf_cal, val_results, ret_msg = trainer(
            labels, nbr_epochs=1, pc_models=[], clf_type='MLP')

        self.assertIsInstance(clf_cal, CalibratedClassifierCV)

    def test_explicit_lr_and_hidden_layers(self):
        """Explicit learning_rate and hidden_layer_sizes override heuristic."""
        X_tr, y_tr, X_ref, y_ref, X_val, y_val = self._make_imbalanced_data()
        labels = _make_mock_training_labels(
            X_tr, y_tr, X_ref, y_ref, X_val, y_val, batch_size=75)

        trainer = MermaidTrainer(
            learning_rate=5e-4,
            hidden_layer_sizes=(50, 25))
        clf_cal, _, _ = trainer(
            labels, nbr_epochs=1, pc_models=[], clf_type='MLP')

        self.assertEqual(
            clf_cal.estimator.hidden_layer_sizes, (50, 25))
        self.assertEqual(clf_cal.estimator.learning_rate_init, 5e-4)

    def test_minibatch_size_controls_gradient_steps(self):
        """minibatch_size controls gradient update size."""
        X_tr, y_tr, X_ref, y_ref, X_val, y_val = self._make_imbalanced_data()
        labels = _make_mock_training_labels(
            X_tr, y_tr, X_ref, y_ref, X_val, y_val, batch_size=75)

        trainer = MermaidTrainer(minibatch_size=25)
        clf_cal, _, ret_msg = trainer(
            labels, nbr_epochs=1, pc_models=[], clf_type='MLP')

        # With 150 training samples and minibatch_size=25,
        # expect 6 gradient steps per epoch (150/25)
        self.assertEqual(len(clf_cal.estimator.loss_curve_), 6)


class TorchMLPClassifierDeviceTest(unittest.TestCase):
    """Tests for device and optimizer selection in TorchMLPClassifier."""

    def test_explicit_cpu_device(self):
        clf = TorchMLPClassifier(
            hidden_layer_sizes=(20,), device='cpu')
        self.assertEqual(clf.device, torch.device('cpu'))

    def test_optimizer_selection(self):
        """Each optimizer string maps to the correct torch class."""
        for name, opt_cls in OPTIMIZERS.items():
            clf = TorchMLPClassifier(
                hidden_layer_sizes=(20,), optimizer=name)
            clf.init_model(10, ['a', 'b'])
            self.assertIsInstance(clf._optimizer, opt_cls)

    def test_weight_decay_passed_through(self):
        clf = TorchMLPClassifier(
            hidden_layer_sizes=(20,), weight_decay=0.01)
        clf.init_model(10, ['a', 'b'])
        # Adam stores weight_decay in param_groups
        self.assertEqual(
            clf._optimizer.param_groups[0]['weight_decay'], 0.01)

    def test_pickle_roundtrip_preserves_optimizer_type(self):
        clf = TorchMLPClassifier(
            hidden_layer_sizes=(20,), optimizer='adamw',
            weight_decay=0.05)
        clf.init_model(10, ['a', 'b', 'c'])
        x_t = torch.randn(5, 10)
        y_t = torch.tensor([0, 1, 2, 0, 1])
        clf.train_step(x_t, y_t)

        data = pickle.dumps(clf, protocol=2)
        clf2 = pickle.loads(data)

        self.assertEqual(clf2.optimizer, 'adamw')
        self.assertEqual(clf2.weight_decay, 0.05)
        # Unpickled model restores on CPU
        self.assertEqual(clf2.device, torch.device('cpu'))


class GradientAccumulationTest(unittest.TestCase):
    """Tests for IO batch / minibatch decoupling and gradient accumulation."""

    def setUp(self):
        _feature_store.clear()
        self._patcher = mock.patch.object(
            ImageFeatures, 'load', side_effect=_mock_load_features)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        _feature_store.clear()

    def _make_imbalanced_data(self):
        rng = np.random.RandomState(42)
        n_features = 20
        X = rng.randn(300, n_features).astype(np.float32)
        y = np.array(['a'] * 200 + ['b'] * 60 + ['c'] * 40)
        perm = rng.permutation(300)
        X, y = X[perm], y[perm]
        return (X[:150], y[:150].tolist(),
                X[150:225], y[150:225].tolist(),
                X[225:], y[225:].tolist())

    def test_gradient_accumulation(self):
        """io_batch_size < minibatch_size triggers gradient accumulation."""
        X_tr, y_tr, X_ref, y_ref, X_val, y_val = self._make_imbalanced_data()
        labels = _make_mock_training_labels(
            X_tr, y_tr, X_ref, y_ref, X_val, y_val, batch_size=75)

        # minibatch_size=75, io_batch_size=25 → 3 accumulation steps
        # 150 samples / 75 effective minibatch = 2 gradient steps
        trainer = MermaidTrainer(minibatch_size=75, io_batch_size=25)
        clf_cal, _, ret_msg = trainer(
            labels, nbr_epochs=1, pc_models=[], clf_type='MLP')

        self.assertEqual(len(clf_cal.estimator.loss_curve_), 2)

    def test_gradient_accumulation_remainder(self):
        """Remainder sub-batches still produce an optimizer step."""
        X_tr, y_tr, X_ref, y_ref, X_val, y_val = self._make_imbalanced_data()
        labels = _make_mock_training_labels(
            X_tr, y_tr, X_ref, y_ref, X_val, y_val, batch_size=75)

        # minibatch_size=100, io_batch_size=50 → accum_steps=2
        # Batches from dataset: 50,50,50 → step after batch 2 (100),
        # remainder of 50 → 1 more step. Total: 2 gradient steps.
        trainer = MermaidTrainer(minibatch_size=100, io_batch_size=50)
        clf_cal, _, _ = trainer(
            labels, nbr_epochs=1, pc_models=[], clf_type='MLP')

        self.assertEqual(len(clf_cal.estimator.loss_curve_), 2)

    def test_no_accumulation_when_io_exceeds_minibatch(self):
        """When io_batch_size >= minibatch_size, no accumulation occurs."""
        X_tr, y_tr, X_ref, y_ref, X_val, y_val = self._make_imbalanced_data()
        labels = _make_mock_training_labels(
            X_tr, y_tr, X_ref, y_ref, X_val, y_val, batch_size=75)

        # io_batch_size=10000 > minibatch_size=50 → effective_io=50,
        # accum_steps=1 → 150/50 = 3 steps
        trainer = MermaidTrainer(minibatch_size=50, io_batch_size=10_000)
        clf_cal, _, _ = trainer(
            labels, nbr_epochs=1, pc_models=[], clf_type='MLP')

        self.assertEqual(len(clf_cal.estimator.loss_curve_), 3)

    def test_forward_backward_no_step(self):
        """forward_backward() accumulates gradients without stepping."""
        X, y, classes = (
            np.random.RandomState(42).randn(50, 10).astype(np.float32),
            ['a', 'b'] * 25,
            ['a', 'b'],
        )
        clf = TorchMLPClassifier(hidden_layer_sizes=(10,))
        clf.init_model(X.shape[1], classes)

        label_to_idx = {l: i for i, l in enumerate(classes)}
        x_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(
            [label_to_idx[l] for l in y], dtype=torch.long)

        # Save initial params
        params_before = {
            k: v.clone() for k, v in clf._model.named_parameters()}

        clf._optimizer.zero_grad()
        clf.forward_backward(x_t, y_t, loss_scale=0.5)

        # Gradients should exist but params unchanged (no step yet)
        for name, param in clf._model.named_parameters():
            self.assertIsNotNone(param.grad)
            torch.testing.assert_close(param.data, params_before[name])

        # After step, params should change
        clf.optimizer_step()
        any_changed = any(
            not torch.equal(param.data, params_before[name])
            for name, param in clf._model.named_parameters())
        self.assertTrue(any_changed)

    def test_serialize_io_batch_size(self):
        trainer = MermaidTrainer(io_batch_size=5000)
        data = trainer.serialize()
        self.assertEqual(data['io_batch_size'], 5000)

    def test_serialize_io_workers(self):
        trainer = MermaidTrainer(io_workers=8)
        data = trainer.serialize()
        self.assertEqual(data['io_workers'], 8)

    def test_gradient_accumulation_with_parallel_workers(self):
        """Gradient accumulation works with io_workers > 1."""
        X_tr, y_tr, X_ref, y_ref, X_val, y_val = self._make_imbalanced_data()
        labels = _make_mock_training_labels(
            X_tr, y_tr, X_ref, y_ref, X_val, y_val, batch_size=75)

        trainer = MermaidTrainer(
            minibatch_size=75, io_batch_size=25, io_workers=2)
        clf_cal, _, _ = trainer(
            labels, nbr_epochs=1, pc_models=[], clf_type='MLP')

        self.assertEqual(len(clf_cal.estimator.loss_curve_), 2)


class PrefetchDataLoaderTest(unittest.TestCase):
    """Tests for PrefetchDataLoader."""

    def test_basic_iteration(self):
        data = list(range(10))
        result = list(PrefetchDataLoader(data))
        self.assertEqual(result, data)

    def test_exception_propagation(self):
        def bad_generator():
            yield 1
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            list(PrefetchDataLoader(bad_generator()))

    def test_empty_iterable(self):
        result = list(PrefetchDataLoader([]))
        self.assertEqual(result, [])


def _make_npz_backed_labels(cache_dir, X_data, y_data, batch_size):
    """Create mock ImageLabels backed by real .npz files on disk."""
    data = {}
    n_features = X_data.shape[1]

    for img_start in range(0, len(X_data), _ANNOS_PER_IMAGE):
        img_end = min(img_start + _ANNOS_PER_IMAGE, len(X_data))
        key = f'features/img_{id(X_data)}_{img_start}.npz'
        loc = DataLocation('filesystem', key=key)

        n_points = img_end - img_start
        # Use local row indices (0..n_points-1) matching the .npz file
        rows = list(range(n_points))
        cols = [0] * n_points
        feat = X_data[img_start:img_end]

        make_npz(cache_dir, key, rows, cols, feat)

        annotations = [
            (rows[j], 0, y_data[img_start + j])
            for j in range(n_points)
        ]
        data[loc] = annotations

    m = mock.Mock()
    m._data = data
    m.keys = lambda: data.keys()
    m.label_count = len(y_data)
    m.classes_set = set(y_data)
    m.label_count_per_class = Counter(y_data)
    m.__len__ = lambda self_: len(data)

    def batch_gen(batch_size=batch_size, random_seed=None):
        for i in range(0, len(X_data), batch_size):
            end = min(i + batch_size, len(X_data))
            x_batch = [X_data[j] for j in range(i, end)]
            y_batch = [y_data[j] for j in range(i, end)]
            yield x_batch, y_batch

    m.load_data_in_batches = batch_gen
    return m


class MaterializedTrainingPathTest(unittest.TestCase):
    """Test MermaidTrainer with materialize_data=True using real .npz files."""

    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir)

    def test_materialized_training_produces_valid_model(self):
        """Full training with materialization produces a calibrated model."""
        rng = np.random.RandomState(42)
        n_features = 20
        X = rng.randn(300, n_features).astype(np.float32)
        y = np.array(['a'] * 200 + ['b'] * 60 + ['c'] * 40)
        perm = rng.permutation(300)
        X, y = X[perm], y[perm]

        X_tr, y_tr = X[:150], y[:150].tolist()
        X_ref, y_ref = X[150:225], y[150:225].tolist()
        X_val, y_val = X[225:], y[225:].tolist()

        labels = mock.Mock()
        labels.train = _make_npz_backed_labels(
            self.tmp_dir, X[:150], y_tr, batch_size=75)
        labels.ref = _make_npz_backed_labels(
            self.tmp_dir, X[150:225], y_ref, batch_size=75)
        labels.val = _make_npz_backed_labels(
            self.tmp_dir, X[225:], y_val, batch_size=75)
        labels.label_count = 300
        labels.set_filesystem_cache = mock.Mock()

        epoch_metrics = []

        trainer = MermaidTrainer(
            materialize_data=True,
            feature_cache_dir=self.tmp_dir,
            on_epoch_end=lambda m: epoch_metrics.append(m),
        )
        clf_cal, val_results, ret_msg = trainer(
            labels, nbr_epochs=2, pc_models=[], clf_type='MLP')

        self.assertIsInstance(clf_cal, CalibratedClassifierCV)
        self.assertIsNotNone(trainer._materialization_seconds)
        self.assertGreater(trainer._materialization_seconds, 0)

        # Verify timing metrics are in epoch callbacks
        self.assertEqual(len(epoch_metrics), 2)
        for m in epoch_metrics:
            self.assertIn('epoch_seconds', m)
            self.assertIn('data_load_seconds', m)
            self.assertIn('gpu_compute_seconds', m)
            self.assertIn('ref_accuracy_seconds', m)

        # Verify materialized dir is cleaned up
        from pathlib import Path
        self.assertFalse(
            Path(self.tmp_dir, '_materialized').exists())

    def test_serialize_includes_materialize_data(self):
        trainer = MermaidTrainer(materialize_data=True)
        data = trainer.serialize()
        self.assertTrue(data['materialize_data'])

        trainer2 = MermaidTrainer(materialize_data=False)
        data2 = trainer2.serialize()
        self.assertFalse(data2['materialize_data'])


class FlattenDataclassTest(unittest.TestCase):
    """Tests for _flatten_dataclass_for_logging."""

    def test_basic_flattening(self):
        from mermaid_classifier.pyspacer.train import (
            _flatten_dataclass_for_logging,
        )

        @dataclasses.dataclass
        class Opts:
            name: str = 'test'
            count: int = 5
            path: str = '/some/dir/file.csv'
            ratio: tuple = (0.1, 0.2)
            missing: str | None = None

        result = _flatten_dataclass_for_logging(Opts(), prefix='p.')
        self.assertEqual(result['p.name'], 'test')
        self.assertEqual(result['p.count'], 5)
        self.assertEqual(result['p.path'], 'file.csv')  # basename
        self.assertEqual(result['p.ratio'], '(0.1, 0.2)')  # tuple→str
        self.assertEqual(result['p.missing'], '')  # None→''

    def test_training_options_all_fields_logged(self):
        from mermaid_classifier.pyspacer.config import TrainingOptions
        from mermaid_classifier.pyspacer.train import (
            _flatten_dataclass_for_logging,
        )
        opts = TrainingOptions(epochs=5, optimizer='adamw')
        result = _flatten_dataclass_for_logging(opts, prefix='training.')
        # Every field in TrainingOptions should appear
        for field in dataclasses.fields(opts):
            self.assertIn(f'training.{field.name}', result)
