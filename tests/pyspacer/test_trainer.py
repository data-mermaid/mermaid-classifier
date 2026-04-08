"""
Tests for mermaid_classifier.pyspacer.trainer.

Verifies that MermaidTrainer._calibrate_in_batches() produces mathematically
identical results to the standard CalibratedClassifierCV(cv='prefit').fit()
approach, tests TorchMLPClassifier, and tests FeatureDataset.
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

from mermaid_classifier.pyspacer.torch_classifier import (
    OPTIMIZERS,
    FeatureDataset,
    TorchMLPClassifier,
)
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
        trainer = MermaidTrainer(io_batch_size=100)
        clf_batched = trainer._calibrate_in_batches(clf, mock_labels)

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
        trainer = MermaidTrainer(io_batch_size=100)
        clf_batched = trainer._calibrate_in_batches(clf, mock_labels)

        self._assert_calibration_equivalent(clf_standard, clf_batched, X_ref)


class FeatureDatasetTest(unittest.TestCase):
    """Tests for FeatureDataset."""

    def test_shapes_and_indexing(self):
        """Verify tensor shapes and label indexing from mock data."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 20).astype(np.float32)
        classes = ['a', 'b', 'c']
        y = rng.choice(classes, size=100).tolist()
        label_to_idx = {label: idx for idx, label in enumerate(classes)}

        mock_labels = _make_mock_labels(X, y, batch_size=30)
        dataset = FeatureDataset(mock_labels, label_to_idx, io_batch_size=30)

        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset.X.shape, (100, 20))
        self.assertEqual(dataset.y.shape, (100,))
        self.assertEqual(dataset.X.dtype, torch.float32)
        self.assertEqual(dataset.y.dtype, torch.long)

        # Verify label mapping is correct
        for i in range(100):
            x_i, y_i = dataset[i]
            self.assertEqual(x_i.shape, (20,))
            self.assertEqual(y_i.item(), label_to_idx[y[i]])


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

        def batch_gen(batch_size=batch_size, random_seed=0):
            for i in range(0, len(X), batch_size):
                end = min(i + batch_size, len(X))
                x_batch = [X[j] for j in range(i, end)]
                y_batch = [y[j] for j in range(i, end)]
                yield x_batch, y_batch

        m.load_data_in_batches = batch_gen
        m.__len__ = lambda self_: 1
        return m

    mock_labels.train = _make_image_labels_mock(X_train, y_train)
    mock_labels.ref = _make_image_labels_mock(X_ref, y_ref)
    mock_labels.val = _make_image_labels_mock(X_val, y_val)
    mock_labels.label_count = len(y_train) + len(y_ref) + len(y_val)
    return mock_labels


class MermaidTrainerIntegrationTest(unittest.TestCase):
    """Test MermaidTrainer with TorchMLPClassifier end-to-end."""

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

        trainer = MermaidTrainer(io_batch_size=75, class_balancing=True)
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

        trainer = MermaidTrainer(io_batch_size=75)
        clf_cal, val_results, ret_msg = trainer(
            labels, nbr_epochs=1, pc_models=[], clf_type='MLP')

        self.assertIsInstance(clf_cal, CalibratedClassifierCV)
        self.assertIsInstance(clf_cal.estimator, TorchMLPClassifier)
        self.assertIsNone(clf_cal.estimator.class_weight)

    def test_serialize_with_class_balancing(self):
        trainer = MermaidTrainer(io_batch_size=5000, class_balancing=True)
        data = trainer.serialize()
        self.assertTrue(data['class_balancing'])

    def test_serialize_without_class_balancing(self):
        trainer = MermaidTrainer(io_batch_size=5000)
        data = trainer.serialize()
        self.assertNotIn('class_balancing', data)

    def test_adamw_optimizer(self):
        """MermaidTrainer with optimizer='adamw' trains successfully."""
        X_tr, y_tr, X_ref, y_ref, X_val, y_val = self._make_imbalanced_data()
        labels = _make_mock_training_labels(
            X_tr, y_tr, X_ref, y_ref, X_val, y_val, batch_size=75)

        trainer = MermaidTrainer(io_batch_size=75, optimizer='adamw')
        clf_cal, val_results, ret_msg = trainer(
            labels, nbr_epochs=1, pc_models=[], clf_type='MLP')

        self.assertIsInstance(clf_cal, CalibratedClassifierCV)

    def test_explicit_lr_and_hidden_layers(self):
        """Explicit learning_rate and hidden_layer_sizes override heuristic."""
        X_tr, y_tr, X_ref, y_ref, X_val, y_val = self._make_imbalanced_data()
        labels = _make_mock_training_labels(
            X_tr, y_tr, X_ref, y_ref, X_val, y_val, batch_size=75)

        trainer = MermaidTrainer(
            io_batch_size=75, learning_rate=5e-4,
            hidden_layer_sizes=(50, 25))
        clf_cal, _, _ = trainer(
            labels, nbr_epochs=1, pc_models=[], clf_type='MLP')

        self.assertEqual(
            clf_cal.estimator.hidden_layer_sizes, (50, 25))
        self.assertEqual(clf_cal.estimator.learning_rate_init, 5e-4)

    def test_minibatch_size_controls_dataloader(self):
        """minibatch_size controls gradient update size, not IO batch."""
        X_tr, y_tr, X_ref, y_ref, X_val, y_val = self._make_imbalanced_data()
        labels = _make_mock_training_labels(
            X_tr, y_tr, X_ref, y_ref, X_val, y_val, batch_size=75)

        # io_batch_size=75 for disk streaming, minibatch_size=25 for training
        trainer = MermaidTrainer(
            io_batch_size=75, minibatch_size=25)
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
