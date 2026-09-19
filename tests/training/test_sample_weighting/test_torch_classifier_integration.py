"""Integration tests for TorchMLPClassifier + class_weight.

These verify that:
  - The internal weight tensor is built in classes_ order.
  - The weight tensor reaches the loss, so weighting changes training.
  - A class present in y with no weight is refused.
"""

from __future__ import annotations

import unittest

import numpy as np

from mermaid_classifier.pyspacer.torch_classifier import TorchMLPClassifier


def _make_imbalanced_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    # Three classes with strongly imbalanced counts.
    X_a = rng.standard_normal((200, 8)).astype(np.float32) + 0.0
    X_b = rng.standard_normal((100, 8)).astype(np.float32) + 3.0
    X_c = rng.standard_normal((10, 8)).astype(np.float32) + 6.0
    X = np.concatenate([X_a, X_b, X_c])
    y = np.array(["a"] * 200 + ["b"] * 100 + ["c"] * 10)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


class TorchClassifierWeightTest(unittest.TestCase):
    def test_class_weight_tensor_in_classes_order(self):
        X, y = _make_imbalanced_dataset(seed=2)
        clf = TorchMLPClassifier(
            hidden_layer_sizes=(8,),
            max_iter=2,
            random_state=0,
            class_weight={"a": 1.0, "b": 5.0, "c": 25.0},
        )
        clf.fit(X, y)
        self.assertIsNotNone(clf._class_weight_tensor)
        # classes_ is sorted alphabetically -> ["a", "b", "c"]
        self.assertEqual(list(clf.classes_), ["a", "b", "c"])
        np.testing.assert_allclose(clf._class_weight_tensor.numpy(), [1.0, 5.0, 25.0])

    def test_class_weight_tensor_reaches_the_loss(self):
        # The tensor being built correctly says nothing about it being passed
        # to cross_entropy; identical seeds and data must still diverge.
        X, y = _make_imbalanced_dataset(seed=3)
        weights = {"a": 1.0, "b": 2.0, "c": 20.0}

        baseline = TorchMLPClassifier(hidden_layer_sizes=(8,), max_iter=5, random_state=42)
        baseline.fit(X, y)

        weighted = TorchMLPClassifier(
            hidden_layer_sizes=(8,), max_iter=5, random_state=42, class_weight=weights
        )
        weighted.fit(X, y)

        self.assertNotAlmostEqual(
            baseline.loss_curve_[-1],
            weighted.loss_curve_[-1],
            places=5,
        )

    def test_missing_weight_for_class_raises(self):
        X, y = _make_imbalanced_dataset(seed=5)
        clf = TorchMLPClassifier(
            hidden_layer_sizes=(4,),
            max_iter=1,
            random_state=0,
            class_weight={"a": 1.0, "b": 1.0},  # missing 'c'
        )
        with self.assertRaisesRegex(ValueError, "class_weight is missing"):
            clf.fit(X, y)


if __name__ == "__main__":
    unittest.main()
