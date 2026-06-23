"""Builds a small fitted CalibratedClassifierCV(TorchMLPClassifier) the same
way MermaidTrainer does, with no network or MLflow. Shared across artifact
tests."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, _fit_calibrator

from mermaid_classifier.pyspacer.torch_classifier import TorchMLPClassifier


def make_calibrated_model(
    n_features: int = 8,
    n_classes: int = 5,
    n_samples: int = 512,
    seed: int = 0,
):
    """Return (fitted CalibratedClassifierCV, representative feature batch).

    Mirrors MermaidTrainer._calibrate_in_batches: train a TorchMLPClassifier,
    then wrap it in CalibratedClassifierCV(cv="prefit") with a single
    sigmoid-calibrated inner classifier fit via _fit_calibrator.
    """
    rng = np.random.default_rng(seed)
    classes = np.array([f"ba{i}::gf{i}" for i in range(n_classes)])

    # Separable-ish features so calibration has signal.
    centers = rng.normal(0, 3, size=(n_classes, n_features)).astype(np.float32)
    y_idx = rng.integers(0, n_classes, size=n_samples)
    X = (centers[y_idx] + rng.normal(0, 1, size=(n_samples, n_features))
         ).astype(np.float32)
    y = classes[y_idx]

    clf = TorchMLPClassifier(hidden_layer_sizes=(16,), random_state=0)
    for _ in range(20):
        clf.partial_fit(X, y, classes=classes.tolist())

    # CalibratedClassifierCV has no decision_function on TorchMLPClassifier,
    # so calibration runs on predict_proba (softmax) outputs.
    predictions = clf.predict_proba(X)
    calibrated_inner = _fit_calibrator(
        clf, predictions, y, clf.classes_, method="sigmoid")
    wrapper = CalibratedClassifierCV(clf, cv="prefit")
    wrapper.calibrated_classifiers_ = [calibrated_inner]
    wrapper.classes_ = clf.classes_

    return wrapper, X
