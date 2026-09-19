"""Builds a small fitted CalibratedClassifierCV(TorchMLPClassifier) via
MermaidTrainer's real calibration path, with no network or MLflow. Shared
across artifact tests."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np

from mermaid_classifier.pyspacer.torch_classifier import TorchMLPClassifier
from mermaid_classifier.pyspacer.trainer import MermaidTrainer


class _BatchedRefLabels:
    """Stands in for spacer's ImageLabels: MermaidTrainer._calibrate_in_batches
    calls only load_data_in_batches on its ref_labels argument, so this
    replays the already-generated feature/label arrays through it in
    fixed-size chunks."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x = x
        self._y = y

    def load_data_in_batches(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, len(self._x), batch_size):
            end = start + batch_size
            yield self._x[start:end], self._y[start:end]


def make_calibrated_model(
    n_features: int = 8,
    n_classes: int = 5,
    n_samples: int = 512,
    seed: int = 0,
):
    """Return (fitted CalibratedClassifierCV, representative feature batch).

    Drives MermaidTrainer._calibrate_in_batches directly instead of
    reassembling its calibration steps, so this fixture tracks production
    by construction. The method reads only self.batch_size off the
    trainer, so it runs unbound against a lightweight stand-in rather than
    a real MermaidTrainer, which would pull in settings and MLflow.
    """
    rng = np.random.default_rng(seed)
    classes = np.array([f"ba{i}::gf{i}" for i in range(n_classes)])

    # Separable-ish features so calibration has signal.
    centers = rng.normal(0, 3, size=(n_classes, n_features)).astype(np.float32)
    y_idx = rng.integers(0, n_classes, size=n_samples)
    X = (centers[y_idx] + rng.normal(0, 1, size=(n_samples, n_features))).astype(np.float32)
    y = classes[y_idx]

    clf = TorchMLPClassifier(hidden_layer_sizes=(16,), random_state=0)
    for _ in range(20):
        clf.partial_fit(X, y, classes=classes.tolist())

    trainer_stub = SimpleNamespace(batch_size=128)
    ref_labels = _BatchedRefLabels(X, y)
    wrapper = MermaidTrainer._calibrate_in_batches(trainer_stub, clf, ref_labels)

    return wrapper, X
