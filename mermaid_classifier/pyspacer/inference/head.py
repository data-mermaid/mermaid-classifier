"""CalibratedHead: a TorchScript-friendly module that reproduces
CalibratedClassifierCV(TorchMLPClassifier, cv='prefit', method='sigmoid').predict_proba.

Pipeline (multiclass, K > 2):
  logits = MLP(features)                    # Linear -> ReLU -> ... -> Linear
  p      = softmax(logits)                  # TorchMLPClassifier.predict_proba
  c_k    = sigmoid(-(a_k * p_k + b_k))      # per-class Platt sigmoid
  proba  = c / c.sum(dim=1)                 # row-normalize; uniform if sum == 0
  proba  = where(1 < proba <= 1+1e-5, 1.0)  # sklearn overshoot clip

Note: ``from __future__ import annotations`` (PEP 563) must NOT appear here.
TorchScript's recursive module compiler calls ``ann_to_type`` on class-level
annotations at import time; PEP 563 turns those annotations into strings
(``'int'`` instead of ``int``), causing "Unknown type annotation: 'int'".
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibratedHead(nn.Module):
    n_classes: int

    def __init__(
        self,
        weights: list[torch.Tensor],
        biases: list[torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        super().__init__()
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError(
                f"Calibration parameters a and b must be 1-D tensors; got"
                f" a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}."
            )
        if a.shape != b.shape:
            raise ValueError(
                f"Calibration parameters a and b must have the same shape; got"
                f" a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}."
            )
        if len(weights) != len(biases):
            raise ValueError(
                f"weights and biases must have the same length; got"
                f" {len(weights)} weights and {len(biases)} biases."
            )
        if len(weights) == 0:
            raise ValueError("weights must contain at least one layer.")
        self.linears = nn.ModuleList()
        for w, bias in zip(weights, biases):
            layer = nn.Linear(int(w.shape[1]), int(w.shape[0]))
            with torch.no_grad():
                layer.weight.copy_(w)
                layer.bias.copy_(bias)
            self.linears.append(layer)
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.n_classes = int(a.shape[0])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features
        n = len(self.linears)
        i = 0
        for linear in self.linears:
            x = linear(x)
            if i < n - 1:
                x = F.relu(x)
            i += 1
        # Computed in float32, matching TorchMLPClassifier._forward_probs; the ~1e-7 residual
        # vs sklearn's float64 path is expected and bounded by the export-time parity gate.
        p = F.softmax(x, dim=1)
        c = torch.sigmoid(-(self.a * p + self.b))
        denom = c.sum(dim=1, keepdim=True)
        # Avoid NaN poisoning: both branches of torch.where are evaluated, so
        # use a safe denominator and select the uniform row where denom == 0
        # (sklearn's edge-case fallback).
        nonzero = denom != 0
        safe_denom = torch.where(nonzero, denom, torch.ones_like(denom))
        uniform = torch.full_like(c, 1.0 / float(self.n_classes))
        proba = torch.where(nonzero, c / safe_denom, uniform)
        proba = torch.where(
            (proba > 1.0) & (proba <= 1.0 + 1e-5),
            torch.ones_like(proba),
            proba,
        )
        return proba


def build_calibrated_head(model) -> CalibratedHead:
    """Construct a CalibratedHead from a fitted CalibratedClassifierCV that
    wraps a TorchMLPClassifier with cv='prefit' and method='sigmoid'."""
    calibrated = model.calibrated_classifiers_
    if len(calibrated) != 1:
        raise ValueError(
            f"Expected exactly one calibrated classifier (cv='prefit'), got"
            f" {len(calibrated)}."
        )
    inner = calibrated[0]
    estimator = inner.estimator
    calibrators = inner.calibrators

    if not np.array_equal(estimator.classes_, model.classes_):
        raise ValueError(
            "estimator.classes_ does not match model.classes_; calibrator"
            " column alignment is only valid when they are identical."
        )
    n_classes = len(model.classes_)
    if n_classes <= 2:
        raise ValueError(
            f"CalibratedHead only supports the multiclass (K > 2) path; got"
            f" K={n_classes}. sklearn stores a single calibrator for K == 2."
        )
    if len(calibrators) != n_classes:
        raise ValueError(
            f"Expected {n_classes} per-class calibrators, got"
            f" {len(calibrators)}."
        )

    module = estimator._module
    weights = [lin.weight.detach().clone().float() for lin in module.linears]
    biases = [lin.bias.detach().clone().float() for lin in module.linears]
    a = torch.tensor([float(c.a_) for c in calibrators], dtype=torch.float32)
    b = torch.tensor([float(c.b_) for c in calibrators], dtype=torch.float32)
    return CalibratedHead(weights, biases, a, b)
