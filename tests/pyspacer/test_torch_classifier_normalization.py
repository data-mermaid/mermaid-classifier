"""Tests for the float64 row-sum renormalization in
TorchMLPClassifier._forward_probs.

Background: softmax is computed in float32 and cast to float64; sklearn's
log_loss validates row sums against a tight float64 tolerance and emits a
"y_pred values do not sum to one" warning at every epoch otherwise. The
classifier now renormalizes after the cast so log_loss stays quiet --
and emits its own RuntimeWarning if the pre-renorm drift is larger than
expected float32 accumulation error, which would indicate a real
numerical issue (NaN/Inf logits, bypassed softmax) rather than rounding.
"""

from __future__ import annotations

import unittest
import warnings
from unittest import mock

import numpy as np
import torch

from mermaid_classifier.pyspacer.torch_classifier import TorchMLPClassifier


def _tiny_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((60, 4)).astype(np.float32)
    y = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def _fit(clf, X, y) -> TorchMLPClassifier:
    clf.fit(X, y)
    return clf


class RowSumRenormalizationTest(unittest.TestCase):
    """Post-renormalization, row sums must be tight enough to pass
    sklearn.metrics.log_loss's check (atol ≈ sqrt(float64 eps) ≈ 1.5e-8).
    Re-accumulation after dividing by the row sum leaves a 1-ULP residual
    (~1e-16), so we assert atol=1e-12 -- comfortably tighter than what
    sklearn requires and four orders of magnitude tighter than the
    pre-renormalization float32 drift (~1e-7) that triggers the warning.
    """

    def test_row_sums_within_sklearn_tolerance(self):
        X, y = _tiny_dataset(seed=1)
        clf = _fit(
            TorchMLPClassifier(
                hidden_layer_sizes=(8,),
                max_iter=2,
                random_state=0,
            ),
            X,
            y,
        )
        probs = clf.predict_proba(X)
        np.testing.assert_allclose(
            probs.sum(axis=1),
            np.ones(len(probs), dtype=np.float64),
            rtol=0,
            atol=1e-12,
        )


class NoSpuriousDriftWarningTest(unittest.TestCase):
    """A normally-trained classifier must NOT trip the drift warning
    on its own softmax output. If it does, the threshold is too tight
    or normalization is broken.
    """

    def test_normal_predict_proba_emits_no_runtime_warning(self):
        X, y = _tiny_dataset(seed=2)
        clf = _fit(
            TorchMLPClassifier(
                hidden_layer_sizes=(8,),
                max_iter=2,
                random_state=0,
            ),
            X,
            y,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf.predict_proba(X)
        drift_warnings = [
            w
            for w in caught
            if issubclass(w.category, RuntimeWarning) and "row sums deviate" in str(w.message)
        ]
        self.assertEqual(
            drift_warnings,
            [],
            f"Unexpected drift warning(s): {[str(w.message) for w in drift_warnings]}",
        )


class AnomalousDriftTriggersWarningTest(unittest.TestCase):
    """If softmax output is sufficiently off (extreme logits, bypassed
    softmax, NaN/Inf), the classifier must raise a RuntimeWarning so the
    user notices instead of silently renormalizing.
    """

    def _fit_minimal(self) -> TorchMLPClassifier:
        X, y = _tiny_dataset(seed=3)
        return _fit(
            TorchMLPClassifier(
                hidden_layer_sizes=(8,),
                max_iter=2,
                random_state=0,
            ),
            X,
            y,
        )

    def test_drifted_rows_trigger_warning_and_still_renormalize(self):
        clf = self._fit_minimal()
        X_eval, _ = _tiny_dataset(seed=4)

        real_softmax = torch.nn.functional.softmax

        def _broken_softmax(logits, dim):
            # Return rows that sum to 1.5 -- a 50% drift, far above the
            # 1e-4 tolerance. Simulates the kind of breakage we want to
            # surface (e.g. softmax bypassed, double-applied, etc.).
            out = real_softmax(logits, dim=dim)
            return out * 1.5

        target = "mermaid_classifier.pyspacer.torch_classifier.F.softmax"
        with (
            mock.patch(target, side_effect=_broken_softmax),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            probs = clf.predict_proba(X_eval)

        drift_warnings = [
            w
            for w in caught
            if issubclass(w.category, RuntimeWarning) and "row sums deviate" in str(w.message)
        ]
        self.assertEqual(
            len(drift_warnings),
            1,
            f"Expected exactly one drift warning, got {len(drift_warnings)}. "
            f"All warnings: {[str(w.message) for w in caught]}",
        )
        # Renormalization must still produce valid output, tight enough
        # for sklearn's log_loss row-sum check (see
        # RowSumRenormalizationTest for the tolerance rationale).
        np.testing.assert_allclose(
            probs.sum(axis=1),
            np.ones(len(probs), dtype=np.float64),
            rtol=0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
