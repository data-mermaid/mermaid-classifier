"""Effective-number-of-samples class weighting (Cui et al. 2019).

This is the sole sample-weighting strategy. The balancing experiments
(see ``docs/research/balancing-experiments.md``) found that balancing the input
distribution via subsampling (``training/subsample``) dominated the
choice of weighting strategy, so the other strategies explored during
experimentation (``tree_balanced``, ``leaf_inverse``, ``decomposed``)
were removed and the registry/ABC indirection collapsed into the single
function below.

Algorithm — weights from the "effective number of samples":

        E_n = (1 - beta^n) / (1 - beta)
        w(class) ~ 1 / E_n(count(class))

where ``beta`` in [0, 1) interpolates between flat (beta=0, equivalent to
uniform weighting) and full inverse-frequency (beta -> 1). Cui et al.
recommend beta in {0.9, 0.99, 0.999, 0.9999} depending on the overall
imbalance ratio. ``beta`` is hard-coded here (see ``BETA``); tuning it is
the natural extension if needed — bump this constant or thread it through
``SampleWeightingOptions``.

This strategy ignores the BA hierarchy and the GF axis; the BA+GF combo
string is treated as an opaque class label.
"""

from __future__ import annotations

from mermaid_classifier.training.sample_weighting.options import (
    SampleWeightingOptions,
)

# Default per Cui et al. 2019 — works well for moderate-to-high imbalance.
# Bump toward 0.99999 for stronger weighting on rare classes.
BETA: float = 0.9999


def compute_class_weights(
    class_counts: dict[str, int],
    options: SampleWeightingOptions,
) -> dict[str, float]:
    """Per-class weights from the effective-number-of-samples formulation.

    Maps each BA+GF combo string in ``class_counts`` to a strictly
    positive weight. Returns ``{}`` when ``options.enabled`` is False or
    ``class_counts`` is empty.

    When ``options.weight_ratio_cap`` is set, any weight above
    ``min_weight * weight_ratio_cap`` is clamped down to that ceiling,
    which bounds how much a single class can dominate the loss.
    """
    if not options.enabled or not class_counts:
        return {}

    weights: dict[str, float] = {}
    for label, count in class_counts.items():
        n = max(int(count), 1)
        effective_n = (1.0 - BETA**n) / (1.0 - BETA)
        weights[label] = 1.0 / max(effective_n, 1e-12)

    # The formula above is always positive; assert the contract the
    # consumer (CrossEntropyLoss weight tensor) relies on.
    for label, weight in weights.items():
        if weight <= 0:
            raise RuntimeError(f"Non-positive weight {weight!r} computed for class {label!r}.")

    # Apply the optional max:min ratio cap.
    cap = options.weight_ratio_cap
    if cap is not None and len(weights) >= 2:
        ceiling = min(weights.values()) * cap
        for label, weight in weights.items():
            if weight > ceiling:
                weights[label] = ceiling

    return weights
