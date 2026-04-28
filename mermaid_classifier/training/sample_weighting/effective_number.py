"""EffectiveNumberOfSamplesStrategy — class-balanced loss (Cui et al. 2019).

Algorithm: weights computed from the "effective number of samples"
formulation:

        E_n = (1 - beta^n) / (1 - beta)
        w(class) ~ 1 / E_n(count(class))

where beta in [0, 1) interpolates between flat (beta=0, equivalent to
uniform weighting) and full inverse-frequency (beta -> 1). The Cui et al.
paper recommends beta in {0.9, 0.99, 0.999, 0.9999} depending on the
overall imbalance ratio.

This strategy ignores the BA hierarchy and the GF axis. ``alpha`` from
the Strategy base class is reinterpreted as a transform on top of the
effective-number weight: w *= alpha-power isn't applied — we let beta
do the smoothing instead. ``alpha`` is therefore unused; we keep it on
the constructor signature for interface uniformity but warn if it is
set away from 1.0.

The ``beta`` is hard-coded here for simplicity (see ``BETA``). Tuning
beta is the natural extension if this strategy is selected; either bump
this constant or thread it through ``SampleWeightingOptions``.

Pros:
  - Theoretically motivated: derives from a sample-coverage argument
    that captures diminishing returns of additional samples.
  - Single-knob smoothing via beta with well-known recommended values.

Cons:
  - Ignores the BA tree entirely; sibling species are independent.
  - Adds a second smoothing knob (beta) decoupled from alpha, which can
    confuse the user. Hence we punt on combining alpha with beta and
    just emit a heads-up.

Other strategies in this package (see their module docstrings for math):
  - tree_balanced  -- BA-tree sibling balance, leaf-flat over GF (default).
  - leaf_inverse   -- ignores hierarchy; w ~ 1/count^alpha at leaf.
  - decomposed     -- BA-side and GF-side both flat.
"""
from __future__ import annotations

import warnings

from mermaid_classifier.training.sample_weighting.base import (
    BALibrary,
    GFLibrary,
    Strategy,
)
from mermaid_classifier.training.sample_weighting.registry import (
    register_strategy,
)


# Default per Cui et al. 2019 — works well for moderate-to-high imbalance.
# Bump to 0.999 or 0.9999 for stronger weighting on rare classes.
BETA: float = 0.999


@register_strategy("effective_number")
class EffectiveNumberOfSamplesStrategy(Strategy):
    """Class-balanced loss from Cui et al. 2019. See module docstring."""

    def __init__(
        self,
        alpha: float = 0.5,
        weight_ratio_cap: float | None = None,
    ):
        super().__init__(
            alpha=alpha,
            weight_ratio_cap=weight_ratio_cap,
        )
        # Warn for any alpha other than the default (0.5) since this
        # strategy ignores alpha and uses BETA instead. Suppressing only
        # when alpha == default avoids spurious warnings on routine runs.
        if abs(self.alpha - 0.5) > 1e-9:
            warnings.warn(
                "EffectiveNumberOfSamplesStrategy uses beta (not alpha) for"
                " smoothing; the alpha argument is ignored. Adjust BETA in"
                " effective_number.py or thread beta through"
                " SampleWeightingOptions if you need to tune.",
                stacklevel=2,
            )

    def compute_raw_weights(
        self,
        class_counts: dict[str, int],
        ba_library: BALibrary,
        gf_library: GFLibrary,
    ) -> dict[str, float]:
        weights: dict[str, float] = {}
        for label, count in class_counts.items():
            n = max(int(count), 1)
            effective_n = (1.0 - BETA ** n) / (1.0 - BETA)
            weights[label] = 1.0 / max(effective_n, 1e-12)
        return weights
