"""LeafInverseStrategy — leaf-only inverse-frequency weighting.

Algorithm:
  - Treats each (BA, GF) pair as the unit; ignores hierarchy entirely.
  - w(class) ~= count(class)^(-alpha)
    - alpha = 0  ->  uniform weighting (no rebalancing)
    - alpha = 1  ->  full inverse-frequency
    - alpha in between ->  smooth interpolation; default 0.5

Pros:
  - Simplest possible weighting; no taxonomy traversal.
  - Easy to reason about: a class with 10x fewer samples gets 10x more
    weight (at alpha=1) or sqrt(10)x more (at alpha=0.5).

Cons:
  - Ignores hierarchy: sibling species are treated as fully independent
    classes, missing the structure the model can exploit.
  - Long-tail rare classes can produce extreme weight ratios.

Other strategies in this package (see their module docstrings for math):
  - tree_balanced     -- BA-tree sibling balance, leaf-flat over GF (default).
  - decomposed        -- BA-side and GF-side both flat.
  - effective_number  -- (1 - beta^n) / (1 - beta) class-balanced loss.
"""
from __future__ import annotations

from mermaid_classifier.training.sample_weighting.base import (
    BALibrary,
    GFLibrary,
    Strategy,
    split_alpha_safe_inverse,
)
from mermaid_classifier.training.sample_weighting.registry import (
    register_strategy,
)


@register_strategy("leaf_inverse")
class LeafInverseStrategy(Strategy):
    """Inverse-frequency weighting at the (BA, GF) leaf. See module docstring."""

    def compute_raw_weights(
        self,
        class_counts: dict[str, int],
        ba_library: BALibrary,
        gf_library: GFLibrary,
    ) -> dict[str, float]:
        return {
            label: split_alpha_safe_inverse(count, self.alpha)
            for label, count in class_counts.items()
        }
