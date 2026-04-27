"""DecomposedBaGfStrategy — flat BA-axis x flat GF-axis.

Algorithm: decompose the (BA, GF) leaf weight into independent BA and GF
factors, both computed flat (no hierarchy traversal).

  - BA-side flat factor (across all BAs that appear in training):

        w_ba(b) = count_ba(b)^(-alpha) / sum_b' count_ba(b')^(-alpha)

  - GF-side flat factor (across all GFs that appear in training):

        w_gf(g) = count_gf(g)^(-alpha) / sum_g' count_gf(g')^(-alpha)

  - Final per-class weight:

        w(class) = w_ba(ba) * w_gf(gf)

Pros:
  - No hierarchy needed; works without a BA tree at hand.
  - Cleanly separates the BA and GF axes — useful when the imbalance
    structure differs between them (e.g. GF is mildly imbalanced but
    BA is severely so).

Cons:
  - Treats BAs as independent rather than as a tree, so closely
    related BAs (sibling species) don't share weighting mass.
  - Two BAs with identical leaf counts get identical weight even if
    one belongs to a small clade and the other to a huge one.

Other strategies in this package (see their module docstrings for math):
  - tree_balanced     -- BA-tree sibling balance, leaf-flat over GF (default).
  - leaf_inverse      -- ignores hierarchy; w ~ 1/count^alpha at leaf.
  - effective_number  -- (1 - beta^n) / (1 - beta) class-balanced loss.
"""
from __future__ import annotations

from collections import defaultdict

from mermaid_classifier.common.benthic_attributes import split_ba_gf
from mermaid_classifier.training.sample_weighting.base import (
    BALibrary,
    GFLibrary,
    Strategy,
    split_alpha_safe_inverse,
    split_alpha_softmax_normalize,
)
from mermaid_classifier.training.sample_weighting.registry import (
    register_strategy,
)


@register_strategy("decomposed")
class DecomposedBaGfStrategy(Strategy):
    """Decomposed flat BA x flat GF. See module docstring."""

    def compute_raw_weights(
        self,
        class_counts: dict[str, int],
        ba_library: BALibrary,
        gf_library: GFLibrary,
    ) -> dict[str, float]:
        ba_counts: dict[str, int] = defaultdict(int)
        gf_counts: dict[str, int] = defaultdict(int)
        parsed: dict[str, tuple[str, str]] = {}
        for bagf, count in class_counts.items():
            ba_id, gf_id = split_ba_gf(bagf)
            parsed[bagf] = (ba_id, gf_id)
            ba_counts[ba_id] += count
            gf_counts[gf_id] += count

        ba_share = split_alpha_softmax_normalize({
            ba_id: split_alpha_safe_inverse(c, self.alpha)
            for ba_id, c in ba_counts.items()
        })
        gf_share = split_alpha_softmax_normalize({
            gf_id: split_alpha_safe_inverse(c, self.alpha)
            for gf_id, c in gf_counts.items()
        })

        return {
            bagf: max(ba_share[ba_id] * gf_share[gf_id], 1e-12)
            for bagf, (ba_id, gf_id) in parsed.items()
        }
