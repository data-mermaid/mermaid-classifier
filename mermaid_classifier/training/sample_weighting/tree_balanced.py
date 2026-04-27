"""TreeBalancedBaFlatGfStrategy — the default sample-weighting strategy.

Algorithm: tree-balanced over the BA hierarchy, leaf-flat over GF.

  - For each non-leaf BA node N with active children c_1..c_K
    (i.e. children that have at least one training class in their
    subtree) and subtree counts n_1..n_K:

        share(c_i) = n_i^(-alpha) / sum_j n_j^(-alpha)
        mass(c_i)  = mass(N) * share(c_i)        # mass(root) = 1

  - alpha = 0  ->  siblings split equally regardless of count
                   (full rebalancing, "give every clade an equal vote")
  - alpha = 1  ->  siblings get share inversely proportional to count
                   (classic inverse-frequency at every level)
  - alpha in between  ->  smooth interpolation; default 0.5

  - For the GF axis (flat across all GFs that appear in training):

        w_gf(g) = count_gf(g)^(-alpha) / sum_h count_gf(h)^(-alpha)

  - Final per-class weight:

        w(class) = mass(ba) * w_gf(gf)

Pros:
  - Uses the BA taxonomy: rare clades get up-weighted as a *clade*,
    not just as individual rare leaves.
  - Tunable via a single alpha that has a consistent meaning (smoothing
    exponent) across both the BA and GF axes.

Cons:
  - Long single-child chains in the BA tree contribute nothing
    distributively, but the multiplicative path can shrink leaf weights
    on deep branches even at equal counts. Mitigated by collapsing
    single-child chains during the descent (see _walk_tree).
  - Relies on the BA hierarchy being correct and reachable for every
    training class.

Other strategies in this package (see their module docstrings for math):
  - leaf_inverse        -- ignores hierarchy; w ~ 1/count^alpha at leaf.
  - decomposed          -- BA-side and GF-side both flat (no hierarchy).
  - effective_number    -- (1 - beta^n) / (1 - beta) class-balanced loss
                            (Cui et al. 2019), no explicit hierarchy term.
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


@register_strategy("tree_balanced_ba_flat_gf")
class TreeBalancedBaFlatGfStrategy(Strategy):
    """Tree-balanced over BA, leaf-flat over GF. See module docstring."""

    def compute_raw_weights(
        self,
        class_counts: dict[str, int],
        ba_library: BALibrary,
        gf_library: GFLibrary,
    ) -> dict[str, float]:
        # Step 1: parse labels.
        parsed: dict[str, tuple[str, str]] = {}
        for bagf in class_counts:
            ba_id, gf_id = split_ba_gf(bagf)
            parsed[bagf] = (ba_id, gf_id)

        # Step 2: GF-side flat factor.
        gf_counts: dict[str, int] = defaultdict(int)
        for bagf, count in class_counts.items():
            _, gf_id = parsed[bagf]
            gf_counts[gf_id] += count
        gf_share = split_alpha_softmax_normalize({
            gf_id: split_alpha_safe_inverse(c, self.alpha)
            for gf_id, c in gf_counts.items()
        })

        # Step 3: BA tree mass.
        ba_mass = self._compute_ba_mass(
            ba_counts=self._ba_subtree_counts(class_counts, parsed),
            active_bas={ba for (ba, _) in parsed.values()},
            ba_library=ba_library,
        )

        # Step 4: combine.
        weights: dict[str, float] = {}
        for bagf, (ba_id, gf_id) in parsed.items():
            ba_w = ba_mass.get(ba_id, 0.0)
            gf_w = gf_share[gf_id]
            # Strategy contract requires strictly positive weights. If an
            # active BA somehow gets ba_w=0 (e.g. taxonomy mismatch), fall
            # back to a tiny epsilon so the contract holds. We log-style
            # behaviour: never silently drop a class via the math.
            weights[bagf] = max(ba_w * gf_w, 1e-12)
        return weights

    @staticmethod
    def _ba_subtree_counts(
        class_counts: dict[str, int],
        parsed: dict[str, tuple[str, str]],
    ) -> dict[str, int]:
        """Sum training counts at each BA leaf in the training-class set.
        Subtree aggregation up the tree happens in _compute_ba_mass."""
        leaf_counts: dict[str, int] = defaultdict(int)
        for bagf, count in class_counts.items():
            ba_id, _ = parsed[bagf]
            leaf_counts[ba_id] += count
        return dict(leaf_counts)

    def _compute_ba_mass(
        self,
        ba_counts: dict[str, int],
        active_bas: set[str],
        ba_library: BALibrary,
    ) -> dict[str, float]:
        """Walk the BA tree from the root downward, distributing mass=1
        among active descendant BAs by sibling-balanced shares.

        ``active_bas`` is the set of BAs that have training classes; only
        these (and their ancestors) participate in the descent. Other
        siblings are simply ignored at each level.
        """
        # Precompute ancestor chains for each active BA (root-first).
        chains: dict[str, list[str]] = {}
        ancestor_set: set[str] = set()
        for ba_id in active_bas:
            anc = ba_library.get_ancestor_ids(ba_id) + [ba_id]
            chains[ba_id] = anc
            ancestor_set.update(anc)

        # Subtree count for any BA in ancestor_set = sum of leaf counts of
        # active descendants whose chain passes through it.
        subtree_count: dict[str, int] = defaultdict(int)
        for ba_id, count in ba_counts.items():
            for ancestor in chains[ba_id]:
                subtree_count[ancestor] += count

        # Build the "active children" map: for each node in ancestor_set,
        # which of its children are also in ancestor_set?
        active_children: dict[str | None, list[str]] = defaultdict(list)
        for ba_id in ancestor_set:
            parent = ba_library.by_id[ba_id].get("parent")
            active_children[parent].append(ba_id)
        # Roots are children of None (per BenthicAttributeLibrary's by_parent
        # convention).

        # Top-down mass distribution starting from the virtual root.
        mass: dict[str, float] = {}
        # The virtual root has mass=1 distributed among root-level BAs.
        self._distribute(
            parent_mass=1.0,
            children=active_children[None],
            subtree_count=subtree_count,
            active_children=active_children,
            mass=mass,
        )

        # Each active BA inherits its allocated mass.
        return {ba: mass[ba] for ba in active_bas}

    def _distribute(
        self,
        parent_mass: float,
        children: list[str],
        subtree_count: dict[str, int],
        active_children: dict[str | None, list[str]],
        mass: dict[str, float],
    ) -> None:
        if not children:
            return
        if len(children) == 1:
            # Single-child chain: don't penalise depth; pass mass through
            # without splitting. This is the "collapse single-child
            # chains" mitigation called out in the module docstring.
            child = children[0]
            mass[child] = parent_mass
            self._distribute(
                parent_mass=parent_mass,
                children=active_children.get(child, []),
                subtree_count=subtree_count,
                active_children=active_children,
                mass=mass,
            )
            return

        raw_shares = {
            c: split_alpha_safe_inverse(subtree_count[c], self.alpha)
            for c in children
        }
        shares = split_alpha_softmax_normalize(raw_shares)
        for c, share in shares.items():
            mass[c] = parent_mass * share
            self._distribute(
                parent_mass=mass[c],
                children=active_children.get(c, []),
                subtree_count=subtree_count,
                active_children=active_children,
                mass=mass,
            )
