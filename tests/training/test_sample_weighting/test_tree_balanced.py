"""Tree-balanced strategy unit tests.

Specifically checks the BA-tree sibling balancing math, the GF flat
factor, and the depth-collapse for single-child chains.
"""
from __future__ import annotations

import math
import unittest

from mermaid_classifier.common.benthic_attributes import combine_ba_gf
from mermaid_classifier.training.sample_weighting import (
    SampleWeightingOptions, compute_class_weights,
)

from .fakes import (
    FakeGFLibrary, small_tree,
)


def _bagf(ba: str, gf: str) -> str:
    return combine_ba_gf(ba, gf)


class TreeBalancedTest(unittest.TestCase):
    def setUp(self):
        self.ba_lib = small_tree()
        self.gf_lib = FakeGFLibrary({"g1": "GF1", "g2": "GF2"})

    def _run(self, counts, **opt_kwargs):
        opts = SampleWeightingOptions(
            strategy="tree_balanced_ba_flat_gf",
            **opt_kwargs,
        )
        return compute_class_weights(
            class_counts=counts,
            ba_library=self.ba_lib,
            gf_library=self.gf_lib,
            options=opts,
        )

    def test_alpha_zero_splits_siblings_equally(self):
        # Two siblings under A (A1, A2) with very different counts,
        # plus B1 under B. With alpha=0, the BA-side mass should be
        # split equally among root-level children (A and B), then
        # equally among A's two children (A1, A2). The GF axis only
        # has one GF (g1), so its share is 1.
        counts = {
            _bagf("A1", "g1"): 1000,
            _bagf("A2", "g1"): 10,
            _bagf("B1", "g1"): 100,
        }
        w = self._run(counts, alpha=0.0)
        # A1: mass(A) * share(A1|A) = 0.5 * 0.5 = 0.25
        # A2: 0.5 * 0.5 = 0.25
        # B1: mass(B) = 0.5 (only child of B)
        self.assertTrue(math.isclose(w[_bagf("A1", "g1")], 0.25, rel_tol=1e-6))
        self.assertTrue(math.isclose(w[_bagf("A2", "g1")], 0.25, rel_tol=1e-6))
        self.assertTrue(math.isclose(w[_bagf("B1", "g1")], 0.5, rel_tol=1e-6))

    def test_alpha_one_inverse_frequency_at_each_level(self):
        # Same counts as above, alpha=1. At root level the active
        # children are A (subtree count=1010) and B (count=100). At
        # alpha=1 the share is 1/1010 / (1/1010 + 1/100) ~ 0.090
        # for A and ~0.910 for B.
        counts = {
            _bagf("A1", "g1"): 1000,
            _bagf("A2", "g1"): 10,
            _bagf("B1", "g1"): 100,
        }
        w = self._run(counts, alpha=1.0)
        a_share = (1.0 / 1010) / (1.0 / 1010 + 1.0 / 100)
        b_share = 1.0 - a_share
        # Within A: A1 has 1000, A2 has 10. share(A1|A) at alpha=1 is
        # (1/1000) / (1/1000 + 1/10) ~ 0.00990, A2 the rest.
        a1_share = (1.0 / 1000) / (1.0 / 1000 + 1.0 / 10)
        a2_share = 1.0 - a1_share
        self.assertTrue(math.isclose(
            w[_bagf("A1", "g1")], a_share * a1_share, rel_tol=1e-6))
        self.assertTrue(math.isclose(
            w[_bagf("A2", "g1")], a_share * a2_share, rel_tol=1e-6))
        self.assertTrue(math.isclose(
            w[_bagf("B1", "g1")], b_share, rel_tol=1e-6))

    def test_intermediate_alpha_interpolates_monotonically_for_rare_subtree(self):
        # Note: monotonicity in alpha holds for classes in *rare
        # subtrees* (where both the subtree-level boost and the
        # within-subtree boost align). For a class that is rare
        # within a common subtree (e.g. A2 with siblings totalling
        # 1010 vs B subtree of 100), the two effects compete and
        # the relationship is non-monotonic.
        # B1 is a rare subtree (B total = 100 vs A total = 1010),
        # so its weight should rise monotonically with alpha.
        counts = {
            _bagf("A1", "g1"): 1000,
            _bagf("A2", "g1"): 10,
            _bagf("B1", "g1"): 100,
        }
        b1_at_zero = self._run(counts, alpha=0.0)[_bagf("B1", "g1")]
        b1_at_half = self._run(counts, alpha=0.5)[_bagf("B1", "g1")]
        b1_at_one = self._run(counts, alpha=1.0)[_bagf("B1", "g1")]
        self.assertLess(b1_at_zero, b1_at_half)
        self.assertLess(b1_at_half, b1_at_one)

    def test_gf_factor_applied_multiplicatively(self):
        # Two GFs, one common (g1, total 200) and one rare (g2,
        # total 20). At alpha=1 GF-side share ratios should be
        # ~10:1 in favour of g2; that ratio carries through into
        # final weights at fixed BA.
        counts = {
            _bagf("A1", "g1"): 200,
            _bagf("A1", "g2"): 20,
        }
        w = self._run(counts, alpha=1.0)
        ratio = w[_bagf("A1", "g2")] / w[_bagf("A1", "g1")]
        self.assertTrue(math.isclose(ratio, 10.0, rel_tol=1e-6))

    def test_weight_ratio_cap_is_noop_when_under_cap(self):
        # A loose cap (well above the natural max/min) should leave
        # the weights unchanged.
        counts = {
            _bagf("A1", "g1"): 1000,
            _bagf("A2", "g1"): 10,
            _bagf("B1", "g1"): 100,
        }
        baseline = self._run(counts, alpha=0.5)
        capped = self._run(counts, alpha=0.5, weight_ratio_cap=1e6)
        for k in baseline:
            self.assertTrue(
                math.isclose(baseline[k], capped[k], rel_tol=1e-9),
                f"weight for {k!r} drifted under no-op cap",
            )

    def test_weight_ratio_cap_one_collapses_kept_weights(self):
        # cap=1.0 forces max == min, so every kept class ends up at the
        # same value (= the original min_kept).
        counts = {
            _bagf("A1", "g1"): 1000,
            _bagf("A2", "g1"): 10,
            _bagf("B1", "g1"): 100,
        }
        baseline = self._run(counts, alpha=1.0)
        capped = self._run(counts, alpha=1.0, weight_ratio_cap=1.0)
        baseline_min = min(baseline.values())
        for v in capped.values():
            self.assertTrue(
                math.isclose(v, baseline_min, rel_tol=1e-9),
                f"capped weight {v!r} != baseline min {baseline_min!r}",
            )

    def test_weight_ratio_cap_only_clips_above_ceiling(self):
        # With alpha=1 and these counts B1's mass dominates: A and B
        # get root shares ~0.090 / 0.910, then within A both leaves
        # split as ~0.0099/0.9901. Apply cap=2.0; weights below
        # min_kept * cap stay untouched.
        counts = {
            _bagf("A1", "g1"): 1000,
            _bagf("A2", "g1"): 10,
            _bagf("B1", "g1"): 100,
        }
        baseline = self._run(counts, alpha=1.0)
        cap = 2.0
        capped = self._run(counts, alpha=1.0, weight_ratio_cap=cap)
        kept = [w for w in capped.values() if w > 0]
        self.assertLessEqual(max(kept) / min(kept), cap + 1e-9)
        ceiling = min(baseline.values()) * cap
        for k, raw_w in baseline.items():
            if raw_w <= ceiling:
                self.assertTrue(
                    math.isclose(capped[k], raw_w, rel_tol=1e-9),
                    f"sub-ceiling weight for {k!r} should be untouched",
                )
            else:
                self.assertTrue(
                    math.isclose(capped[k], ceiling, rel_tol=1e-9),
                    f"super-ceiling weight for {k!r} should equal ceiling",
                )

    def test_collapses_single_child_chain(self):
        # C -> C1 -> C1a is a single-child chain. With alpha=0 and
        # only C1a active in that branch, the depth-collapse should
        # ensure C1a gets the full mass(C) (no extra penalty for the
        # chain). With three root branches active (A1, B1, C1a),
        # each root gets mass=1/3, and C1a's leaf weight = 1/3.
        counts = {
            _bagf("A1", "g1"): 100,
            _bagf("B1", "g1"): 100,
            _bagf("C1a", "g1"): 100,
        }
        w = self._run(counts, alpha=0.0)
        self.assertTrue(math.isclose(
            w[_bagf("C1a", "g1")], 1.0 / 3.0, rel_tol=1e-6))
        # And the other two should also be 1/3 (single-child A subtree
        # doesn't apply here — A has two children, but A1 is the only
        # active one. The "active children" set restricts traversal,
        # so A is single-child-active and collapses to A1.)
        self.assertTrue(math.isclose(
            w[_bagf("A1", "g1")], 1.0 / 3.0, rel_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
