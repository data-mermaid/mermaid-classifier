"""Invariants for compute_class_weights (effective-number weighting)."""

from __future__ import annotations

import unittest

from mermaid_classifier.common.benthic_attributes import combine_ba_gf
from mermaid_classifier.training.sample_weighting import (
    SampleWeightingOptions,
    compute_class_weights,
)


def _toy_class_counts() -> dict[str, int]:
    """Six classes with a spread of counts (the rare-class drop/merge
    decision is upstream of weighting, so this exercises the weighting
    layer with the kept-class set the data pipeline produces)."""
    return {
        combine_ba_gf("A1", "g1"): 100,
        combine_ba_gf("A2", "g1"): 100,
        combine_ba_gf("B1", "g1"): 50,
        combine_ba_gf("C1a", "g2"): 30,
        combine_ba_gf("A1", "g2"): 20,
        combine_ba_gf("B1", "g2"): 15,
    }


class WeightInvariantsTest(unittest.TestCase):
    def setUp(self):
        self.counts = _toy_class_counts()

    def test_returns_weight_for_every_class(self):
        weights = compute_class_weights(self.counts, SampleWeightingOptions())
        self.assertEqual(set(weights), set(self.counts))

    def test_no_zero_weights(self):
        weights = compute_class_weights(self.counts, SampleWeightingOptions())
        for label, w in weights.items():
            self.assertGreater(w, 0.0, f"weight for {label!r} is non-positive: {w!r}")

    def test_disabled_returns_empty(self):
        weights = compute_class_weights(self.counts, SampleWeightingOptions(enabled=False))
        self.assertEqual(weights, {})

    def test_weight_ratio_cap_bounds_weight_spread(self):
        # weight_ratio_cap=R must ensure max/min <= R. Weights are
        # universally positive, so the cap applies to the full set.
        cap = 5.0
        tol = 1e-9
        weights = compute_class_weights(self.counts, SampleWeightingOptions(weight_ratio_cap=cap))
        self.assertGreaterEqual(len(weights), 2)
        ws = list(weights.values())
        self.assertLessEqual(max(ws) / min(ws), cap + tol)

    def test_cap_of_one_is_accepted_and_equalises_every_weight(self):
        # 1.0 is the inclusive floor of the validator and the degenerate end
        # of the clamp: every weight collapses onto the minimum.
        weights = compute_class_weights(self.counts, SampleWeightingOptions(weight_ratio_cap=1.0))
        self.assertEqual(len(set(weights.values())), 1)


if __name__ == "__main__":
    unittest.main()
