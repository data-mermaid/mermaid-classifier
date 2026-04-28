"""DecomposedBaGfStrategy unit tests."""
from __future__ import annotations

import math
import unittest

from mermaid_classifier.common.benthic_attributes import combine_ba_gf
from mermaid_classifier.training.sample_weighting import (
    SampleWeightingOptions, compute_class_weights,
)

from tests.training.test_sample_weighting.fakes import (
    FakeGFLibrary, small_tree,
)


class DecomposedTest(unittest.TestCase):
    def setUp(self):
        self.ba_lib = small_tree()
        self.gf_lib = FakeGFLibrary({"g1": "GF1", "g2": "GF2"})

    def _run(self, counts, **kw):
        opts = SampleWeightingOptions(
            strategy="decomposed",
            **kw,
        )
        return compute_class_weights(
            counts, self.ba_lib, self.gf_lib, opts)

    def test_decomposed_separates_axes(self):
        # With alpha=1, BA totals: A1=100+50=150, A2=100. GF totals:
        # g1=100+100=200, g2=50. Each axis is normalized
        # independently, then multiplied per leaf.
        counts = {
            combine_ba_gf("A1", "g1"): 100,
            combine_ba_gf("A1", "g2"): 50,
            combine_ba_gf("A2", "g1"): 100,
        }
        w = self._run(counts, alpha=1.0)
        # Manually compute expected:
        ba_inv = {"A1": 1.0 / 150.0, "A2": 1.0 / 100.0}
        ba_total = sum(ba_inv.values())
        ba_share = {k: v / ba_total for k, v in ba_inv.items()}
        gf_inv = {"g1": 1.0 / 200.0, "g2": 1.0 / 50.0}
        gf_total = sum(gf_inv.values())
        gf_share = {k: v / gf_total for k, v in gf_inv.items()}
        for ba, gf in [("A1", "g1"), ("A1", "g2"), ("A2", "g1")]:
            expected = ba_share[ba] * gf_share[gf]
            self.assertTrue(math.isclose(
                w[combine_ba_gf(ba, gf)], expected, rel_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
