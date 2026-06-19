"""LeafInverseStrategy unit tests."""
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


class LeafInverseTest(unittest.TestCase):
    def setUp(self):
        self.ba_lib = small_tree()
        self.gf_lib = FakeGFLibrary({"g1": "GF1"})

    def _run(self, counts, **kw):
        opts = SampleWeightingOptions(
            strategy="leaf_inverse",
            **kw,
        )
        return compute_class_weights(
            counts, self.ba_lib, self.gf_lib, opts)

    def test_alpha_zero_uniform(self):
        counts = {
            combine_ba_gf("A1", "g1"): 100,
            combine_ba_gf("A2", "g1"): 10,
        }
        w = self._run(counts, alpha=0.0)
        # All weights should be equal (count^0 = 1 for every class).
        self.assertTrue(math.isclose(
            w[combine_ba_gf("A1", "g1")],
            w[combine_ba_gf("A2", "g1")],
            rel_tol=1e-9))

    def test_alpha_one_inverse_frequency(self):
        counts = {
            combine_ba_gf("A1", "g1"): 100,
            combine_ba_gf("A2", "g1"): 10,
        }
        w = self._run(counts, alpha=1.0)
        # Rare class should be weighted 10x more.
        ratio = w[combine_ba_gf("A2", "g1")] / w[combine_ba_gf("A1", "g1")]
        self.assertTrue(math.isclose(ratio, 10.0, rel_tol=1e-9))

    def test_alpha_half_sqrt_inverse(self):
        counts = {
            combine_ba_gf("A1", "g1"): 100,
            combine_ba_gf("A2", "g1"): 4,
        }
        w = self._run(counts, alpha=0.5)
        ratio = w[combine_ba_gf("A2", "g1")] / w[combine_ba_gf("A1", "g1")]
        # sqrt(100/4) = 5
        self.assertTrue(math.isclose(ratio, 5.0, rel_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
