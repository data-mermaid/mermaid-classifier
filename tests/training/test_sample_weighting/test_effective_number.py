"""EffectiveNumberOfSamplesStrategy unit tests (Cui et al. 2019)."""
from __future__ import annotations

import math
import unittest
import warnings

from mermaid_classifier.common.benthic_attributes import combine_ba_gf
from mermaid_classifier.training.sample_weighting import (
    SampleWeightingOptions, compute_class_weights,
)
from mermaid_classifier.training.sample_weighting.effective_number import (
    BETA,
)

from tests.training.test_sample_weighting.fakes import (
    FakeGFLibrary, small_tree,
)


class EffectiveNumberTest(unittest.TestCase):
    def setUp(self):
        self.ba_lib = small_tree()
        self.gf_lib = FakeGFLibrary({"g1": "GF1"})

    def _run(self, counts, **kw):
        opts = SampleWeightingOptions(
            strategy="effective_number",
            rare_policy=kw.pop("rare_policy", "keep"),
            **kw,
        )
        return compute_class_weights(
            counts, self.ba_lib, self.gf_lib, opts)

    def test_matches_cui_formula(self):
        counts = {
            combine_ba_gf("A1", "g1"): 100,
            combine_ba_gf("A2", "g1"): 10,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = self._run(counts, alpha=0.5)

        def expected(n):
            en = (1.0 - BETA ** n) / (1.0 - BETA)
            return 1.0 / en

        self.assertTrue(math.isclose(
            w[combine_ba_gf("A1", "g1")], expected(100), rel_tol=1e-9))
        self.assertTrue(math.isclose(
            w[combine_ba_gf("A2", "g1")], expected(10), rel_tol=1e-9))

    def test_rare_class_outweighs_common(self):
        counts = {
            combine_ba_gf("A1", "g1"): 1000,
            combine_ba_gf("A2", "g1"): 5,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = self._run(counts, alpha=0.5)
        self.assertGreater(
            w[combine_ba_gf("A2", "g1")],
            w[combine_ba_gf("A1", "g1")],
        )


if __name__ == "__main__":
    unittest.main()
