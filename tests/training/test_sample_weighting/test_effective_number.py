"""compute_class_weights unit tests (effective number of samples, Cui 2019)."""

from __future__ import annotations

import math
import unittest

from mermaid_classifier.common.benthic_attributes import combine_ba_gf
from mermaid_classifier.training.sample_weighting import (
    SampleWeightingOptions,
    compute_class_weights,
)


class EffectiveNumberTest(unittest.TestCase):
    def _run(self, counts, **kw):
        return compute_class_weights(counts, SampleWeightingOptions(**kw))

    def test_matches_cui_formula(self):
        # w = (1 - beta) / (1 - beta**n) at beta = 0.9999, evaluated once by
        # hand so a change to the formula cannot move both sides together.
        counts = {
            combine_ba_gf("A1", "g1"): 100,
            combine_ba_gf("A2", "g1"): 10,
        }
        w = self._run(counts)

        self.assertTrue(
            math.isclose(w[combine_ba_gf("A1", "g1")], 0.010049583329027612, rel_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(w[combine_ba_gf("A2", "g1")], 0.10004500825041404, rel_tol=1e-9)
        )

    def test_rare_class_outweighs_common(self):
        counts = {
            combine_ba_gf("A1", "g1"): 1000,
            combine_ba_gf("A2", "g1"): 5,
        }
        w = self._run(counts)
        self.assertGreater(
            w[combine_ba_gf("A2", "g1")],
            w[combine_ba_gf("A1", "g1")],
        )


if __name__ == "__main__":
    unittest.main()
