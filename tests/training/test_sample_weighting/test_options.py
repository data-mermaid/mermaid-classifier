"""SampleWeightingOptions validation tests."""
from __future__ import annotations

import unittest

from mermaid_classifier.training.sample_weighting.options import (
    SampleWeightingOptions,
)


class OptionsValidationTest(unittest.TestCase):
    def test_defaults_valid(self):
        opts = SampleWeightingOptions()
        self.assertTrue(opts.enabled)
        self.assertEqual(opts.strategy, "tree_balanced_ba_flat_gf")
        self.assertEqual(opts.alpha, 0.5)
        self.assertEqual(opts.min_count, 10)
        self.assertEqual(opts.rare_policy, "drop")

    def test_alpha_below_zero_rejected(self):
        with self.assertRaisesRegex(ValueError, "alpha"):
            SampleWeightingOptions(alpha=-0.01)

    def test_alpha_above_one_rejected(self):
        with self.assertRaisesRegex(ValueError, "alpha"):
            SampleWeightingOptions(alpha=1.5)

    def test_min_count_below_one_rejected(self):
        with self.assertRaisesRegex(ValueError, "min_count"):
            SampleWeightingOptions(min_count=0)

    def test_unknown_rare_policy_rejected(self):
        with self.assertRaisesRegex(ValueError, "rare_policy"):
            SampleWeightingOptions(rare_policy="bogus")

    def test_to_log_dict_flat(self):
        opts = SampleWeightingOptions(
            enabled=True,
            strategy="leaf_inverse",
            alpha=0.3,
            min_count=5,
            rare_policy="keep",
        )
        d = opts.to_log_dict()
        self.assertEqual(d["weighting/strategy"], "leaf_inverse")
        self.assertEqual(d["weighting/alpha"], 0.3)
        self.assertEqual(d["weighting/min_count"], 5)
        self.assertEqual(d["weighting/rare_policy"], "keep")
        self.assertTrue(d["weighting/enabled"])


if __name__ == "__main__":
    unittest.main()
