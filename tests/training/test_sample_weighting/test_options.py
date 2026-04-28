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
        self.assertIsNone(opts.weight_ratio_cap)

    def test_alpha_below_zero_rejected(self):
        with self.assertRaisesRegex(ValueError, "alpha"):
            SampleWeightingOptions(alpha=-0.01)

    def test_alpha_above_one_rejected(self):
        with self.assertRaisesRegex(ValueError, "alpha"):
            SampleWeightingOptions(alpha=1.5)

    def test_weight_ratio_cap_default_none(self):
        opts = SampleWeightingOptions()
        self.assertIsNone(opts.weight_ratio_cap)

    def test_weight_ratio_cap_one_accepted(self):
        opts = SampleWeightingOptions(weight_ratio_cap=1.0)
        self.assertEqual(opts.weight_ratio_cap, 1.0)

    def test_weight_ratio_cap_below_one_rejected(self):
        with self.assertRaisesRegex(ValueError, "weight_ratio_cap"):
            SampleWeightingOptions(weight_ratio_cap=0.5)

    def test_to_log_dict_flat(self):
        opts = SampleWeightingOptions(
            enabled=True,
            strategy="leaf_inverse",
            alpha=0.3,
        )
        d = opts.to_log_dict()
        self.assertEqual(d["weighting/strategy"], "leaf_inverse")
        self.assertEqual(d["weighting/alpha"], 0.3)
        self.assertTrue(d["weighting/enabled"])
        self.assertIsNone(d["weighting/weight_ratio_cap"])

    def test_to_log_dict_includes_weight_ratio_cap_when_set(self):
        opts = SampleWeightingOptions(weight_ratio_cap=10.0)
        self.assertEqual(opts.to_log_dict()["weighting/weight_ratio_cap"], 10.0)


if __name__ == "__main__":
    unittest.main()
