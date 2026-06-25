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
        self.assertIsNone(opts.weight_ratio_cap)

    def test_weight_ratio_cap_one_accepted(self):
        opts = SampleWeightingOptions(weight_ratio_cap=1.0)
        self.assertEqual(opts.weight_ratio_cap, 1.0)

    def test_weight_ratio_cap_below_one_rejected(self):
        with self.assertRaisesRegex(ValueError, "weight_ratio_cap"):
            SampleWeightingOptions(weight_ratio_cap=0.5)

    def test_to_log_dict_flat(self):
        opts = SampleWeightingOptions(enabled=True)
        d = opts.to_log_dict()
        self.assertTrue(d["weighting/enabled"])
        self.assertIsNone(d["weighting/weight_ratio_cap"])
        # strategy/alpha keys were removed when the subsystem collapsed to
        # the single effective_number strategy.
        self.assertNotIn("weighting/strategy", d)
        self.assertNotIn("weighting/alpha", d)

    def test_to_log_dict_includes_weight_ratio_cap_when_set(self):
        opts = SampleWeightingOptions(weight_ratio_cap=10.0)
        self.assertEqual(
            opts.to_log_dict()["weighting/weight_ratio_cap"], 10.0)


if __name__ == "__main__":
    unittest.main()
