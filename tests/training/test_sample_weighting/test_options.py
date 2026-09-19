"""SampleWeightingOptions validation tests."""

from __future__ import annotations

import unittest

from mermaid_classifier.training.sample_weighting.options import (
    SampleWeightingOptions,
)


class OptionsValidationTest(unittest.TestCase):
    def test_weight_ratio_cap_below_one_rejected(self):
        with self.assertRaisesRegex(ValueError, "weight_ratio_cap"):
            SampleWeightingOptions(weight_ratio_cap=0.5)


if __name__ == "__main__":
    unittest.main()
