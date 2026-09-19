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


class LogDictTest(unittest.TestCase):
    """to_log_dict is what TrainingRunner hands mlflow.log_params, so these
    keys are the names a run is read back by."""

    def test_every_option_reaches_the_logged_params(self):
        logged = SampleWeightingOptions(enabled=True, weight_ratio_cap=250.0).to_log_dict()

        self.assertEqual(
            logged,
            {"weighting/enabled": True, "weighting/weight_ratio_cap": 250.0},
        )


if __name__ == "__main__":
    unittest.main()
