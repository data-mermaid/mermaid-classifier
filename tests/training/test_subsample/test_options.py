"""SubsampleOptions validation tests."""

from __future__ import annotations

import unittest

from mermaid_classifier.training.subsample.options import SubsampleOptions


class OptionsValidationTest(unittest.TestCase):
    def test_stratified_defaults_require_total(self):
        # 'stratified' is the default strategy and needs total_annotations.
        with self.assertRaisesRegex(ValueError, "total_annotations"):
            SubsampleOptions()

    def test_balanced_requires_total(self):
        with self.assertRaisesRegex(ValueError, "balanced"):
            SubsampleOptions(strategy="balanced")

    def test_unknown_strategy_rejected(self):
        with self.assertRaisesRegex(ValueError, "strategy"):
            SubsampleOptions(strategy="nope", total_annotations=100)

    def test_zero_total_rejected(self):
        with self.assertRaisesRegex(ValueError, "total_annotations"):
            SubsampleOptions(strategy="stratified", total_annotations=0)

    def test_negative_min_per_class_rejected(self):
        with self.assertRaisesRegex(ValueError, "min_per_class"):
            SubsampleOptions(
                strategy="stratified",
                total_annotations=100,
                min_per_class=-1,
            )


class LogDictTest(unittest.TestCase):
    """to_log_dict is what TrainingRunner hands mlflow.log_params, so these
    keys are the names a run is read back by."""

    def test_every_option_reaches_the_logged_params(self):
        options = SubsampleOptions(strategy="balanced", total_annotations=1000, min_per_class=10)

        self.assertEqual(
            options.to_log_dict(),
            {
                "subsample/enabled": True,
                "subsample/strategy": "balanced",
                "subsample/total_annotations": 1000,
                "subsample/min_per_class": 10,
            },
        )


if __name__ == "__main__":
    unittest.main()
