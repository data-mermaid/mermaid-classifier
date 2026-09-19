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


if __name__ == "__main__":
    unittest.main()
