"""SubsampleOptions validation + to_log_dict tests."""
from __future__ import annotations

import unittest

from mermaid_classifier.training.subsample.options import (
    SUBSAMPLE_STRATEGIES,
    SubsampleOptions,
)


class OptionsValidationTest(unittest.TestCase):
    def test_stratified_defaults_require_total(self):
        # 'stratified' is the default strategy and needs total_annotations.
        with self.assertRaisesRegex(ValueError, "total_annotations"):
            SubsampleOptions()

    def test_stratified_with_total_valid(self):
        opts = SubsampleOptions(strategy="stratified", total_annotations=100)
        self.assertEqual(opts.strategy, "stratified")
        self.assertEqual(opts.total_annotations, 100)
        self.assertEqual(opts.min_per_class, 0)

    def test_balanced_with_total_valid(self):
        opts = SubsampleOptions(strategy="balanced", total_annotations=100)
        self.assertEqual(opts.strategy, "balanced")
        self.assertEqual(opts.total_annotations, 100)

    def test_balanced_requires_total(self):
        with self.assertRaisesRegex(ValueError, "balanced"):
            SubsampleOptions(strategy="balanced")

    def test_unknown_strategy_rejected(self):
        with self.assertRaisesRegex(ValueError, "strategy"):
            SubsampleOptions(strategy="nope", total_annotations=100)

    def test_negative_total_rejected(self):
        with self.assertRaisesRegex(ValueError, "total_annotations"):
            SubsampleOptions(strategy="stratified", total_annotations=-1)

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

    def test_to_log_dict_shape(self):
        opts = SubsampleOptions(
            strategy="stratified",
            total_annotations=400_000,
            min_per_class=5,
        )
        d = opts.to_log_dict()
        self.assertTrue(d["subsample/enabled"])
        self.assertEqual(d["subsample/strategy"], "stratified")
        self.assertEqual(d["subsample/total_annotations"], 400_000)
        self.assertEqual(d["subsample/min_per_class"], 5)
        # target_per_class / balance_alpha / seed were removed when the
        # soft_balanced experiment was dropped.
        self.assertNotIn("subsample/target_per_class", d)
        self.assertNotIn("subsample/balance_alpha", d)
        self.assertNotIn("subsample/seed", d)

    def test_strategies_constant_includes_known_strategies(self):
        # Sanity check that we don't accidentally drop a strategy from
        # the public constant.
        self.assertIn("stratified", SUBSAMPLE_STRATEGIES)
        self.assertIn("balanced", SUBSAMPLE_STRATEGIES)
        self.assertNotIn("soft_balanced", SUBSAMPLE_STRATEGIES)


if __name__ == "__main__":
    unittest.main()
