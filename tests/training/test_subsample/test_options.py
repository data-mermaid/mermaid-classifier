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
        self.assertIsNone(opts.target_per_class)

    def test_balanced_with_target_per_class_valid(self):
        opts = SubsampleOptions(strategy="balanced", target_per_class=50)
        self.assertEqual(opts.strategy, "balanced")
        self.assertEqual(opts.target_per_class, 50)

    def test_balanced_with_total_only_valid(self):
        opts = SubsampleOptions(strategy="balanced", total_annotations=100)
        self.assertEqual(opts.total_annotations, 100)

    def test_balanced_requires_one_budget(self):
        with self.assertRaisesRegex(ValueError, "balanced"):
            SubsampleOptions(strategy="balanced")

    def test_balanced_rejects_both_budgets(self):
        with self.assertRaisesRegex(ValueError, "either"):
            SubsampleOptions(
                strategy="balanced",
                total_annotations=100,
                target_per_class=10,
            )

    def test_stratified_rejects_target_per_class(self):
        with self.assertRaisesRegex(ValueError, "target_per_class"):
            SubsampleOptions(
                strategy="stratified",
                total_annotations=100,
                target_per_class=10,
            )

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

    def test_zero_target_per_class_rejected(self):
        with self.assertRaisesRegex(ValueError, "target_per_class"):
            SubsampleOptions(strategy="balanced", target_per_class=0)

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
        self.assertIsNone(d["subsample/target_per_class"])
        self.assertIsNone(d["subsample/balance_alpha"])
        # ``seed`` defaults to 0 and is logged for visibility even though
        # the built-in allocators are deterministic by SQL ordering.
        self.assertEqual(d["subsample/seed"], 0)

    def test_to_log_dict_includes_explicit_seed(self):
        opts = SubsampleOptions(
            strategy="stratified",
            total_annotations=400_000,
            seed=42,
        )
        self.assertEqual(opts.to_log_dict()["subsample/seed"], 42)

    def test_strategies_constant_includes_known_strategies(self):
        # Sanity check that we don't accidentally drop a strategy from
        # the public constant.
        self.assertIn("stratified", SUBSAMPLE_STRATEGIES)
        self.assertIn("balanced", SUBSAMPLE_STRATEGIES)
        self.assertIn("soft_balanced", SUBSAMPLE_STRATEGIES)


class SoftBalancedOptionsTest(unittest.TestCase):
    def test_soft_balanced_with_total_and_alpha_valid(self):
        opts = SubsampleOptions(
            strategy="soft_balanced",
            total_annotations=100,
            balance_alpha=0.5,
        )
        self.assertEqual(opts.strategy, "soft_balanced")
        self.assertEqual(opts.total_annotations, 100)
        self.assertEqual(opts.balance_alpha, 0.5)

    def test_soft_balanced_requires_total_annotations(self):
        with self.assertRaisesRegex(ValueError, "total_annotations"):
            SubsampleOptions(strategy="soft_balanced", balance_alpha=0.5)

    def test_soft_balanced_requires_balance_alpha(self):
        with self.assertRaisesRegex(ValueError, "balance_alpha"):
            SubsampleOptions(
                strategy="soft_balanced", total_annotations=100,
            )

    def test_soft_balanced_alpha_below_zero_rejected(self):
        with self.assertRaisesRegex(ValueError, "balance_alpha"):
            SubsampleOptions(
                strategy="soft_balanced",
                total_annotations=100,
                balance_alpha=-0.1,
            )

    def test_soft_balanced_alpha_above_one_rejected(self):
        with self.assertRaisesRegex(ValueError, "balance_alpha"):
            SubsampleOptions(
                strategy="soft_balanced",
                total_annotations=100,
                balance_alpha=1.5,
            )

    def test_soft_balanced_rejects_target_per_class(self):
        with self.assertRaisesRegex(ValueError, "target_per_class"):
            SubsampleOptions(
                strategy="soft_balanced",
                total_annotations=100,
                balance_alpha=0.5,
                target_per_class=10,
            )

    def test_balance_alpha_rejected_for_stratified(self):
        with self.assertRaisesRegex(ValueError, "balance_alpha"):
            SubsampleOptions(
                strategy="stratified",
                total_annotations=100,
                balance_alpha=0.5,
            )

    def test_balance_alpha_rejected_for_balanced(self):
        with self.assertRaisesRegex(ValueError, "balance_alpha"):
            SubsampleOptions(
                strategy="balanced",
                total_annotations=100,
                balance_alpha=0.5,
            )

    def test_soft_balanced_to_log_dict_includes_alpha(self):
        opts = SubsampleOptions(
            strategy="soft_balanced",
            total_annotations=100,
            balance_alpha=0.25,
        )
        d = opts.to_log_dict()
        self.assertEqual(d["subsample/strategy"], "soft_balanced")
        self.assertEqual(d["subsample/balance_alpha"], 0.25)


if __name__ == "__main__":
    unittest.main()
