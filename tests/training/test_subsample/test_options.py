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

    def test_within_class_defaults_to_ordered_with_seed_zero(self):
        options = SubsampleOptions(strategy="balanced", total_annotations=100)
        self.assertEqual(options.within_class, "ordered")
        self.assertEqual(options.seed, 0)

    def test_unknown_within_class_rejected(self):
        with self.assertRaisesRegex(ValueError, "within_class"):
            SubsampleOptions(
                strategy="balanced",
                total_annotations=100,
                within_class="random",  # pyright: ignore[reportArgumentType]
            )

    def test_negative_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            SubsampleOptions(strategy="balanced", total_annotations=100, seed=-1)


class MultiplierValidationTest(unittest.TestCase):
    def _make(self, multipliers):
        return SubsampleOptions(
            strategy="balanced",
            total_annotations=1000,
            class_target_multipliers=multipliers,
        )

    def test_key_without_separator_rejected(self):
        with self.assertRaisesRegex(ValueError, "class_target_multipliers"):
            self._make({"abc": 3.0})

    def test_key_with_empty_benthic_attribute_rejected(self):
        with self.assertRaisesRegex(ValueError, "class_target_multipliers"):
            self._make({"::gf": 3.0})

    def test_non_positive_value_rejected(self):
        for bad in (0.0, -1.0):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "> 0"):
                self._make({"a::": bad})

    def test_non_finite_value_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite"):
                self._make({"a::": bad})

    def test_empty_growth_form_key_accepted(self):
        self.assertEqual(self._make({"a::": 3.0}).multipliers_by_class(), {("a", ""): 3.0})


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
                "subsample/within_class": "ordered",
                "subsample/seed": 0,
                "subsample/class_target_multipliers": "",
            },
        )

    def test_multipliers_are_logged_as_sorted_json(self):
        options = SubsampleOptions(
            strategy="balanced",
            total_annotations=1000,
            class_target_multipliers={"b::g": 2.0, "a::": 3.0},
        )

        self.assertEqual(
            options.to_log_dict()["subsample/class_target_multipliers"],
            '{"a::": 3.0, "b::g": 2.0}',
        )

    def test_keys_are_lower_cased_and_exposed_as_class_keys(self):
        options = SubsampleOptions(
            strategy="balanced",
            total_annotations=1000,
            class_target_multipliers={"ABC-1::GF": 3},
        )

        self.assertEqual(options.multipliers_by_class(), {("abc-1", "gf"): 3.0})

    def test_source_spread_and_seed_reach_the_logged_params(self):
        options = SubsampleOptions(
            strategy="balanced", total_annotations=1000, within_class="source_spread", seed=7
        )

        logged = options.to_log_dict()

        self.assertEqual(logged["subsample/within_class"], "source_spread")
        self.assertEqual(logged["subsample/seed"], 7)


if __name__ == "__main__":
    unittest.main()
