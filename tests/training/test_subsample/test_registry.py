"""compute_per_class_targets allocator behavior tests."""

from __future__ import annotations

import unittest

from mermaid_classifier.training.subsample.options import SubsampleOptions
from mermaid_classifier.training.subsample.registry import (
    compute_per_class_targets,
)


class StratifiedTest(unittest.TestCase):
    def test_proportional_split_sums_to_target(self):
        counts = {("ba1", "gf1"): 1000, ("ba2", "gf2"): 200, ("ba3", "gf3"): 50}
        opts = SubsampleOptions(strategy="stratified", total_annotations=125)
        targets = compute_per_class_targets(opts, counts)
        # Sum should equal the budget within rounding tolerance and (with
        # _trim_overshoot) exactly equal it when the data allows.
        self.assertEqual(sum(targets.values()), 125)
        # Class proportions are roughly preserved (1000:200:50 ~ 80%:16%:4%).
        self.assertGreater(targets[("ba1", "gf1")], targets[("ba2", "gf2")])
        self.assertGreater(targets[("ba2", "gf2")], targets[("ba3", "gf3")])

    def test_skewed_distribution_preserved(self):
        # 1 class is 90% of data; it should still get ~90% of the budget.
        counts = {("ba", "gf"): 9000, ("ba", "gf2"): 1000}
        opts = SubsampleOptions(strategy="stratified", total_annotations=100)
        targets = compute_per_class_targets(opts, counts)
        self.assertEqual(targets[("ba", "gf")], 90)
        self.assertEqual(targets[("ba", "gf2")], 10)

    def test_never_oversamples(self):
        # If a class has only 5 rows, target must not exceed 5 even if
        # the proportional allocation says more.
        counts = {("a", ""): 5, ("b", ""): 1_000_000}
        opts = SubsampleOptions(strategy="stratified", total_annotations=10)
        targets = compute_per_class_targets(opts, counts)
        self.assertLessEqual(targets[("a", "")], 5)

    def test_min_per_class_floor(self):
        # Tiny class would round to 0 without a floor.
        counts = {("big", ""): 10_000, ("tiny", ""): 1}
        opts = SubsampleOptions(
            strategy="stratified",
            total_annotations=100,
            min_per_class=1,
        )
        targets = compute_per_class_targets(opts, counts)
        self.assertEqual(targets[("tiny", "")], 1)

    def test_empty_class_counts(self):
        opts = SubsampleOptions(strategy="stratified", total_annotations=100)
        self.assertEqual(compute_per_class_targets(opts, {}), {})

    def test_overshoot_never_trims_below_min_per_class(self):
        # 5 abundant classes, tiny budget, min_per_class=10. Each class
        # floors to 10 (sum 50), overshooting the budget of 10. The trim
        # must NOT pull any class below the floor -- it accepts the
        # residual overshoot rather than dropping rare classes to 0.
        counts = {(f"c{i}", ""): 1000 for i in range(5)}
        opts = SubsampleOptions(
            strategy="stratified",
            total_annotations=10,
            min_per_class=10,
        )
        targets = compute_per_class_targets(opts, counts)
        self.assertEqual(len(targets), 5)
        for cls, t in targets.items():
            self.assertGreaterEqual(t, 10, f"{cls} trimmed below min_per_class floor")

    def test_overshoot_trims_largest_class(self):
        # Three equal-size classes, budget=100. Naive rounding gives
        # 33+33+33=99; the trim helper accepts the 1-row undershoot
        # rather than over-allocating. This documents the chosen
        # behavior: never grow targets to hit an exact budget.
        counts = {("a", ""): 1000, ("b", ""): 1000, ("c", ""): 1000}
        opts = SubsampleOptions(strategy="stratified", total_annotations=100)
        targets = compute_per_class_targets(opts, counts)
        self.assertLessEqual(sum(targets.values()), 100)
        self.assertGreaterEqual(sum(targets.values()), 99)


class BalancedTest(unittest.TestCase):
    def test_caps_per_class_at_budget_share(self):
        # total // num_classes = 300 // 3 = 100. Classes above the share
        # are capped at 100; classes below are kept in full.
        counts = {("a", ""): 1000, ("b", ""): 30, ("c", ""): 500}
        opts = SubsampleOptions(strategy="balanced", total_annotations=300)
        targets = compute_per_class_targets(opts, counts)
        self.assertEqual(targets[("a", "")], 100)
        self.assertEqual(targets[("b", "")], 30)
        self.assertEqual(targets[("c", "")], 100)

    def test_balanced_respects_min_per_class(self):
        # min_per_class is the floor: even if min(per, n_c) is smaller,
        # the allocator returns at least min_per_class. Here per =
        # 20 // 2 = 10. The SQL apply step then keeps min(target, n_c)
        # rows, which is the documented behavior (floor expresses
        # *intent*; the SQL caps at availability).
        counts = {("a", ""): 1000, ("b", ""): 1}
        opts = SubsampleOptions(
            strategy="balanced",
            total_annotations=20,
            min_per_class=5,
        )
        targets = compute_per_class_targets(opts, counts)
        self.assertEqual(targets[("a", "")], 10)
        self.assertEqual(targets[("b", "")], 5)


class MultiplierTest(unittest.TestCase):
    def _opts(self, multipliers, strategy="balanced", total=300, **kw):
        return SubsampleOptions(
            strategy=strategy,
            total_annotations=total,
            class_target_multipliers=multipliers,
            **kw,
        )

    def test_balanced_multiplier_raises_target_to_a_multiple_of_the_cap(self):
        # cap = 300 // 3 = 100; a 3.0 multiplier gives 300.
        counts = {("a", ""): 1000, ("b", ""): 1000, ("c", ""): 1000}
        targets = compute_per_class_targets(self._opts({"a::": 3.0}), counts)
        self.assertEqual(targets, {("a", ""): 300, ("b", ""): 100, ("c", ""): 100})

    def test_target_is_clamped_to_the_available_rows(self):
        counts = {("a", ""): 150, ("b", ""): 1000, ("c", ""): 1000}
        targets = compute_per_class_targets(self._opts({"a::": 3.0}), counts)
        self.assertEqual(targets[("a", "")], 150)

    def test_stratified_boost_is_not_trimmed_back_to_the_budget(self):
        # Proportional targets are 100 / 100 / 100 (sum 300 = budget); the
        # boosted class ends at 300 and the realised total exceeds 300.
        counts = {("a", ""): 1000, ("b", ""): 1000, ("c", ""): 1000}
        opts = self._opts({"a::": 3.0}, strategy="stratified")
        targets = compute_per_class_targets(opts, counts)
        self.assertEqual(targets, {("a", ""): 300, ("b", ""): 100, ("c", ""): 100})
        self.assertGreater(sum(targets.values()), 300)

    def test_classes_without_a_multiplier_are_unchanged(self):
        counts = {("a", ""): 1000, ("b", ""): 30, ("c", ""): 500}
        base = compute_per_class_targets(self._opts({}), counts)
        boosted = compute_per_class_targets(self._opts({"a::": 2.0}), counts)
        self.assertEqual(boosted[("b", "")], base[("b", "")])
        self.assertEqual(boosted[("c", "")], base[("c", "")])

    def test_unknown_key_raises_and_lists_the_key(self):
        counts = {("a", ""): 1000}
        with self.assertRaisesRegex(ValueError, "typo::"):
            compute_per_class_targets(self._opts({"typo::": 3.0}), counts)

    def test_multiplier_below_one_keeps_the_min_per_class_floor(self):
        # cap = 200 // 2 = 100; 0.01 * 100 rounds to 1, floored at 5.
        counts = {("a", ""): 1000, ("b", ""): 1000}
        opts = self._opts({"a::": 0.01}, total=200, min_per_class=5)
        targets = compute_per_class_targets(opts, counts)
        self.assertEqual(targets[("a", "")], 5)
        self.assertEqual(targets[("b", "")], 100)

    def test_empty_counts_return_empty_with_multipliers_set(self):
        self.assertEqual(compute_per_class_targets(self._opts({"a::": 3.0}), {}), {})


if __name__ == "__main__":
    unittest.main()
