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

    def test_total_split_equally(self):
        counts = {("a", ""): 1000, ("b", ""): 1000, ("c", ""): 1000}
        opts = SubsampleOptions(strategy="balanced", total_annotations=300)
        targets = compute_per_class_targets(opts, counts)
        self.assertEqual(targets[("a", "")], 100)
        self.assertEqual(targets[("b", "")], 100)
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

    def test_single_class(self):
        # per = 50 // 1 = 50, capped at the available 100.
        counts = {("solo", ""): 100}
        opts = SubsampleOptions(strategy="balanced", total_annotations=50)
        self.assertEqual(
            compute_per_class_targets(opts, counts),
            {("solo", ""): 50},
        )


class DispatchTest(unittest.TestCase):
    def test_stratified_dispatches(self):
        # Smoke test: dispatch reaches _stratified.
        opts = SubsampleOptions(strategy="stratified", total_annotations=10)
        targets = compute_per_class_targets(opts, {("a", ""): 100})
        self.assertEqual(targets[("a", "")], 10)

    def test_balanced_dispatches(self):
        opts = SubsampleOptions(strategy="balanced", total_annotations=5)
        targets = compute_per_class_targets(opts, {("a", ""): 100})
        self.assertEqual(targets[("a", "")], 5)

    def test_empty_counts_returns_empty(self):
        opts = SubsampleOptions(strategy="stratified", total_annotations=10)
        self.assertEqual(compute_per_class_targets(opts, {}), {})


if __name__ == "__main__":
    unittest.main()
