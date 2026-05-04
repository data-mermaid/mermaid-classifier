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
    def test_target_per_class_explicit(self):
        counts = {("a", ""): 1000, ("b", ""): 30, ("c", ""): 500}
        opts = SubsampleOptions(strategy="balanced", target_per_class=100)
        targets = compute_per_class_targets(opts, counts)
        # Capped at min(target, n_c).
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
        # min_per_class is the floor: even if min(target, n_c) is
        # smaller, the allocator returns at least min_per_class. The
        # SQL apply step then keeps min(target, n_c) rows, which is
        # the documented behavior (floor expresses *intent*; the SQL
        # caps at availability).
        counts = {("a", ""): 1000, ("b", ""): 1}
        opts = SubsampleOptions(
            strategy="balanced",
            target_per_class=10,
            min_per_class=5,
        )
        targets = compute_per_class_targets(opts, counts)
        self.assertEqual(targets[("a", "")], 10)
        self.assertEqual(targets[("b", "")], 5)

    def test_single_class(self):
        counts = {("solo", ""): 100}
        opts = SubsampleOptions(strategy="balanced", target_per_class=50)
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
        opts = SubsampleOptions(strategy="balanced", target_per_class=5)
        targets = compute_per_class_targets(opts, {("a", ""): 100})
        self.assertEqual(targets[("a", "")], 5)

    def test_soft_balanced_dispatches(self):
        opts = SubsampleOptions(
            strategy="soft_balanced",
            total_annotations=10,
            balance_alpha=0.5,
        )
        targets = compute_per_class_targets(opts, {("a", ""): 100})
        self.assertEqual(targets[("a", "")], 10)

    def test_empty_counts_returns_empty(self):
        opts = SubsampleOptions(strategy="stratified", total_annotations=10)
        self.assertEqual(compute_per_class_targets(opts, {}), {})


class SoftBalancedTest(unittest.TestCase):
    """Allocator behavior at the limits of balance_alpha and at 0.5."""

    def test_alpha_zero_matches_stratified(self):
        # alpha=0 -> target_c proportional to n_c -> identical to
        # _stratified for the same total_annotations.
        counts = {("a", ""): 1000, ("b", ""): 200, ("c", ""): 50}
        soft = SubsampleOptions(
            strategy="soft_balanced",
            total_annotations=125,
            balance_alpha=0.0,
        )
        strat = SubsampleOptions(
            strategy="stratified", total_annotations=125,
        )
        self.assertEqual(
            compute_per_class_targets(soft, counts),
            compute_per_class_targets(strat, counts),
        )

    def test_alpha_one_distributes_equally(self):
        # alpha=1 -> target_c proportional to 1 -> equal split, capped
        # at n_c. With abundant classes the result equals balanced
        # with total_annotations split evenly across classes.
        counts = {("a", ""): 1000, ("b", ""): 1000, ("c", ""): 1000}
        soft = SubsampleOptions(
            strategy="soft_balanced",
            total_annotations=300,
            balance_alpha=1.0,
        )
        bal = SubsampleOptions(
            strategy="balanced", total_annotations=300,
        )
        self.assertEqual(
            compute_per_class_targets(soft, counts),
            compute_per_class_targets(bal, counts),
        )

    def test_alpha_half_is_sqrt_proportional(self):
        # alpha=0.5 -> target_c proportional to sqrt(n_c).
        # 100:25 sqrt-ratio is 10:5 -> 2:1, so at 60 total we get 40:20.
        counts = {("a", ""): 100, ("b", ""): 25}
        opts = SubsampleOptions(
            strategy="soft_balanced",
            total_annotations=60,
            balance_alpha=0.5,
        )
        targets = compute_per_class_targets(opts, counts)
        self.assertEqual(targets[("a", "")], 40)
        self.assertEqual(targets[("b", "")], 20)

    def test_soft_balanced_caps_at_available(self):
        # Rare class has 5 rows. soft_balanced must not oversample it.
        counts = {("rare", ""): 5, ("common", ""): 1_000_000}
        opts = SubsampleOptions(
            strategy="soft_balanced",
            total_annotations=100,
            balance_alpha=0.8,
        )
        targets = compute_per_class_targets(opts, counts)
        self.assertLessEqual(targets[("rare", "")], 5)

    def test_soft_balanced_min_per_class_floor(self):
        # alpha=0.0 (proportional) on a 1-row tiny class would round
        # to 0; min_per_class=1 must keep it.
        counts = {("big", ""): 10_000, ("tiny", ""): 1}
        opts = SubsampleOptions(
            strategy="soft_balanced",
            total_annotations=100,
            balance_alpha=0.0,
            min_per_class=1,
        )
        targets = compute_per_class_targets(opts, counts)
        self.assertEqual(targets[("tiny", "")], 1)

    def test_soft_balanced_alpha_between_collapses(self):
        # At 0 < alpha < 1, max:min target ratio should land strictly
        # between the stratified ratio and the balanced ratio.
        # Budget is sized so neither class hits its availability cap,
        # so the cap-clamping doesn't distort the ratio.
        counts = {("big", ""): 10_000, ("small", ""): 1_000}
        # stratified ratio is 10, balanced is 1.
        soft = SubsampleOptions(
            strategy="soft_balanced",
            total_annotations=440,
            balance_alpha=0.5,
        )
        targets = compute_per_class_targets(soft, counts)
        ratio = targets[("big", "")] / targets[("small", "")]
        self.assertLess(ratio, 10)  # less skewed than stratified
        self.assertGreater(ratio, 1)  # more skewed than balanced
        # sqrt(10000/1000) ~ 3.16, so target ratio should be near it.
        self.assertAlmostEqual(ratio, 3.16, delta=0.3)


if __name__ == "__main__":
    unittest.main()
