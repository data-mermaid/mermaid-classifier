"""water_fill: the per-group quota allocator behind within_class='source_spread'."""

from __future__ import annotations

import unittest

from mermaid_classifier.training.subsample import water_fill


class WaterFillTest(unittest.TestCase):
    def test_equal_quotas_for_groups_larger_than_the_level(self):
        # Level L=30: the 10-point group is kept whole, the others get 30.
        quotas = water_fill({"a": 10, "b": 100, "c": 500}, 70)
        self.assertEqual(quotas, {"a": 10, "b": 30, "c": 30})

    def test_remainder_goes_to_groups_below_their_size_in_key_order(self):
        # L=3 uses 9 of 11; the 2 left over go to the first two keys.
        quotas = water_fill({"c": 50, "a": 50, "b": 50}, 11)
        self.assertEqual(quotas, {"a": 4, "b": 4, "c": 3})

    def test_remainder_skips_groups_already_at_their_size(self):
        # L=3: "a" is full at 2; the single leftover goes to "b", not "a".
        quotas = water_fill({"a": 2, "b": 50, "c": 50}, 9)
        self.assertEqual(quotas, {"a": 2, "b": 4, "c": 3})

    def test_target_at_or_above_total_keeps_every_group_whole(self):
        counts = {"a": 3, "b": 4}
        self.assertEqual(water_fill(counts, 7), counts)
        self.assertEqual(water_fill(counts, 1000), counts)

    def test_zero_target_gives_zero_quotas(self):
        self.assertEqual(water_fill({"a": 3, "b": 4}, 0), {"a": 0, "b": 0})

    def test_single_group(self):
        self.assertEqual(water_fill({"a": 10}, 4), {"a": 4})

    def test_empty(self):
        self.assertEqual(water_fill({}, 5), {})

    def test_tuple_keys(self):
        quotas = water_fill({("coralnet", "1"): 5, ("mermaid", "9"): 5}, 3)
        self.assertEqual(quotas, {("coralnet", "1"): 2, ("mermaid", "9"): 1})

    def test_properties_over_a_grid_of_inputs(self):
        counts = {"a": 1, "b": 7, "c": 7, "d": 40, "e": 41}
        total = sum(counts.values())
        for target in range(0, total + 5):
            with self.subTest(target=target):
                quotas = water_fill(counts, target)
                self.assertEqual(sum(quotas.values()), min(target, total))
                self.assertEqual(set(quotas), set(counts))
                for g, q in quotas.items():
                    self.assertLessEqual(q, counts[g])
                    self.assertGreaterEqual(q, 0)
                # Water-filling: a group that did not fill is within 1 of
                # every other unfilled group.
                unfilled = [q for g, q in quotas.items() if q < counts[g]]
                if unfilled:
                    self.assertLessEqual(max(unfilled) - min(unfilled), 1)
                    # A filled group never exceeds the unfilled level by more
                    # than the one remainder row.
                    for g, q in quotas.items():
                        if q == counts[g]:
                            self.assertLessEqual(q, max(unfilled) + 1)

    def test_input_order_does_not_change_the_result(self):
        forward = water_fill({"a": 50, "b": 50, "c": 50}, 100)
        backward = water_fill({"c": 50, "b": 50, "a": 50}, 100)
        self.assertEqual(forward, backward)


if __name__ == "__main__":
    unittest.main()
