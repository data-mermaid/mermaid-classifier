"""Contract tests parametrized over every registered Strategy.

Adding a new strategy automatically picks up these tests as long as it
is registered via ``@register_strategy(...)``.
"""
from __future__ import annotations

import math
import unittest

from mermaid_classifier.common.benthic_attributes import combine_ba_gf
from mermaid_classifier.training.sample_weighting import (
    SampleWeightingOptions,
    STRATEGY_REGISTRY,
    compute_class_weights,
)

from tests.training.test_sample_weighting.fakes import (
    FakeBALibrary, FakeGFLibrary, small_tree,
)


def _toy_class_counts() -> dict[str, int]:
    """Six classes with one rare (count=2)."""
    return {
        combine_ba_gf("A1", "g1"): 100,
        combine_ba_gf("A2", "g1"): 100,
        combine_ba_gf("B1", "g1"): 50,
        combine_ba_gf("C1a", "g2"): 30,
        combine_ba_gf("A1", "g2"): 20,
        combine_ba_gf("B1", "g2"): 2,        # rare
    }


class StrategyContractTest(unittest.TestCase):
    """Every strategy in the registry must satisfy these invariants."""

    def setUp(self):
        self.ba_lib = small_tree()
        self.gf_lib = FakeGFLibrary({"g1": "GF1", "g2": "GF2"})
        self.counts = _toy_class_counts()

    def _run_for_each(self, options_kwargs):
        for name in sorted(STRATEGY_REGISTRY):
            with self.subTest(strategy=name):
                opts = SampleWeightingOptions(strategy=name, **options_kwargs)
                weights = compute_class_weights(
                    class_counts=self.counts,
                    ba_library=self.ba_lib,
                    gf_library=self.gf_lib,
                    options=opts,
                )
                yield name, weights

    def test_returns_weight_for_every_class(self):
        for _, weights in self._run_for_each(dict(rare_policy="keep")):
            self.assertEqual(set(weights.keys()), set(self.counts.keys()))

    def test_keep_policy_no_zero_weights(self):
        for _, weights in self._run_for_each(dict(rare_policy="keep")):
            for label, w in weights.items():
                self.assertGreater(
                    w, 0.0,
                    f"weight for {label!r} is non-positive: {w!r}",
                )

    def test_drop_policy_zeros_only_rare_classes(self):
        for _, weights in self._run_for_each(
            dict(rare_policy="drop", min_count=10),
        ):
            rare_label = combine_ba_gf("B1", "g2")
            for label, w in weights.items():
                if self.counts[label] < 10:
                    self.assertEqual(
                        w, 0.0,
                        f"rare class {label!r} should be zeroed, got {w!r}",
                    )
                else:
                    self.assertGreater(w, 0.0)
            self.assertEqual(weights[rare_label], 0.0)

    def test_deterministic(self):
        for name in sorted(STRATEGY_REGISTRY):
            with self.subTest(strategy=name):
                opts = SampleWeightingOptions(
                    strategy=name, rare_policy="keep")
                w1 = compute_class_weights(
                    self.counts, self.ba_lib, self.gf_lib, opts)
                w2 = compute_class_weights(
                    self.counts, self.ba_lib, self.gf_lib, opts)
                for k in w1:
                    self.assertTrue(
                        math.isclose(w1[k], w2[k]),
                        f"strategy {name}: weight for {k!r} drifted across calls",
                    )

    def test_disabled_returns_empty(self):
        opts = SampleWeightingOptions(enabled=False)
        weights = compute_class_weights(
            self.counts, self.ba_lib, self.gf_lib, opts)
        self.assertEqual(weights, {})

    def test_unknown_strategy_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown weighting strategy"):
            opts = SampleWeightingOptions.__new__(SampleWeightingOptions)
            # Bypass __post_init__ validation since we're testing the
            # registry's secondary defence.
            opts.enabled = True
            opts.strategy = "no_such_strategy"
            opts.alpha = 0.5
            opts.min_count = 10
            opts.rare_policy = "drop"
            compute_class_weights(
                self.counts, self.ba_lib, self.gf_lib, opts)

    def test_merge_policy_raises_not_implemented(self):
        for name in sorted(STRATEGY_REGISTRY):
            with self.subTest(strategy=name):
                opts = SampleWeightingOptions(
                    strategy=name, rare_policy="merge")
                with self.assertRaises(NotImplementedError):
                    compute_class_weights(
                        self.counts, self.ba_lib, self.gf_lib, opts)


if __name__ == "__main__":
    unittest.main()
