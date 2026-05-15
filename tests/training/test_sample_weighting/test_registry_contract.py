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

from .fakes import (
    FakeGFLibrary, small_tree,
)


def _toy_class_counts() -> dict[str, int]:
    """Six classes, all with reasonable counts (the rare-class drop/merge
    decision is now upstream of weighting, so this contract tests the
    weighting layer with the kept-class set the data pipeline produces)."""
    return {
        combine_ba_gf("A1", "g1"): 100,
        combine_ba_gf("A2", "g1"): 100,
        combine_ba_gf("B1", "g1"): 50,
        combine_ba_gf("C1a", "g2"): 30,
        combine_ba_gf("A1", "g2"): 20,
        combine_ba_gf("B1", "g2"): 15,
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
        for _, weights in self._run_for_each({}):
            self.assertEqual(set(weights.keys()), set(self.counts.keys()))

    def test_no_zero_weights(self):
        # Universal contract: every class gets a strictly positive
        # weight.
        for _, weights in self._run_for_each({}):
            for label, w in weights.items():
                self.assertGreater(
                    w, 0.0,
                    f"weight for {label!r} is non-positive: {w!r}",
                )

    def test_deterministic(self):
        for name in sorted(STRATEGY_REGISTRY):
            with self.subTest(strategy=name):
                opts = SampleWeightingOptions(strategy=name)
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
            opts.weight_ratio_cap = None
            compute_class_weights(
                self.counts, self.ba_lib, self.gf_lib, opts)

    def test_weight_ratio_cap_bounds_weight_spread(self):
        # Setting weight_ratio_cap=R must ensure max/min <= R for every
        # registered strategy, including any that override compute() in
        # the future. Since weights are universally positive (no
        # zero-weighting in this layer), the cap applies to the full
        # weight set.
        cap = 5.0
        tol = 1e-9
        for _, weights in self._run_for_each({"weight_ratio_cap": cap}):
            self.assertGreaterEqual(len(weights), 2)
            ws = list(weights.values())
            self.assertLessEqual(max(ws) / min(ws), cap + tol)


if __name__ == "__main__":
    unittest.main()
