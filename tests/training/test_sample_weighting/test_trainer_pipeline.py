"""TrainingRunner._compute_class_weights wiring test.

Exercises the orchestration path that connects DatasetOptions.weighting
through to compute_class_weights and the per-class log structure. We
bypass the real BenthicAttributeLibrary/GrowthFormLibrary by
monkeypatching the module globals in train.py with fakes — those real
classes hit the MERMAID API on construction, which is unsuitable for
unit tests.
"""
from __future__ import annotations

import types
import unittest
from collections import Counter
from unittest import mock

from mermaid_classifier.common.benthic_attributes import combine_ba_gf
from mermaid_classifier.training.sample_weighting import (
    SampleWeightingOptions,
)

from .fakes import (
    FakeGFLibrary, small_tree,
)


def _fake_labels(class_counts: dict[str, int]):
    """Mimic the shape of pyspacer's ImageLabels container expected by
    ``TrainingRunner._compute_class_weights``: it only reads
    ``labels.train.label_count_per_class``."""
    train = types.SimpleNamespace(
        label_count_per_class=Counter(class_counts),
    )
    return types.SimpleNamespace(train=train)


class TrainerPipelineTest(unittest.TestCase):
    """Contract: DatasetOptions.weighting -> _compute_class_weights ->
    weight dict + summary log structure."""

    @classmethod
    def setUpClass(cls):
        # Import train.py with the MERMAID API singletons patched out.
        # This avoids any network call at import time.
        with mock.patch(
            "mermaid_classifier.common.benthic_attributes."
            "BenthicAttributeLibrary",
            return_value=small_tree(),
        ), mock.patch(
            "mermaid_classifier.common.benthic_attributes.GrowthFormLibrary",
            return_value=FakeGFLibrary({"g1": "GF1", "g2": "GF2"}),
        ):
            from mermaid_classifier.pyspacer import train as train_mod
        cls.train_mod = train_mod

    def setUp(self):
        # Replace the cached taxonomy-library accessors with ones that
        # return our fakes, for the duration of each test. train.py calls
        # get_benthic_attribute_library() / get_growth_form_library()
        # (the real ones hit the MERMAID API on construction).
        fake_ba = small_tree()
        fake_gf = FakeGFLibrary({"g1": "GF1", "g2": "GF2"})
        self._patches = [
            mock.patch.object(
                self.train_mod, "get_benthic_attribute_library",
                lambda: fake_ba),
            mock.patch.object(
                self.train_mod, "get_growth_form_library",
                lambda: fake_gf),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def _make_runner(self, weighting):
        DatasetOptions = self.train_mod.DatasetOptions
        TrainingRunner = self.train_mod.TrainingRunner
        return TrainingRunner(
            dataset_options=DatasetOptions(
                include_mermaid=False,
                weighting=weighting,
            ),
        )

    def test_weighting_none_returns_none_and_disabled_log(self):
        runner = self._make_runner(weighting=None)
        labels = _fake_labels({combine_ba_gf("A1", "g1"): 100})
        weights, log = runner._compute_class_weights(labels)
        self.assertIsNone(weights)
        self.assertEqual(log, {"enabled": False})

    def test_weighting_disabled_skips_computation(self):
        runner = self._make_runner(
            weighting=SampleWeightingOptions(enabled=False))
        labels = _fake_labels({combine_ba_gf("A1", "g1"): 100})
        weights, log = runner._compute_class_weights(labels)
        self.assertIsNone(weights)
        self.assertEqual(log, {"enabled": False})

    def test_default_options_produce_weights_for_every_class(self):
        runner = self._make_runner(weighting=SampleWeightingOptions())
        counts = {
            combine_ba_gf("A1", "g1"): 100,
            combine_ba_gf("A2", "g1"): 50,
            combine_ba_gf("B1", "g2"): 30,
        }
        labels = _fake_labels(counts)
        weights, log = runner._compute_class_weights(labels)

        self.assertIsNotNone(weights)
        self.assertEqual(set(weights), set(counts))
        for label, w in weights.items():
            self.assertGreater(
                w, 0.0, f"weight for {label!r} should be positive")
        self.assertTrue(log["enabled"])

    def test_log_structure_contains_required_summary_keys(self):
        runner = self._make_runner(weighting=SampleWeightingOptions())
        counts = {
            combine_ba_gf("A1", "g1"): 100,
            combine_ba_gf("A2", "g1"): 50,
        }
        weights, log = runner._compute_class_weights(_fake_labels(counts))
        self.assertIn("per_class_df", log)
        self.assertIn("summary", log)
        for key in (
            "weight_mean", "weight_median", "weight_p5", "weight_p95",
            "weight_max_min_ratio", "n_classes",
        ):
            self.assertIn(key, log["summary"])
        # Per-class DataFrame no longer carries a rare_action column —
        # rare-class accounting lives in the label-transforms artifact.
        df = log["per_class_df"]
        self.assertNotIn("rare_action", df.columns)
        self.assertIn("bagf_id", df.columns)
        self.assertIn("count", df.columns)
        self.assertIn("weight", df.columns)


if __name__ == "__main__":
    unittest.main()
