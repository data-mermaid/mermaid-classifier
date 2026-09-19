"""TrainingRunner._compute_class_weights wiring test.

Exercises the orchestration path that connects DatasetOptions.weighting
through to compute_class_weights and the per-class log structure.

_compute_class_weights reads one field, labels.train.label_count_per_class,
and reaches no taxonomy library, so no BA/GF double is needed here.
"""

from __future__ import annotations

import types
import unittest
from collections import Counter

from mermaid_classifier.common.benthic_attributes import combine_ba_gf
from mermaid_classifier.pyspacer.options import DatasetOptions
from mermaid_classifier.pyspacer.runner import TrainingRunner
from mermaid_classifier.training.sample_weighting import (
    SampleWeightingOptions,
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

    def _make_runner(self, weighting):
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
        runner = self._make_runner(weighting=SampleWeightingOptions(enabled=False))
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
            self.assertGreater(w, 0.0, f"weight for {label!r} should be positive")
        self.assertTrue(log["enabled"])


if __name__ == "__main__":
    unittest.main()
