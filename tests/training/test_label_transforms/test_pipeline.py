"""apply_label_transforms / registry resolution tests."""
from __future__ import annotations

import unittest

from spacer.data_classes import DataLocation, ImageLabels
from spacer.task_utils import TrainingTaskLabels

from mermaid_classifier.common.benthic_attributes import combine_ba_gf
from mermaid_classifier.training.label_transforms import (
    LabelTransformsOptions,
    TRANSFORM_REGISTRY,
    TransformSpec,
    apply_label_transforms,
)

from tests.training.test_sample_weighting.fakes import (
    FakeGFLibrary, small_tree,
)


def _bagf(ba: str, gf: str) -> str:
    return combine_ba_gf(ba, gf)


def _make_split(items: dict[str, list[tuple[int, int, str]]]) -> ImageLabels:
    out = ImageLabels()
    for k, anns in items.items():
        out.add_image(DataLocation("filesystem", key=k), anns)
    return out


class RegistryTest(unittest.TestCase):
    def test_drop_and_merge_registered(self):
        self.assertIn("drop_rare", TRANSFORM_REGISTRY)
        self.assertIn("merge_rare", TRANSFORM_REGISTRY)

    def test_unknown_transform_raises(self):
        labels = TrainingTaskLabels(
            train=_make_split({"img1": [(0, 0, _bagf("A1", "g1"))] * 5}),
            ref=_make_split({"img1": [(0, 0, _bagf("A1", "g1"))]}),
            val=_make_split({"img1": [(0, 0, _bagf("A1", "g1"))]}),
        )
        opts = LabelTransformsOptions(
            enabled=True,
            pipeline=[TransformSpec("no_such_transform", {})],
        )
        with self.assertRaisesRegex(ValueError, "Unknown label transform"):
            apply_label_transforms(
                labels, opts, small_tree(), FakeGFLibrary({"g1": "GF1"}))


class PipelineCompositionTest(unittest.TestCase):
    def test_two_stage_pipeline_applies_in_order(self):
        # Stage 1: drop_rare(min_count=10) — kills the rarest classes.
        # Stage 2: merge_rare(min_count=200) — but every remaining class
        # is also under 200; their parent is the root (per small_tree),
        # so merge falls back to drop. Result: everything is dropped
        # except classes with count >= 200 (none in this test). The
        # final label space is empty.
        labels = TrainingTaskLabels(
            train=_make_split({
                "img_t": (
                    [(0, 0, _bagf("A1", "g1"))] * 100
                    + [(1, 1, _bagf("A2", "g1"))] * 5  # rare
                ),
            }),
            ref=_make_split({"img_r": [(0, 0, _bagf("A1", "g1"))]}),
            val=_make_split({"img_v": [(0, 0, _bagf("A1", "g1"))]}),
        )
        opts = LabelTransformsOptions(
            enabled=True,
            pipeline=[
                TransformSpec("drop_rare", {"min_count": 10}),
                TransformSpec("merge_rare", {"min_count": 200}),
            ],
        )
        new_labels, plans = apply_label_transforms(
            labels, opts, small_tree(), FakeGFLibrary({"g1": "GF1"}))

        # First stage's plan dropped A2. Second stage saw {A1: 100}
        # and tried to merge A1 (count 100 < 200), but A1's only
        # ancestor is the root — fall back to drop.
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0].actions.get(_bagf("A2", "g1")), "drop")
        self.assertEqual(plans[1].actions.get(_bagf("A1", "g1")), "drop")
        # Everything dropped — train split is empty.
        self.assertEqual(len(new_labels.train.classes_set), 0)


if __name__ == "__main__":
    unittest.main()
