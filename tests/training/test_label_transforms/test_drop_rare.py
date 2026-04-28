"""DropRareClasses transform tests."""
from __future__ import annotations

import unittest

from spacer.data_classes import DataLocation, ImageLabels
from spacer.task_utils import TrainingTaskLabels

from mermaid_classifier.common.benthic_attributes import combine_ba_gf
from mermaid_classifier.training.label_transforms import (
    LabelTransformsOptions,
    TransformSpec,
    apply_label_transforms,
)
from mermaid_classifier.training.label_transforms.drop_rare import (
    DropRareClasses,
)

from tests.training.test_sample_weighting.fakes import (
    FakeGFLibrary, small_tree,
)


def _bagf(ba: str, gf: str) -> str:
    return combine_ba_gf(ba, gf)


def _make_split(annotations_per_image: dict[str, list[tuple[int, int, str]]]) -> ImageLabels:
    """Build an ImageLabels from a dict of {image_key: [(row, col, bagf), ...]}."""
    out = ImageLabels()
    for key, anns in annotations_per_image.items():
        out.add_image(DataLocation("filesystem", key=key), anns)
    return out


class DropRareClassesPlanTest(unittest.TestCase):
    def setUp(self):
        self.ba_lib = small_tree()
        self.gf_lib = FakeGFLibrary({"g1": "GF1"})

    def test_below_threshold_marked_drop(self):
        counts = {
            _bagf("A1", "g1"): 100,
            _bagf("A2", "g1"): 5,         # rare
            _bagf("B1", "g1"): 3,         # rare
        }
        plan = DropRareClasses(min_count=10).plan(counts, self.ba_lib, self.gf_lib)
        self.assertEqual(plan.actions[_bagf("A2", "g1")], "drop")
        self.assertEqual(plan.actions[_bagf("B1", "g1")], "drop")
        # Common class is left out of actions (implicit keep).
        self.assertNotIn(_bagf("A1", "g1"), plan.actions)
        self.assertEqual(plan.remap, {})

    def test_threshold_boundary_inclusive_below(self):
        # min_count=10: a class with exactly 10 is kept; 9 is dropped.
        counts = {
            _bagf("A1", "g1"): 10,
            _bagf("A2", "g1"): 9,
        }
        plan = DropRareClasses(min_count=10).plan(counts, self.ba_lib, self.gf_lib)
        self.assertNotIn(_bagf("A1", "g1"), plan.actions)
        self.assertEqual(plan.actions[_bagf("A2", "g1")], "drop")

    def test_min_count_below_one_rejected(self):
        with self.assertRaisesRegex(ValueError, "min_count"):
            DropRareClasses(min_count=0)


class DropRareClassesApplyTest(unittest.TestCase):
    def setUp(self):
        self.ba_lib = small_tree()
        self.gf_lib = FakeGFLibrary({"g1": "GF1"})

    def _make_labels(
        self,
        train_anns: dict[str, list[tuple[int, int, str]]],
        ref_anns: dict[str, list[tuple[int, int, str]]],
        val_anns: dict[str, list[tuple[int, int, str]]],
    ) -> TrainingTaskLabels:
        return TrainingTaskLabels(
            train=_make_split(train_anns),
            ref=_make_split(ref_anns),
            val=_make_split(val_anns),
        )

    def test_drops_rare_classes_from_all_three_splits(self):
        common = _bagf("A1", "g1")
        rare = _bagf("A2", "g1")

        labels = self._make_labels(
            train_anns={
                "img1": [(0, 0, common)] * 100 + [(1, 1, rare)] * 5,
            },
            ref_anns={
                "img2": [(0, 0, common), (1, 1, rare)],
            },
            val_anns={
                "img3": [(0, 0, common), (1, 1, rare)],
            },
        )

        opts = LabelTransformsOptions(
            enabled=True,
            pipeline=[TransformSpec("drop_rare", {"min_count": 10})],
        )
        new_labels, plans = apply_label_transforms(
            labels, opts, self.ba_lib, self.gf_lib)

        self.assertEqual(len(plans), 1)
        # Rare class is gone from every split.
        for set_name in ("train", "ref", "val"):
            self.assertNotIn(rare, new_labels[set_name].classes_set)
            self.assertIn(common, new_labels[set_name].classes_set)
        # Train counts dropped by exactly 5 (the rare-class samples).
        self.assertEqual(new_labels.train.label_count_per_class[common], 100)
        self.assertEqual(new_labels.train.label_count_per_class[rare], 0)

    def test_image_with_only_rare_annotations_disappears(self):
        common = _bagf("A1", "g1")
        rare = _bagf("A2", "g1")
        labels = self._make_labels(
            train_anns={
                "img_common": [(0, 0, common)] * 50,
                "img_rare_only": [(0, 0, rare)] * 3,
            },
            ref_anns={"img_ref": [(0, 0, common)] * 5},
            val_anns={"img_val": [(0, 0, common)] * 5},
        )

        opts = LabelTransformsOptions(
            enabled=True,
            pipeline=[TransformSpec("drop_rare", {"min_count": 10})],
        )
        new_labels, _ = apply_label_transforms(
            labels, opts, self.ba_lib, self.gf_lib)

        # The rare-only image was dropped wholesale.
        train_keys = {loc.key for loc in new_labels.train.keys()}
        self.assertIn("img_common", train_keys)
        self.assertNotIn("img_rare_only", train_keys)

    def test_disabled_options_is_noop(self):
        labels = self._make_labels(
            train_anns={"img1": [(0, 0, _bagf("A1", "g1"))] * 5},
            ref_anns={"img2": [(0, 0, _bagf("A1", "g1"))]},
            val_anns={"img3": [(0, 0, _bagf("A1", "g1"))]},
        )
        opts = LabelTransformsOptions()  # disabled by default
        new_labels, plans = apply_label_transforms(
            labels, opts, self.ba_lib, self.gf_lib)
        self.assertIs(new_labels, labels)
        self.assertEqual(plans, [])


if __name__ == "__main__":
    unittest.main()
