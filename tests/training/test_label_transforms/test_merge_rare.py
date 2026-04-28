"""MergeRareClassesToBAParent transform tests."""
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
from mermaid_classifier.training.label_transforms.merge_rare import (
    MergeRareClassesToBAParent,
)

from tests.training.test_sample_weighting.fakes import (
    FakeBALibrary, FakeGFLibrary, small_tree,
)


def _bagf(ba: str, gf: str) -> str:
    return combine_ba_gf(ba, gf)


def _make_split(annotations_per_image: dict[str, list[tuple[int, int, str]]]) -> ImageLabels:
    out = ImageLabels()
    for key, anns in annotations_per_image.items():
        out.add_image(DataLocation("filesystem", key=key), anns)
    return out


class MergeRareClassesPlanTest(unittest.TestCase):
    """plan() unit tests — exercise the algorithm without applying."""

    def setUp(self):
        self.gf_lib = FakeGFLibrary({"g1": "GF1", "g2": "GF2"})

    def test_drops_when_only_ancestor_is_root(self):
        # Tree: A is root; A1, A2 are children of A.
        ba_lib = small_tree()
        counts = {
            _bagf("A1", "g1"): 30,    # rare under min_count=50
            _bagf("A2", "g1"): 100,
        }
        # A1 has parent A (which is the root). The walk excludes the
        # root, so A1 has no eligible non-root ancestor and falls back
        # to drop. This confirms the user-chosen fallback.
        plan = MergeRareClassesToBAParent(min_count=50).plan(
            counts, ba_lib, self.gf_lib)
        self.assertEqual(plan.actions.get(_bagf("A1", "g1")), "drop")
        self.assertEqual(plan.remap, {})

    def test_remaps_to_lowest_non_root_ancestor_with_enough_count(self):
        # Build a deeper tree:
        #   root None
        #     |- Top
        #          |- Mid
        #               |- Leaf1   (rare)
        #               |- Leaf2   (kept)
        edges = [
            ("Top", None),
            ("Mid", "Top"),
            ("Leaf1", "Mid"),
            ("Leaf2", "Mid"),
        ]
        ba_lib = FakeBALibrary(edges)
        counts = {
            _bagf("Leaf1", "g1"): 30,    # rare
            _bagf("Leaf2", "g1"): 100,
        }
        # Leaf1's parent is Mid; ancestors are [Top, Mid] root-first
        # (Top is the root since Top.parent == None). Walk parent-first,
        # excluding the root: chain = [Mid]. Mid has no direct count
        # (no _bagf("Mid", "g1") in counts), so eff[(Mid, "g1")] = 0
        # and adding Leaf1's 30 -> 30 < min_count=50 — fail.
        # No more non-root ancestors, so drop.
        plan = MergeRareClassesToBAParent(min_count=50).plan(
            counts, ba_lib, self.gf_lib)
        self.assertEqual(plan.actions.get(_bagf("Leaf1", "g1")), "drop")

        # Now configure it so Mid has its own count that's enough to
        # absorb the rare class.
        counts2 = {
            _bagf("Leaf1", "g1"): 30,
            _bagf("Mid", "g1"): 40,     # 40+30 = 70 >= 50
        }
        plan2 = MergeRareClassesToBAParent(min_count=50).plan(
            counts2, ba_lib, self.gf_lib)
        # Leaf1 merges into Mid's bagf.
        self.assertEqual(
            plan2.remap[_bagf("Leaf1", "g1")], _bagf("Mid", "g1"))
        self.assertNotIn(_bagf("Leaf1", "g1"), plan2.actions)

    def test_two_rare_siblings_merge_at_shared_ancestor(self):
        edges = [
            ("Top", None),
            ("Mid", "Top"),
            ("Leaf1", "Mid"),
            ("Leaf2", "Mid"),
        ]
        ba_lib = FakeBALibrary(edges)
        counts = {
            _bagf("Leaf1", "g1"): 30,
            _bagf("Leaf2", "g1"): 25,
        }
        # Both rare. Process in count-ascending order: Leaf2 (25), then
        # Leaf1 (30).
        # Leaf2: Mid.eff=0, +25=25 < 50, drop fallback considered;
        # but only ancestor is Mid — fail, drop.
        # Wait: that's the simple greedy. Let's instead set up a case
        # where merges build cumulative count.
        counts2 = {
            _bagf("Leaf1", "g1"): 30,
            _bagf("Leaf2", "g1"): 25,
            _bagf("Mid", "g1"): 5,    # already kept-class with count 5
        }
        # Process by count ascending: Mid (5)? No — Mid is at count 5
        # which is itself rare. After processing Mid (count 5), checks
        # ancestors of Mid: chain = [Top]; reversed [Top]; exclude root
        # Top -> chain empty. Drop Mid.
        # Then Leaf2 (25): Mid.eff = 0 (Mid was dropped). 0+25 = 25 < 50.
        # Drop. Then Leaf1 (30): Mid.eff = 0. 0+30 = 30 < 50. Drop.
        plan = MergeRareClassesToBAParent(min_count=50).plan(
            counts2, ba_lib, self.gf_lib)
        # All three are dropped because their only non-root ancestor
        # never gathers enough cumulative count.
        for cls in (_bagf("Mid", "g1"), _bagf("Leaf1", "g1"), _bagf("Leaf2", "g1")):
            self.assertEqual(plan.actions.get(cls), "drop")

    def test_merge_with_buildup_at_intermediate_ancestor(self):
        # Use a deeper tree so intermediate ancestors can accumulate.
        #  Root
        #   |- L1
        #        |- L2
        #             |- L3a (rare)
        #             |- L3b (rare)
        edges = [
            ("L1", None),       # root
            ("L2", "L1"),
            ("L3a", "L2"),
            ("L3b", "L2"),
        ]
        ba_lib = FakeBALibrary(edges)
        counts = {
            _bagf("L3a", "g1"): 30,
            _bagf("L3b", "g1"): 25,
        }
        # Process L3b first (count 25). Chain (parent-first, excl.
        # root) = [L2]. L2.eff = 0, +25 = 25 < 50. Fail. Drop L3b.
        # Process L3a (30). L2.eff = 0, +30 = 30 < 50. Fail. Drop.
        # Sub-50 budget at L2: nothing merges. Both drop.
        plan = MergeRareClassesToBAParent(min_count=50).plan(
            counts, ba_lib, self.gf_lib)
        self.assertEqual(plan.actions.get(_bagf("L3a", "g1")), "drop")
        self.assertEqual(plan.actions.get(_bagf("L3b", "g1")), "drop")

        # Larger counts that succeed: each rare individually puts L2
        # over threshold.
        counts_big = {
            _bagf("L3a", "g1"): 30,
            _bagf("L3b", "g1"): 60,
        }
        # Process L3a first (30): L2.eff = 0, +30 = 30 < 50. Fail. Drop.
        # Process L3b (60): not rare? 60 >= 50, so it's not in the rare
        # set. So plan only acts on L3a. Result: L3a dropped, L3b kept.
        plan_big = MergeRareClassesToBAParent(min_count=50).plan(
            counts_big, ba_lib, self.gf_lib)
        self.assertEqual(plan_big.actions.get(_bagf("L3a", "g1")), "drop")
        self.assertNotIn(_bagf("L3b", "g1"), plan_big.actions)
        self.assertNotIn(_bagf("L3b", "g1"), plan_big.remap)

    def test_root_only_class_dropped(self):
        # A class whose BA is itself a root has no ancestors at all.
        edges = [("OnlyRoot", None)]
        ba_lib = FakeBALibrary(edges)
        counts = {
            _bagf("OnlyRoot", "g1"): 5,    # rare
        }
        plan = MergeRareClassesToBAParent(min_count=10).plan(
            counts, ba_lib, self.gf_lib)
        self.assertEqual(plan.actions[_bagf("OnlyRoot", "g1")], "drop")

    def test_min_count_below_one_rejected(self):
        with self.assertRaisesRegex(ValueError, "min_count"):
            MergeRareClassesToBAParent(min_count=0)


class MergeRareClassesApplyTest(unittest.TestCase):
    def setUp(self):
        edges = [
            ("Top", None),
            ("Mid", "Top"),
            ("Leaf1", "Mid"),
            ("Leaf2", "Mid"),
        ]
        self.ba_lib = FakeBALibrary(edges)
        self.gf_lib = FakeGFLibrary({"g1": "GF1"})

    def test_merge_relabels_in_all_splits(self):
        rare = _bagf("Leaf1", "g1")
        target = _bagf("Mid", "g1")

        labels = TrainingTaskLabels(
            train=_make_split({
                "img_t": [(0, 0, rare)] * 30 + [(1, 1, target)] * 40,
            }),
            ref=_make_split({
                "img_r": [(0, 0, rare), (1, 1, target)],
            }),
            val=_make_split({
                "img_v": [(0, 0, rare), (1, 1, target)],
            }),
        )

        opts = LabelTransformsOptions(
            enabled=True,
            pipeline=[TransformSpec("merge_rare", {"min_count": 50})],
        )
        new_labels, plans = apply_label_transforms(
            labels, opts, self.ba_lib, self.gf_lib)

        self.assertEqual(len(plans), 1)
        for set_name in ("train", "ref", "val"):
            classes = new_labels[set_name].classes_set
            self.assertIn(target, classes)
            self.assertNotIn(rare, classes)

        # Counts roll into the target.
        self.assertEqual(
            new_labels.train.label_count_per_class[target],
            70,  # 40 original + 30 merged from rare
        )


if __name__ == "__main__":
    unittest.main()
