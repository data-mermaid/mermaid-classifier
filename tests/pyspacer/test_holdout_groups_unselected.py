"""_holdout_groups_unselected across several coralnet datasets."""

import unittest
from io import StringIO

from segdata_spec.manifest import FeatureSetPin, Pin

from mermaid_classifier.pyspacer.dataset import _holdout_groups_unselected
from mermaid_classifier.pyspacer.label_specs import HoldoutGroups
from mermaid_classifier.pyspacer.options import DatasetOptions, PinnedDataset


def _coralnet_dataset(name: str) -> PinnedDataset:
    pin = Pin(name=name, version="20261124-5ab21b3", manifest_sha256="a" * 64)
    features = FeatureSetPin(dataset=pin, family="efficientnet-v1", manifest_sha256="d" * 64)
    return PinnedDataset(pin=pin, features=features, labels="coralnet")


class HoldoutGroupsUnselectedTest(unittest.TestCase):
    def test_a_group_is_unselected_only_when_no_coralnet_dataset_reads_it(self):
        options = DatasetOptions(datasets=[_coralnet_dataset("cn-a"), _coralnet_dataset("cn-b")])
        holdout = HoldoutGroups(StringIO("site,group_id,fold\ncoralnet,5,test\n"))
        cases = [
            ("one dataset reads every group", None, frozenset({"s1"}), {}),
            ("both selections leave s5 out", frozenset({"s1"}), frozenset({"s2"}), {"s5": "test"}),
            ("one selection reads s5", frozenset({"s5"}), frozenset({"s2"}), {}),
        ]
        for name, selection_a, selection_b, expected in cases:
            with self.subTest(name):
                groups = {"cn-a": selection_a, "cn-b": selection_b}
                self.assertEqual(_holdout_groups_unselected(holdout, options, groups), expected)


if __name__ == "__main__":
    unittest.main()
