import unittest

from mermaid_classifier.common.benthic_attributes import (
    BenthicAttributeLibrary,
    GrowthFormLibrary,
)
from mermaid_classifier.model_review import cli

_BA = [
    {"id": "hc", "name": "Hard coral", "parent": None},
    {"id": "acr", "name": "Acropora", "parent": "hc"},
]


def _fake_ba():
    lib = BenthicAttributeLibrary.__new__(BenthicAttributeLibrary)
    lib.raw_results = list(_BA)
    lib.by_id = {r["id"]: r for r in _BA}
    lib.by_name = {r["name"]: r for r in _BA}
    lib.by_parent = {}
    for r in _BA:
        lib.by_parent.setdefault(r["parent"], []).append(r)
    return lib


def _fake_gf():
    lib = GrowthFormLibrary.__new__(GrowthFormLibrary)
    lib.by_id = {"br": "Branching"}
    return lib


class CliTest(unittest.TestCase):
    def test_label_name_builds_full_path(self):
        name = cli.make_label_name(_fake_ba(), _fake_gf())
        self.assertEqual(name("acr::br"), "Hard coral::Acropora::Branching")

    def test_label_name_ba_only(self):
        name = cli.make_label_name(_fake_ba(), _fake_gf())
        self.assertEqual(name("hc::"), "Hard coral")
