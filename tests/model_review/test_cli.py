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
    def test_label_path_builds_full_path_list(self):
        path = cli.make_label_path(_fake_ba(), _fake_gf())
        self.assertEqual(path("acr::br"), ["Hard coral", "Acropora", "Branching"])

    def test_label_path_ba_only(self):
        path = cli.make_label_path(_fake_ba(), _fake_gf())
        self.assertEqual(path("hc::"), ["Hard coral"])

    def test_path_to_bagf_roundtrips_with_growth_form(self):
        to_bagf = cli.make_path_to_bagf(_fake_ba(), _fake_gf())
        self.assertEqual(to_bagf(["Hard coral", "Acropora", "Branching"]), "acr::br")

    def test_path_to_bagf_ba_only(self):
        to_bagf = cli.make_path_to_bagf(_fake_ba(), _fake_gf())
        self.assertEqual(to_bagf(["Hard coral"]), "hc::")

    def test_label_path_and_inverse_roundtrip(self):
        path = cli.make_label_path(_fake_ba(), _fake_gf())
        to_bagf = cli.make_path_to_bagf(_fake_ba(), _fake_gf())
        for bagf in ("acr::br", "hc::", "acr::"):
            self.assertEqual(to_bagf(path(bagf)), bagf)
