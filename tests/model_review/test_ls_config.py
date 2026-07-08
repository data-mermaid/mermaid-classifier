import unittest
import xml.etree.ElementTree as ET

from mermaid_classifier.common.benthic_attributes import (
    BenthicAttributeLibrary,
    GrowthFormLibrary,
)
from mermaid_classifier.model_review import ls_config

# tiny hierarchy: root "Hard coral" -> child "Acropora"
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


class LsConfigTest(unittest.TestCase):
    def test_taxonomy_config_is_valid_xml_with_perregion(self):
        xml = ls_config.build_taxonomy_config(_fake_ba(), _fake_gf())
        root = ET.fromstring(xml)
        tax = root.find("Taxonomy")
        self.assertIsNotNone(tax)
        self.assertEqual(tax.get("perRegion"), "true")
        self.assertEqual(tax.get("pathSeparator"), "::")
        self.assertIsNotNone(root.find("KeyPoint"))
        self.assertIsNotNone(root.find("TextArea"))

    def test_taxonomy_nests_children_and_growth_forms(self):
        xml = ls_config.build_taxonomy_config(_fake_ba(), _fake_gf())
        root = ET.fromstring(xml)
        names = [c.get("value") for c in root.iter("Choice")]
        self.assertIn("Hard coral", names)
        self.assertIn("Acropora", names)
        self.assertIn("Branching", names)  # growth forms attached as leaves

    def test_flat_config_lists_labels(self):
        xml = ls_config.build_flat_config(["Hard coral", "Sand"])
        root = ET.fromstring(xml)
        labels = [c.get("value") for c in root.iter("Label")]
        self.assertEqual(labels, ["Hard coral", "Sand"])
        self.assertIsNotNone(root.find("TextArea"))
