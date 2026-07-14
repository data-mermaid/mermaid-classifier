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

    def test_v1_label_mapper_rolls_and_filters_to_classes(self):
        raw = {"1": "ba1::", "2": "baX::gf", "3": None}  # coralnet "3" is unmappable
        rollup = {"ba1::": "ROLLED::", "baX::gf": "OUT::"}  # baX::gf rolls outside V1
        classes = {"ROLLED::"}
        m = cli.make_v1_label_mapper(lambda c: raw.get(c), lambda b: rollup.get(b, b), classes)
        self.assertEqual(m("1"), "ROLLED::")  # mapped, rolled, in V1 classes
        self.assertIsNone(m("2"))  # rolls to a non-class -> dropped (like training)
        self.assertIsNone(m("3"))  # unmappable coralnet id -> dropped

    def test_build_tasks_defaults_come_from_image_set(self):
        from mermaid_classifier.model_review.image_set import IMAGE_SET

        parser = cli.build_parser()
        args = parser.parse_args(["build-tasks"])  # no flags -> all defaults
        self.assertEqual(args.n_images, IMAGE_SET.n_images)
        self.assertEqual(args.seed, IMAGE_SET.seed)
        self.assertEqual(args.min_points, IMAGE_SET.min_points)
        self.assertEqual(args.classifier, IMAGE_SET.classifier)  # S3, not required
        self.assertEqual(args.name, IMAGE_SET.name)

    def test_image_set_from_args_roundtrips(self):
        from mermaid_classifier.model_review.image_set import IMAGE_SET

        parser = cli.build_parser()
        args = parser.parse_args(["build-tasks", "--n-images", "7", "--name", "custom"])
        image_set = cli.image_set_from_args(args)
        self.assertEqual(image_set.n_images, 7)
        self.assertEqual(image_set.name, "custom")
        self.assertEqual(image_set.classifier, IMAGE_SET.classifier)  # untouched default
        self.assertEqual(image_set.image_prefix, IMAGE_SET.image_prefix)
