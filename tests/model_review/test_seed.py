import unittest

from mermaid_classifier.model_review import seed
from mermaid_classifier.model_review.ls_config import UNLABELED_TOPLEVEL


def _gt_result():
    # a GT prediction result: paired top-level keypointlabels + taxonomy per point
    return [
        {
            "id": "pt-0",
            "type": "keypointlabels",
            "from_name": "toplevel",
            "to_name": "image",
            "original_width": 1000,
            "original_height": 500,
            "value": {"x": 20.0, "y": 30.0, "width": 0.5, "keypointlabels": ["Hard coral"]},
        },
        {
            "id": "pt-0",
            "type": "taxonomy",
            "from_name": "label",
            "to_name": "image",
            "value": {"taxonomy": [["Hard coral"]]},
        },
    ]


class SeedTest(unittest.TestCase):
    def test_blank_keeps_keypoints_grey_and_drops_taxonomy(self):
        blank = seed.blank_annotation_result(_gt_result())
        self.assertEqual(len(blank), 1)  # only the keypoint region
        self.assertEqual(blank[0]["type"], "keypointlabels")
        self.assertEqual(blank[0]["id"], "pt-0")
        self.assertEqual(blank[0]["value"]["x"], 20.0)  # position preserved
        self.assertEqual(blank[0]["value"]["keypointlabels"], [UNLABELED_TOPLEVEL])

    def test_blank_does_not_mutate_input(self):
        ref = _gt_result()
        seed.blank_annotation_result(ref)
        self.assertEqual(ref[0]["value"]["keypointlabels"], ["Hard coral"])  # untouched
        self.assertEqual(len(ref), 2)
