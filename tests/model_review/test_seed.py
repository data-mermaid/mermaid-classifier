import unittest

from mermaid_classifier.model_review import seed


class SeedTest(unittest.TestCase):
    def test_blank_keeps_keypoints_and_drops_taxonomy_labels(self):
        # a GT prediction result: paired keypoint + taxonomy per point
        gt_result = [
            {
                "id": "pt-0",
                "type": "keypoint",
                "from_name": "kp",
                "to_name": "image",
                "original_width": 1000,
                "original_height": 500,
                "value": {"x": 20.0, "y": 30.0, "width": 0.5},
            },
            {
                "id": "pt-0",
                "type": "taxonomy",
                "from_name": "label",
                "to_name": "image",
                "value": {"taxonomy": [["Hard coral"]]},
            },
        ]
        blank = seed.blank_annotation_result(gt_result)
        # only the keypoint geometry remains; no taxonomy label
        self.assertEqual(len(blank), 1)
        self.assertEqual(blank[0]["type"], "keypoint")
        self.assertEqual(blank[0]["id"], "pt-0")
        self.assertEqual(blank[0]["value"]["x"], 20.0)  # position preserved
        self.assertNotIn("taxonomy", blank[0]["value"])

    def test_blank_does_not_mutate_input(self):
        gt_result = [
            {"id": "pt-0", "type": "keypoint", "value": {"x": 1.0}},
            {"id": "pt-0", "type": "taxonomy", "value": {"taxonomy": [["X"]]}},
        ]
        seed.blank_annotation_result(gt_result)
        self.assertEqual(len(gt_result), 2)  # original untouched
