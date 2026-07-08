import unittest

from mermaid_classifier.model_review import seed


class SeedTest(unittest.TestCase):
    def test_blank_result_places_every_point_unlabeled(self):
        # a GT prediction result reused as the point skeleton
        gt_result = [
            {
                "id": "pt-0",
                "type": "keypointlabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": 1000,
                "original_height": 500,
                "value": {"x": 20.0, "y": 20.0, "width": 0.5, "keypointlabels": ["Hard coral"]},
            },
        ]
        blank = seed.blank_annotation_result(gt_result)
        self.assertEqual(blank[0]["value"]["keypointlabels"], ["Unlabeled"])
        self.assertEqual(blank[0]["value"]["x"], 20.0)  # position preserved
        self.assertEqual(blank[0]["id"], "pt-0")
