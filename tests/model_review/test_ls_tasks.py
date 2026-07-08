import unittest

from mermaid_classifier.model_review import ls_tasks
from mermaid_classifier.model_review.sample import ReviewPoint


def _label_name(bagf):  # identity for test
    return bagf


def _url(source_id, image_id):
    return f"https://example/{source_id}/{image_id}.jpg"


def _size(source_id, image_id):
    return (1000, 500)  # width, height


class LsTasksTest(unittest.TestCase):
    def setUp(self):
        self.points = [
            ReviewPoint("A", "109", 250, 500, "gtA1::", "k", "b"),  # row=250,col=500
            ReviewPoint("A", "109", 100, 200, "gtA2::", "k", "b"),
        ]
        self.v1 = {("A", 250, 500): "v1A1::", ("A", 100, 200): "v1A2::"}

    def test_task_has_two_prediction_sets(self):
        task = ls_tasks.build_task("A", self.points, self.v1, _label_name, _url, _size)
        versions = {p["model_version"] for p in task["predictions"]}
        self.assertEqual(versions, {"ground-truth", "v1"})

    def test_coordinates_converted_to_percent(self):
        task = ls_tasks.build_task("A", self.points, self.v1, _label_name, _url, _size)
        gt = next(p for p in task["predictions"] if p["model_version"] == "ground-truth")
        first = gt["result"][0]  # points sorted by (row,col): (100,200) first
        self.assertAlmostEqual(first["value"]["x"], 200 / 1000 * 100)  # col/width
        self.assertAlmostEqual(first["value"]["y"], 100 / 500 * 100)  # row/height
        self.assertEqual(first["original_width"], 1000)
        self.assertEqual(first["original_height"], 500)

    def test_original_points_preserved_in_pixels(self):
        task = ls_tasks.build_task("A", self.points, self.v1, _label_name, _url, _size)
        op = task["data"]["original_points"]
        self.assertEqual(op[0], {"row": 100, "col": 200, "gt": "gtA2::", "v1": "v1A2::"})

    def test_gt_and_v1_labels_come_from_resolver(self):
        task = ls_tasks.build_task("A", self.points, self.v1, str.upper, _url, _size)
        v1 = next(p for p in task["predictions"] if p["model_version"] == "v1")
        self.assertEqual(v1["result"][0]["value"]["keypointlabels"], ["V1A2::"])
