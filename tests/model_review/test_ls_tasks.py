import unittest

from mermaid_classifier.model_review import ls_tasks
from mermaid_classifier.model_review.sample import ReviewPoint


def _label_path(bagf):  # test resolver: a single-element name path
    return [bagf]


def _toplevel(bagf):  # test resolver: top-level name = the BA id part
    return bagf.split("::")[0]


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

    def _task(self, label_path=_label_path):
        return ls_tasks.build_task("A", self.points, self.v1, label_path, _toplevel, _url, _size)

    def test_task_has_blank_and_reference_prediction_sets(self):
        versions = {p["model_version"] for p in self._task()["predictions"]}
        self.assertEqual(versions, {"blank", "ground-truth", "v1"})

    def test_blank_prediction_is_unlabelled_fixed_points(self):
        blank = next(p for p in self._task()["predictions"] if p["model_version"] == "blank")
        # one keypoint per point, all Unlabeled, and NO taxonomy (blind starting layer)
        self.assertEqual(len(blank["result"]), 2)  # 2 points
        self.assertTrue(all(r["type"] == "keypointlabels" for r in blank["result"]))
        self.assertTrue(all(r["value"]["keypointlabels"] == ["Unlabeled"] for r in blank["result"]))
        self.assertFalse(any(r["type"] == "taxonomy" for r in blank["result"]))
        # geometry preserved (pt-0 = (100,200))
        self.assertAlmostEqual(blank["result"][0]["value"]["x"], 200 / 1000 * 100)

    def test_each_point_has_toplevel_keypoint_and_taxonomy_sharing_id(self):
        gt = next(p for p in self._task()["predictions"] if p["model_version"] == "ground-truth")
        kps = [r for r in gt["result"] if r["type"] == "keypointlabels"]
        taxes = [r for r in gt["result"] if r["type"] == "taxonomy"]
        self.assertEqual(len(kps), 2)  # 2 points
        self.assertEqual(len(taxes), 2)
        # first point (100,200 after sort) shares region id pt-0 across both entries
        self.assertEqual(kps[0]["id"], "pt-0")
        self.assertEqual(taxes[0]["id"], "pt-0")
        self.assertEqual(kps[0]["from_name"], "toplevel")
        self.assertEqual(taxes[0]["from_name"], "label")

    def test_toplevel_keypointlabel_is_colored_category(self):
        gt = next(p for p in self._task()["predictions"] if p["model_version"] == "ground-truth")
        kp0 = next(r for r in gt["result"] if r["type"] == "keypointlabels")  # pt-0 -> gtA2
        self.assertEqual(kp0["value"]["keypointlabels"], ["gtA2"])  # top-level name

    def test_coordinates_on_keypoint_in_percent(self):
        gt = next(p for p in self._task()["predictions"] if p["model_version"] == "ground-truth")
        kp0 = next(r for r in gt["result"] if r["type"] == "keypointlabels")  # pt-0 = (100,200)
        self.assertAlmostEqual(kp0["value"]["x"], 200 / 1000 * 100)  # col/width
        self.assertAlmostEqual(kp0["value"]["y"], 100 / 500 * 100)  # row/height
        self.assertEqual(kp0["original_width"], 1000)
        self.assertEqual(kp0["original_height"], 500)

    def test_taxonomy_value_is_wrapped_path_from_resolver(self):
        task = self._task(label_path=lambda b: ["Root", b])
        v1 = next(p for p in task["predictions"] if p["model_version"] == "v1")
        tax0 = next(r for r in v1["result"] if r["type"] == "taxonomy")  # pt-0 -> v1A2
        self.assertEqual(tax0["value"]["taxonomy"], [["Root", "v1A2::"]])

    def test_original_points_preserved_in_pixels_with_both_labels(self):
        op = self._task()["data"]["original_points"]
        self.assertEqual(op[0], {"row": 100, "col": 200, "gt": "gtA2::", "v1": "v1A2::"})
