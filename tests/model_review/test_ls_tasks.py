import unittest

from mermaid_classifier.model_review import ls_tasks
from mermaid_classifier.model_review.sample import ReviewPoint


def _label_path(bagf):  # test resolver: a single-element name path
    return [bagf]


def _toplevel(bagf):  # test resolver: top-level name = the BA id part
    return bagf.split("::")[0]


def _url(point):
    return f"https://example/{point.source_id}/{point.image_id}.jpg"


def _size(point):
    return (1000, 500)  # width, height


class LsTasksTest(unittest.TestCase):
    def setUp(self):
        self.points = [
            ReviewPoint("coralnet", "A", "109", 250, 500, "gtA1::", "k", "b"),  # row=250,col=500
            ReviewPoint("coralnet", "A", "109", 100, 200, "gtA2::", "k", "b"),
        ]
        self.v1 = {("A", 250, 500): "v1A1::", ("A", 100, 200): "v1A2::"}
        self.beta = {("A", 250, 500): "btA1::", ("A", 100, 200): "btA2::"}

    def _task(self, label_path=_label_path):
        return ls_tasks.build_task(
            "A", self.points, self.v1, self.beta, label_path, _toplevel, _url, _size
        )

    def test_prediction_order_places_beta_after_ground_truth_in_the_ui(self):
        # Label Studio assigns prediction ids in array order and lists the tabs by
        # DESCENDING id, so this array is what puts Beta between ground-truth and the
        # blank starting layer on screen. Order, not just membership, is the contract.
        versions = [p["model_version"] for p in self._task()["predictions"]]
        self.assertEqual(
            versions,
            [ls_tasks.BLANK_MODEL_VERSION, ls_tasks.BETA_MODEL_VERSION, "ground-truth", "v1"],
        )

    def test_blank_prediction_is_unlabelled_fixed_points(self):
        blank = next(
            p
            for p in self._task()["predictions"]
            if p["model_version"] == ls_tasks.BLANK_MODEL_VERSION
        )
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

    def test_beta_points_pair_keypoint_and_taxonomy_sharing_id(self):
        beta = next(
            p
            for p in self._task()["predictions"]
            if p["model_version"] == ls_tasks.BETA_MODEL_VERSION
        )
        kps = [r for r in beta["result"] if r["type"] == "keypointlabels"]
        taxes = [r for r in beta["result"] if r["type"] == "taxonomy"]
        self.assertEqual(len(kps), 2)  # 2 points
        self.assertEqual(len(taxes), 2)
        self.assertEqual(kps[0]["id"], "pt-0")
        self.assertEqual(taxes[0]["id"], "pt-0")
        # pt-0 = (100,200) -> btA2::, rendered as its own top-level colour
        self.assertEqual(kps[0]["value"]["keypointlabels"], ["btA2"])
        self.assertEqual(taxes[0]["value"]["taxonomy"], [["btA2::"]])

    def test_original_points_preserved_in_pixels_with_all_reference_labels(self):
        op = self._task()["data"]["original_points"]
        self.assertEqual(
            op[0],
            {"row": 100, "col": 200, "gt": "gtA2::", "v1": "v1A2::", "beta": "btA2::"},
        )

    def test_build_tasks_stamps_image_set_into_data(self):
        tasks = ls_tasks.build_tasks(
            self.points,
            self.v1,
            self.beta,
            _label_path,
            _toplevel,
            _url,
            _size,
            image_set="reef-batch-2",
        )
        self.assertTrue(tasks)
        self.assertTrue(all(t["data"]["image_set"] == "reef-batch-2" for t in tasks))

    def test_build_tasks_image_set_defaults_empty(self):
        tasks = ls_tasks.build_tasks(
            self.points, self.v1, self.beta, _label_path, _toplevel, _url, _size
        )
        self.assertTrue(all(t["data"]["image_set"] == "" for t in tasks))


class SiteProvenanceTest(unittest.TestCase):
    def test_task_data_records_the_points_site(self):
        points = [ReviewPoint("mermaid", "uuid-1", "", 1, 2, "gt::", "k", "b")]
        task = ls_tasks.build_task(
            "uuid-1",
            points,
            {("uuid-1", 1, 2): "v1::"},
            {("uuid-1", 1, 2): "beta::"},
            _label_path,
            _toplevel,
            _url,
            _size,
        )
        self.assertEqual(task["data"]["source"], "mermaid")
        self.assertEqual(task["data"]["source_id"], "")  # MERMAID has no source

    def test_image_id_shared_across_sites_is_rejected(self):
        points = [
            ReviewPoint("coralnet", "dup", "109", 1, 2, "gt::", "k", "b"),
            ReviewPoint("mermaid", "dup", "", 3, 4, "gt::", "k", "b"),
        ]
        v1 = {("dup", 1, 2): "v1::", ("dup", 3, 4): "v1::"}
        beta = {("dup", 1, 2): "beta::", ("dup", 3, 4): "beta::"}
        with self.assertRaises(ValueError):
            ls_tasks.build_tasks(points, v1, beta, _label_path, _toplevel, _url, _size)
