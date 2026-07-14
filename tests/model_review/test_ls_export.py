import unittest

from mermaid_classifier.model_review import ls_export


def _path_to_bagf(path):  # test resolver: last element is the "bagf"
    return path[-1]


def _kp(idx, toplevel="Hard coral"):
    # region is created by the colored top-level KeyPointLabels control
    return {
        "id": f"pt-{idx}",
        "type": "keypointlabels",
        "from_name": "toplevel",
        "value": {"x": 1.0, "y": 2.0, "keypointlabels": [toplevel]},
    }


def _tax(idx, path):
    return {"id": f"pt-{idx}", "type": "taxonomy", "value": {"taxonomy": [path]}}


_EXPORT = [
    {
        "data": {
            "image_id": "A",
            "original_points": [
                {"row": 100, "col": 200, "gt": "g1::", "v1": "v1::"},
                {"row": 300, "col": 400, "gt": "g2::", "v1": "v2::"},
            ],
        },
        "annotations": [
            {
                "completed_by": 7,
                "result": [
                    _kp(0),
                    _tax(0, ["Hard coral", "exp1::"]),  # pt-0 labeled
                    _kp(1),  # pt-1 keypoint present but NO taxonomy -> unlabeled
                    {
                        "type": "textarea",
                        "from_name": "notes",
                        "value": {"text": ["GT looks wrong on point 2"]},
                    },
                ],
            }
        ],
    }
]


class LsExportTest(unittest.TestCase):
    def test_labels_join_by_id_and_reconstruct_bagf(self):
        labels, _ = ls_export.parse_export(_EXPORT, _path_to_bagf)
        by_rc = {(lbl.row, lbl.col): lbl.bagf for lbl in labels}
        self.assertEqual(by_rc[(100, 200)], "exp1::")  # pt-0 via resolver
        self.assertEqual(by_rc[(300, 400)], ls_export.UNLABELED)  # pt-1 unlabeled, kept

    def test_every_original_point_yields_one_label(self):
        labels, _ = ls_export.parse_export(_EXPORT, _path_to_bagf)
        self.assertEqual(len(labels), 2)  # one per original point

    def test_deleted_keypoint_becomes_unlabeled_not_dropped(self):
        export = [
            {
                "data": {
                    "image_id": "A",
                    "original_points": [
                        {"row": 1, "col": 1, "gt": "g::", "v1": "v::"},
                        {"row": 2, "col": 2, "gt": "g::", "v1": "v::"},
                    ],
                },
                "annotations": [{"completed_by": 7, "result": [_kp(0), _tax(0, ["x::"])]}],
            }
        ]
        labels, _ = ls_export.parse_export(export, _path_to_bagf)
        by_rc = {(lbl.row, lbl.col): lbl.bagf for lbl in labels}
        self.assertEqual(by_rc[(1, 1)], "x::")
        self.assertEqual(by_rc[(2, 2)], ls_export.UNLABELED)

    def test_expert_id_captured(self):
        labels, _ = ls_export.parse_export(_EXPORT, _path_to_bagf)
        self.assertTrue(all(lbl.expert == "7" for lbl in labels))

    def test_expert_resolved_to_email_via_user_map(self):
        user_map = {"7": "alice@datamermaid.org"}
        labels, notes = ls_export.parse_export(_EXPORT, _path_to_bagf, user_map=user_map)
        self.assertTrue(all(lbl.expert == "alice@datamermaid.org" for lbl in labels))
        self.assertEqual(notes[0].expert, "alice@datamermaid.org")

    def test_expert_falls_back_to_id_when_unmapped(self):
        labels, _ = ls_export.parse_export(_EXPORT, _path_to_bagf, user_map={"999": "x@y"})
        self.assertTrue(all(lbl.expert == "7" for lbl in labels))

    def test_expert_from_embedded_completed_by_object(self):
        export = [
            {
                "data": {"image_id": "A", "original_points": [{"row": 1, "col": 1, "gt": "g::", "v1": "v::"}]},
                "annotations": [
                    {"completed_by": {"id": 7, "email": "bob@datamermaid.org"},
                     "result": [_kp(0), _tax(0, ["x::"])]}
                ],
            }
        ]
        labels, _ = ls_export.parse_export(export, _path_to_bagf)
        self.assertEqual(labels[0].expert, "bob@datamermaid.org")

    def test_notes_extracted(self):
        _, notes = ls_export.parse_export(_EXPORT, _path_to_bagf)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].note, "GT looks wrong on point 2")
        self.assertEqual(notes[0].image_id, "A")
