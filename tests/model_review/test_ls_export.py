import unittest

from mermaid_classifier.model_review import ls_export

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
                    {"type": "keypointlabels", "value": {"keypointlabels": ["exp1::"]}},
                    {"type": "keypointlabels", "value": {"keypointlabels": ["Unlabeled"]}},
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
    def test_labels_zip_positionally_with_original_points(self):
        labels, _ = ls_export.parse_export(_EXPORT)
        by_rc = {(lbl.row, lbl.col): lbl.bagf for lbl in labels}
        self.assertEqual(by_rc[(100, 200)], "exp1::")
        self.assertEqual(by_rc[(300, 400)], "Unlabeled")  # kept, not dropped

    def test_expert_id_captured(self):
        labels, _ = ls_export.parse_export(_EXPORT)
        self.assertTrue(all(lbl.expert == "7" for lbl in labels))

    def test_notes_extracted(self):
        _, notes = ls_export.parse_export(_EXPORT)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].note, "GT looks wrong on point 2")
        self.assertEqual(notes[0].image_id, "A")

    def test_keypoint_count_mismatch_raises(self):
        export = [
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
                            {
                                "type": "keypointlabels",
                                "value": {"keypointlabels": ["exp1::"]},
                            },
                        ],
                    }
                ],
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            ls_export.parse_export(export)
        message = str(ctx.exception)
        self.assertIn("A", message)
        self.assertIn("7", message)
