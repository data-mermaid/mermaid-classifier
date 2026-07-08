import unittest

from mermaid_classifier.model_review import synthesis
from mermaid_classifier.model_review.ls_export import ExpertLabel


# roll: strip to BA id (identity-ish for test); "Unlabeled" -> None
def _roll(bagf):
    if bagf == "Unlabeled":
        return None
    return bagf.split("::")[0]


_TASKS = [
    {
        "data": {
            "image_id": "A",
            "original_points": [
                {"row": 1, "col": 1, "gt": "hc::", "v1": "hc::"},  # v1 matches gt
                {"row": 2, "col": 2, "gt": "hc::", "v1": "sand::"},  # v1 differs
            ],
        }
    },
]
_EXPERTS = [
    ExpertLabel("A", "e1", 1, 1, "hc::"),
    ExpertLabel("A", "e1", 2, 2, "hc::"),  # expert agrees with gt
    ExpertLabel("A", "e2", 1, 1, "hc::"),
    ExpertLabel("A", "e2", 2, 2, "sand::"),  # expert disagrees with e1
]


class SynthesisTest(unittest.TestCase):
    def test_point_table_rolls_up(self):
        df = synthesis.build_point_table(_TASKS, _EXPERTS, _roll)
        row = df[(df.image_id == "A") & (df.row == 2) & (df.expert == "e1")].iloc[0]
        self.assertEqual(row.gt_top, "hc")
        self.assertEqual(row.v1_top, "sand")
        self.assertEqual(row.expert_top, "hc")

    def test_v1_vs_gt_match_rate(self):
        df = synthesis.build_point_table(_TASKS, _EXPERTS, _roll)
        summary = synthesis.agreement_summary(df)
        self.assertAlmostEqual(summary["v1_vs_gt"], 0.5)  # 1 of 2 points match

    def test_expert_vs_gt_match_rate(self):
        df = synthesis.build_point_table(_TASKS, _EXPERTS, _roll)
        summary = synthesis.agreement_summary(df)
        # e1: 2/2 match gt; e2: 1/2 match gt -> 3/4
        self.assertAlmostEqual(summary["expert_vs_gt"], 0.75)

    def test_expert_vs_expert_agreement(self):
        df = synthesis.build_point_table(_TASKS, _EXPERTS, _roll)
        summary = synthesis.agreement_summary(df)
        # 2 shared points; agree on point1, disagree on point2 -> 0.5
        self.assertAlmostEqual(summary["expert_vs_expert"], 0.5)
