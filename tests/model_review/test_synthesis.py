import unittest

import pandas as pd

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

    def test_all_expert_top_none_raises(self):
        # Simulates the name<->id rollup collapse: every expert label rolls to
        # None, so all three comparisons involving experts would silently
        # become NaN. This must raise instead of silently collapsing.
        records = [
            {
                "image_id": "A",
                "row": 1,
                "col": 1,
                "gt_top": "hc",
                "v1_top": "hc",
                "expert": "e1",
                "expert_top": None,
            },
            {
                "image_id": "A",
                "row": 2,
                "col": 2,
                "gt_top": "hc",
                "v1_top": "sand",
                "expert": "e2",
                "expert_top": None,
            },
        ]
        df = pd.DataFrame.from_records(records)
        with self.assertRaises(ValueError):
            synthesis.agreement_summary(df)

    def test_worked_example_does_not_raise(self):
        # The existing worked-example fixture has non-None expert_top values,
        # so the total-collapse guard must not trip on legitimate data.
        df = synthesis.build_point_table(_TASKS, _EXPERTS, _roll)
        synthesis.agreement_summary(df)  # should not raise

    def test_none_expert_top_excluded_from_match_rates(self):
        # An expert label that rolls to None (e.g. "Unlabeled") must be
        # excluded from the match-rate denominators entirely -- it should not
        # count as agreement nor disagreement. At least one expert_top stays
        # non-None so Fix 1's total-collapse guard does not trip.
        records = [
            {
                "image_id": "A",
                "row": 1,
                "col": 1,
                "gt_top": "hc",
                "v1_top": "hc",
                "expert": "e1",
                "expert_top": "hc",
            },
            {
                "image_id": "A",
                "row": 1,
                "col": 1,
                "gt_top": "hc",
                "v1_top": "hc",
                "expert": "e2",
                "expert_top": "sand",
            },
            {
                "image_id": "A",
                "row": 2,
                "col": 2,
                "gt_top": "hc",
                "v1_top": "hc",
                "expert": "e1",
                "expert_top": "hc",
            },
            {
                "image_id": "A",
                "row": 2,
                "col": 2,
                "gt_top": "hc",
                "v1_top": "hc",
                "expert": "e3",
                "expert_top": None,  # e.g. rolled from bagf "Unlabeled"
            },
        ]
        df = pd.DataFrame.from_records(records)
        summary = synthesis.agreement_summary(df)
        # expert_vs_gt: 3 valid (expert_top, gt_top) pairs (e3's None excluded),
        # 2 match ((A,1,1)/e1 and (A,2,2)/e1) -> 2/3, not 2/4.
        self.assertAlmostEqual(summary["expert_vs_gt"], 2 / 3)
        # expert_vs_expert: point (A,1,1) pair (e1,e2) -> disagree, counted.
        # point (A,2,2) pair (e1,e3) -> e3 is None, pair skipped entirely.
        # So total=1, agree=0 -> 0.0, not nan and not 0.5.
        self.assertAlmostEqual(summary["expert_vs_expert"], 0.0)
