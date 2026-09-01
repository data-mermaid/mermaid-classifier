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
                # v1 matches gt here, Beta does not
                {"row": 1, "col": 1, "gt": "hc::", "v1": "hc::", "beta": "sand::"},
                # neither matches gt
                {"row": 2, "col": 2, "gt": "hc::", "v1": "sand::", "beta": "sand::"},
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
        self.assertEqual(row.beta_top, "sand")
        self.assertEqual(row.expert_top, "hc")

    def test_v1_vs_gt_match_rate(self):
        df = synthesis.build_point_table(_TASKS, _EXPERTS, _roll)
        summary = synthesis.agreement_summary(df, _TASKS, _roll)
        self.assertAlmostEqual(summary["v1_vs_gt"], 0.5)  # 1 of 2 points match

    def test_v1_vs_gt_covers_points_no_expert_reviewed(self):
        # Experts labeled only point (1,1); v1_vs_gt must still cover BOTH points
        # (it comes from the tasks, not the expert df).
        experts = [ExpertLabel("A", "e1", 1, 1, "hc::")]
        df = synthesis.build_point_table(_TASKS, experts, _roll)
        summary = synthesis.agreement_summary(df, _TASKS, _roll)
        self.assertAlmostEqual(summary["v1_vs_gt"], 0.5)  # 1 of 2, not 1 of 1

    def test_beta_vs_gt_is_scored_separately_from_v1(self):
        df = synthesis.build_point_table(_TASKS, _EXPERTS, _roll)
        summary = synthesis.agreement_summary(df, _TASKS, _roll)
        self.assertAlmostEqual(summary["v1_vs_gt"], 0.5)  # 1 of 2 points match
        self.assertAlmostEqual(summary["beta_vs_gt"], 0.0)  # neither point matches

    def test_beta_vs_gt_covers_points_no_expert_reviewed(self):
        experts = [e for e in _EXPERTS if (e.row, e.col) == (1, 1)]
        df = synthesis.build_point_table(_TASKS, experts, _roll)
        summary = synthesis.agreement_summary(df, _TASKS, _roll)
        self.assertAlmostEqual(summary["beta_vs_gt"], 0.0)  # 0 of 2, not 0 of 1

    def test_tasks_without_a_beta_layer_still_synthesize(self):
        # A tasks file with no Beta reference layer stays synthesizable.
        tasks = [
            {
                "data": {
                    "image_id": "A",
                    "original_points": [{"row": 1, "col": 1, "gt": "hc::", "v1": "hc::"}],
                }
            }
        ]
        df = synthesis.build_point_table(tasks, _EXPERTS, _roll)
        self.assertTrue(df["beta_top"].isna().all())
        summary = synthesis.agreement_summary(df, tasks, _roll)
        self.assertTrue(pd.isna(summary["beta_vs_gt"]))

    def test_v1_vs_gt_rate_dedupes_and_uses_all_tasks(self):
        # standalone helper, no experts involved at all
        self.assertAlmostEqual(synthesis.v1_vs_gt_rate(_TASKS, _roll), 0.5)

    def test_expert_vs_gt_match_rate(self):
        df = synthesis.build_point_table(_TASKS, _EXPERTS, _roll)
        summary = synthesis.agreement_summary(df, _TASKS, _roll)
        # e1: 2/2 match gt; e2: 1/2 match gt -> 3/4
        self.assertAlmostEqual(summary["expert_vs_gt"], 0.75)

    def test_expert_vs_expert_agreement(self):
        df = synthesis.build_point_table(_TASKS, _EXPERTS, _roll)
        summary = synthesis.agreement_summary(df, _TASKS, _roll)
        # 2 shared points; agree on point1, disagree on point2 -> 0.5
        self.assertAlmostEqual(summary["expert_vs_expert"], 0.5)

    def test_all_expert_top_none_raises(self):
        # Simulates the name<->id rollup collapse: every expert label rolls to
        # None, so the expert comparisons would silently become NaN. Must raise.
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
            synthesis.agreement_summary(df, _TASKS, _roll)

    def test_worked_example_does_not_raise(self):
        df = synthesis.build_point_table(_TASKS, _EXPERTS, _roll)
        synthesis.agreement_summary(df, _TASKS, _roll)  # should not raise

    def test_none_expert_top_excluded_from_match_rates(self):
        # An expert label that rolls to None must be excluded from denominators.
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
                "expert_top": None,
            },  # e.g. rolled from "Unlabeled"
        ]
        df = pd.DataFrame.from_records(records)
        summary = synthesis.agreement_summary(df, _TASKS, _roll)
        # expert_vs_gt: 3 valid pairs (e3's None excluded), 2 match -> 2/3
        self.assertAlmostEqual(summary["expert_vs_gt"], 2 / 3)
        # expert_vs_expert: (A,1,1) pair (e1,e2) disagree; (A,2,2) pair has e3=None
        # -> skipped. total=1, agree=0 -> 0.0.
        self.assertAlmostEqual(summary["expert_vs_expert"], 0.0)

    def test_point_table_carries_image_provenance(self):
        tasks = [
            {
                "data": {
                    "image_id": "A",
                    "image_set": "reef-batch-2",
                    "source": "coralnet",
                    "source_id": "109",
                    "image_url": "s3://bucket/coralnet-public-images/s109/images/A.jpg",
                    "original_points": [
                        {"row": 1, "col": 1, "gt": "hc::", "v1": "hc::", "beta": "hc::"}
                    ],
                }
            }
        ]
        experts = [ExpertLabel("A", "e1", 1, 1, "hc::")]
        df = synthesis.build_point_table(tasks, experts, _roll)
        row = df.iloc[0]
        self.assertEqual(row.image_set, "reef-batch-2")
        self.assertEqual(row.site, "coralnet")
        self.assertEqual(row.source_id, "109")
        self.assertEqual(row.image_url, "s3://bucket/coralnet-public-images/s109/images/A.jpg")
        self.assertEqual(
            list(df.columns),
            [
                "image_set",
                "site",
                "image_id",
                "source_id",
                "image_url",
                "row",
                "col",
                "gt_top",
                "v1_top",
                "beta_top",
                "expert",
                "expert_top",
            ],
        )

    def test_point_table_defaults_provenance_when_absent(self):
        # Existing-style tasks without provenance keys must still work (empty strings).
        df = synthesis.build_point_table(_TASKS, _EXPERTS, _roll)
        self.assertTrue((df.image_set == "").all())
        self.assertTrue((df.site == "").all())
        self.assertTrue((df.source_id == "").all())
        self.assertTrue((df.image_url == "").all())


def _sited_task(image_id, site, gt, v1, beta=None):
    return {
        "data": {
            "image_id": image_id,
            "source": site,
            "original_points": [
                {"row": 1, "col": 1, "gt": gt, "v1": v1, "beta": beta if beta else v1}
            ],
        }
    }


_MIXED_TASKS = [
    _sited_task("A", "coralnet", "hc::", "hc::"),  # v1 matches gt
    _sited_task("M", "mermaid", "hc::", "sand::"),  # v1 differs
]


class BySiteTest(unittest.TestCase):
    def test_rates_are_computed_per_site(self):
        experts = [ExpertLabel("A", "e1", 1, 1, "hc::"), ExpertLabel("M", "e1", 1, 1, "sand::")]
        df = synthesis.build_point_table(_MIXED_TASKS, experts, _roll)
        by_site = synthesis.agreement_by_site(df, _MIXED_TASKS, _roll)
        self.assertEqual(sorted(by_site), ["coralnet", "mermaid"])
        self.assertAlmostEqual(by_site["coralnet"]["v1_vs_gt"], 1.0)
        self.assertAlmostEqual(by_site["mermaid"]["v1_vs_gt"], 0.0)
        self.assertAlmostEqual(by_site["coralnet"]["expert_vs_gt"], 1.0)  # hc vs hc
        self.assertAlmostEqual(by_site["mermaid"]["expert_vs_gt"], 0.0)  # sand vs hc

    def test_a_site_with_no_expert_labels_yields_nan_without_raising(self):
        experts = [ExpertLabel("A", "e1", 1, 1, "hc::")]  # nothing for the mermaid image
        df = synthesis.build_point_table(_MIXED_TASKS, experts, _roll)
        by_site = synthesis.agreement_by_site(df, _MIXED_TASKS, _roll)
        self.assertNotEqual(by_site["mermaid"]["expert_vs_gt"], by_site["mermaid"]["expert_vs_gt"])
        self.assertAlmostEqual(by_site["mermaid"]["v1_vs_gt"], 0.0)  # GT/v1 need no expert

    def test_overall_summary_still_guards_against_nothing_rolling_up(self):
        experts = [ExpertLabel("A", "e1", 1, 1, "Unlabeled")]
        df = synthesis.build_point_table(_MIXED_TASKS, experts, _roll)
        with self.assertRaises(ValueError):
            synthesis.agreement_summary(df, _MIXED_TASKS, _roll)

    def test_image_counts_by_site(self):
        self.assertEqual(
            synthesis.image_counts_by_site(_MIXED_TASKS), {"coralnet": 1, "mermaid": 1}
        )
