"""Tests for scripts/compare_holdout_runs.py on the offline CSV path."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import pandas as pd
from support.paths import add_scripts_to_path

add_scripts_to_path()

import compare_holdout_runs as chr_  # noqa: E402

HEADER = "row,col,image_id,site,bucket,project_id,feature_vector,benthic_attribute_id,growth_form_id,pred_bagf,in_model_classes\n"
POINTS = [
    ("1", "i1", "coralnet", "1", "a"),
    ("2", "i1", "coralnet", "1", "a"),
    ("3", "i2", "coralnet", "2", "b"),
    ("4", "i3", "mermaid", "9", "c"),
]


def _holdout_csv(preds: list[str]) -> str:
    lines = [
        f'{row},1,{img},{site},bk,{proj},{proj}/{img}.fv,{gt},"",{pred}::,true\n'
        for (row, img, site, proj, gt), pred in zip(POINTS, preds, strict=True)
    ]
    return HEADER + "".join(lines)


ROLLUP = textwrap.dedent("""\
    from_ba_id,from_ba_name,from_gf_id,from_gf_name,to_ba_id,to_ba_name,to_gf_id,to_gf_name
    a,A,,,T,Top,,
    b,B,,,T,Top,,
    c,C,,,U,Other,,
""")
TOP_LEVEL = "id,name\nT,Top level T\nU,Top level U\n"
COUNTS = textwrap.dedent("""\
    benthic_attribute_id,growth_form_id,available,realized,ba_name,gf_name
    a,"",20,2,A,""
    b,"",1,1,B,""
    c,"",1,1,C,""
""")


class OfflineCompareTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        files = {
            "a.csv": _holdout_csv(["a", "b", "b", "c"]),
            "b.csv": _holdout_csv(["a", "a", "b", "a"]),
            "rollup.csv": ROLLUP,
            "counts.csv": COUNTS,
            "top_level.csv": TOP_LEVEL,
        }
        for name, text in files.items():
            (self.dir / name).write_text(text)

    def _args(self, *extra: str) -> list[str]:
        d = self.dir
        return [
            "--csv-a",
            str(d / "a.csv"),
            "--csv-b",
            str(d / "b.csv"),
            "--out-dir",
            str(d / "out"),
            "--reps",
            "50",
            *extra,
        ]

    def test_writes_class_and_top_level_comparison(self):
        chr_.main(
            self._args(
                "--top-level-rollup-csv",
                str(self.dir / "rollup.csv"),
                "--top-level-classes-csv",
                str(self.dir / "top_level.csv"),
                "--real-mix",
                "--class-counts-csv",
                str(self.dir / "counts.csv"),
            )
        )
        out = self.dir / "out"
        metrics = pd.read_csv(out / "compare_metrics.csv")
        self.assertEqual(set(metrics["level"]), {"class", "top_level"})
        self.assertEqual(set(metrics["scope"]), {"all", "coralnet", "mermaid"})
        recall = metrics[
            (metrics["scope"] == "all")
            & (metrics["level"] == "class")
            & (metrics["metric"] == "recall")
        ].set_index("class_key")
        # A misses one of the two a-points; B gets both but misses the c-point.
        self.assertAlmostEqual(recall.loc["a::", "diff"], 0.5)
        self.assertAlmostEqual(recall.loc["c::", "diff"], -1.0)
        self.assertIn("aligned: 4", (out / "point_set_check.txt").read_text())
        self.assertEqual(set(metrics["weighting"]), {"real_mix"})
        self.assertIn("weighting: real_mix", (out / "point_set_check.txt").read_text())
        md = (out / "compare_metrics.md").read_text()
        self.assertIn("Weighting: real_mix", md)
        self.assertIn("## all (top_level)", md)
        self.assertIn("| Top level T |", md)
        self.assertIn("Top level U", set(metrics["class_name"]))

    def test_class_counts_without_real_mix_is_rejected(self):
        with self.assertRaises(SystemExit):
            chr_.main(self._args("--class-counts-csv", str(self.dir / "counts.csv")))
        self.assertFalse((self.dir / "out").exists())

    def test_unweighted_run_records_capped_weighting(self):
        chr_.main(self._args())
        out = self.dir / "out"
        metrics = pd.read_csv(out / "compare_metrics.csv")
        self.assertEqual(set(metrics["weighting"]), {"capped"})
        self.assertIn("weighting: capped", (out / "point_set_check.txt").read_text())
        self.assertIn("Weighting: capped", (out / "compare_metrics.md").read_text())

    def test_mismatched_point_sets_stop_the_comparison(self):
        (self.dir / "b.csv").write_text(_holdout_csv(["a", "a", "b", "a"]).rsplit("\n", 2)[0])
        with self.assertRaisesRegex(SystemExit, "only in A"):
            chr_.main(self._args())


if __name__ == "__main__":
    unittest.main()
