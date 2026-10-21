"""Tests for scripts/select_val_groups.py, run offline through --counts-csv."""

from __future__ import annotations

import csv
import tempfile
import textwrap
import unittest
from pathlib import Path

import pandas as pd
from support.paths import add_scripts_to_path

add_scripts_to_path()

import select_val_groups as svg  # noqa: E402

CONFIG_YAML = textwrap.dedent("""
    dataset:
      include_mermaid: false
      coralnet_manifest_uri: s3://bucket/manifest.parquet
      feature_extractor_weights: s3://bucket/efficientnet.pt
      holdout_groups_csv: holdout_groups.csv
      holdout_fold: test
      val_groups_csv: val_groups.csv
      split_mode: images
      ref_val_ratios: [0.1, 0.0]
    training:
      epochs: 5
    mlflow:
      experiment_name: my-experiment
      model_name: my-model
""").lstrip()

TEST_IDS = ["3", "7", "12"]


def _write_counts(path: Path) -> None:
    rows = []
    for i in range(1, 31):
        rows.append(("a", "coralnet", str(i), 40))
        rows.append(("b", "coralnet", str(i), 40))
    rows.append(("rare", "coralnet", "5", 100))
    pd.DataFrame(rows, columns=["class_key", "site", "project_id", "n"]).to_csv(path, index=False)


class SelectValGroupsScriptTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = self.root / "cfg"
        self.cfg.mkdir()
        (self.cfg / "training_config.yaml").write_text(CONFIG_YAML)
        lines = ["site,group_id,fold"]
        lines += [f"coralnet,{i},test" for i in TEST_IDS]
        lines += [f"coralnet,{i},0" for i in ("1", "2", "4")]
        self.holdout = self.cfg / "holdout_groups.csv"
        self.holdout.write_text("\n".join(lines) + "\n")
        self.counts = self.root / "counts.csv"
        _write_counts(self.counts)

    def _run(self, *extra: str) -> None:
        # 20% point cap: three 80-point groups fit under it.
        args = ["--config-dir", str(self.cfg), "--counts-csv", str(self.counts)]
        svg.main([*args, "--max-val-point-frac", "0.2", *extra])

    def _read_out(self) -> list[tuple[str, str]]:
        with (self.cfg / "val_groups.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ["site", "group_id"])
            return [(r["site"], r["group_id"]) for r in reader]

    def test_writes_sorted_val_csv_at_the_configs_val_groups_path(self):
        self._run()
        rows = self._read_out()
        # 27 non-test sources at 10% rounds to 3 groups.
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows, sorted(rows))

    def test_a_test_group_never_appears_in_the_output(self):
        self._run()
        self.assertFalse({gid for _, gid in self._read_out()} & set(TEST_IDS))

    def test_holdout_groups_csv_is_byte_identical_after_the_run(self):
        before = self.holdout.read_bytes()
        self._run()
        self.assertEqual(self.holdout.read_bytes(), before)

    def test_out_equal_to_the_holdout_file_is_refused(self):
        before = self.holdout.read_bytes()
        with self.assertRaises(SystemExit):
            self._run("--out", str(self.holdout))
        self.assertEqual(self.holdout.read_bytes(), before)

    def test_coverage_report_has_val_columns_and_the_classes_kept_groups(self):
        self._run()
        report = pd.read_csv(self.cfg / "val_groups_coverage.csv").set_index("class_key")
        self.assertIn("val_points", report.columns)
        self.assertNotIn("test_points", report.columns)
        val_groups = int(report.loc["a", "val_groups"])
        self.assertEqual(val_groups, 3)
        # Class 'a' spans 27 non-test groups; none of the three val groups trains.
        self.assertEqual(int(report.loc["a", "train_groups"]), 24)
        # 'rare' lives in one group, which never goes to val.
        self.assertEqual(int(report.loc["rare", "val_groups"]), 0)

    def test_short_selection_exits_without_writing_unless_allowed(self):
        with self.assertRaises(SystemExit):
            self._run("--max-val-point-frac", "0.05")
        self.assertFalse((self.cfg / "val_groups.csv").exists())
        self._run("--max-val-point-frac", "0.05", "--allow-short-val")
        self.assertEqual(len(self._read_out()), 1)

    def test_out_of_range_parameters_are_rejected(self):
        for flag, value in (
            ("--max-class-group-loss", "1.0"),
            ("--max-class-group-loss", "-0.1"),
            ("--val-frac", "1.5"),
            ("--max-val-point-frac", "-0.2"),
        ):
            with self.subTest(flag=flag, value=value), self.assertRaises(SystemExit):
                svg.parse_args(["--config-dir", "cfg", flag, value])

    def test_counts_csv_with_a_class_outside_included_labels_is_refused(self):
        (self.cfg / "included.csv").write_text("ba_id,gf_id\nzzz,\n")
        yaml_text = CONFIG_YAML.replace(
            "  holdout_fold: test\n",
            "  holdout_fold: test\n  included_labels_csv: included.csv\n",
        )
        (self.cfg / "training_config.yaml").write_text(yaml_text)
        with self.assertRaisesRegex(SystemExit, "included_labels_csv"):
            self._run()
        self.assertFalse((self.cfg / "val_groups.csv").exists())

    def test_out_named_holdout_groups_csv_elsewhere_is_refused(self):
        other = self.root / "other"
        other.mkdir()
        with self.assertRaises(SystemExit):
            self._run("--out", str(other / "holdout_groups.csv"))
        self.assertFalse((other / "holdout_groups.csv").exists())

    def test_out_over_an_existing_fold_file_is_refused(self):
        target = self.root / "mine.csv"
        target.write_text("site,group_id,fold\ncoralnet,1,0\n")
        before = target.read_bytes()
        with self.assertRaises(SystemExit):
            self._run("--out", str(target))
        self.assertEqual(target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
