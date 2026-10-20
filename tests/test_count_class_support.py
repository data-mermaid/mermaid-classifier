"""Tests for scripts/count_class_support.py: CLI parsing, CSV writing and the dataset flow."""

from __future__ import annotations

import csv
import dataclasses
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import duckdb
from support.paths import add_scripts_to_path

add_scripts_to_path()

import count_class_support as ccs  # noqa: E402


@dataclasses.dataclass(frozen=True)
class _Options:
    holdout_groups_csv: str | None
    holdout_fold: int | str


def _dataset():
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE annotations (benthic_attribute_id VARCHAR, growth_form_id VARCHAR,"
        " site VARCHAR, project_id VARCHAR)"
    )
    conn.executemany(
        "INSERT INTO annotations VALUES (?, ?, ?, ?)",
        [
            ("ba1", "", "coralnet", "10"),
            ("ba1", "", "coralnet", "10"),
            ("ba1", "", "coralnet", "11"),
            ("ba2", "gf", "coralnet", "10"),
        ],
    )
    return SimpleNamespace(duck_conn=conn, cleanup=mock.Mock())


class ParseArgsTest(unittest.TestCase):
    def test_requires_config_dir_and_out(self):
        with self.assertRaises(SystemExit):
            ccs.parse_args(["--config-dir", "cfg"])
        with self.assertRaises(SystemExit):
            ccs.parse_args(["--out", "x.csv"])

    def test_reads_both_arguments(self):
        args = ccs.parse_args(["--config-dir", "cfg", "--out", "x.csv"])
        self.assertEqual((args.config_dir, args.out), ("cfg", "x.csv"))


class MainTest(unittest.TestCase):
    def _run(self, td: str):
        dataset = _dataset()
        options = _Options(holdout_groups_csv="old.csv", holdout_fold=0)
        config = mock.Mock()
        config.build_options.return_value = (options, None, None)
        dataset_cls = mock.Mock()
        dataset_cls.filtered_annotations_only.return_value = dataset
        out = Path(td) / "support.csv"
        with (
            mock.patch.object(ccs, "_resolve_local_aws_credentials"),
            mock.patch.object(ccs, "load_config_then_pyspacer", return_value=(config, dataset_cls)),
        ):
            ccs.main(["--config-dir", td, "--out", str(out)])
        return out, dataset, dataset_cls

    def test_writes_counts_csv_with_empty_gf_keys_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as td:
            out, dataset, _ = self._run(td)
            with out.open(newline="") as f:
                rows = list(csv.reader(f))
        self.assertEqual(
            rows,
            [
                ["class_key", "site", "project_id", "n"],
                ["ba1::", "coralnet", "10", "2"],
                ["ba1::", "coralnet", "11", "1"],
                ["ba2::gf", "coralnet", "10", "1"],
            ],
        )
        dataset.cleanup.assert_called_once()

    def test_dataset_is_built_with_holdout_tagging_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, dataset_cls = self._run(td)
        (options,), _ = dataset_cls.filtered_annotations_only.call_args
        self.assertIsNone(options.holdout_groups_csv)
        self.assertEqual(options.holdout_fold, "test")


class SummaryTest(unittest.TestCase):
    def test_reports_points_and_distinct_groups_per_class(self):
        import pandas as pd

        counts = pd.DataFrame(
            {
                "class_key": ["a::", "a::", "b::"],
                "site": ["coralnet", "coralnet", "mermaid"],
                "project_id": ["1", "2", "1"],
                "n": [5, 7, 3],
            }
        )
        summary = ccs.class_summary(counts)
        self.assertEqual(summary.loc["a::", "points"], 12)
        self.assertEqual(summary.loc["a::", "groups"], 2)
        self.assertEqual(summary.loc["b::", "groups"], 1)


if __name__ == "__main__":
    unittest.main()
