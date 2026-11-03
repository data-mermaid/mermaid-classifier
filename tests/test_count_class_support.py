"""Tests for scripts/count_class_support.py: CLI parsing, CSV writing and the dataset flow."""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import io
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
    include_mermaid: bool = False
    mermaid_project_lookup_uri: str | None = None
    included_labels_csv: str | None = None


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
    def _run(self, td: str, *extra: str, options: _Options | None = None):
        dataset = _dataset()
        options = options or _Options(holdout_groups_csv="old.csv", holdout_fold=0)
        config = mock.Mock()
        config.build_options.return_value = (options, None, None)
        dataset_cls = mock.Mock()
        dataset_cls.filtered_annotations_only.return_value = dataset
        out = Path(td) / "support.csv"
        with (
            mock.patch.object(ccs, "_resolve_local_aws_credentials"),
            mock.patch.object(ccs, "load_config_then_pyspacer", return_value=(config, dataset_cls)),
        ):
            ccs.main(["--config-dir", td, "--out", str(out), *extra])
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

    def test_mermaid_rows_without_a_project_lookup_stop_the_count(self):
        options = _Options(holdout_groups_csv=None, holdout_fold="test", include_mermaid=True)
        with (
            tempfile.TemporaryDirectory() as td,
            self.assertRaisesRegex(SystemExit, "mermaid_project_lookup_uri"),
        ):
            self._run(td, options=options)

    def test_check_failure_exits_nonzero_after_writing_the_csv(self):
        # ba1 has 3 points in 2 groups; ba2 has 1 point in 1 group.
        with tempfile.TemporaryDirectory() as td:
            stdout = io.StringIO()
            with (
                contextlib.redirect_stdout(stdout),
                self.assertRaises(SystemExit) as raised,
            ):
                self._run(td, "--check-min-points", "2", "--check-min-groups", "1")
            self.assertTrue((Path(td) / "support.csv").exists())
        self.assertNotIn(raised.exception.code, (0, None))
        self.assertIn("ba2::gf", stdout.getvalue())

    def _run_checked(self, td: str, labels: str | None, *floors: str):
        path = None
        if labels is not None:
            path = str(Path(td) / "included.csv")
            Path(path).write_text(labels)
        options = _Options(holdout_groups_csv=None, holdout_fold="test", included_labels_csv=path)
        self._run(td, *floors, options=options)

    def test_included_class_with_zero_annotations_fails_the_check(self):
        labels = "ba_id,gf_id\nba1,\nba3,\n"
        with tempfile.TemporaryDirectory() as td:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit):
                self._run_checked(td, labels, "--check-min-points", "1")
        self.assertIn("ba3::", stdout.getvalue())

    def test_included_class_with_a_growth_form_is_keyed_with_it(self):
        labels = "ba_id,gf_id\nba1,\nba2,gf\n"
        with tempfile.TemporaryDirectory() as td:
            self._run_checked(td, labels, "--check-min-points", "1")

    def test_no_included_labels_csv_says_zero_annotation_classes_are_unchecked(self):
        with tempfile.TemporaryDirectory() as td:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self._run_checked(td, None, "--check-min-points", "1")
        self.assertIn("cannot be checked", stdout.getvalue())

    def test_empty_counts_fail_the_check_when_classes_are_included(self):
        import pandas as pd

        empty = pd.DataFrame({"class_key": [], "site": [], "project_id": [], "n": []})
        labels = "ba_id,gf_id\nba1,\n"
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch(
                "mermaid_classifier.training.class_support.class_group_counts", return_value=empty
            ),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            self._run_checked(td, labels, "--check-min-points", "1")

    def test_check_passing_returns_normally(self):
        with tempfile.TemporaryDirectory() as td:
            self._run(td, "--check-min-points", "1", "--check-min-groups", "1")


class BelowFloorsTest(unittest.TestCase):
    def _summary(self):
        import pandas as pd

        return pd.DataFrame(
            {"points": [600, 400, 900], "groups": [12, 15, 9]},
            index=pd.Index(["ok::", "few_points::", "few_groups::"], name="class_key"),
        )

    def test_lists_classes_below_either_floor_with_their_counts(self):
        failing = ccs.below_floors(self._summary(), min_points=500, min_groups=10)
        self.assertEqual(
            failing.to_dict("index"),
            {
                "few_points::": {"points": 400, "groups": 15},
                "few_groups::": {"points": 900, "groups": 9},
            },
        )

    def test_a_class_exactly_at_both_floors_passes(self):
        failing = ccs.below_floors(self._summary(), min_points=400, min_groups=9)
        self.assertTrue(failing.empty)


class PoritesUnionTest(unittest.TestCase):
    def _counts(self, rows: list[tuple[str, str, int]]):
        import pandas as pd

        return pd.DataFrame(
            {
                "class_key": [r[0] for r in rows],
                "site": ["s"] * len(rows),
                "project_id": [r[1] for r in rows],
                "n": [r[2] for r in rows],
            }
        )

    def test_thin_empty_gf_class_is_judged_with_its_growth_form_buckets(self):
        counts = self._counts(
            [("P::", "1", 10)]
            + [("P::a", str(i), 200) for i in (2, 3, 4)]
            + [("P::b", str(i), 200) for i in (5, 6, 7)]
        )
        failing = ccs.below_floors(ccs.class_summary(counts), 500, 3)
        self.assertTrue(failing.empty)

    def test_growth_form_buckets_are_still_judged_individually(self):
        counts = self._counts([("P::", "1", 10), ("P::a", "2", 300), ("P::b", "3", 300)])
        failing = ccs.below_floors(ccs.class_summary(counts), 500, 1)
        self.assertEqual(set(failing.index), {"P::a", "P::b"})

    def test_zero_point_empty_gf_class_passes_on_its_buckets(self):
        counts = self._counts([("P::a", "2", 300), ("P::b", "3", 300)])
        summary = ccs.class_summary(counts, extra_keys=["P::"])
        self.assertEqual(set(ccs.below_floors(summary, 500, 2).index), {"P::a", "P::b"})
        self.assertNotIn("P::", ccs.below_floors(summary, 500, 2).index)

    def test_extra_key_with_no_rows_has_zero_support(self):
        summary = ccs.class_summary(self._counts([("a::", "1", 5)]), extra_keys=["z::"])
        self.assertEqual(summary.loc["z::"].to_dict(), {"points": 0, "groups": 0})


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
