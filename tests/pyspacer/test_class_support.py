"""Tests for class_group_counts: per-(class, group) point counts from a dataset's annotations."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import duckdb

from mermaid_classifier.training.class_support import class_group_counts

COLUMNS = ["class_key", "site", "project_id", "n"]


def _dataset(rows: list[tuple[str, str, str, str]]):
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE annotations (benthic_attribute_id VARCHAR, growth_form_id VARCHAR,"
        " site VARCHAR, project_id VARCHAR)"
    )
    if rows:
        conn.executemany("INSERT INTO annotations VALUES (?, ?, ?, ?)", rows)
    return SimpleNamespace(duck_conn=conn)


class ClassGroupCountsTest(unittest.TestCase):
    def test_counts_points_per_class_and_group_with_empty_gf_key(self):
        dataset = _dataset(
            [
                ("ba1", "gf1", "coralnet", "10"),
                ("ba1", "gf1", "coralnet", "10"),
                ("ba1", "", "coralnet", "10"),
                ("ba1", "gf1", "mermaid", "all"),
            ]
        )
        df = class_group_counts(dataset)
        self.assertEqual(list(df.columns), COLUMNS)
        self.assertEqual(
            df.values.tolist(),
            [
                ["ba1::", "coralnet", "10", 1],
                ["ba1::gf1", "coralnet", "10", 2],
                ["ba1::gf1", "mermaid", "all", 1],
            ],
        )

    def test_row_order_is_independent_of_insertion_order(self):
        rows = [("b", "", "coralnet", "2"), ("a", "", "coralnet", "9"), ("a", "", "coralnet", "10")]
        forward = class_group_counts(_dataset(rows))
        backward = class_group_counts(_dataset(rows[::-1]))
        self.assertTrue(forward.equals(backward))
        self.assertEqual(forward["project_id"].tolist(), ["10", "9", "2"])

    def test_no_annotations_gives_an_empty_frame_with_the_columns(self):
        df = class_group_counts(_dataset([]))
        self.assertEqual(list(df.columns), COLUMNS)
        self.assertEqual(len(df), 0)


if __name__ == "__main__":
    unittest.main()
