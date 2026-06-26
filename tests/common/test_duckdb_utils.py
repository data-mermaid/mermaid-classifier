"""Characterization unit tests for common/duckdb_utils.py.

Each TestCase covers one public function. All tests use in-memory DuckDB
connections and assert exact observed behavior — these are characterization
tests intended to guard refactoring in issues #73 and #74.
"""

import unittest

import duckdb
import pandas as pd

from mermaid_classifier.common.duckdb_utils import (
    duckdb_add_column,
    duckdb_batched_rows,
    duckdb_filter_on_column,
    duckdb_grouped_rows,
    duckdb_replace_column,
    duckdb_temp_table_name,
    duckdb_transform_column,
)


def _make_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


class TempTableNameTest(unittest.TestCase):
    """Tests for duckdb_temp_table_name."""

    def test_yields_temp_prefixed_name_with_no_base(self):
        conn = _make_conn()
        with duckdb_temp_table_name(conn) as name:
            self.assertTrue(name.startswith("temp_"), msg=f"Expected temp_ prefix, got {name!r}")

    def test_yields_deterministic_name_with_base_name(self):
        conn = _make_conn()
        with duckdb_temp_table_name(conn, base_name="foo") as name:
            self.assertEqual(name, "temp_foo")

    def test_table_is_dropped_after_context_exits(self):
        conn = _make_conn()
        captured_name = None
        with duckdb_temp_table_name(conn) as name:
            captured_name = name
            # Create the table inside the context.
            conn.execute(f"CREATE TABLE {name} (x INTEGER)")
            # Verify it exists.
            result = conn.execute(
                f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{name}'"
            ).fetchone()[0]
            self.assertEqual(result, 1)

        # After exit, the table should no longer exist.
        result = conn.execute(
            f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{captured_name}'"
        ).fetchone()[0]
        self.assertEqual(result, 0, msg="Table should be dropped after context manager exits")

    def test_table_dropped_even_if_never_created(self):
        """No-op DROP IF EXISTS — should not raise if table was never created."""
        conn = _make_conn()
        with duckdb_temp_table_name(conn, base_name="never_created") as name:
            self.assertEqual(name, "temp_never_created")
        # If we get here, no exception was raised.


class ReplaceColumnTest(unittest.TestCase):
    """Tests for duckdb_replace_column."""

    def test_old_column_replaced_by_new_column(self):
        conn = _make_conn()
        df = pd.DataFrame({"k": ["a", "b"], "new_k": ["A", "B"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        duckdb_replace_column(conn, "t", column_name="k", new_values_column_name="new_k")

        cols = [row[0] for row in conn.execute("DESCRIBE t").fetchall()]
        self.assertIn("k", cols, msg="Renamed column should be called 'k'")
        self.assertNotIn("new_k", cols, msg="Original new_k column should be gone")

        values = sorted(row[0] for row in conn.execute("SELECT k FROM t").fetchall())
        self.assertEqual(values, ["A", "B"], msg="Column should hold the new values")

    def test_original_column_name_is_reused(self):
        conn = _make_conn()
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        duckdb_replace_column(conn, "t", column_name="x", new_values_column_name="y")

        cols = [row[0] for row in conn.execute("DESCRIBE t").fetchall()]
        # After replace, only 'x' should remain (y was renamed to x).
        self.assertEqual(cols, ["x"])
        values = sorted(row[0] for row in conn.execute("SELECT x FROM t").fetchall())
        self.assertEqual(values, [10, 20, 30])


class TransformColumnTest(unittest.TestCase):
    """Tests for duckdb_transform_column."""

    def test_transform_applied_to_all_values(self):
        conn = _make_conn()
        df = pd.DataFrame({"name": ["alice", "bob", "carol"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        duckdb_transform_column(conn, "t", "name", lambda v: (v or "").upper())

        values = sorted(row[0] for row in conn.execute("SELECT name FROM t").fetchall())
        self.assertEqual(values, ["ALICE", "BOB", "CAROL"])

    def test_transform_of_empty_string(self):
        """Empty string '' is a distinct value (not NULL). The func receives ''."""
        conn = _make_conn()
        received: list[str | None] = []

        df = pd.DataFrame({"val": ["x", ""]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        def spy(v: str | None) -> str | None:
            received.append(v)
            return v

        duckdb_transform_column(conn, "t", "val", spy)

        # '' should be one of the values passed to the function.
        self.assertIn("", received, msg="Empty string should be passed to transform_func")

    def test_transform_result_applied_to_all_rows(self):
        """Duplicate values: both rows with value 'a' get the same transform result."""
        conn = _make_conn()
        df = pd.DataFrame({"v": ["a", "a", "b"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        duckdb_transform_column(conn, "t", "v", lambda v: (v or "").upper())

        values = sorted(row[0] for row in conn.execute("SELECT v FROM t").fetchall())
        self.assertEqual(values, ["A", "A", "B"])

    def test_column_name_unchanged_after_transform(self):
        conn = _make_conn()
        df = pd.DataFrame({"status": ["open", "closed"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        duckdb_transform_column(conn, "t", "status", lambda v: v)

        cols = [row[0] for row in conn.execute("DESCRIBE t").fetchall()]
        self.assertEqual(cols, ["status"])


class AddColumnTest(unittest.TestCase):
    """Tests for duckdb_add_column."""

    def test_new_column_added_and_base_column_intact(self):
        conn = _make_conn()
        df = pd.DataFrame({"code": ["A", "B", "C"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        duckdb_add_column(conn, "t", "code", "label", lambda v: f"label_{v}")

        rows = conn.execute("SELECT code, label FROM t ORDER BY code").fetchall()
        self.assertEqual(rows, [("A", "label_A"), ("B", "label_B"), ("C", "label_C")])

    def test_both_columns_present_after_call(self):
        conn = _make_conn()
        df = pd.DataFrame({"id": ["x"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        duckdb_add_column(conn, "t", "id", "derived", lambda v: (v or "").upper())

        cols = [row[0] for row in conn.execute("DESCRIBE t").fetchall()]
        self.assertIn("id", cols)
        self.assertIn("derived", cols)

    def test_new_column_computed_from_base(self):
        conn = _make_conn()
        df = pd.DataFrame({"n": [1, 2, 3]})  # noqa: F841
        conn.execute("CREATE TABLE t (n INTEGER)")
        conn.execute("INSERT INTO t SELECT * FROM df")

        # DuckDB add_column works via distinct-value mapping.
        # n is an integer; we cast via str in the func.
        duckdb_add_column(conn, "t", "n", "doubled", lambda v: str(int(v) * 2) if v else None)

        rows = conn.execute("SELECT n, doubled FROM t ORDER BY n").fetchall()
        self.assertEqual(rows, [(1, "2"), (2, "4"), (3, "6")])


class FilterOnColumnTest(unittest.TestCase):
    """Tests for duckdb_filter_on_column."""

    def test_only_matching_rows_remain(self):
        conn = _make_conn()
        df = pd.DataFrame({"val": ["keep", "drop", "keep", "drop"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        duckdb_filter_on_column(conn, "t", "val", lambda v: v == "keep")

        values = sorted(row[0] for row in conn.execute("SELECT val FROM t").fetchall())
        self.assertEqual(values, ["keep", "keep"])

    def test_excluded_rows_gone(self):
        conn = _make_conn()
        df = pd.DataFrame({"status": ["a", "b", "c", "a"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        duckdb_filter_on_column(conn, "t", "status", lambda v: v != "b")

        values = sorted(row[0] for row in conn.execute("SELECT status FROM t").fetchall())
        self.assertEqual(values, ["a", "a", "c"])

    def test_included_column_not_left_on_table(self):
        """The temp 'included' column must NOT appear on the table after the call."""
        conn = _make_conn()
        df = pd.DataFrame({"x": ["p", "q"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        duckdb_filter_on_column(conn, "t", "x", lambda v: True)

        cols = [row[0] for row in conn.execute("DESCRIBE t").fetchall()]
        self.assertNotIn(
            "included",
            cols,
            msg="The 'included' temp column should not be left on the table",
        )

    def test_all_rows_kept_when_func_always_true(self):
        conn = _make_conn()
        df = pd.DataFrame({"v": ["a", "b", "c"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        duckdb_filter_on_column(conn, "t", "v", lambda v: True)

        count = conn.execute("SELECT count(*) FROM t").fetchone()[0]
        self.assertEqual(count, 3)

    def test_all_rows_removed_when_func_always_false(self):
        conn = _make_conn()
        df = pd.DataFrame({"v": ["a", "b", "c"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        duckdb_filter_on_column(conn, "t", "v", lambda v: False)

        count = conn.execute("SELECT count(*) FROM t").fetchone()[0]
        self.assertEqual(count, 0)


class BatchedRowsTest(unittest.TestCase):
    """Tests for duckdb_batched_rows."""

    def test_all_rows_yielded(self):
        conn = _make_conn()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        relation = conn.execute("SELECT * FROM t ORDER BY a")
        rows = list(duckdb_batched_rows(relation))

        self.assertEqual(len(rows), 5)

    def test_row_field_values_correct(self):
        conn = _make_conn()
        df = pd.DataFrame({"id": [42], "name": ["hello"]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        relation = conn.execute("SELECT * FROM t")
        rows = list(duckdb_batched_rows(relation))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], 42)
        self.assertEqual(rows[0]["name"], "hello")

    def test_empty_table_yields_no_rows(self):
        conn = _make_conn()
        conn.execute("CREATE TABLE t (x INTEGER)")

        relation = conn.execute("SELECT * FROM t")
        rows = list(duckdb_batched_rows(relation))

        self.assertEqual(rows, [])

    def test_yields_pandas_series(self):
        conn = _make_conn()
        df = pd.DataFrame({"v": [99]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        relation = conn.execute("SELECT * FROM t")
        rows = list(duckdb_batched_rows(relation))

        import pandas

        self.assertIsInstance(rows[0], pandas.Series)


class GroupedRowsTest(unittest.TestCase):
    """Tests for duckdb_grouped_rows."""

    def test_two_groups_yielded(self):
        conn = _make_conn()
        df = pd.DataFrame(  # noqa: F841
            {"grp": ["a", "a", "b", "b", "b"], "val": [1, 2, 3, 4, 5]}
        )
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        groups = list(duckdb_grouped_rows(conn, "t", ["grp"]))

        self.assertEqual(len(groups), 2)

    def test_group_row_counts_correct(self):
        conn = _make_conn()
        df = pd.DataFrame(  # noqa: F841
            {"grp": ["a", "a", "b", "b", "b"], "val": [1, 2, 3, 4, 5]}
        )
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        groups = list(duckdb_grouped_rows(conn, "t", ["grp"]))
        # Ordered by grp column → 'a' first, then 'b'.
        self.assertEqual(len(groups[0]), 2, msg="Group 'a' should have 2 rows")
        self.assertEqual(len(groups[1]), 3, msg="Group 'b' should have 3 rows")

    def test_groups_ordered_by_grouping_column(self):
        conn = _make_conn()
        df = pd.DataFrame(  # noqa: F841
            {"grp": ["z", "z", "a", "m"], "n": [1, 2, 3, 4]}
        )
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        groups = list(duckdb_grouped_rows(conn, "t", ["grp"]))
        # ORDER BY grp → a, m, z
        first_grp = groups[0][0]["grp"]
        self.assertEqual(first_grp, "a")

    def test_single_row_group(self):
        conn = _make_conn()
        df = pd.DataFrame({"g": ["only"], "v": [7]})  # noqa: F841
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        groups = list(duckdb_grouped_rows(conn, "t", ["g"]))

        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 1)
        self.assertEqual(groups[0][0]["v"], 7)

    def test_multi_column_grouping(self):
        conn = _make_conn()
        df = pd.DataFrame(  # noqa: F841
            {
                "site": ["cn", "cn", "mm", "mm"],
                "proj": ["p1", "p1", "p2", "p2"],
                "val": [1, 2, 3, 4],
            }
        )
        conn.execute("CREATE TABLE t AS SELECT * FROM df")

        groups = list(duckdb_grouped_rows(conn, "t", ["site", "proj"]))

        self.assertEqual(len(groups), 2)
        for group in groups:
            self.assertEqual(len(group), 2)


if __name__ == "__main__":
    unittest.main()
