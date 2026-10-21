"""HoldoutGroups and ValGroups: the CSV specs naming test, CV-fold and val
groups, and the is_holdout / is_excluded / is_val tags they write onto the
annotations table.

A group is (site, project_id): a CoralNet source or a MERMAID project.
"""

import unittest
from io import StringIO

import duckdb
import pandas as pd

from mermaid_classifier.pyspacer.label_specs import HoldoutGroups, ValGroups, tag_exclude_clause

_LOGGER = "mermaid_classifier.pyspacer.label_specs"
_CSV = "site,group_id,fold\ncoralnet,10,test\ncoralnet,11,0\nmermaid,pA,1\nmermaid,pB,0\n"


def _seed(conn: duckdb.DuckDBPyConnection) -> None:
    rows = [
        ("coralnet", "10", "i1"),
        ("coralnet", "10", "i2"),
        ("coralnet", "11", "i3"),
        ("coralnet", "12", "i4"),  # in no group
        ("mermaid", "pA", "i5"),
        ("mermaid", "pB", "i6"),
    ]
    df = pd.DataFrame(rows, columns=["site", "project_id", "image_id"])  # noqa: F841 -- DuckDB scans by name
    conn.execute("CREATE TABLE annotations AS SELECT * FROM df")


def _tags(conn: duckdb.DuckDBPyConnection) -> dict[str, tuple[bool, bool]]:
    return {
        image_id: (is_holdout, is_excluded)
        for image_id, is_holdout, is_excluded in conn.execute(
            "SELECT image_id, is_holdout, is_excluded FROM annotations"
        ).fetchall()
    }


class HoldoutGroupsParsingTest(unittest.TestCase):
    def test_groups_map_site_and_group_id_to_fold(self):
        spec = HoldoutGroups(StringIO(_CSV))
        self.assertEqual(
            spec.group_folds,
            {
                ("coralnet", "10"): "test",
                ("coralnet", "11"): "0",
                ("mermaid", "pA"): "1",
                ("mermaid", "pB"): "0",
            },
        )

    def test_fold_zero_survives_pandas_integer_inference(self):
        """An all-numeric fold column reads as ints; 0 must not read as blank."""
        spec = HoldoutGroups(StringIO("site,group_id,fold\ncoralnet,5,0\ncoralnet,6,1\n"))
        self.assertEqual(spec.group_folds[("coralnet", "5")], "0")

    def test_empty_csv_is_empty(self):
        self.assertTrue(HoldoutGroups(StringIO("")).is_empty())

    def test_fold_must_be_test_or_non_negative_integer(self):
        for bad in ("train", "-1", "1.5"):
            with self.subTest(fold=bad), self.assertRaises(ValueError):
                HoldoutGroups(StringIO(f"site,group_id,fold\ncoralnet,5,{bad}\n"))

    def test_group_in_two_folds_is_rejected(self):
        with self.assertRaises(ValueError):
            HoldoutGroups(StringIO("site,group_id,fold\ncoralnet,5,0\ncoralnet,5,1\n"))

    def test_missing_column_is_rejected(self):
        with self.assertRaises(ValueError):
            HoldoutGroups(StringIO("site,group_id\ncoralnet,5\n"))


class HoldoutFoldValidationTest(unittest.TestCase):
    def setUp(self):
        self.spec = HoldoutGroups(StringIO(_CSV))

    def test_test_and_present_folds_are_accepted(self):
        self.spec.validate_holdout_fold("test")
        self.spec.validate_holdout_fold("0")
        self.spec.validate_holdout_fold("1")

    def test_absent_fold_is_rejected(self):
        with self.assertRaises(ValueError):
            self.spec.validate_holdout_fold("2")

    def test_non_test_fold_without_any_groups_is_rejected(self):
        with self.assertRaises(ValueError):
            HoldoutGroups(StringIO("")).validate_holdout_fold("0")

    def test_test_fold_without_any_groups_is_accepted(self):
        HoldoutGroups(StringIO("")).validate_holdout_fold("test")


class HoldoutTaggingTest(unittest.TestCase):
    def setUp(self):
        self.conn = duckdb.connect()
        _seed(self.conn)
        self.spec = HoldoutGroups(StringIO(_CSV))

    def test_fold_test_marks_test_groups_as_holdout_and_excludes_nothing(self):
        self.spec.tag_in_duckdb(self.conn, "annotations", "test")
        self.assertEqual(
            _tags(self.conn),
            {
                "i1": (True, False),
                "i2": (True, False),
                "i3": (False, False),
                "i4": (False, False),
                "i5": (False, False),
                "i6": (False, False),
            },
        )

    def test_fold_k_marks_fold_k_as_holdout_and_test_groups_as_excluded(self):
        self.spec.tag_in_duckdb(self.conn, "annotations", "0")
        self.assertEqual(
            _tags(self.conn),
            {
                "i1": (False, True),
                "i2": (False, True),
                "i3": (True, False),
                "i4": (False, False),
                "i5": (False, False),
                "i6": (True, False),
            },
        )

    def test_same_group_id_on_another_site_is_not_tagged(self):
        conn = duckdb.connect()
        df = pd.DataFrame(  # noqa: F841 -- DuckDB scans by name
            {"site": ["mermaid", "coralnet"], "project_id": ["10", "10"], "image_id": ["x", "y"]}
        )
        conn.execute("CREATE TABLE annotations AS SELECT * FROM df")
        self.spec.tag_in_duckdb(conn, "annotations", "test")
        self.assertEqual(_tags(conn), {"x": (False, False), "y": (True, False)})

    def test_empty_spec_tags_everything_false_and_keeps_every_row(self):
        HoldoutGroups(StringIO("")).tag_in_duckdb(self.conn, "annotations", "test")
        tags = _tags(self.conn)
        self.assertEqual(len(tags), 6)
        self.assertEqual(set(tags.values()), {(False, False)})

    def test_columns_are_boolean(self):
        self.spec.tag_in_duckdb(self.conn, "annotations", "test")
        types = {r[0]: r[1] for r in self.conn.execute("DESCRIBE annotations").fetchall()}
        self.assertEqual(types["is_holdout"], "BOOLEAN")
        self.assertEqual(types["is_excluded"], "BOOLEAN")


class UnmatchedHoldoutGroupTest(unittest.TestCase):
    """A listed group with no rows: a typo, or a MERMAID project never looked up."""

    def setUp(self):
        self.conn = duckdb.connect()
        _seed(self.conn)

    def _spec(self, extra_row: str) -> HoldoutGroups:
        return HoldoutGroups(StringIO(_CSV + extra_row))

    def test_unmatched_test_fold_group_is_rejected_by_name(self):
        with self.assertRaisesRegex(ValueError, r"coralnet, 99"):
            self._spec("coralnet,99,test\n").tag_in_duckdb(self.conn, "annotations", "test")

    def test_unmatched_test_fold_group_is_rejected_in_a_cv_run(self):
        with self.assertRaisesRegex(ValueError, r"coralnet, 99"):
            self._spec("coralnet,99,test\n").tag_in_duckdb(self.conn, "annotations", "0")

    def test_unmatched_active_fold_group_is_rejected_by_name(self):
        with self.assertRaisesRegex(ValueError, r"coralnet, 98"):
            self._spec("coralnet,98,0\n").tag_in_duckdb(self.conn, "annotations", "0")

    def test_unmatched_group_in_another_fold_is_a_warning_by_name(self):
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            self._spec("coralnet,97,1\n").tag_in_duckdb(self.conn, "annotations", "test")
        self.assertIn("coralnet, 97", "\n".join(logs.output))

    def test_test_group_emptied_by_filtering_is_a_warning_not_an_error(self):
        groups_read = {("coralnet", "10"), ("coralnet", "11"), ("coralnet", "12")}
        groups_read |= {("mermaid", "pA"), ("mermaid", "pB"), ("coralnet", "96")}
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            self._spec("coralnet,96,test\n").tag_in_duckdb(
                self.conn, "annotations", "test", groups_read=groups_read
            )
        self.assertIn("coralnet, 96", "\n".join(logs.output))


class ValGroupsParsingTest(unittest.TestCase):
    def test_rows_are_site_and_group_id_pairs_and_repeats_are_idempotent(self):
        spec = ValGroups(StringIO("site,group_id\ncoralnet,10\nmermaid,pA\ncoralnet,10\n"))
        self.assertEqual(spec.groups, {("coralnet", "10"), ("mermaid", "pA")})

    def test_empty_csv_is_empty(self):
        self.assertTrue(ValGroups(StringIO("")).is_empty())
        self.assertFalse(ValGroups(StringIO("site,group_id\ncoralnet,10\n")).is_empty())

    def test_missing_column_is_rejected(self):
        with self.assertRaises(ValueError):
            ValGroups(StringIO("site\ncoralnet\n"))

    def test_unknown_group_is_rejected(self):
        # 'unknown' collects MERMAID images with no project; those always train.
        with self.assertRaisesRegex(ValueError, "unknown"):
            ValGroups(StringIO("site,group_id\nmermaid,unknown\n"))


class ValGroupsDisjointTest(unittest.TestCase):
    def setUp(self):
        self.holdout = HoldoutGroups(StringIO(_CSV))

    def _check(self, val_rows: str, holdout_fold: str) -> None:
        ValGroups(StringIO("site,group_id\n" + val_rows)).check_disjoint(self.holdout, holdout_fold)

    def test_val_group_in_the_test_fold_is_rejected_in_every_run(self):
        for holdout_fold in ("test", "0"):
            with (
                self.subTest(holdout_fold=holdout_fold),
                self.assertRaisesRegex(ValueError, r"coralnet, 10"),
            ):
                self._check("coralnet,10\n", holdout_fold)

    def test_val_group_in_the_held_out_cv_fold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\(coralnet, 11\).*val file per fold"):
            self._check("coralnet,11\n", "0")

    def test_every_offending_group_is_named(self):
        with self.assertRaises(ValueError) as raised:
            self._check("coralnet,10\nmermaid,pB\n", "0")
        self.assertIn("coralnet, 10", str(raised.exception))
        self.assertIn("mermaid, pB", str(raised.exception))

    def test_val_group_in_a_training_fold_or_in_no_fold_is_accepted(self):
        self._check("coralnet,11\ncoralnet,12\n", "test")
        self._check("coralnet,11\n", "1")


def _val_tags(conn: duckdb.DuckDBPyConnection) -> dict[str, bool]:
    return dict(conn.execute("SELECT image_id, is_val FROM annotations").fetchall())


class ValGroupsTaggingTest(unittest.TestCase):
    def setUp(self):
        self.conn = duckdb.connect()
        _seed(self.conn)

    def _tag(self, val_rows: str, holdout_fold: str = "test", groups_read=None) -> None:
        HoldoutGroups(StringIO(_CSV)).tag_in_duckdb(self.conn, "annotations", holdout_fold)
        ValGroups(StringIO("site,group_id\n" + val_rows)).tag_in_duckdb(
            self.conn, "annotations", groups_read=groups_read
        )

    def test_listed_groups_are_tagged_in_a_cv_run(self):
        self._tag("coralnet,12\nmermaid,pA\n", holdout_fold="0")
        self.assertEqual(
            _val_tags(self.conn),
            {"i1": False, "i2": False, "i3": False, "i4": True, "i5": True, "i6": False},
        )

    def test_holdout_rows_are_never_val(self):
        self._tag("coralnet,10\ncoralnet,12\n")
        self.assertEqual(_val_tags(self.conn)["i1"], False)
        self.assertEqual(_val_tags(self.conn)["i4"], True)

    def test_same_group_id_on_another_site_is_not_tagged(self):
        conn = duckdb.connect()
        df = pd.DataFrame(  # noqa: F841 -- DuckDB scans by name
            {
                "site": ["mermaid", "coralnet"],
                "project_id": ["10", "10"],
                "image_id": ["x", "y"],
                "is_holdout": [False, False],
            }
        )
        conn.execute("CREATE TABLE annotations AS SELECT * FROM df")
        ValGroups(StringIO("site,group_id\ncoralnet,10\n")).tag_in_duckdb(conn, "annotations")
        self.assertEqual(_val_tags(conn), {"x": False, "y": True})

    def test_empty_spec_tags_everything_false_and_keeps_every_row(self):
        HoldoutGroups(StringIO("")).tag_in_duckdb(self.conn, "annotations", "test")
        ValGroups(StringIO("")).tag_in_duckdb(self.conn, "annotations")
        self.assertEqual(
            _val_tags(self.conn), dict.fromkeys(["i1", "i2", "i3", "i4", "i5", "i6"], False)
        )
        types = {r[0]: r[1] for r in self.conn.execute("DESCRIBE annotations").fetchall()}
        self.assertEqual(types["is_val"], "BOOLEAN")

    def test_unmatched_group_is_rejected_by_name(self):
        with self.assertRaisesRegex(ValueError, r"coralnet, 99"):
            self._tag("coralnet,12\ncoralnet,99\n")

    def test_group_emptied_by_filtering_is_a_warning_not_an_error(self):
        groups_read = {("coralnet", "12"), ("coralnet", "96")}
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            self._tag("coralnet,12\ncoralnet,96\n", groups_read=groups_read)
        self.assertIn("coralnet, 96", "\n".join(logs.output))
        self.assertEqual(_val_tags(self.conn)["i4"], True)


class TagExcludeClauseTest(unittest.TestCase):
    def test_excludes_only_the_tag_columns_the_table_has(self):
        schemas = [
            "image_id VARCHAR, is_holdout BOOLEAN, is_excluded BOOLEAN, is_val BOOLEAN",
            "image_id VARCHAR, is_holdout BOOLEAN, is_excluded BOOLEAN",
            "image_id VARCHAR",
        ]
        for schema in schemas:
            with self.subTest(schema=schema):
                conn = duckdb.connect()
                conn.execute(f"CREATE TABLE annotations ({schema})")
                clause = tag_exclude_clause(conn, "annotations")
                columns = [
                    d[0] for d in conn.execute(f"SELECT * {clause} FROM annotations").description
                ]
                self.assertEqual(columns, ["image_id"])


if __name__ == "__main__":
    unittest.main()
