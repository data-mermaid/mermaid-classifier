"""Characterization tests for LabelFilter, LabelRollupSpec, and CNSourceFilter.

All three classes live in mermaid_classifier.pyspacer.label_specs and subclass CsvSpec.
Tests cover both the pure-Python methods and the in-DuckDB pipeline methods.

Empty growth form is the empty string '' (never NULL) per the BA+GF convention.
BA+GF separator is '::' (BAGF_SEP).
"""

import unittest
from io import StringIO

import duckdb
import pandas as pd

from mermaid_classifier.pyspacer.label_specs import CNSourceFilter, LabelFilter, LabelRollupSpec


def _make_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def _seed_annotations(
    conn: duckdb.DuckDBPyConnection,
    ba_ids: list[str],
    gf_ids: list[str],
) -> None:
    """Seed a minimal annotations table with benthic_attribute_id + growth_form_id."""
    assert len(ba_ids) == len(gf_ids)
    df = pd.DataFrame(  # noqa: F841 — referenced by name in DuckDB SQL
        {
            "benthic_attribute_id": ba_ids,
            "growth_form_id": gf_ids,
        }
    )
    conn.execute("CREATE OR REPLACE TABLE annotations AS SELECT * FROM df")


# ---------------------------------------------------------------------------
# LabelFilter
# ---------------------------------------------------------------------------


_FILTER_CSV = "ba_id,gf_id\nBA1,GF1\nBA2,\n"


class LabelFilterInclusionTest(unittest.TestCase):
    """Tests for LabelFilter with inclusion=True (default)."""

    def setUp(self):
        self.f = LabelFilter(StringIO(_FILTER_CSV), inclusion=True)

    def test_known_ba_gf_accepted(self):
        self.assertTrue(self.f.accepts_bagf("BA1::GF1"))

    def test_known_ba_empty_gf_accepted(self):
        """BA2 with no growth form is in the spec (empty gf_id cell)."""
        self.assertTrue(self.f.accepts_bagf("BA2::"))

    def test_unknown_bagf_rejected(self):
        self.assertFalse(self.f.accepts_bagf("BA9::GF9"))

    def test_none_rejected(self):
        """None bagf_id is not accepted by an inclusion filter."""
        self.assertFalse(self.f.accepts_bagf(None))


class LabelFilterExclusionTest(unittest.TestCase):
    """Tests for LabelFilter with inclusion=False."""

    def setUp(self):
        self.f = LabelFilter(StringIO(_FILTER_CSV), inclusion=False)

    def test_known_ba_gf_excluded(self):
        """BA1::GF1 is in the exclusion set → not accepted."""
        self.assertFalse(self.f.accepts_bagf("BA1::GF1"))

    def test_unknown_bagf_accepted(self):
        """BA9::GF9 is NOT in the exclusion set → accepted."""
        self.assertTrue(self.f.accepts_bagf("BA9::GF9"))

    def test_none_accepted(self):
        """None bagf_id passes through an exclusion filter."""
        self.assertTrue(self.f.accepts_bagf(None))


class LabelFilterInDuckDBTest(unittest.TestCase):
    """Tests for LabelFilter.filter_in_duckdb."""

    def test_only_included_rows_remain(self):
        conn = _make_conn()
        _seed_annotations(conn, ["BA1", "BA2", "BA9"], ["GF1", "", "GF9"])

        f = LabelFilter(StringIO(_FILTER_CSV), inclusion=True)
        f.filter_in_duckdb(conn, "annotations")

        rows = conn.execute(
            "SELECT benthic_attribute_id, growth_form_id FROM annotations"
            " ORDER BY benthic_attribute_id"
        ).fetchall()
        self.assertEqual(rows, [("BA1", "GF1"), ("BA2", "")])

    def test_excluded_row_removed(self):
        conn = _make_conn()
        _seed_annotations(conn, ["BA9"], ["GF9"])

        f = LabelFilter(StringIO(_FILTER_CSV), inclusion=True)
        f.filter_in_duckdb(conn, "annotations")

        count = conn.execute("SELECT count(*) FROM annotations").fetchone()[0]
        self.assertEqual(count, 0)

    def test_bagf_id_temp_column_absent_after_filter(self):
        """The temp 'bagf_id' column added during filtering must be dropped."""
        conn = _make_conn()
        _seed_annotations(conn, ["BA1", "BA2"], ["GF1", ""])

        f = LabelFilter(StringIO(_FILTER_CSV), inclusion=True)
        f.filter_in_duckdb(conn, "annotations")

        cols = [row[0] for row in conn.execute("DESCRIBE annotations").fetchall()]
        self.assertNotIn("bagf_id", cols, msg="bagf_id temp column must be removed after filter")

    def test_exclusion_filter_keeps_unknown_rows(self):
        conn = _make_conn()
        _seed_annotations(conn, ["BA1", "BA9"], ["GF1", "GF9"])

        f = LabelFilter(StringIO(_FILTER_CSV), inclusion=False)
        f.filter_in_duckdb(conn, "annotations")

        rows = conn.execute("SELECT benthic_attribute_id FROM annotations").fetchall()
        self.assertEqual(rows, [("BA9",)])


# ---------------------------------------------------------------------------
# LabelRollupSpec
# ---------------------------------------------------------------------------


_ROLLUP_CSV = "from_ba_id,from_gf_id,to_ba_id,to_gf_id\nBA1,GF1,BA10,GF10\nBA2,,BA20,\n"


class LabelRollupSpecRollUpTest(unittest.TestCase):
    """Tests for LabelRollupSpec.roll_up (pure Python)."""

    def setUp(self):
        self.r = LabelRollupSpec(StringIO(_ROLLUP_CSV))

    def test_mapped_bagf_rolled_up(self):
        self.assertEqual(self.r.roll_up("BA1::GF1"), "BA10::GF10")

    def test_empty_gf_rolled_up(self):
        """BA2 with empty GF maps to BA20 with empty GF."""
        self.assertEqual(self.r.roll_up("BA2::"), "BA20::")

    def test_unmapped_bagf_unchanged(self):
        """A BA+GF not in the rollup spec is returned unchanged."""
        self.assertEqual(self.r.roll_up("BA9::GF9"), "BA9::GF9")

    def test_none_returns_none(self):
        self.assertIsNone(self.r.roll_up(None))


class LabelRollupSpecInDuckDBTest(unittest.TestCase):
    """Tests for LabelRollupSpec.roll_up_in_duckdb."""

    def test_mapped_rows_have_updated_ba_gf(self):
        conn = _make_conn()
        _seed_annotations(conn, ["BA1", "BA2", "BA9"], ["GF1", "", "GF9"])

        r = LabelRollupSpec(StringIO(_ROLLUP_CSV))
        r.roll_up_in_duckdb(conn, "annotations")

        rows = conn.execute(
            "SELECT benthic_attribute_id, growth_form_id FROM annotations ORDER BY benthic_attribute_id"
        ).fetchall()
        # BA1::GF1 → BA10::GF10; BA2:: → BA20::; BA9::GF9 unchanged
        self.assertIn(("BA10", "GF10"), rows)
        self.assertIn(("BA20", ""), rows)
        self.assertIn(("BA9", "GF9"), rows)

    def test_bagf_id_temp_column_absent_after_rollup(self):
        """The temp 'bagf_id' column added during rollup must be dropped."""
        conn = _make_conn()
        _seed_annotations(conn, ["BA1"], ["GF1"])

        r = LabelRollupSpec(StringIO(_ROLLUP_CSV))
        r.roll_up_in_duckdb(conn, "annotations")

        cols = [row[0] for row in conn.execute("DESCRIBE annotations").fetchall()]
        self.assertNotIn("bagf_id", cols, msg="bagf_id temp column must be removed after rollup")

    def test_unmapped_row_count_unchanged(self):
        """Rollup does not drop unmapped rows — count stays the same."""
        conn = _make_conn()
        _seed_annotations(conn, ["BA1", "BA9"], ["GF1", "GF9"])

        r = LabelRollupSpec(StringIO(_ROLLUP_CSV))
        r.roll_up_in_duckdb(conn, "annotations")

        count = conn.execute("SELECT count(*) FROM annotations").fetchone()[0]
        self.assertEqual(count, 2)


# ---------------------------------------------------------------------------
# CNSourceFilter
# ---------------------------------------------------------------------------


class CNSourceFilterTest(unittest.TestCase):
    """Tests for CNSourceFilter."""

    def test_is_empty_false_when_sources_present(self):
        f = CNSourceFilter(StringIO("id\n123\n456\n"))
        self.assertFalse(f.is_empty())

    def test_source_id_list_length(self):
        f = CNSourceFilter(StringIO("id\n123\n456\n"))
        self.assertEqual(len(f.source_id_list), 2)

    def test_source_id_list_values(self):
        """Actual characterization: pandas reads integer IDs as numpy int64."""
        f = CNSourceFilter(StringIO("id\n123\n456\n"))
        # The ids are numeric (pandas infers int64 from all-numeric column)
        # and compare equal to plain Python ints.
        self.assertEqual(int(f.source_id_list[0]), 123)
        self.assertEqual(int(f.source_id_list[1]), 456)

    def test_is_empty_true_when_header_only(self):
        """A CSV with only a header row (no data rows) yields is_empty() == True."""
        f = CNSourceFilter(StringIO("id\n"))
        self.assertTrue(f.is_empty())

    def test_is_empty_true_when_empty_csv(self):
        """A completely empty CSV yields is_empty() == True."""
        f = CNSourceFilter(StringIO(""))
        self.assertTrue(f.is_empty())

    def test_source_id_list_empty_when_no_data(self):
        f = CNSourceFilter(StringIO("id\n"))
        self.assertEqual(f.source_id_list, [])


if __name__ == "__main__":
    unittest.main()
