"""Characterization unit tests for common/csv_utils.py (CsvSpec / ColumnSpec).

Uses a minimal concrete subclass defined in this module. All tests assert
exact observed behavior, pinning which parsed value lands under which key so
a rewrite of the header handling has to reproduce it.
"""

import unittest
from io import StringIO

from mermaid_classifier.common.csv_utils import ColumnSpec, CsvSpec


class _Spec(CsvSpec):
    """Minimal concrete subclass for testing CsvSpec."""

    column_specs = [
        ColumnSpec(name="id", allow_blank=False),
        ColumnSpec(name=["gf", "growth_form"]),
    ]

    def __init__(self, csv_file):
        self.rows: list[dict] = []
        super().__init__(csv_file=csv_file)

    def per_row_init_action(self, row: dict) -> None:
        self.rows.append(row)


class HappyPathTest(unittest.TestCase):
    """Normal CSV input: per_row_init_action is called once per row."""

    def test_every_row_is_collected_with_its_values(self):
        spec = _Spec(StringIO("id,gf\n42,Encrusting\n43,Massive\n"))
        self.assertEqual(
            [(r["id"], r["gf"]) for r in spec.rows],
            [(42, "Encrusting"), (43, "Massive")],
        )


class EmptyToNoneTest(unittest.TestCase):
    """Empty CSV cells are normalised to None by the base class."""

    def test_empty_gf_cell_is_none(self):
        """An empty gf cell should arrive as None in per_row_init_action."""
        spec = _Spec(StringIO("id,gf\n99,\n"))
        self.assertIsNone(spec.rows[0]["gf"], msg="Blank gf cell should become None")


class AlternateColumnNamesTest(unittest.TestCase):
    """ColumnSpec with name list resolves to the first matching header."""

    def test_value_accessible_under_actual_header_key(self):
        """The dict key in per_row_init_action uses the actual header name found."""
        spec = _Spec(StringIO("id,growth_form\n1,Branching\n"))
        # Header was 'growth_form', so the key is 'growth_form', not 'gf'.
        self.assertIn("growth_form", spec.rows[0])
        self.assertEqual(spec.rows[0]["growth_form"], "Branching")

    def test_primary_column_name_still_used_when_present(self):
        """If 'gf' is in the header, the key is 'gf'."""
        spec = _Spec(StringIO("id,gf\n1,Massive\n"))
        self.assertIn("gf", spec.rows[0])
        self.assertEqual(spec.rows[0]["gf"], "Massive")


class MissingRequiredColumnTest(unittest.TestCase):
    """A CSV missing a required column raises ValueError."""

    def test_missing_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            _Spec(StringIO("name,gf\nfoo,Branching\n"))

    def test_missing_all_alternate_names_raises_value_error(self):
        """If neither 'gf' nor 'growth_form' is present, ValueError is raised."""
        with self.assertRaises(ValueError):
            _Spec(StringIO("id,something_else\n1,x\n"))


class EmptyFileTest(unittest.TestCase):
    """An empty or header-only CSV produces no rows and no exception."""

    def test_header_only_no_rows_collected(self):
        spec = _Spec(StringIO("id,gf\n"))
        self.assertEqual(spec.rows, [], msg="per_row_init_action should not be called")

    def test_completely_empty_file_no_rows(self):
        """Completely empty CSV (no content at all) — no exception, no rows."""
        spec = _Spec(StringIO(""))
        self.assertEqual(spec.rows, [])
        self.assertTrue(spec.csv_dataframe.empty)


if __name__ == "__main__":
    unittest.main()
