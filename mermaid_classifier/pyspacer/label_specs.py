"""
CSV-defined label specifications for the training pipeline.

- LabelFilter: include/exclude specific BA+GF combos from training data.
- LabelRollupSpec: roll up fine-grained BA+GF combos to coarser categories.
- CNSourceFilter: specify which CoralNet sources to include.
"""

import typing

import duckdb

from mermaid_classifier.common.benthic_attributes import (
    BAGF_SEP,
    combine_ba_gf,
    split_ba_gf,
)
from mermaid_classifier.common.csv_utils import ColumnSpec, CsvSpec
from mermaid_classifier.common.duckdb_utils import (
    duckdb_filter_on_column,
    duckdb_replace_column,
    duckdb_transform_column,
)


class LabelFilter(CsvSpec):
    """
    A CSV-defined spec which says what benthic attribute + growth form
    combos to include in, or exclude from, training data.
    """

    column_specs = [
        ColumnSpec(name="ba_id", allow_blank=False),
        ColumnSpec(name="gf_id"),
    ]

    def __init__(self, csv_file: typing.TextIO, inclusion: bool = True):
        self.bagf_set: set[tuple[str, str]] = set()

        super().__init__(csv_file=csv_file)

        self.inclusion = inclusion

    def per_row_init_action(self, row: dict[str, str | None]) -> None:
        # Ensure absent values are just '', not '' or None.
        self.bagf_set.add((row["ba_id"] or "", row.get("gf_id") or ""))

    def accepts_bagf(self, bagf_id: str | None) -> bool:
        if bagf_id is None:
            return not self.inclusion
        ba_id, gf_id = split_ba_gf(bagf_id)

        if self.inclusion:
            return (ba_id, gf_id) in self.bagf_set
        return (ba_id, gf_id) not in self.bagf_set

    def filter_in_duckdb(
        self,
        duck_conn: duckdb.DuckDBPyConnection,
        duck_table_name: str,
        ba_id_column_name: str = "benthic_attribute_id",
        gf_id_column_name: str = "growth_form_id",
    ):
        """
        Filter down the rows in the given DuckDB table, based on the
        benthic attribute ID and growth form ID columns, and this
        instance's filter rules.
        """
        # Concatenate BA+GF so that we can define the filter as a
        # single-column operation.
        # https://duckdb.org/docs/stable/sql/functions/text#concat_wsseparator-string-
        duck_conn.execute(
            f"CREATE OR REPLACE TABLE {duck_table_name} AS"
            f" SELECT"
            f"  *,"
            f"  concat_ws("
            f"   '{BAGF_SEP}', {ba_id_column_name}, {gf_id_column_name})"
            f"   AS bagf_id"
            f" FROM {duck_table_name}"
        )

        # Filter.
        duckdb_filter_on_column(
            duck_conn=duck_conn,
            duck_table_name=duck_table_name,
            column_name="bagf_id",
            inclusion_func=self.accepts_bagf,
        )

        # Don't need the combined BAGF column anymore.
        duck_conn.execute(f"ALTER TABLE {duck_table_name} DROP bagf_id")


class LabelRollupSpec(CsvSpec):
    """
    A CSV-defined spec which says what BA+GF combos to roll up to
    what other BA+GF combos.
    """

    column_specs = [
        ColumnSpec(name="from_ba_id", allow_blank=False),
        ColumnSpec(name="from_gf_id"),
        ColumnSpec(name="to_ba_id", allow_blank=False),
        ColumnSpec(name="to_gf_id"),
    ]

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        self.lookup: dict[tuple[str, str], tuple[str, str]] = {}

        super().__init__(*args, **kwargs)

    def per_row_init_action(self, row: dict[str, str | None]) -> None:
        # Ensure absent values are just '', not '' or None.
        key = (row["from_ba_id"] or "", row.get("from_gf_id") or "")
        value = (row["to_ba_id"] or "", row.get("to_gf_id") or "")
        self.lookup[key] = value

    def roll_up(self, bagf_id: str | None) -> str | None:
        if bagf_id is None:
            return None
        ba_id, gf_id = split_ba_gf(bagf_id)

        if (ba_id, gf_id) in self.lookup:
            new_ba_id, new_gf_id = self.lookup[(ba_id, gf_id)]
            return combine_ba_gf(new_ba_id, new_gf_id)
        # If this BAGF is not in the rollup spec, then we leave the
        # BAGF as is.
        return bagf_id

    def roll_up_in_duckdb(
        self,
        duck_conn: duckdb.DuckDBPyConnection,
        duck_table_name: str,
        ba_id_column_name: str = "benthic_attribute_id",
        gf_id_column_name: str = "growth_form_id",
    ):
        """
        Roll up the BA IDs and GF IDs in the given DuckDB table,
        based on this instance's rollup rules.
        """
        # Concatenate BA+GF so that we can define the rollup as a
        # single-column transform.
        # https://duckdb.org/docs/stable/sql/functions/text#concat_wsseparator-string-
        # If there's no GF, then the result is the BA plus separator.
        duck_conn.execute(
            f"CREATE OR REPLACE TABLE {duck_table_name} AS"
            f" SELECT"
            f"  *,"
            f"  concat_ws("
            f"   '{BAGF_SEP}', {ba_id_column_name}, {gf_id_column_name})"
            f"   AS bagf_id"
            f" FROM {duck_table_name}"
        )

        # Apply the rollup.
        duckdb_transform_column(
            duck_conn=duck_conn,
            duck_table_name=duck_table_name,
            column_name="bagf_id",
            transform_func=self.roll_up,
        )

        # Propagate rolled-up BAGF back to the split BA-GF fields.
        # https://duckdb.org/docs/stable/sql/functions/text#split_partstring-separator-index
        duck_conn.execute(
            f"CREATE OR REPLACE TABLE {duck_table_name} AS"
            f" SELECT"
            f"  *,"
            f"  bagf_id.split_part('{BAGF_SEP}', 1)"
            f"   AS rollup_{ba_id_column_name},"
            f"  bagf_id.split_part('{BAGF_SEP}', 2)"
            f"   AS rollup_{gf_id_column_name}"
            f" FROM {duck_table_name}"
        )
        duckdb_replace_column(
            duck_conn=duck_conn,
            duck_table_name=duck_table_name,
            column_name=ba_id_column_name,
            new_values_column_name=f"rollup_{ba_id_column_name}",
        )
        duckdb_replace_column(
            duck_conn=duck_conn,
            duck_table_name=duck_table_name,
            column_name=gf_id_column_name,
            new_values_column_name=f"rollup_{gf_id_column_name}",
        )

        # Don't need the combined BAGF column anymore.
        duck_conn.execute(f"ALTER TABLE {duck_table_name} DROP bagf_id")


class CNSourceFilter(CsvSpec):
    column_specs = [
        ColumnSpec(name="id", allow_blank=False),
    ]

    source_id_list: list[str]

    def __init__(self, csv_file: typing.TextIO):
        """
        Initialize using a CSV file that specifies a set
        of CoralNet sources.
        """
        self.source_id_list = []

        super().__init__(csv_file=csv_file)

    def per_row_init_action(self, row: dict[str, str | None]) -> None:
        self.source_id_list.append(row["id"] or "")

    def is_empty(self) -> bool:
        return len(self.source_id_list) == 0
