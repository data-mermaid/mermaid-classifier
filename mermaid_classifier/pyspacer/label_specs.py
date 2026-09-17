"""
CSV-defined label specifications for the training pipeline.

- LabelFilter: include/exclude specific BA+GF combos from training data.
- LabelRollupSpec: roll up fine-grained BA+GF combos to coarser categories.
- CNSourceFilter: specify which CoralNet sources to include.
- ImageExclusionFilter: withhold specific images from training data.
"""

import typing

import duckdb
import pandas as pd

from mermaid_classifier.common.benthic_attributes import (
    BAGF_SEP,
    combine_ba_gf,
    split_ba_gf,
)
from mermaid_classifier.common.csv_utils import ColumnSpec, CsvSpec
from mermaid_classifier.common.duckdb_utils import (
    duckdb_filter_on_column,
    duckdb_replace_column,
    duckdb_temp_table_name,
    duckdb_transform_column,
)
from mermaid_classifier.pyspacer.utils import logging_config_for_script

logger = logging_config_for_script("train")

# How many unmatched image ids a warning names. Enough to recognise an id
# format that has drifted, short enough to read in a training log.
UNMATCHED_ID_SAMPLE_SIZE = 10


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


class ImageExclusionFilter(CsvSpec):
    """
    A CSV-defined spec listing image IDs to withhold from training data
    entirely, regardless of label. Used to keep specific images — such
    as a frozen evaluation probe's held-out set — out of training.
    """

    column_specs = [
        ColumnSpec(name="image_id", allow_blank=False),
    ]

    def __init__(self, csv_file: typing.TextIO):
        self.excluded_image_ids: set[str] = set()

        super().__init__(csv_file=csv_file)

    def per_row_init_action(self, row: dict[str, str | None]) -> None:
        # str() guards against pandas inferring int64 for an all-numeric
        # image_id column; the annotations table's image_id is always a
        # string, so the comparison in filter_in_duckdb must be too.
        self.excluded_image_ids.add(str(row["image_id"]))

    def is_empty(self) -> bool:
        return len(self.excluded_image_ids) == 0

    def filter_in_duckdb(
        self,
        duck_conn: duckdb.DuckDBPyConnection,
        duck_table_name: str,
        image_id_column_name: str = "image_id",
    ) -> None:
        """
        Remove rows from the given DuckDB table whose image_id is in
        this spec's exclusion list.

        Logs the number of annotations and distinct images removed, and
        every listed id that matched no row, a sample of them by name.
        Any unmatched id is a warning: an id format that has drifted from
        the table's image_id leaves those images in training while the run
        reads as though they were withheld, and a list that matches nothing
        at all carries the harder message.
        """
        if self.is_empty():
            return

        excluded_images_df = pd.DataFrame(  # noqa: F841 — referenced by name in DuckDB SQL via Python-scope scanning  # pyright: ignore[reportUnusedVariable]
            {image_id_column_name: sorted(self.excluded_image_ids)}
        )

        with duckdb_temp_table_name(duck_conn, base_name="excluded_images") as excluded_table_name:
            duck_conn.execute(
                f"CREATE TABLE {excluded_table_name} AS SELECT * FROM excluded_images_df"
            )

            # A COUNT(*) query always returns exactly one row, so fetchall()[0]
            # avoids fetchone()'s `tuple[Any, ...] | None` return type.
            annotations_removed, images_removed = duck_conn.execute(
                f"SELECT count(*), count(DISTINCT t.{image_id_column_name})"
                f" FROM {duck_table_name} t"
                f" JOIN {excluded_table_name} e"
                f"  USING ({image_id_column_name})"
            ).fetchall()[0]

            unmatched_count = len(self.excluded_image_ids) - images_removed
            unmatched_sample: list[str] = []
            if unmatched_count > 0:
                # Sampled before the delete: afterwards no listed id
                # matches a row, whether it did or not.
                unmatched_sample = [
                    row[0]
                    for row in duck_conn.execute(
                        f"SELECT e.{image_id_column_name}"
                        f" FROM {excluded_table_name} e"
                        f" LEFT JOIN {duck_table_name} t"
                        f"  USING ({image_id_column_name})"
                        f" WHERE t.{image_id_column_name} IS NULL"
                        f" ORDER BY e.{image_id_column_name}"
                        f" LIMIT {UNMATCHED_ID_SAMPLE_SIZE}"
                    ).fetchall()
                ]

            duck_conn.execute(
                f"CREATE OR REPLACE TABLE {duck_table_name} AS"
                f" SELECT t.*"
                f" FROM {duck_table_name} t"
                f" LEFT JOIN {excluded_table_name} e"
                f"  USING ({image_id_column_name})"
                f" WHERE e.{image_id_column_name} IS NULL"
            )

        if images_removed == 0:
            logger.warning(
                "Image exclusion spec listed %s image id(s), but none"
                " matched an image_id in this dataset — the exclusion"
                " had no effect. Unmatched ids include: %s",
                len(self.excluded_image_ids),
                ", ".join(unmatched_sample),
            )
        elif unmatched_count > 0:
            logger.warning(
                "Image exclusion spec removed %s annotation(s) across %s"
                " image(s), but %s of %s listed id(s) matched no image in"
                " this dataset and stay in the training data."
                " Unmatched ids include: %s",
                annotations_removed,
                images_removed,
                unmatched_count,
                len(self.excluded_image_ids),
                ", ".join(unmatched_sample),
            )
        else:
            logger.info(
                "Image exclusion spec removed %s annotation(s) across"
                " %s image(s); every one of the %s listed id(s) matched.",
                annotations_removed,
                images_removed,
                len(self.excluded_image_ids),
            )
