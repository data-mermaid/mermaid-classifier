"""
CSV-defined label specifications for the training pipeline.

- LabelFilter: include/exclude specific BA+GF combos from training data.
- LabelRollupSpec: roll up fine-grained BA+GF combos to coarser categories.
- ImageExclusionFilter: withhold specific images from training data.
- HoldoutGroups: assign (site, project) groups to the test set or a CV fold.
- ValGroups: name (site, project) groups whose rows form the val set.
"""

import logging
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
from mermaid_classifier.training.holdout_selection import UNKNOWN_GROUP

logger = logging.getLogger(__name__)


def _project_bagf_column(
    duck_conn: duckdb.DuckDBPyConnection,
    duck_table_name: str,
    ba_id_column_name: str,
    gf_id_column_name: str,
) -> None:
    """
    Add a bagf_id column to the given DuckDB table, concatenating the BA
    and GF id columns so a filter or rollup can operate on a single column.
    Table and column name arguments are interpolated directly into SQL and
    must be trusted identifiers (code constants), never config or user values.
    """
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


def _drop_bagf_column(
    duck_conn: duckdb.DuckDBPyConnection,
    duck_table_name: str,
) -> None:
    """Drop the bagf_id column added by `_project_bagf_column`.
    Table name argument is interpolated directly into SQL and must be a
    trusted identifier (code constant), never config or user values.
    """
    duck_conn.execute(f"ALTER TABLE {duck_table_name} DROP bagf_id")


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
        _project_bagf_column(
            duck_conn=duck_conn,
            duck_table_name=duck_table_name,
            ba_id_column_name=ba_id_column_name,
            gf_id_column_name=gf_id_column_name,
        )

        # Filter.
        duckdb_filter_on_column(
            duck_conn=duck_conn,
            duck_table_name=duck_table_name,
            column_name="bagf_id",
            inclusion_func=self.accepts_bagf,
        )

        _drop_bagf_column(duck_conn=duck_conn, duck_table_name=duck_table_name)


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
        _project_bagf_column(
            duck_conn=duck_conn,
            duck_table_name=duck_table_name,
            ba_id_column_name=ba_id_column_name,
            gf_id_column_name=gf_id_column_name,
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

        _drop_bagf_column(duck_conn=duck_conn, duck_table_name=duck_table_name)


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
        this spec's exclusion list. A NULL image_id can never be a listed
        exclusion (the image_id column disallows blanks), so it always
        survives this filter.

        Logs how many of the listed ids matched an image_id in this
        dataset. An id format that has drifted from the table's image_id
        leaves those images in training while the run reads as though
        they were withheld, so this count is worth seeing even when it's
        zero -- and a spec that matches nothing at all logs at WARNING,
        since that means a held-out set is silently missing from the data
        it's meant to protect.
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
            matched_count = duck_conn.execute(
                f"SELECT count(DISTINCT t.{image_id_column_name})"
                f" FROM {duck_table_name} t"
                f" JOIN {excluded_table_name} e"
                f"  USING ({image_id_column_name})"
            ).fetchall()[0][0]

            # An explicit anti-join, not duckdb_filter_on_column: that
            # helper's JOIN ... USING never matches NULL to NULL, so it
            # drops a NULL image_id outright. Excluded ids are never NULL
            # (image_id disallows blanks), so NOT IN against them is safe.
            duck_conn.execute(
                f"CREATE OR REPLACE TABLE {duck_table_name} AS"
                f" SELECT * FROM {duck_table_name}"
                f" WHERE {image_id_column_name} IS NULL"
                f"  OR {image_id_column_name} NOT IN"
                f"   (SELECT {image_id_column_name} FROM {excluded_table_name})"
            )

        logger.log(
            logging.WARNING if matched_count == 0 else logging.INFO,
            "Image exclusion spec matched %s of %s listed image id(s) in this dataset.",
            matched_count,
            len(self.excluded_image_ids),
        )


HOLDOUT_TEST_FOLD = "test"

# Columns the group specs add to the annotations table; never part of an artifact.
TAG_COLUMNS = ("is_holdout", "is_excluded", "is_val")


def tag_exclude_clause(duck_conn: duckdb.DuckDBPyConnection, duck_table_name: str) -> str:
    """
    An `EXCLUDE (...)` clause naming the TAG_COLUMNS the table has, or "" for none.
    DuckDB's EXCLUDE fails on a missing column, so only present ones are named.
    """
    present = {row[0] for row in duck_conn.execute(f"DESCRIBE {duck_table_name}").fetchall()}
    tags = [name for name in TAG_COLUMNS if name in present]
    return f"EXCLUDE ({', '.join(tags)})" if tags else ""


def _create_groups_table(
    duck_conn: duckdb.DuckDBPyConnection,
    groups_table_name: str,
    columns: dict[str, list[str]],
) -> None:
    """
    Create `groups_table_name` with one VARCHAR column per key of `columns`,
    holding its values. The explicit schema keeps the columns VARCHAR when
    there are no rows. Names are interpolated, so they must be code constants.
    """
    groups_df = pd.DataFrame(columns)  # noqa: F841 — referenced by name in DuckDB SQL via Python-scope scanning  # pyright: ignore[reportUnusedVariable]
    column_sql = ", ".join(f"{name} VARCHAR" for name in columns)
    duck_conn.execute(f"CREATE TABLE {groups_table_name} ({column_sql})")
    duck_conn.execute(f"INSERT INTO {groups_table_name} SELECT * FROM groups_df")


def _describe_groups(groups: list[tuple[str, str]]) -> str:
    return ", ".join(f"({site}, {group_id})" for site, group_id in groups)


class HoldoutGroups(CsvSpec):
    """
    A CSV-defined spec assigning groups to the held-out test set or a
    cross-validation fold. A group is a (site, project_id) pair: a
    CoralNet source, or a MERMAID project.

    Columns: site, group_id, fold. `fold` is "test" or a non-negative
    integer. Every group appears in at most one fold.
    """

    column_specs = [
        ColumnSpec(name="site", allow_blank=False),
        ColumnSpec(name="group_id", allow_blank=False),
        ColumnSpec(name="fold", allow_blank=False),
    ]
    # Fold 0 and numeric group ids must not read as blank.
    read_dtype = str

    def __init__(self, csv_file: typing.TextIO):
        self.group_folds: dict[tuple[str, str], str] = {}

        super().__init__(csv_file=csv_file)

    def per_row_init_action(self, row: dict[str, str | None]) -> None:
        site = str(row["site"])
        group_id = str(row["group_id"])
        fold = str(row["fold"])
        if not (fold == HOLDOUT_TEST_FOLD or (fold.isascii() and fold.isdigit())):
            raise ValueError(
                f"Holdout fold must be '{HOLDOUT_TEST_FOLD}' or a non-negative"
                f" integer; got {fold!r} for group ({site}, {group_id})."
            )
        existing = self.group_folds.get((site, group_id))
        if existing is not None and existing != fold:
            raise ValueError(
                f"Group ({site}, {group_id}) is assigned to two folds: {existing!r} and {fold!r}."
            )
        self.group_folds[(site, group_id)] = fold

    def is_empty(self) -> bool:
        return len(self.group_folds) == 0

    def folds(self) -> set[str]:
        return set(self.group_folds.values())

    def validate_holdout_fold(self, holdout_fold: str) -> None:
        """
        Raise unless `holdout_fold` is "test" or a fold present in this spec.
        """
        if holdout_fold == HOLDOUT_TEST_FOLD:
            return
        if holdout_fold not in self.folds():
            raise ValueError(
                f"holdout_fold {holdout_fold!r} must be '{HOLDOUT_TEST_FOLD}' or one of"
                f" the folds in the holdout groups CSV: {sorted(self.folds())}."
            )

    def tag_in_duckdb(
        self,
        duck_conn: duckdb.DuckDBPyConnection,
        duck_table_name: str,
        holdout_fold: str,
        groups_read: set[tuple[str, str]] | None = None,
    ) -> None:
        """
        Add boolean columns to the given table, joining on (site, project_id):
        - is_holdout: the row's group is in the `holdout_fold` fold.
        - is_excluded: the row's group is in the test fold while
          `holdout_fold` is a CV fold, so the row is neither trained on
          nor evaluated.
        Every row is kept. Rows in no listed group are tagged FALSE for both.

        A listed group with no rows in the table would otherwise train
        silently. `groups_read` is the (site, project_id) set before label
        filtering: a group in it lost its rows to filtering, which is
        logged as a warning. Any other unmatched group in the test fold or
        in `holdout_fold` raises ValueError; one in another CV fold is
        training data in this run, and is logged as a warning.
        """
        self.validate_holdout_fold(holdout_fold)

        ordered_groups = sorted(self.group_folds.items())

        with duckdb_temp_table_name(duck_conn, base_name="holdout_groups") as groups_table_name:
            _create_groups_table(
                duck_conn,
                groups_table_name,
                {
                    "site": [site for (site, _), _ in ordered_groups],
                    "group_id": [group_id for (_, group_id), _ in ordered_groups],
                    "fold": [fold for _, fold in ordered_groups],
                },
            )

            # holdout_fold is validated above to be 'test' or a listed
            # integer string, so interpolating it is safe.
            duck_conn.execute(
                f"CREATE OR REPLACE TABLE {duck_table_name} AS"
                f" SELECT a.*,"
                f"  COALESCE(g.fold = '{holdout_fold}', FALSE) AS is_holdout,"
                f"  COALESCE(g.fold = '{HOLDOUT_TEST_FOLD}'"
                f"   AND '{holdout_fold}' <> '{HOLDOUT_TEST_FOLD}', FALSE) AS is_excluded"
                f" FROM {duck_table_name} a"
                f"  LEFT JOIN {groups_table_name} g"
                f"   ON a.site = g.site AND a.project_id = g.group_id"
            )

            matched_groups = {
                (str(site), str(group_id))
                for site, group_id in duck_conn.execute(
                    f"SELECT DISTINCT g.site, g.group_id"
                    f" FROM {duck_table_name} a"
                    f"  JOIN {groups_table_name} g"
                    f"   ON a.site = g.site AND a.project_id = g.group_id"
                ).fetchall()
            }

        if self.is_empty():
            return
        logger.log(
            logging.WARNING if not matched_groups else logging.INFO,
            "Holdout groups matched %s of %s listed group(s) in this dataset.",
            len(matched_groups),
            len(self.group_folds),
        )
        self._check_unmatched_groups(matched_groups, holdout_fold, groups_read or set())

    def _check_unmatched_groups(
        self,
        matched_groups: set[tuple[str, str]],
        holdout_fold: str,
        groups_read: set[tuple[str, str]],
    ) -> None:
        unmatched = sorted(set(self.group_folds) - matched_groups)
        filtered_out = [g for g in unmatched if g in groups_read]
        absent = [g for g in unmatched if g not in groups_read]
        evaluated_folds = {HOLDOUT_TEST_FOLD, holdout_fold}
        required = [g for g in absent if self.group_folds[g] in evaluated_folds]
        other = [g for g in absent if self.group_folds[g] not in evaluated_folds]

        def describe(groups: list[tuple[str, str]]) -> str:
            return ", ".join(
                f"({site}, {gid}): {self.group_folds[(site, gid)]}" for site, gid in groups
            )

        if required:
            raise ValueError(
                f"{len(required)} holdout group(s) in the test fold or holdout_fold"
                f" {holdout_fold!r} match no rows even before label filtering"
                f" (site, group_id): fold: {describe(required)}. Rows meant for them"
                f" would train. Check the group ids, and that MERMAID groups have a"
                f" mermaid_project_lookup_uri. Of the other unmatched groups,"
                f" {len(filtered_out)} lost every row to filtering, and {len(other)}"
                f" in other folds match nothing."
            )
        if other:
            logger.warning(
                "%d holdout group(s) in other folds match no rows in the data read"
                " (site, group_id): fold: %s",
                len(other),
                describe(other),
            )
        if filtered_out:
            logger.warning(
                "%d holdout group(s) lost every row to rollup, label filtering, image"
                " exclusion or dedupe (site, group_id): fold: %s",
                len(filtered_out),
                describe(filtered_out),
            )


class ValGroups(CsvSpec):
    """
    A CSV-defined spec naming the groups whose rows form the val
    (early-stopping) set. A group is a (site, project_id) pair, as in
    `HoldoutGroups`.

    Columns: site, group_id. Repeated rows are idempotent. The 'unknown'
    group (MERMAID images with no project) is rejected: its rows always train.
    """

    column_specs = [
        ColumnSpec(name="site", allow_blank=False),
        ColumnSpec(name="group_id", allow_blank=False),
    ]
    # Numeric group ids must not read as blank.
    read_dtype = str

    def __init__(self, csv_file: typing.TextIO):
        self.groups: set[tuple[str, str]] = set()

        super().__init__(csv_file=csv_file)

    def per_row_init_action(self, row: dict[str, str | None]) -> None:
        site = str(row["site"])
        group_id = str(row["group_id"])
        if group_id == UNKNOWN_GROUP:
            raise ValueError(
                f"Val group ({site}, {group_id}) is the group of images with no project;"
                f" those rows always train and cannot be val."
            )
        self.groups.add((site, group_id))

    def is_empty(self) -> bool:
        return len(self.groups) == 0

    def check_disjoint(self, holdout: HoldoutGroups, holdout_fold: str) -> None:
        """
        Raise ValueError naming every val group in the holdout test fold, and,
        when `holdout_fold` is a CV fold, every val group in that fold.
        """
        in_test = sorted(g for g in self.groups if holdout.group_folds.get(g) == HOLDOUT_TEST_FOLD)
        in_fold = []
        if holdout_fold != HOLDOUT_TEST_FOLD:
            in_fold = sorted(g for g in self.groups if holdout.group_folds.get(g) == holdout_fold)
        problems: list[str] = []
        if in_test:
            problems.append(
                f"{len(in_test)} val group(s) are in the holdout test fold (site, group_id):"
                f" {_describe_groups(in_test)}."
            )
        if in_fold:
            problems.append(
                f"{len(in_fold)} val group(s) are in holdout_fold {holdout_fold!r}"
                f" (site, group_id): {_describe_groups(in_fold)}. CV with grouped val"
                f" needs a val file per fold that lists none of that fold's groups."
            )
        if problems:
            raise ValueError(" ".join(problems))

    def tag_in_duckdb(
        self,
        duck_conn: duckdb.DuckDBPyConnection,
        duck_table_name: str,
        groups_read: set[tuple[str, str]] | None = None,
    ) -> None:
        """
        Add a boolean is_val column to the given table: the row's
        (site, project_id) is listed and the row is not is_holdout. The table
        must already have is_holdout (see `HoldoutGroups.tag_in_duckdb`).
        Every row is kept.

        A listed group with no rows would leave the val set silently smaller.
        `groups_read` is the (site, project_id) set before label filtering: a
        group in it lost its rows to filtering, which is logged as a warning;
        any other unmatched group raises ValueError.
        """
        ordered_groups = sorted(self.groups)

        with duckdb_temp_table_name(duck_conn, base_name="val_groups") as groups_table_name:
            _create_groups_table(
                duck_conn,
                groups_table_name,
                {
                    "site": [site for site, _ in ordered_groups],
                    "group_id": [group_id for _, group_id in ordered_groups],
                },
            )

            # Listed groups are unique, so the join cannot fan rows out.
            duck_conn.execute(
                f"CREATE OR REPLACE TABLE {duck_table_name} AS"
                f" SELECT a.*, (g.site IS NOT NULL AND NOT a.is_holdout) AS is_val"
                f" FROM {duck_table_name} a"
                f"  LEFT JOIN {groups_table_name} g"
                f"   ON a.site = g.site AND a.project_id = g.group_id"
            )

            matched_groups = {
                (str(site), str(group_id))
                for site, group_id in duck_conn.execute(
                    f"SELECT DISTINCT g.site, g.group_id"
                    f" FROM {duck_table_name} a"
                    f"  JOIN {groups_table_name} g"
                    f"   ON a.site = g.site AND a.project_id = g.group_id"
                ).fetchall()
            }

        if self.is_empty():
            return
        logger.log(
            logging.WARNING if not matched_groups else logging.INFO,
            "Val groups matched %s of %s listed group(s) in this dataset.",
            len(matched_groups),
            len(self.groups),
        )
        unmatched = sorted(self.groups - matched_groups)
        read = groups_read or set()
        filtered_out = [g for g in unmatched if g in read]
        absent = [g for g in unmatched if g not in read]
        if absent:
            raise ValueError(
                f"{len(absent)} val group(s) match no rows even before label filtering"
                f" (site, group_id): {_describe_groups(absent)}. Check the group ids, and"
                f" that MERMAID groups have a mermaid_project_lookup_uri."
            )
        if filtered_out:
            logger.warning(
                "%d val group(s) lost every row to rollup, label filtering, image"
                " exclusion or dedupe (site, group_id): %s",
                len(filtered_out),
                _describe_groups(filtered_out),
            )
