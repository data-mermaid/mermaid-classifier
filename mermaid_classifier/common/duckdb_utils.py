from contextlib import contextmanager
import typing
import uuid

import duckdb
import pandas as pd


@contextmanager
def duckdb_temp_table_name(
    duck_conn: duckdb.DuckDBPyConnection,
    base_name: str | None = None,
):
    """
    Context manager for temporary DuckDB table names with automatic cleanup.

    The end of this CM cleans up the table if it exists.
    However, the caller is responsible for table creation, because it's not
    easy for this function to accommodate the different CREATE TABLE
    statements one might want to use. Such DuckDB statements could even
    reference local Python variables, such as "CREATE TABLE my_table
    AS SELECT * FROM my_dataframe" (in this case, my_dataframe).

    So, think of this CM as guaranteeing the availability of the table name
    for the duration of the CM, and not longer than that.
    """
    table_name = f"temp_{base_name or uuid.uuid4().hex[:8]}"

    try:
        # The wrapped code can create a table of this name and do anything
        # with that table.
        yield table_name
    finally:
        # Clean up the table regardless of whether the wrapped code
        # runs without errors, or whether the table was even created.
        duck_conn.execute(f"DROP TABLE IF EXISTS {table_name}")


@contextmanager
def _duckdb_temp_transform_table(
    duck_conn: duckdb.DuckDBPyConnection,
    duck_table_name: str,
    source_column_name: str,
    target_column_name: str,
    transform_func: typing.Callable,
):
    """
    Internal helper: given an existing DuckDB table and column, creates a
    temp table mapping that column's values to a new column,
    using the provided Python function.

    Yields the name of the created temp table, and then ensures the temp
    table is cleaned up.
    """
    # Get all unique values of the existing column.
    tuples_with_unique_values = list(
        duck_conn.execute(
            f"SELECT DISTINCT {source_column_name} FROM {duck_table_name}"
        ).fetchall()
    )
    unique_values = [tup[0] for tup in tuples_with_unique_values]

    # Build mapping DataFrame using pandas. This is more efficient and less
    # error-prone for inserting values compared to building a large
    # INSERT INTO statement.
    mapping_df = pd.DataFrame({
        source_column_name: unique_values,
        target_column_name: [transform_func(v) for v in unique_values]
    })

    with duckdb_temp_table_name(duck_conn) as temp_table_name:
        duck_conn.execute(
            f"CREATE TABLE {temp_table_name} AS SELECT * FROM mapping_df"
        )
        yield temp_table_name


@contextmanager
def duckdb_allow_nullable_column_join(
    duck_conn: duckdb.DuckDBPyConnection,
    duck_table_names: list[str],
    column_name: str,
    null_placeholder: str = 'NULL_PLACEHOLDER',
):
    """
    In the given column of the given tables, temporarily stand in placeholder
    values instead of NULL for the duration of this context manager.
    This way, if a JOIN is done on the column within this context manager,
    the NULL values will be eligible for joining like any other value.
    Normally, NULL values mean the row will instead get dropped from the
    join result entirely.

    null_placeholder is the placeholder value, and must not collide with real
    non-null values in the column.
    """
    # Apply placeholder.
    for table_name in duck_table_names:
        duckdb_replace_value_in_column(
            duck_conn,
            table_name,
            column_name,
            None,
            null_placeholder,
        )

    yield

    # Un-apply placeholder.
    for table_name in duck_table_names:
        duckdb_replace_value_in_column(
            duck_conn,
            table_name,
            column_name,
            null_placeholder,
            None,
        )


def duckdb_replace_column(
    duck_conn: duckdb.DuckDBPyConnection,
    duck_table_name: str,
    column_name: str,
    new_values_column_name: str,
):
    """
    Replace the column_name column
    with the new_values_column_name column.
    """
    # Drop the original column.
    # https://duckdb.org/docs/stable/sql/statements/alter_table
    duck_conn.execute(
        f"ALTER TABLE {duck_table_name} DROP {column_name}"
    )
    # New-values column becomes the new column.
    duck_conn.execute(
        f"ALTER TABLE {duck_table_name}"
        f" RENAME {new_values_column_name} TO {column_name}"
    )


def duckdb_transform_column(
    duck_conn: duckdb.DuckDBPyConnection,
    duck_table_name: str,
    column_name: str,
    transform_func: typing.Callable[[str|None], str|None],
):
    """
    Transform the values of the specified DuckDB column using the given
    function.
    """
    with _duckdb_temp_transform_table(
        duck_conn,
        duck_table_name,
        column_name,
        f'transformed_{column_name}',
        transform_func,
    ) as transform_table_name:

        with duckdb_allow_nullable_column_join(
            duck_conn,
            [duck_table_name, transform_table_name],
            column_name,
        ):

            # Use JOIN ... USING to apply the transform.
            duck_conn.execute(
                f"CREATE OR REPLACE TABLE {duck_table_name} AS"
                f" SELECT *"
                f" FROM {duck_table_name}"
                f"  JOIN {transform_table_name}"
                f"  USING ({column_name})"
            )

    # Post-transform column becomes the new column.
    duckdb_replace_column(
        duck_conn,
        duck_table_name,
        column_name,
        f'transformed_{column_name}',
    )


def duckdb_replace_value_in_column(
    duck_conn: duckdb.DuckDBPyConnection,
    duck_table_name: str,
    column_name: str,
    old_value: str|None,
    new_value: str|None,
):
    """
    Replace old_value with new_value in the column_name column.
    """
    if old_value is None:
        where_predicate = "IS NULL"
    else:
        where_predicate = f"= '{old_value}'"

    if new_value is None:
        new_value_sql = "NULL"
    else:
        new_value_sql = f"'{new_value}'"

    duck_conn.execute(
        f"UPDATE {duck_table_name}"
        f" SET {column_name} = {new_value_sql}"
        f" WHERE {column_name} {where_predicate}"
    )


def duckdb_add_column(
    duck_conn: duckdb.DuckDBPyConnection,
    duck_table_name: str,
    base_column_name: str,
    new_column_name: str,
    base_to_new_func: typing.Callable[[str|None], str|None],
):
    """
    Transform the values of the specified DuckDB 'base column' into
    values for a new column.
    """
    with _duckdb_temp_transform_table(
        duck_conn,
        duck_table_name,
        base_column_name,
        new_column_name,
        base_to_new_func,
    ) as transform_table_name:

        with duckdb_allow_nullable_column_join(
            duck_conn,
            [duck_table_name, transform_table_name],
            base_column_name,
        ):

            # Use JOIN ... USING to add the new column.
            # https://duckdb.org/docs/stable/sql/query_syntax/from#conditional-joins
            duck_conn.execute(
                f"CREATE OR REPLACE TABLE {duck_table_name} AS"
                f" SELECT *"
                f" FROM {duck_table_name}"
                f"  JOIN {transform_table_name}"
                f"  USING ({base_column_name})"
            )


def duckdb_filter_on_column(
    duck_conn: duckdb.DuckDBPyConnection,
    duck_table_name: str,
    column_name: str,
    inclusion_func: typing.Callable[[str|None], bool],
):
    """
    Filter the rows of the specified DuckDB table by passing the
    specified column through a boolean function.
    """
    with _duckdb_temp_transform_table(
        duck_conn,
        duck_table_name,
        column_name,
        'included',
        inclusion_func,
    ) as transform_table_name:

        with duckdb_allow_nullable_column_join(
            duck_conn,
            [duck_table_name, transform_table_name],
            column_name,
        ):

            # Use JOIN to determine which annotations to keep, and
            # WHERE to filter things down.
            #
            # SELECT only from the original table since we don't need
            # the included column after this.
            duck_conn.execute(
                f"CREATE OR REPLACE TABLE {duck_table_name} AS"
                f" SELECT {duck_table_name}.*"
                f" FROM {duck_table_name}"
                f"  JOIN {transform_table_name}"
                f"  USING ({column_name})"
                f" WHERE included = true"
            )


def duckdb_batched_rows(
    rows: duckdb.DuckDBPyRelation,
) -> typing.Generator[pd.core.series.Series, None, None]:
    """
    Reads from a DuckDB relation (chunkifying to avoid memory issues),
    and generates pandas dataframe rows.
    """
    # Fetch in chunks, not all at once, to avoid running out of
    # memory.
    # An example chunk size is 2048 rows x 12 columns.
    #
    # fetchmany() is similar, but returns lists of tuples instead.
    # A dataframe provides dict-like access which is a bit nicer.
    while True:
        dataframe = rows.fetch_df_chunk()
        if dataframe.shape[0] == 0:
            # Empty dataframe; no more chunks left.
            break

        # Some ways to inspect the pandas dataframe in a debugger:
        # dataframe
        # dataframe.columns    # Column names
        # dataframe.iloc[0]    # First row
        # set([row['growth_form_name']
        #     for _index, row in dataframe.iterrows()])
        # [(row['row'], row['col']) for _index, row in dataframe.iterrows()
        #  if row['image_id'] == '0032dba6-8357-42e2-bace-988f99032286']

        for _index, row in dataframe.iterrows():
            # With this generator behavior, the chunkifying detail
            # is invisible to the caller.
            yield row


def duckdb_grouped_rows(
    duck_conn: duckdb.DuckDBPyConnection,
    duck_table_name: str,
    grouping_column_names: list[str],
) -> typing.Generator[pd.core.series.Series, None, None]:
    """
    Fetch rows from the given DuckDB table in groups, with each group
    consisting of all rows that have a particular set of values in the
    given columns.
    """

    # Ordering by a set of columns should mean that, for any set of values
    # for those columns, all the rows with that set of values are all
    # together - which is our goal here.
    columns_str = ", ".join(grouping_column_names)
    rows = duck_conn.execute(
        f"SELECT * FROM {duck_table_name}"
        f" ORDER BY {columns_str}"
    )

    grouping_values = None
    group_rows = []

    for row in duckdb_batched_rows(rows):
        if grouping_values != [row[col] for col in grouping_column_names]:
            if grouping_values:
                # End of group.
                yield group_rows

            # Start of group.
            grouping_values = [row[col] for col in grouping_column_names]
            group_rows = []

        group_rows.append(row)

    # End of last group.
    yield group_rows
