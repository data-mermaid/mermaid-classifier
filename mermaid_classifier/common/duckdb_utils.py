from contextlib import contextmanager
import typing
import uuid

import pandas as pd


@contextmanager
def temp_duckdb_table(duck_conn, base_name=None):
    """Context manager for temporary DuckDB tables with automatic cleanup."""
    table_name = f"temp_{base_name or uuid.uuid4().hex[:8]}"
    try:
        yield table_name
    finally:
        try:
            duck_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        except Exception:
            pass  # Best effort cleanup


def _duckdb_apply_function_via_temp_table(
    duck_conn: 'DuckDBPyConnection',
    duck_table_name: str,
    source_column_name: str,
    target_column_name: str,
    transform_func: typing.Callable,
) -> str:
    """
    Internal helper: creates a temp table with source->target column mapping
    using the provided Python function.

    Returns the name of the created temp table.
    """
    # Get unique values
    tuples_with_unique_values = list(
        duck_conn.execute(
            f"SELECT DISTINCT {source_column_name} FROM {duck_table_name}"
        ).fetchall()
    )
    unique_values = [tup[0] for tup in tuples_with_unique_values]

    # Build mapping dataframe using pandas (more efficient than string building)
    mapping_df = pd.DataFrame({
        source_column_name: unique_values,
        target_column_name: [transform_func(v) for v in unique_values]
    })

    # Create temp table with safe name
    temp_table_name = f"temp_{uuid.uuid4().hex[:8]}"

    # Use DuckDB's native DataFrame support (safer and faster than string building)
    duck_conn.execute(
        f"CREATE TABLE {temp_table_name} AS SELECT * FROM mapping_df"
    )

    return temp_table_name


def duckdb_replace_column(
    duck_conn: 'DuckDBPyConnection',
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
    duck_conn: 'DuckDBPyConnection',
    duck_table_name: str,
    column_name: str,
    transform_func: typing.Callable[[str|None], str|None],
):
    """
    Transform the values of the specified DuckDB column using the given
    function.
    Only supports text columns.
    """
    transformed_col = f"transformed_{column_name}"
    temp_table = _duckdb_apply_function_via_temp_table(
        duck_conn, duck_table_name, column_name, transformed_col, transform_func
    )

    try:
        # Use JOIN ... USING to apply the transform.
        duck_conn.execute(
            f"CREATE OR REPLACE TABLE {duck_table_name} AS"
            f" SELECT *"
            f" FROM {duck_table_name}"
            f"  JOIN {temp_table}"
            f"  USING ({column_name})"
        )

        # Post-transform column becomes the new column.
        duckdb_replace_column(
            duck_conn,
            duck_table_name,
            column_name,
            transformed_col,
        )
    finally:
        # Cleanup temp table
        duck_conn.execute(f"DROP TABLE IF EXISTS {temp_table}")


def duckdb_replace_value_in_column(
    duck_conn: 'DuckDBPyConnection',
    duck_table_name: str,
    column_name: str,
    old_value: str|None,
    new_value: str|None,
):
    """
    Replace old_value with new_value in the column_name column.
    """
    if old_value is None:
        transform_func = lambda x: new_value if x is None else x
    else:
        transform_func = lambda x: new_value if x == old_value else x

    duckdb_transform_column(
        duck_conn=duck_conn,
        duck_table_name=duck_table_name,
        column_name=column_name,
        transform_func=transform_func,
    )


def duckdb_add_column(
    duck_conn: 'DuckDBPyConnection',
    duck_table_name: str,
    base_column_name: str,
    new_column_name: str,
    base_to_new_func: typing.Callable[[str|None], str|None],
):
    """
    Transform the values of the specified DuckDB 'base column' into
    values for a new column.
    Only supports text columns.
    """
    temp_table = _duckdb_apply_function_via_temp_table(
        duck_conn, duck_table_name, base_column_name, new_column_name, base_to_new_func
    )

    try:
        # Use JOIN ... USING to add the new column.
        # https://duckdb.org/docs/stable/sql/query_syntax/from#conditional-joins
        duck_conn.execute(
            f"CREATE OR REPLACE TABLE {duck_table_name} AS"
            f" SELECT *"
            f" FROM {duck_table_name}"
            f"  JOIN {temp_table}"
            f"  USING ({base_column_name})"
        )
    finally:
        # Cleanup temp table
        duck_conn.execute(f"DROP TABLE IF EXISTS {temp_table}")


def duckdb_filter_on_column(
    duck_conn: 'DuckDBPyConnection',
    duck_table_name: str,
    column_name: str,
    inclusion_func: typing.Callable[[str|None], bool],
):
    """
    Filter the rows of the specified DuckDB table by passing the
    specified column through a boolean function.
    Only supports text columns.
    """
    temp_table = _duckdb_apply_function_via_temp_table(
        duck_conn, duck_table_name, column_name, 'included', inclusion_func
    )

    try:
        # Use JOIN to determine which annotations to keep, and
        # WHERE to filter things down.
        duck_conn.execute(
            f"CREATE OR REPLACE TABLE {duck_table_name} AS"
            f" SELECT *"
            f" FROM {duck_table_name}"
            f"  JOIN {temp_table}"
            f"  USING ({column_name})"
            f" WHERE included = true"
        )
        # Don't need the included column anymore.
        duck_conn.execute(
            f"ALTER TABLE {duck_table_name} DROP included"
        )
    finally:
        # Cleanup temp table
        duck_conn.execute(f"DROP TABLE IF EXISTS {temp_table}")


def duckdb_batched_rows(
    rows: 'DuckDBPyRelation',
) -> typing.Generator['pandas.core.series.Series', None, None]:
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
    duck_conn: 'DuckDBPyConnection',
    duck_table_name: str,
    grouping_column_names: list[str],
) -> typing.Generator['pandas.core.series.Series', None, None]:

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
