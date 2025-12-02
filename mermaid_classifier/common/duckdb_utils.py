import typing


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
    tuples_with_unique_values = list(
        duck_conn.execute(
            f"SELECT DISTINCT {column_name} FROM {duck_table_name}"
        ).fetchall()
    )
    unique_values = [tup[0] for tup in tuples_with_unique_values]

    entries = [
        (value, transform_func(value))
        for value in unique_values
    ]
    entries_str = ', '.join(
        f"('{v1}', {'NULL' if v2 is None else '\''+v2+'\''})"
        for v1, v2 in entries
    )

    # Create a DuckDB table for the transform spec.
    duck_conn.execute(
        f"CREATE TABLE transform_table"
        f" ({column_name} VARCHAR,"
        f"  transformed_{column_name} VARCHAR)"
    )
    duck_conn.execute(
        f"INSERT INTO transform_table VALUES"
        f" {entries_str}"
    )

    # Use JOIN ... USING to apply the transform.
    duck_conn.execute(
        f"CREATE OR REPLACE TABLE {duck_table_name} AS"
        f" SELECT *"
        f" FROM {duck_table_name}"
        f"  JOIN transform_table"
        f"  USING ({column_name})"
    )

    # Post-transform column becomes the new column.
    duckdb_replace_column(
        duck_conn,
        duck_table_name,
        column_name,
        f"transformed_{column_name}",
    )
    # Don't need the temporary table anymore.
    duck_conn.execute(
        f"DROP TABLE transform_table"
    )


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
    tuples_with_unique_values = list(
        duck_conn.execute(
            f"SELECT DISTINCT {base_column_name} FROM {duck_table_name}"
        ).fetchall()
    )
    unique_values = [tup[0] for tup in tuples_with_unique_values]
    entries = [
        (value, base_to_new_func(value))
        for value in unique_values
    ]
    entries_str = ', '.join(
        f"('{v1}', {'NULL' if v2 is None else '\''+v2+'\''})"
        for v1, v2 in entries
    )

    # Create a DuckDB table for the transform spec.
    duck_conn.execute(
        f"CREATE TABLE transform_table"
        f" ({base_column_name} VARCHAR,"
        f"  {new_column_name} VARCHAR)"
    )
    duck_conn.execute(
        f"INSERT INTO transform_table VALUES"
        f" {entries_str}"
    )

    # Use JOIN ... USING to add the new column.
    # https://duckdb.org/docs/stable/sql/query_syntax/from#conditional-joins
    duck_conn.execute(
        f"CREATE OR REPLACE TABLE {duck_table_name} AS"
        f" SELECT *"
        f" FROM {duck_table_name}"
        f"  JOIN transform_table"
        f"  USING ({base_column_name})"
    )

    # Don't need the temporary table anymore.
    duck_conn.execute(
        f"DROP TABLE transform_table"
    )


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
    tuples_with_unique_values = list(
        duck_conn.execute(
            f"SELECT DISTINCT {column_name} FROM {duck_table_name}"
        ).fetchall()
    )
    unique_values = [tup[0] for tup in tuples_with_unique_values]
    entries = [
        (value, inclusion_func(value))
        for value in unique_values
    ]
    entries_str = ', '.join(
        f"('{value}', {'true' if included else 'false'})"
        for value, included in entries
    )

    duck_conn.execute(
        f"CREATE TABLE {duck_table_name}_filter"
        f" ({column_name} VARCHAR,"
        f"  included BOOLEAN)"
    )
    duck_conn.execute(
        f"INSERT INTO {duck_table_name}_filter VALUES"
        f" {entries_str}"
    )

    # Use JOIN to determine which annotations to keep, and
    # WHERE to filter things down.
    duck_conn.execute(
        f"CREATE OR REPLACE TABLE {duck_table_name} AS"
        f" SELECT *"
        f" FROM {duck_table_name}"
        f"  JOIN {duck_table_name}_filter"
        f"  USING ({column_name})"
        f" WHERE included = true"
    )
    # Don't need the included column anymore.
    duck_conn.execute(
        f"ALTER TABLE {duck_table_name} DROP included"
    )
    # Or the *_filter table.
    duck_conn.execute(
        f"DROP TABLE {duck_table_name}_filter"
    )


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
