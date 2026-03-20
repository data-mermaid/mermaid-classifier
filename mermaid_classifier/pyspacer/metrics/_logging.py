"""MLflow logging helpers for metrics artifacts."""

import tempfile

import duckdb
import mlflow
import pandas as pd

from mermaid_classifier.common.duckdb_utils import duckdb_temp_table_name


def log_dataframe(
    duck_conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    filestem: str,
):
    """Log a DataFrame as a CSV artifact to MLflow via DuckDB.

    MLflow's log_table() saves .json, which isn't easily inspectable
    with external tools like Excel. This uses DuckDB + log_text() to
    produce a proper .csv instead.
    """
    with duckdb_temp_table_name(duck_conn) as table_name:
        duck_conn.execute(
            f"CREATE TABLE {table_name} AS SELECT * FROM df"
        )

        with tempfile.NamedTemporaryFile(
            mode='w+t', suffix='.csv', delete_on_close=False,
        ) as f:
            # DuckDB will reopen the file, so close first.
            f.close()
            duck_conn.execute(f"COPY {table_name} TO '{f.name}'")

            # Need to open yet again to get the DuckDB-written contents.
            with open(f.name) as f_new:
                mlflow.log_text(f_new.read(), filestem + '.csv')

            # As this context manager finishes, the temp file will be
            # deleted.
