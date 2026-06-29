"""Build a raw-image CoralNet training manifest from the ETL parquets.

The manifest is a per-annotation-point parquet whose schema matches the
segmentation training manifest, but every row references the *raw* original
image (``image_s3_key`` = the image's ``s3_key``, ``uses_resized_image`` =
False). It is a pure "which images/points" dataset definition: label mapping,
rollups, and filtering stay at train time.
"""

from __future__ import annotations

import duckdb

MANIFEST_COLUMNS: list[str] = [
    "source_id",
    "image_id",
    "row",
    "col",
    "coralnet_id",
    "source_label_name",
    "image_s3_key",
    "load_width",
    "load_height",
    "uses_resized_image",
]


def _source_filter_sql(source_ids: list[str] | None) -> str:
    if not source_ids:
        return ""
    ids = ", ".join(str(int(s)) for s in source_ids)  # int() guards against injection
    return f" AND a.source_id IN ({ids})"


def build_manifest_relation(
    conn: duckdb.DuckDBPyConnection,
    annotations_uri: str,
    images_uri: str,
    source_ids: list[str] | None = None,
) -> duckdb.DuckDBPyRelation:
    """Join annotations x images, keep valid raw images, project to the schema.

    Inclusion: ``header_status = 'ok'`` and ``s3_key`` non-null. No
    feature-vector / audit / label filtering.
    """
    sql = f"""
        SELECT
            a.source_id                       AS source_id,
            a.image_id                        AS image_id,
            a.row                             AS row,
            a.col                             AS col,
            a.coralnet_id                     AS coralnet_id,
            CAST(NULL AS VARCHAR)             AS source_label_name,
            i.s3_key                          AS image_s3_key,
            i.width                           AS load_width,
            i.height                          AS load_height,
            CAST(FALSE AS BOOLEAN)            AS uses_resized_image
        FROM read_parquet('{annotations_uri}') a
        JOIN read_parquet('{images_uri}') i
          ON a.source_id = i.source_id AND a.image_id = i.image_id
        WHERE i.header_status = 'ok'
          AND i.s3_key IS NOT NULL
          {_source_filter_sql(source_ids)}
    """
    return conn.sql(sql)


def write_manifest(relation: duckdb.DuckDBPyRelation, output_uri: str) -> None:
    """Write the relation to a parquet file (local path or s3:// URI)."""
    relation.write_parquet(output_uri)


def summarize_build(
    conn: duckdb.DuckDBPyConnection,
    annotations_uri: str,
    images_uri: str,
    source_ids: list[str] | None = None,
) -> dict[str, int]:
    """Return point and source counts for the manifest build.

    Returns a dict with keys: points_in, points_kept, points_dropped_no_image,
    sources_out.
    """
    source_filter = _source_filter_sql(source_ids)
    points_in = conn.sql(
        f"SELECT count(*) FROM read_parquet('{annotations_uri}') a WHERE TRUE {source_filter}"
    ).fetchone()[0]
    rel = build_manifest_relation(conn, annotations_uri, images_uri, source_ids)
    points_kept = rel.count("*").fetchone()[0]
    sources_out = rel.aggregate("count(DISTINCT source_id)").fetchone()[0]
    return {
        "points_in": int(points_in),
        "points_kept": int(points_kept),
        "points_dropped_no_image": int(points_in) - int(points_kept),
        "sources_out": int(sources_out),
    }
