"""Build a raw-image CoralNet manifest parquet from the ETL parquets.

Example
-------
    uv run python scripts/build_coralnet_manifest.py \\
        --annotations-uri s3://dev-datamermaid-sm-sources/etl-outputs/coralnet/<run>/coralnet_annotations_<run>.parquet \\
        --images-uri      s3://dev-datamermaid-sm-sources/etl-outputs/coralnet/<run>/coralnet_images_<run>.parquet \\
        --sources-csv     sagemaker/configs/coralnet_top108_full/sources.csv \\
        --output-uri      s3://.../coralnet_classifier_manifest_<run>.parquet
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys

import duckdb

from mermaid_classifier.coralnet.manifest import (
    build_manifest_relation,
    summarize_build,
    write_manifest,
)

log = logging.getLogger("build_coralnet_manifest")


def _configure_duckdb_s3(conn: duckdb.DuckDBPyConnection, region: str) -> None:
    """Load the httpfs extension and configure credential_chain-based S3 auth.

    Mirrors the pattern in mermaid_classifier/pyspacer/dataset.py (~lines 711-749).
    The credential_chain provider resolves AWS credentials from the environment or
    instance role at query time — no key/secret arguments are needed here.
    """
    try:
        conn.load_extension("httpfs")
    except duckdb.IOException:
        # Extension not yet installed in this DuckDB installation.
        conn.install_extension("httpfs")
        conn.load_extension("httpfs")

    conn.execute(
        f"CREATE OR REPLACE SECRET secret ( TYPE s3, PROVIDER credential_chain, REGION '{region}')"
    )


def _load_source_ids(sources_csv: str | None, source_ids: str | None) -> list[str] | None:
    if sources_csv:
        with open(sources_csv, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "id" not in reader.fieldnames:
                raise ValueError(
                    f"--sources-csv '{sources_csv}' must contain an 'id' column"
                    f" (got columns: {list(reader.fieldnames or [])})"
                )
            return [r["id"] for r in reader if r["id"]]
    if source_ids:
        return [s.strip() for s in source_ids.split(",") if s.strip()]
    return None


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    p.add_argument("--annotations-uri", required=True)
    p.add_argument("--images-uri", required=True)
    p.add_argument("--audit-uri")
    p.add_argument("--output-uri", required=True)
    p.add_argument(
        "--aws-region", default="us-east-1", help="AWS region for S3 access (default: us-east-1)"
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sources-csv")
    g.add_argument("--source-ids")
    args = p.parse_args(argv)

    source_ids = _load_source_ids(args.sources_csv, args.source_ids)
    conn = duckdb.connect()

    s3_uris = [args.annotations_uri, args.images_uri, args.output_uri]
    if args.audit_uri:
        s3_uris.append(args.audit_uri)
    if any(uri.startswith("s3://") for uri in s3_uris):
        _configure_duckdb_s3(conn, args.aws_region)

    summary = summarize_build(conn, args.annotations_uri, args.images_uri, source_ids)
    log.info(
        "points_in=%d kept=%d dropped_invalid_image=%d sources_out=%d",
        summary["points_in"],
        summary["points_kept"],
        summary["points_dropped_invalid_image"],
        summary["sources_out"],
    )
    if args.audit_uri:
        flagged = conn.execute(
            "SELECT source_id FROM read_parquet(?) WHERE is_complete = FALSE OR image_count_match = FALSE",
            [args.audit_uri],
        ).df()
        if not flagged.empty:
            log.warning(
                "audit-flagged sources (reported, not filtered): %s",
                list(flagged["source_id"]),
            )

    if summary["points_kept"] == 0:
        log.error("Manifest would be empty after filtering; not writing.")
        sys.exit(1)

    rel = build_manifest_relation(conn, args.annotations_uri, args.images_uri, source_ids)
    write_manifest(rel, args.output_uri)
    log.info("Wrote manifest: %s", args.output_uri)


if __name__ == "__main__":
    main()
