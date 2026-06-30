"""Unit tests for scripts/build_coralnet_manifest.py."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pyarrow.parquet as pq

# Make scripts/ importable (same pattern as tests/pyspacer/test_build_feature_bucket.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_coralnet_manifest import _configure_duckdb_s3, main  # noqa: E402

from coralnet.test_manifest import _annotations, _images  # noqa: E402


class ConfigureDuckdbS3Test(unittest.TestCase):
    """_configure_duckdb_s3 should load httpfs and create the credential_chain secret offline.

    The credential_chain provider validates credentials at secret-creation time, so
    executing the real SQL without AWS creds would raise. We therefore verify that
    _configure_duckdb_s3 issues the correct SQL rather than running live against AWS.
    """

    def test_secret_sql_is_well_formed(self):
        """The credential_chain CREATE SECRET SQL must use TYPE s3, PROVIDER credential_chain, and the given region."""
        conn = MagicMock(spec=duckdb.DuckDBPyConnection)
        _configure_duckdb_s3(conn, "ap-southeast-2")
        # httpfs must be loaded first.
        conn.load_extension.assert_called_once_with("httpfs")
        # A single execute call must follow.
        conn.execute.assert_called_once()
        sql: str = conn.execute.call_args[0][0]
        self.assertIn("TYPE s3", sql)
        self.assertIn("PROVIDER credential_chain", sql)
        self.assertIn("ap-southeast-2", sql)
        self.assertIn("CREATE OR REPLACE SECRET", sql)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp = self.tmp_dir.name
        self.addCleanup(self.tmp_dir.cleanup)
        self.ann = os.path.join(self.tmp, "ann.parquet")
        pq.write_table(_annotations(), self.ann)
        self.img = os.path.join(self.tmp, "img.parquet")
        pq.write_table(_images(), self.img)
        self.out = os.path.join(self.tmp, "manifest.parquet")

    def test_writes_manifest(self):
        main(
            [
                "--annotations-uri",
                self.ann,
                "--images-uri",
                self.img,
                "--output-uri",
                self.out,
            ]
        )
        self.assertTrue(os.path.exists(self.out))
        df = duckdb.connect().sql(f"SELECT * FROM read_parquet('{self.out}')").df()
        self.assertEqual(sorted(df["image_id"]), ["a", "c"])
        self.assertFalse(df["uses_resized_image"].any())

    def test_audit_uri_writes_manifest_and_reports_flagged(self):
        # Write an audit parquet with one flagged source (source_id=2) and one clean (source_id=1).
        audit_path = os.path.join(self.tmp, "audit.parquet")
        import pyarrow as pa

        audit_table = pa.table(
            {
                "source_id": pa.array([1, 2], pa.int32()),
                "is_complete": pa.array([True, False], pa.bool_()),
                "image_count_match": pa.array([True, True], pa.bool_()),
            }
        )
        pq.write_table(audit_table, audit_path)

        main(
            [
                "--annotations-uri",
                self.ann,
                "--images-uri",
                self.img,
                "--output-uri",
                self.out,
                "--audit-uri",
                audit_path,
            ]
        )
        # Manifest must still be written (audit reports, does not filter).
        self.assertTrue(os.path.exists(self.out))
        df = duckdb.connect().sql(f"SELECT * FROM read_parquet('{self.out}')").df()
        self.assertEqual(sorted(df["image_id"]), ["a", "c"])
