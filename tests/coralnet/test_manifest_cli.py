"""Unit tests for scripts/build_coralnet_manifest.py."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

# Make scripts/ importable (same pattern as tests/pyspacer/test_build_feature_bucket.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_coralnet_manifest import main  # noqa: E402

from coralnet.test_manifest import _annotations, _images  # noqa: E402


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
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
