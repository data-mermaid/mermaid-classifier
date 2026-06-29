import os
import tempfile
import unittest

import duckdb
import pyarrow.parquet as pq
from coralnet.test_manifest import _annotations, _images

from mermaid_classifier.coralnet.manifest import build_manifest_relation, write_manifest


class ManifestNormalizationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        ann = os.path.join(self.tmp, "a.parquet")
        pq.write_table(_annotations(), ann)
        img = os.path.join(self.tmp, "i.parquet")
        pq.write_table(_images(), img)
        self.manifest = os.path.join(self.tmp, "m.parquet")
        conn = duckdb.connect()
        write_manifest(build_manifest_relation(conn, ann, img), self.manifest)

    def test_normalized_columns_match_contract(self):
        # The exact SELECT used by read_coralnet_manifest (see implementation).
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE annotations AS SELECT"
            " row, col, CAST(image_id AS VARCHAR) AS image_id,"
            " CAST(coralnet_id AS VARCHAR) AS label_id,"
            " 'coralnet' AS site, 'b' AS bucket,"
            " CAST(source_id AS VARCHAR) AS project_id,"
            " 's' || CAST(source_id AS VARCHAR) || '/features/i' || image_id"
            "  || '.featurevector' AS feature_vector"
            f" FROM read_parquet('{self.manifest}')"
        )
        cols = [c[0] for c in conn.sql("DESCRIBE annotations").fetchall()]
        self.assertEqual(
            cols,
            [
                "row",
                "col",
                "image_id",
                "label_id",
                "site",
                "bucket",
                "project_id",
                "feature_vector",
            ],
        )
        fv = conn.sql("SELECT feature_vector FROM annotations WHERE image_id='a'").fetchone()[0]
        self.assertEqual(fv, "s1/features/ia.featurevector")
