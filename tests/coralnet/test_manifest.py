import os
import shutil
import tempfile
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from mermaid_classifier.coralnet.manifest import (
    MANIFEST_COLUMNS,
    build_manifest_relation,
    summarize_build,
)


def _write(tmp, name, table):
    path = os.path.join(tmp, name)
    pq.write_table(table, path)
    return path


def _annotations():
    return pa.table(
        {
            "source_id": pa.array([1, 1, 2], pa.int32()),
            "image_id": pa.array(["a", "b", "c"], pa.string()),
            "row": pa.array([10, 20, 30], pa.int32()),
            "col": pa.array([11, 21, 31], pa.int32()),
            "coralnet_id": pa.array([100, 100, 200], pa.int32()),
            "status": pa.array(["Confirmed", "Confirmed", None], pa.string()),
        }
    )


def _images():
    # image 'b' has a failed header and must be dropped.
    return pa.table(
        {
            "source_id": pa.array([1, 1, 2], pa.int32()),
            "image_id": pa.array(["a", "b", "c"], pa.string()),
            "s3_key": pa.array(
                [
                    "coralnet-public-images/s1/images/a.jpg",
                    "coralnet-public-images/s1/images/b.jpg",
                    "coralnet-public-images/s2/images/c.jpg",
                ],
                pa.string(),
            ),
            "width": pa.array([4000, 4000, 800], pa.int32()),
            "height": pa.array([3000, 3000, 600], pa.int32()),
            "longest_edge": pa.array([4000, 4000, 800], pa.int32()),
            "file_size": pa.array([1, 1, 1], pa.int64()),
            "needs_resize": pa.array([True, True, False], pa.bool_()),
            "header_status": pa.array(["ok", "header_read_failed", "ok"], pa.string()),
            "error_message": pa.array([None, "bad", None], pa.string()),
        }
    )


class BuildManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.ann = _write(self.tmp, "ann.parquet", _annotations())
        self.img = _write(self.tmp, "img.parquet", _images())
        self.conn = duckdb.connect()

    def test_schema_and_filtering(self):
        rel = build_manifest_relation(self.conn, self.ann, self.img)
        self.assertEqual(rel.columns, MANIFEST_COLUMNS)
        df = rel.df()
        # 'b' dropped (header_read_failed) -> points a and c remain.
        self.assertEqual(sorted(df["image_id"]), ["a", "c"])
        # raw key carried through; uses_resized_image always False.
        self.assertTrue((~df["uses_resized_image"]).all())
        row_a = df[df["image_id"] == "a"].iloc[0]
        self.assertEqual(row_a["image_s3_key"], "coralnet-public-images/s1/images/a.jpg")
        self.assertEqual(row_a["load_width"], 4000)
        self.assertEqual(row_a["load_height"], 3000)
        self.assertTrue(df["source_label_name"].isna().all())

    def test_source_subsetting(self):
        rel = build_manifest_relation(self.conn, self.ann, self.img, source_ids=["2"])
        df = rel.df()
        self.assertEqual(list(df["image_id"]), ["c"])
        self.assertEqual(list(df["source_id"]), [2])

    def test_summary_counts(self):
        s = summarize_build(self.conn, self.ann, self.img)
        self.assertEqual(s["points_in"], 3)
        self.assertEqual(s["points_kept"], 2)
        self.assertEqual(s["points_dropped_invalid_image"], 1)
        self.assertEqual(s["sources_out"], 2)

    def test_source_id_non_integer_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_manifest_relation(self.conn, self.ann, self.img, source_ids=["abc"])
        self.assertIn("integer-valued", str(ctx.exception))

    def test_null_s3_key_drops_annotation(self):
        """
        An image row with header_status='ok' but s3_key=NULL must be excluded
        from the manifest (the second half of the inclusion rule).
        Its annotation point must not appear in the output.
        """
        # Minimal images table: one valid image 'a' and one with null s3_key 'z'.
        images_null_key = pa.table(
            {
                "source_id": pa.array([1, 1], pa.int32()),
                "image_id": pa.array(["a", "z"], pa.string()),
                "s3_key": pa.array(
                    ["coralnet-public-images/s1/images/a.jpg", None],
                    pa.string(),
                ),
                "width": pa.array([4000, 800], pa.int32()),
                "height": pa.array([3000, 600], pa.int32()),
                "longest_edge": pa.array([4000, 800], pa.int32()),
                "file_size": pa.array([1, 1], pa.int64()),
                "needs_resize": pa.array([False, False], pa.bool_()),
                "header_status": pa.array(["ok", "ok"], pa.string()),
                "error_message": pa.array([None, None], pa.string()),
            }
        )
        # Matching annotations: one point for 'a', one for 'z'.
        annotations_null_key = pa.table(
            {
                "source_id": pa.array([1, 1], pa.int32()),
                "image_id": pa.array(["a", "z"], pa.string()),
                "row": pa.array([10, 20], pa.int32()),
                "col": pa.array([11, 21], pa.int32()),
                "coralnet_id": pa.array([100, 200], pa.int32()),
                "status": pa.array(["Confirmed", "Confirmed"], pa.string()),
            }
        )
        ann_path = _write(self.tmp, "ann_null.parquet", annotations_null_key)
        img_path = _write(self.tmp, "img_null.parquet", images_null_key)
        conn = duckdb.connect()
        rel = build_manifest_relation(conn, ann_path, img_path)
        df = rel.df()
        # Only 'a' should survive; 'z' must be dropped because s3_key is NULL.
        self.assertEqual(list(df["image_id"]), ["a"])
        self.assertNotIn("z", df["image_id"].values)
