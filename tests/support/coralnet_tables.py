"""The CoralNet ETL parquet tables the manifest builder reads.

There is no production writer for these, so the shape lives here rather
than being derived from one.
"""

import pyarrow as pa


def annotations_table():
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


def images_table():
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
