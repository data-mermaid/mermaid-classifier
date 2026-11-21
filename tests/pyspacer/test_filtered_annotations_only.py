"""TrainingDataset.filtered_annotations_only builds the filtered annotations
table without listing or downloading feature vectors."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb
import pandas as pd
from support.settings import override_settings

from mermaid_classifier.pyspacer.dataset import TrainingDataset
from mermaid_classifier.pyspacer.options import DatasetOptions


def _write_parquet(directory: str, name: str, df: pd.DataFrame) -> str:
    path = str(Path(directory) / name)
    conn = duckdb.connect()
    conn.register("df", df)
    conn.execute(f"COPY (SELECT * FROM df) TO '{path}' (FORMAT parquet)")
    conn.close()
    return path


class FilteredAnnotationsOnlyTest(unittest.TestCase):
    def test_builds_table_without_touching_features(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        mermaid = pd.DataFrame(
            {
                "image_id": ["i1", "i1", "i2"],
                "row": [1, 2, 1],
                "col": [1, 2, 1],
                "benthic_attribute_id": ["ba_a", "ba_b", "ba_a"],
                "growth_form_id": ["", "", ""],
            }
        )
        lookup = pd.DataFrame({"image_id": ["i1", "i2"], "project_id": ["p1", "p2"]})
        options = DatasetOptions(
            mermaid_project_lookup_uri=_write_parquet(tmp.name, "lookup.parquet", lookup)
        )
        with (
            override_settings(
                aws_anonymous="True",
                mermaid_annotations_parquet_pattern=_write_parquet(
                    tmp.name, "mermaid.parquet", mermaid
                ),
            ),
            mock.patch.object(TrainingDataset, "feature_source_prefixes") as prefixes,
            mock.patch("s3fs.core.S3FileSystem.find") as find,
            mock.patch("mermaid_classifier.pyspacer.dataset.download_features_parallel") as dl,
        ):
            dataset = TrainingDataset.filtered_annotations_only(options)
            self.addCleanup(dataset.cleanup)
            rows = dataset.duck_conn.execute(
                "SELECT project_id, benthic_attribute_id, is_holdout, is_excluded"
                " FROM annotations ORDER BY project_id, row"
            ).fetchall()

        self.assertEqual(
            rows,
            [
                ("p1", "ba_a", False, False),
                ("p1", "ba_b", False, False),
                ("p2", "ba_a", False, False),
            ],
        )
        prefixes.assert_not_called()
        find.assert_not_called()
        dl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
