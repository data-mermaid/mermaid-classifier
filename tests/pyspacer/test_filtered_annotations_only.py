"""TrainingDataset.filtered_annotations_only builds the filtered annotations
table without fetching a feature index or a feature file."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from segdata_spec.manifest import FeatureSetPin

from mermaid_classifier.pyspacer.dataset import TrainingDataset
from mermaid_classifier.pyspacer.options import DatasetOptions, PinnedDataset
from pyspacer.segdata_fixtures import (
    FakeS3,
    Image,
    label_row,
    load_training_dataset,
    publish_dataset,
)

# CoralNet label 100 maps to ba-1, 200 to ba-2.
CORALNET_MAPPING = [
    {
        "provider_id": str(label_id),
        "provider_label": f"label {label_id}",
        "benthic_attribute_id": ba,
        "benthic_attribute_name": ba,
        "growth_form_id": None,
        "growth_form_name": None,
    }
    for label_id, ba in ((100, "ba-1"), (200, "ba-2"))
]


class FilteredAnnotationsOnlyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, name: str, text: str) -> str:
        path = self.tmp / name
        path.write_text(text)
        return str(path)

    def test_rolls_up_filters_and_tags_without_touching_features(self):
        fake = FakeS3()
        pin = publish_dataset(
            fake,
            "coralnet-raw",
            [Image("i1", "s1"), Image("i2", "s2")],
            [("i1", 1, 1, 100), ("i1", 2, 2, 200), ("i2", 1, 1, 200)],
            [label_row(100), label_row(200)],
        )
        options = DatasetOptions(
            datasets=[
                PinnedDataset(
                    pin=pin,
                    features=FeatureSetPin(
                        dataset=pin, family="efficientnet-v1", manifest_sha256="0"
                    ),
                    labels="coralnet",
                )
            ],
            # ba-2 rolls up to ba-9, and only ba-9 is kept.
            label_rollup_spec_csv=self._write(
                "rollups.csv", "from_ba_id,from_gf_id,to_ba_id,to_gf_id\nba-2,,ba-9,\n"
            ),
            included_labels_csv=self._write("included.csv", "ba_id,gf_id\nba-9,\n"),
            holdout_groups_csv=self._write("holdout.csv", "site,group_id,fold\ncoralnet,2,test\n"),
        )
        refuse = AssertionError("filtered_annotations_only must not touch feature sets")
        with (
            mock.patch.object(TrainingDataset, "fetch_feature_indexes", side_effect=refuse),
            mock.patch.object(TrainingDataset, "sync_feature_vectors", side_effect=refuse),
            mock.patch("mermaid_classifier.pyspacer.dataset.sync_feature_set", side_effect=refuse),
        ):
            dataset = load_training_dataset(
                fake,
                options,
                CORALNET_MAPPING,
                filtered_only=True,
                feature_cache_dir=str(self.tmp / "cache"),
            )
        self.addCleanup(dataset.cleanup)

        rows = dataset.duck_conn.execute(
            "SELECT image_id, benthic_attribute_id, is_holdout FROM annotations ORDER BY image_id"
        ).fetchall()
        self.assertEqual(rows, [("i1", "ba-9", False), ("i2", "ba-9", True)])
        self.assertEqual(fake.feature_gets(), [])


if __name__ == "__main__":
    unittest.main()
