"""The manifest parquet must satisfy the contract read_coralnet_manifest depends on.

Two subsystems meet here: scripts/build_coralnet_manifest.py writes the parquet,
and TrainingDataset.read_coralnet_manifest normalizes it into the open-data
column layout. The assertion runs the real normalization against a manifest built
by the real writer, so a column renamed on either side fails.
"""

import os
import tempfile
import unittest
from unittest import mock

import duckdb
import pyarrow.parquet as pq
from support.coralnet_tables import annotations_table, images_table
from support.dataset import NoInitDataset
from support.settings import override_settings

from mermaid_classifier.common.benthic_attributes import CoralNetMermaidMapping
from mermaid_classifier.coralnet.manifest import build_manifest_relation, write_manifest
from mermaid_classifier.pyspacer.options import DatasetOptions

_OPEN_DATA_COLUMNS = [
    "row",
    "col",
    "image_id",
    "label_id",
    "site",
    "bucket",
    "project_id",
    "feature_vector",
    "benthic_attribute_id",
    "growth_form_id",
]

# Mocked: this test is about the manifest's column contract, not about
# CoralNet label resolution, which test_train.py covers.
_MAPPING = [
    {
        "benthic_attribute_id": f"ba-{provider_id}",
        "growth_form_id": None,
        "benthic_attribute_name": f"BA{provider_id}",
        "growth_form_name": None,
        "provider_id": provider_id,
        "provider_label": f"Label{provider_id}",
    }
    # The CoralNet label ids carried by the fixture manifest; a row whose
    # label does not map is dropped, which would empty the table.
    for provider_id in ("100", "200")
]


class ManifestNormalizationTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        tmp = self.tmp_dir.name
        ann = os.path.join(tmp, "a.parquet")
        pq.write_table(annotations_table(), ann)
        img = os.path.join(tmp, "i.parquet")
        pq.write_table(images_table(), img)
        self.manifest = os.path.join(tmp, "m.parquet")
        write_manifest(build_manifest_relation(duckdb.connect(), ann, img), self.manifest)

        self.override = override_settings(aws_anonymous="True", coralnet_train_data_bucket="b")
        self.override.__enter__()
        self.addCleanup(self.override.__exit__, None, None, None)

    def test_manifest_normalizes_to_the_open_data_layout(self):
        dataset = NoInitDataset()
        dataset.options = DatasetOptions(coralnet_manifest_uri=self.manifest)

        with mock.patch.object(CoralNetMermaidMapping, "_download_mapping", return_value=_MAPPING):
            dataset.read_coralnet_manifest()

        cols = [c[0] for c in dataset.duck_conn.sql("DESCRIBE annotations").fetchall()]
        self.assertEqual(cols, _OPEN_DATA_COLUMNS)

        row = dataset.duck_conn.sql(
            "SELECT site, bucket, project_id, feature_vector FROM annotations WHERE image_id = 'a'"
        ).fetchone()
        self.assertEqual(row, ("coralnet", "b", "1", "s1/features/ia.featurevector"))


if __name__ == "__main__":
    unittest.main()
