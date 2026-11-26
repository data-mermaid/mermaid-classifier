"""The training-data pins a run records: MLflow params and the model.json block."""

import unittest

from segdata_spec.manifest import FeatureSetPin, Pin

from mermaid_classifier.pyspacer.options import (
    DatasetOptions,
    PinnedDataset,
    lineage_params,
    training_data_lineage,
)

CORALNET = Pin(name="coralnet-raw", version="20261124-5ab21b3", manifest_sha256="a" * 64)
MERMAID = Pin(name="mermaid", version="20261124-cfc389d", manifest_sha256="b" * 64)
TAXONOMY = Pin(name="mermaid-benthic", version="20261124-e6abfad", manifest_sha256="c" * 64)

OPTIONS = DatasetOptions(
    datasets=[
        PinnedDataset(
            pin=CORALNET,
            features=FeatureSetPin(
                dataset=CORALNET, family="efficientnet-v1", manifest_sha256="d" * 64
            ),
            labels="coralnet",
            groups_csv="/configs/x/groups.csv",
        ),
        PinnedDataset(
            pin=MERMAID,
            features=FeatureSetPin(
                dataset=MERMAID, family="efficientnet-v1", manifest_sha256="e" * 64
            ),
            labels="mermaid",
        ),
    ],
    taxonomy=TAXONOMY,
)


# sha256 of b"s7\ns9": the sorted, newline-joined selection, hand-computed.
S7_S9_SHA256 = "b769571b2eec930bc93173cabf88edf5102b787006b4ef5e33e805317ec525a0"
GROUPS = {"coralnet-raw": frozenset({"s9", "s7"}), "mermaid": None}


class TrainingDataLineageTest(unittest.TestCase):
    def test_model_json_block_carries_every_pin(self):
        self.assertEqual(
            training_data_lineage(OPTIONS, GROUPS),
            {
                "datasets": [
                    {
                        "name": "coralnet-raw",
                        "version": "20261124-5ab21b3",
                        "manifest_sha256": "a" * 64,
                        "labels": "coralnet",
                        "features": {"family": "efficientnet-v1", "manifest_sha256": "d" * 64},
                        "groups": {"sha256": S7_S9_SHA256, "n": 2},
                    },
                    {
                        "name": "mermaid",
                        "version": "20261124-cfc389d",
                        "manifest_sha256": "b" * 64,
                        "labels": "mermaid",
                        "features": {"family": "efficientnet-v1", "manifest_sha256": "e" * 64},
                        "groups": None,
                    },
                ],
                "taxonomy": {
                    "name": "mermaid-benthic",
                    "version": "20261124-e6abfad",
                    "manifest_sha256": "c" * 64,
                },
            },
        )

    def test_mlflow_params_carry_every_pin(self):
        self.assertEqual(
            lineage_params(training_data_lineage(OPTIONS, GROUPS)),
            {
                "dataset/coralnet-raw/version": "20261124-5ab21b3",
                "dataset/coralnet-raw/manifest_sha256": "a" * 64,
                "dataset/coralnet-raw/labels": "coralnet",
                "dataset/coralnet-raw/features/family": "efficientnet-v1",
                "dataset/coralnet-raw/features/manifest_sha256": "d" * 64,
                "dataset/coralnet-raw/groups_sha256": S7_S9_SHA256,
                "dataset/coralnet-raw/groups_n": "2",
                "dataset/mermaid/version": "20261124-cfc389d",
                "dataset/mermaid/manifest_sha256": "b" * 64,
                "dataset/mermaid/labels": "mermaid",
                "dataset/mermaid/features/family": "efficientnet-v1",
                "dataset/mermaid/features/manifest_sha256": "e" * 64,
                "dataset/mermaid/groups_sha256": "all",
                "dataset/mermaid/groups_n": "all",
                "taxonomy/mermaid-benthic/version": "20261124-e6abfad",
                "taxonomy/mermaid-benthic/manifest_sha256": "c" * 64,
            },
        )

    def test_a_run_without_a_taxonomy_records_none(self):
        options = DatasetOptions(datasets=OPTIONS.datasets[:1])
        lineage = training_data_lineage(options, {"coralnet-raw": None})
        self.assertIsNone(lineage["taxonomy"])
        self.assertNotIn("taxonomy/mermaid-benthic/version", lineage_params(lineage))


if __name__ == "__main__":
    unittest.main()
