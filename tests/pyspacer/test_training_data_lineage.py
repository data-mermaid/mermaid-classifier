"""The training-data pins a run records: MLflow params and the model.json block."""

import dataclasses
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
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


MAPPING_CSV = "benthic_attribute_id,provider_id\nba-1,100\nba-2,200\n"
MAPPING_SHA256 = hashlib.sha256(MAPPING_CSV.encode()).hexdigest()


def _mapping(rows: list[tuple[str, str]], columns: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns))


class CoralNetMappingLineageTest(unittest.TestCase):
    def _mapping_block(self, mapping: pd.DataFrame, options: DatasetOptions = OPTIONS):
        groups = {source.pin.name: None for source in options.datasets}
        return training_data_lineage(options, groups, coralnet_mapping=mapping)

    def test_the_block_hashes_the_canonical_csv_and_counts_rows(self):
        mapping = _mapping(
            [("ba-1", "100"), ("ba-2", "200")], ("benthic_attribute_id", "provider_id")
        )
        self.assertEqual(
            self._mapping_block(mapping)["coralnet_label_mapping"],
            {"sha256": MAPPING_SHA256, "n": 2},
        )

    def test_row_and_column_order_do_not_change_the_hash(self):
        shuffled = _mapping(
            [("200", "ba-2"), ("100", "ba-1")], ("provider_id", "benthic_attribute_id")
        )
        self.assertEqual(
            self._mapping_block(shuffled)["coralnet_label_mapping"],
            {"sha256": MAPPING_SHA256, "n": 2},
        )

    def test_a_changed_cell_changes_the_hash(self):
        changed = _mapping(
            [("ba-1", "100"), ("ba-9", "200")], ("benthic_attribute_id", "provider_id")
        )
        block = self._mapping_block(changed)["coralnet_label_mapping"]
        self.assertNotEqual(block["sha256"], MAPPING_SHA256)
        self.assertEqual(block["n"], 2)

    def test_a_run_without_coralnet_labels_records_no_block(self):
        mapping = _mapping(
            [("ba-1", "100"), ("ba-2", "200")], ("benthic_attribute_id", "provider_id")
        )
        mermaid_only = DatasetOptions(datasets=OPTIONS.datasets[1:], taxonomy=TAXONOMY)
        self.assertNotIn("coralnet_label_mapping", self._mapping_block(mapping, mermaid_only))

    def test_mlflow_params_carry_the_mapping_hash_and_row_count(self):
        mapping = _mapping(
            [("ba-1", "100"), ("ba-2", "200")], ("benthic_attribute_id", "provider_id")
        )
        params = lineage_params(self._mapping_block(mapping))
        self.assertEqual(params["coralnet_label_mapping/sha256"], MAPPING_SHA256)
        self.assertEqual(params["coralnet_label_mapping/n"], "2")


# sha256 of each fixture file's bytes, hand-computed with `shasum -a 256`.
LOOKUP_BYTES = b"lookup parquet bytes"
LOOKUP_SHA256 = "fb2997bdcafedbb64070f9e564ef93450cba64fb418879aa1f55487ebfe6c9af"
HOLDOUT_CSV = "site,group_id,fold\ncoralnet,7,test\nmermaid,P1,0\n"
HOLDOUT_SHA256 = "8636d2978221810fccfed347edb4806ba281ec7b136b40ba1ead59de5dcca34a"


class HoldoutLineageTest(unittest.TestCase):
    """The lookup and holdout CSV a run reads are recorded by content hash."""

    def test_lookup_and_holdout_are_recorded_and_flattened(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        lookup = tmp / "lookup.parquet"
        lookup.write_bytes(LOOKUP_BYTES)
        holdout = tmp / "holdout_groups.csv"
        holdout.write_text(HOLDOUT_CSV)
        options = dataclasses.replace(
            OPTIONS,
            mermaid_project_lookup_uri=str(lookup),
            holdout_groups_csv=str(holdout),
            holdout_fold="0",
        )

        lineage = training_data_lineage(options, GROUPS)
        params = lineage_params(lineage)

        self.assertEqual(
            lineage["mermaid_project_lookup"], {"uri": str(lookup), "sha256": LOOKUP_SHA256}
        )
        self.assertEqual(lineage["holdout_groups"], {"sha256": HOLDOUT_SHA256, "n": 2, "fold": "0"})
        self.assertEqual(
            {
                key: value
                for key, value in params.items()
                if not key.startswith(("dataset/", "taxonomy/"))
            },
            {
                "mermaid_project_lookup/uri": str(lookup),
                "mermaid_project_lookup/sha256": LOOKUP_SHA256,
                "holdout_groups/sha256": HOLDOUT_SHA256,
                "holdout_groups/n": "2",
                "holdout_groups/fold": "0",
            },
        )


if __name__ == "__main__":
    unittest.main()
