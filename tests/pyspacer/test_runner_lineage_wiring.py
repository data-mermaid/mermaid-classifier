"""MLflowTrainingRunner.run records the training-data lineage it is given.

Training itself (TrainingRunner.run) and the export are replaced: the export
stand-in captures its keyword arguments and stops the run there, so the test
reads what reached model.json. MLflow logs to a file store in a temp dir.
"""

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import duckdb
import mlflow
import numpy as np
import pandas as pd
from segdata_spec.manifest import FeatureSetPin, Pin
from support.extractor import make_extractor_spec
from support.settings import override_settings

from mermaid_classifier.pyspacer.dataset import _read_holdout_groups, resolve_groups
from mermaid_classifier.pyspacer.options import DatasetOptions, MLflowOptions, PinnedDataset
from mermaid_classifier.pyspacer.runner import MLflowTrainingRunner, TrainingRunner

PIN = Pin(name="coralnet-raw", version="20261124-5ab21b3", manifest_sha256="a" * 64)
FEATURES = FeatureSetPin(dataset=PIN, family="efficientnet-v1", manifest_sha256="d" * 64)
# sha256 of b"s7\ns9", hand-computed.
S7_S9_SHA256 = "b769571b2eec930bc93173cabf88edf5102b787006b4ef5e33e805317ec525a0"
# sha256 of b"benthic_attribute_id,provider_id\nba-1,100\n", hand-computed.
MAPPING_SHA256 = "e9c13ff2f000866b54bbd37aedef18457710e576fa5987dda63484bc066c5e4f"
# sha256 of b"site,group_id,fold\nmermaid,P1,test\n", hand-computed: the bytes the dataset read.
HOLDOUT_SHA256 = "a7c864ec103963d620ca389108cf3143a849fb7d2ae84e38da0a6d55a0c42020"
LOOKUP_SHA256 = "f" * 64


class _ExportReached(Exception):
    def __init__(self, kwargs: dict) -> None:
        super().__init__("export reached")
        self.kwargs = kwargs


def _stop_at_export(*_args, **kwargs):
    raise _ExportReached(kwargs)


class _Library:
    def id_to_name(self, value):
        return value


class RunnerLineageWiringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # The runner points MLflow at the temp file store for the whole process.
        self.addCleanup(mlflow.set_tracking_uri, mlflow.get_tracking_uri())
        self.groups_csv = self.tmp / "groups.csv"
        self.groups_csv.write_text("s9\ns7\n")
        self.holdout_csv = self.tmp / "holdout_groups.csv"
        self.holdout_csv.write_text("site,group_id,fold\nmermaid,P1,test\n")
        self.options = DatasetOptions(
            mermaid_project_lookup_uri="s3://lookup-bucket/lookup.parquet",
            holdout_groups_csv=str(self.holdout_csv),
            datasets=[
                PinnedDataset(
                    pin=PIN, features=FEATURES, labels="coralnet", groups_csv=str(self.groups_csv)
                )
            ],
        )
        # False stands in an empty holdout spec; True reads holdout_csv as the dataset does.
        self.read_holdout = False

    def _holdout_state(self) -> dict:
        if not self.read_holdout:
            return {
                "holdout_groups": SimpleNamespace(
                    is_empty=lambda: True, group_folds={("mermaid", "P1"): "test"}
                ),
                "holdout_groups_sha256": HOLDOUT_SHA256,
                "holdout_groups_bytes": None,
            }
        holdout_groups, sha256, data = _read_holdout_groups(str(self.holdout_csv))
        return {
            "holdout_groups": holdout_groups,
            "holdout_groups_sha256": sha256,
            "holdout_groups_bytes": data,
        }

    def _fake_training(self, runner, run_name=None, cleanup_dataset=True):
        duck_conn = duckdb.connect()
        duck_conn.execute(
            "CREATE TABLE annotations"
            " (training_set VARCHAR, is_holdout BOOLEAN, is_excluded BOOLEAN)"
        )
        runner.dataset = SimpleNamespace(
            groups=resolve_groups(self.options),
            artifacts=SimpleNamespace(
                coralnet_label_mapping=pd.DataFrame(
                    {"provider_id": ["100"], "benthic_attribute_id": ["ba-1"]}
                ),
                mermaid_project_stats=pd.DataFrame(),
                coralnet_project_stats=pd.DataFrame(),
                train_summary_stats={},
                ba_counts=pd.DataFrame(),
                bagf_counts=pd.DataFrame(),
                unmapped_labels=pd.DataFrame(),
            ),
            coralnet_source_ids=[],
            label_filter=SimpleNamespace(inclusion=True, csv_text=""),
            rollup_spec=SimpleNamespace(csv_text=""),
            **self._holdout_state(),
            _subsample_audit_df=None,
            _subsample_group_audit_df=None,
            _subsample_realized_total=None,
            _holdout_audit_df=None,
            mermaid_project_lookup_sha256=LOOKUP_SHA256,
            compute_project_stats=lambda **_kwargs: pd.DataFrame(),
            labels=SimpleNamespace(
                val=SimpleNamespace(
                    load_data_in_batches=lambda: iter([((np.zeros(8),), ("ba::",))])
                )
            ),
            extractor_spec=make_extractor_spec(8),
            duck_conn=duck_conn,
            cleanup=lambda: None,
        )
        # The real base run() logs dataset artifacts right after building the dataset.
        with mock.patch.object(runner, "log_dataframe"):
            runner.log_dataset_artifacts()
        # What MLflow holds at this point is what a run that fails in training keeps.
        self.params_before_training = dict(
            mlflow.get_run(mlflow.active_run().info.run_id).data.params
        )
        runner.profiled_sections = [{"section": "training", "seconds": 0.0}]
        # The file changes after the dataset read it; the run must still
        # describe the selection the dataset trained on.
        self.groups_csv.write_text("s1\n")
        self.holdout_csv.write_text("site,group_id,fold\nmermaid,P2,test\n")
        return None, None, SimpleNamespace(serialize=dict)

    def _run(self) -> dict:
        with (
            override_settings(mlflow_tracking_server=f"file:{self.tmp / 'mlruns'}"),
            mock.patch.object(mlflow, "enable_system_metrics_logging"),
            mock.patch.object(
                TrainingRunner, "run", autospec=True, side_effect=self._fake_training
            ),
            mock.patch(
                "mermaid_classifier.pyspacer.runner.export_artifact", side_effect=_stop_at_export
            ),
            mock.patch(
                "mermaid_classifier.pyspacer.runner.get_benthic_attribute_library",
                return_value=_Library(),
            ),
            mock.patch(
                "mermaid_classifier.pyspacer.runner.get_growth_form_library",
                return_value=_Library(),
            ),
        ):
            runner = MLflowTrainingRunner(
                dataset_options=self.options,
                mlflow_options=MLflowOptions(experiment_name="wiring", model_name="Wiring"),
            )
            with self.assertRaises(_ExportReached) as ctx:
                runner.run()
        self.run_id = mlflow.last_active_run().info.run_id
        return ctx.exception.kwargs

    def test_the_lineage_reaches_the_export_and_the_mlflow_run(self):
        export_kwargs = self._run()

        [entry] = export_kwargs["training_data"]["datasets"]
        self.assertEqual(entry["groups"], {"sha256": S7_S9_SHA256, "n": 2})
        self.assertEqual(entry["features"]["manifest_sha256"], "d" * 64)
        self.assertEqual(
            export_kwargs["training_data"]["coralnet_label_mapping"],
            {"sha256": MAPPING_SHA256, "n": 1},
        )

        self.assertEqual(
            self.params_before_training["dataset/coralnet-raw/groups_sha256"], S7_S9_SHA256
        )
        run = mlflow.get_run(self.run_id)
        self.assertEqual(run.data.params["dataset/coralnet-raw/groups_sha256"], S7_S9_SHA256)
        self.assertEqual(run.data.params["dataset/coralnet-raw/manifest_sha256"], "a" * 64)
        groups_file = mlflow.artifacts.download_artifacts(
            run_id=self.run_id, artifact_path="groups/coralnet-raw.csv", dst_path=str(self.tmp)
        )
        self.assertEqual(Path(groups_file).read_text(), "s7\ns9\n")

    def test_lookup_and_holdout_hashes_are_those_of_the_bytes_the_dataset_read(self):
        training_data = self._run()["training_data"]

        self.assertEqual(
            training_data["mermaid_project_lookup"],
            {"uri": "s3://lookup-bucket/lookup.parquet", "sha256": LOOKUP_SHA256},
        )
        self.assertEqual(
            training_data["holdout_groups"], {"sha256": HOLDOUT_SHA256, "n": 1, "fold": "test"}
        )

    def test_the_logged_holdout_csv_hashes_to_the_lineage_sha256(self):
        cases = {
            "CRLF line endings": b"site,group_id,fold\r\nmermaid,P1,test\r\n",
            "UTF-8 BOM": b"\xef\xbb\xbfsite,group_id,fold\nmermaid,P1,test\n",
        }
        for name, data in cases.items():
            with self.subTest(name):
                self.holdout_csv.write_bytes(data)
                self.read_holdout = True
                with mock.patch.object(mlflow, "log_text", wraps=mlflow.log_text) as log_text:
                    training_data = self._run()["training_data"]

                [logged] = [
                    c.args[0]
                    for c in log_text.call_args_list
                    if c.args[1:] == ("holdout_groups.csv",)
                ]
                self.assertEqual(
                    hashlib.sha256(logged.encode("utf-8")).hexdigest(),
                    training_data["holdout_groups"]["sha256"],
                )


if __name__ == "__main__":
    unittest.main()
