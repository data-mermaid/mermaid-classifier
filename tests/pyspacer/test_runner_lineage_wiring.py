"""MLflowTrainingRunner.run records the training-data lineage it is given.

Training itself (TrainingRunner.run) and the export are replaced: the export
stand-in captures its keyword arguments and stops the run there, so the test
reads what reached model.json. MLflow logs to a file store in a temp dir.
"""

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

from mermaid_classifier.pyspacer.dataset import resolve_groups
from mermaid_classifier.pyspacer.options import DatasetOptions, MLflowOptions, PinnedDataset
from mermaid_classifier.pyspacer.runner import MLflowTrainingRunner, TrainingRunner

PIN = Pin(name="coralnet-raw", version="20261124-5ab21b3", manifest_sha256="a" * 64)
FEATURES = FeatureSetPin(dataset=PIN, family="efficientnet-v1", manifest_sha256="d" * 64)
# sha256 of b"s7\ns9", hand-computed.
S7_S9_SHA256 = "b769571b2eec930bc93173cabf88edf5102b787006b4ef5e33e805317ec525a0"
# sha256 of b"benthic_attribute_id,provider_id\nba-1,100\n", hand-computed.
MAPPING_SHA256 = "e9c13ff2f000866b54bbd37aedef18457710e576fa5987dda63484bc066c5e4f"


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
        self.options = DatasetOptions(
            datasets=[
                PinnedDataset(
                    pin=PIN, features=FEATURES, labels="coralnet", groups_csv=str(self.groups_csv)
                )
            ]
        )

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
            holdout_groups=SimpleNamespace(is_empty=lambda: True),
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


if __name__ == "__main__":
    unittest.main()
