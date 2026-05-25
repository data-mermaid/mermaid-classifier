"""Tests for scripts.launch_training (mermaid-classifier)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launch_training as lt  # type: ignore


def _minimal_yaml() -> str:
    return """
job:
  name_prefix: mermaid-test
  image: mermaid-classifier-jobs:training-smoke
  entrypoint: scripts/sagemaker_train_entrypoint.py
  instance_type: ml.m5.4xlarge
  volume_gb: 200
  max_runtime_hours: 24
  env:
    MY_VAR: "1"
  tags:
    Owner: greg
"""


class ExpandImageTest(unittest.TestCase):
    def test_short_form_expands(self):
        result = lt.expand_image_uri("mermaid-classifier-jobs:training-latest")
        self.assertEqual(
            result,
            "554812291621.dkr.ecr.us-east-1.amazonaws.com"
            "/mermaid-classifier-jobs:training-latest",
        )

    def test_full_uri_passes_through(self):
        full = "111111111111.dkr.ecr.us-east-1.amazonaws.com/other:tag"
        self.assertEqual(lt.expand_image_uri(full), full)

    def test_short_form_rejects_unknown_repo(self):
        # The launcher only knows the classifier ECR for short-form
        # expansion. Unknown short-form repos must raise so users get
        # a clear error instead of a silently-wrong URI.
        with self.assertRaises(ValueError):
            lt.expand_image_uri("some-other-repo:latest")


class MakeRunIdTest(unittest.TestCase):
    @patch("launch_training.datetime")
    def test_run_id_format(self, mock_dt):
        mock_dt.now.return_value.strftime.return_value = "20260525T120000Z"
        self.assertEqual(lt.make_run_id("mermaid-test"), "mermaid-test-20260525T120000Z")


class BuildEstimatorKwargsTest(unittest.TestCase):
    @patch("launch_training.datetime")
    def test_kwargs_match_expectation(self, mock_dt):
        mock_dt.now.return_value.strftime.return_value = "20260525T120000Z"

        from mermaid_classifier.sagemaker.launcher_config import parse_run_config
        cfg = parse_run_config(_minimal_yaml(), kind="training", strict=False)
        kwargs = lt.build_estimator_kwargs(
            cfg=cfg,
            run_id="mermaid-test-20260525T120000Z",
            staging_bucket="dev-datamermaid-sm-data",
            mlflow_uri="arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-2OMU4VP53ZS2",
            sm_session=MagicMock(),
        )
        self.assertEqual(kwargs["instance_type"], "ml.m5.4xlarge")
        self.assertEqual(kwargs["instance_count"], 1)
        self.assertEqual(kwargs["volume_size"], 200)
        self.assertEqual(kwargs["max_run"], 24 * 3600)
        self.assertEqual(
            kwargs["role"],
            "arn:aws:iam::554812291621:role/dev-sm-execution-role",
        )
        self.assertEqual(
            kwargs["output_path"],
            "s3://dev-datamermaid-sm-data/runs/mermaid-test-20260525T120000Z/output/",
        )
        self.assertEqual(
            kwargs["image_uri"],
            "554812291621.dkr.ecr.us-east-1.amazonaws.com"
            "/mermaid-classifier-jobs:training-smoke",
        )
        # MLflow URI is injected as env, NOT YAML-overridable.
        self.assertEqual(kwargs["environment"]["MLFLOW_TRACKING_SERVER"],
            "arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-2OMU4VP53ZS2")
        # YAML env preserved:
        self.assertEqual(kwargs["environment"]["MY_VAR"], "1")
