"""Tests for scripts.launch_training (mermaid-classifier)."""

from __future__ import annotations

import importlib.util
import os
import unittest
from unittest.mock import MagicMock, patch

from support.paths import add_scripts_to_path

add_scripts_to_path()

# Importing the SDK makes botocore resolve credentials, which probes the EC2
# instance-metadata endpoint. These tests drive the SDK entirely through
# MagicMock, so the probe is pure latency on a laptop and a real IMDS call on
# an EC2 runner.
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

# launch_training imports the SageMaker SDK at module level. That SDK lives in
# the `sagemaker` extra, so skip this module entirely when it's absent.
# Check `sagemaker.estimator` specifically: `sagemaker-mlflow` (a training-extra
# dep) provides a partial `sagemaker` namespace without the full SDK.
_HAS_SAGEMAKER = importlib.util.find_spec("sagemaker.estimator") is not None
if _HAS_SAGEMAKER:
    import launch_training as lt  # type: ignore


def setUpModule():
    # CI installs the `sagemaker` extra, so a missing SDK there means the extra
    # was dropped from the workflow or a tests/<name>/ package is shadowing it
    # again -- both regress this module back to silently skipped. Locally, the
    # extra is optional, so a missing SDK there is expected and just skips.
    if _HAS_SAGEMAKER:
        return
    if os.environ.get("CI"):
        raise RuntimeError(
            "sagemaker.estimator does not import under CI, where the `sagemaker` "
            "extra is installed: either the extra was dropped from the workflow, "
            "or a tests/<name>/ package is shadowing the SDK."
        )
    raise unittest.SkipTest("sagemaker SDK not installed (`sagemaker` extra)")


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
            "554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-jobs:training-latest",
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
            "554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-jobs:training-smoke",
        )
        # Environment carries the MLflow URI passed to build_estimator_kwargs;
        # test_a_yaml_env_block_cannot_redirect_launcher_owned_env_keys covers
        # that a YAML env block cannot override it.
        self.assertEqual(
            kwargs["environment"]["MLFLOW_TRACKING_SERVER"],
            "arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-2OMU4VP53ZS2",
        )
        # YAML env preserved:
        self.assertEqual(kwargs["environment"]["MY_VAR"], "1")
        # The container entrypoint shim dispatches on this.
        self.assertEqual(
            kwargs["environment"]["CONTAINER_ENTRYPOINT_SCRIPT"],
            "scripts/sagemaker_train_entrypoint.py",
        )

    def test_a_yaml_env_block_cannot_redirect_launcher_owned_env_keys(self):
        """CONTAINER_ENTRYPOINT_SCRIPT, MLFLOW_TRACKING_SERVER and
        AWS_DEFAULT_REGION are the launcher's to set, so a job's own env
        block must not be able to point any of them somewhere else."""
        from mermaid_classifier.sagemaker.launcher_config import parse_run_config

        yaml_text = _minimal_yaml().replace(
            '    MY_VAR: "1"',
            '    MY_VAR: "1"\n'
            "    CONTAINER_ENTRYPOINT_SCRIPT: scripts/somewhere_else.py\n"
            "    MLFLOW_TRACKING_SERVER: https://attacker.example/mlflow\n"
            "    AWS_DEFAULT_REGION: us-west-2",
        )
        cfg = parse_run_config(yaml_text, kind="training", strict=False)
        kwargs = lt.build_estimator_kwargs(
            cfg=cfg,
            run_id="mermaid-test-20260525T120000Z",
            staging_bucket="dev-datamermaid-sm-data",
            mlflow_uri="arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-2OMU4VP53ZS2",
            sm_session=MagicMock(),
        )
        self.assertEqual(
            kwargs["environment"]["CONTAINER_ENTRYPOINT_SCRIPT"],
            "scripts/sagemaker_train_entrypoint.py",
        )
        self.assertEqual(
            kwargs["environment"]["MLFLOW_TRACKING_SERVER"],
            "arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-2OMU4VP53ZS2",
        )
        self.assertEqual(kwargs["environment"]["AWS_DEFAULT_REGION"], "us-east-1")
