"""Tests for scripts.launch_training (mermaid-classifier)."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from support.paths import add_scripts_to_path

add_scripts_to_path()

# Importing the SDK makes botocore resolve credentials, which probes the EC2
# instance-metadata endpoint. These tests never reach AWS, so the probe is pure
# latency on a laptop and a real IMDS call on an EC2 runner.
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")


# launch_training imports sagemaker-core at module level. That package lives in
# the `sagemaker` extra, so skip this module entirely when it's absent.
# `sagemaker-mlflow` (a training-extra dep) installs only `sagemaker_mlflow`, but
# a bare `sagemaker` can still resolve as a namespace package (e.g. the repo's
# own sagemaker/ directory when it is on sys.path). find_spec then raises
# ModuleNotFoundError on the missing `sagemaker.core` parent instead of
# returning None, so treat both outcomes as absent.
def _has_sagemaker_core() -> bool:
    try:
        return importlib.util.find_spec("sagemaker.core.resources") is not None
    except ModuleNotFoundError:
        return False


_HAS_SAGEMAKER = _has_sagemaker_core()
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
            "sagemaker.core.resources does not import under CI, where the `sagemaker` "
            "extra is installed: either the extra was dropped from the workflow, "
            "or a tests/<name>/ package is shadowing the SDK."
        )
    raise unittest.SkipTest("sagemaker-core not installed (`sagemaker` extra)")


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


MLFLOW_ARN = "arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-2OMU4VP53ZS2"
RUN_ID = "mermaid-test-20260525T120000Z"


def _build(yaml_text: str) -> dict:
    from mermaid_classifier.sagemaker.launcher_config import parse_run_config

    cfg = parse_run_config(yaml_text, kind="training", strict=False)
    return lt.build_training_job_kwargs(
        cfg=cfg,
        run_id=RUN_ID,
        staging_bucket="dev-datamermaid-sm-data",
        mlflow_uri=MLFLOW_ARN,
    )


class BuildTrainingJobKwargsTest(unittest.TestCase):
    def test_kwargs_match_expectation(self):
        kwargs = _build(_minimal_yaml())

        self.assertEqual(kwargs["training_job_name"], RUN_ID)
        self.assertEqual(
            kwargs["role_arn"],
            "arn:aws:iam::554812291621:role/dev-sm-execution-role",
        )
        self.assertEqual(
            kwargs["algorithm_specification"].training_image,
            "554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-jobs:training-smoke",
        )
        self.assertEqual(kwargs["algorithm_specification"].training_input_mode, "File")
        resources = kwargs["resource_config"]
        self.assertEqual(resources.instance_type, "ml.m5.4xlarge")
        self.assertEqual(resources.instance_count, 1)
        self.assertEqual(resources.volume_size_in_gb, 200)
        self.assertEqual(kwargs["stopping_condition"].max_runtime_in_seconds, 24 * 3600)
        self.assertEqual(
            kwargs["output_data_config"].s3_output_path,
            f"s3://dev-datamermaid-sm-data/runs/{RUN_ID}/output/",
        )
        self.assertEqual(
            [(t.key, t.value) for t in kwargs["tags"]],
            [("Owner", "greg")],
        )
        # Environment carries the MLflow URI passed to the builder;
        # test_a_yaml_env_block_cannot_redirect_launcher_owned_env_keys covers
        # that a YAML env block cannot override it.
        self.assertEqual(kwargs["environment"]["MLFLOW_TRACKING_SERVER"], MLFLOW_ARN)
        # YAML env preserved:
        self.assertEqual(kwargs["environment"]["MY_VAR"], "1")
        # The container entrypoint shim dispatches on this.
        self.assertEqual(
            kwargs["environment"]["CONTAINER_ENTRYPOINT_SCRIPT"],
            "scripts/sagemaker_train_entrypoint.py",
        )

    def test_on_demand_job_sets_no_spot_fields(self):
        kwargs = _build(_minimal_yaml())

        self.assertNotIn("enable_managed_spot_training", kwargs)
        self.assertNotIn("max_wait_time_in_seconds", kwargs["stopping_condition"].model_fields_set)

    def test_spot_job_enables_managed_spot_with_max_wait_one_hour_over_runtime(self):
        kwargs = _build(_minimal_yaml() + "  use_spot: true\n")

        self.assertIs(kwargs["enable_managed_spot_training"], True)
        self.assertEqual(kwargs["stopping_condition"].max_runtime_in_seconds, 24 * 3600)
        self.assertEqual(kwargs["stopping_condition"].max_wait_time_in_seconds, 24 * 3600 + 3600)

    def test_config_channel_is_staged_under_the_run_prefix_in_file_mode(self):
        kwargs = _build(_minimal_yaml())

        (channel,) = kwargs["input_data_config"]
        self.assertEqual(channel.channel_name, "config")
        self.assertEqual(channel.input_mode, "File")
        s3 = channel.data_source.s3_data_source
        self.assertEqual(s3.s3_data_type, "S3Prefix")
        self.assertEqual(s3.s3_uri, f"s3://dev-datamermaid-sm-data/runs/{RUN_ID}/config/")
        self.assertEqual(s3.s3_data_distribution_type, "FullyReplicated")

    def test_training_channels_follow_config_with_their_own_input_mode(self):
        yaml_text = _minimal_yaml() + (
            "training:\n"
            "  channels:\n"
            "    train:\n"
            "      s3_uri: s3://bucket/train/\n"
            "      input_mode: FastFile\n"
            "    val:\n"
            "      s3_uri: s3://bucket/val/\n"
        )
        kwargs = _build(yaml_text)

        got = [
            (
                c.channel_name,
                c.input_mode,
                c.data_source.s3_data_source.s3_uri,
            )
            for c in kwargs["input_data_config"]
        ]
        self.assertEqual(
            got,
            [
                ("config", "File", f"s3://dev-datamermaid-sm-data/runs/{RUN_ID}/config/"),
                ("train", "FastFile", "s3://bucket/train/"),
                ("val", "File", "s3://bucket/val/"),
            ],
        )

    def test_a_yaml_env_block_cannot_redirect_launcher_owned_env_keys(self):
        """CONTAINER_ENTRYPOINT_SCRIPT, MLFLOW_TRACKING_SERVER and
        AWS_DEFAULT_REGION are the launcher's to set, so a job's own env
        block must not be able to point any of them somewhere else."""
        yaml_text = _minimal_yaml().replace(
            '    MY_VAR: "1"',
            '    MY_VAR: "1"\n'
            "    CONTAINER_ENTRYPOINT_SCRIPT: scripts/somewhere_else.py\n"
            "    MLFLOW_TRACKING_SERVER: https://attacker.example/mlflow\n"
            "    AWS_DEFAULT_REGION: us-west-2",
        )
        env = _build(yaml_text)["environment"]

        self.assertEqual(
            env["CONTAINER_ENTRYPOINT_SCRIPT"],
            "scripts/sagemaker_train_entrypoint.py",
        )
        self.assertEqual(env["MLFLOW_TRACKING_SERVER"], MLFLOW_ARN)
        self.assertEqual(env["AWS_DEFAULT_REGION"], "us-east-1")


class MainSubmitTest(unittest.TestCase):
    def test_main_submits_job_named_run_id_and_waits_with_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_config = Path(tmp) / "run.yaml"
            run_config.write_text(_minimal_yaml())
            config_dir = Path(tmp) / "cfg"
            config_dir.mkdir()

            job = MagicMock()
            region_at_wait = []
            job.wait.side_effect = lambda **_: region_at_wait.append(
                os.environ.get("AWS_DEFAULT_REGION")
            )
            with (
                patch.dict(os.environ),
                patch.object(lt, "make_run_id", return_value=RUN_ID),
                patch.object(lt, "_upload_config_dir") as upload,
                patch.object(lt.boto3, "Session"),
                patch.object(lt.TrainingJob, "create", return_value=job) as create,
            ):
                lt.main(
                    [
                        "--run-config",
                        str(run_config),
                        "--config-dir",
                        str(config_dir),
                        "--mlflow-tracking-uri",
                        MLFLOW_ARN,
                    ]
                )

        upload.assert_called_once()
        self.assertEqual(upload.call_args.args[1], RUN_ID)
        self.assertEqual(create.call_args.kwargs["training_job_name"], RUN_ID)
        job.wait.assert_called_once_with(logs=True)
        self.assertEqual(region_at_wait, [lt.REGION])
