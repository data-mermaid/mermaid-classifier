"""Unit tests for scripts/launch_training_sagemaker.py.

The launcher is a script; tests import it by path. SageMaker SDK and
boto3 are mocked so tests don't hit AWS.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LAUNCHER_PATH = REPO_ROOT / "scripts" / "launch_training_sagemaker.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "launch_training_sagemaker", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MINIMAL_YAML = textwrap.dedent("""
    dataset:
      include_mermaid: false
      coralnet_sources_csv: sources.csv
      label_rollup_spec_csv: rollups.csv
      included_labels_csv: included_labels.csv
      subsample:
        strategy: balanced
        total_annotations: 100
    training:
      epochs: 1
    mlflow:
      experiment_name: test
      model_name: T
    env:
      WEIGHTS_LOCATION: s3://x/weights.pt
""").lstrip()


class _Fixture:
    """Lazy fixture: a TemporaryDirectory with a valid config dir."""
    def __init__(self):
        self._td = TemporaryDirectory()
        self.path = Path(self._td.name)
        (self.path / "training_config.yaml").write_text(MINIMAL_YAML)
        (self.path / "sources.csv").write_text("id\n1\n")
        (self.path / "rollups.csv").write_text(
            "from_ba_id,from_gf_id,to_ba_id,to_gf_id\n")
        (self.path / "included_labels.csv").write_text("ba_id,gf_id\n")

    def cleanup(self):
        self._td.cleanup()


def _base_args(cfg_dir: str) -> list[str]:
    return [
        "--config-dir", cfg_dir,
        "--mlflow-tracking-uri", "arn:aws:sagemaker:us-east-1:1:mlflow-app/A",
        "--role-arn", "arn:aws:iam::1:role/MermaidTrainer",
        "--ecr-image-uri", "1.dkr.ecr.us-east-1.amazonaws.com/training:latest",
        "--staging-bucket", "my-staging-bucket",
    ]


class DryRunTest(unittest.TestCase):

    def test_dry_run_does_not_call_sdk(self):
        module = _load_launcher()
        fixture = _Fixture()
        try:
            with patch.object(module, "Estimator") as Estimator:
                with patch.object(module, "_upload_config_dir") as upload:
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        module.main(_base_args(str(fixture.path)) + [
                            "--dry-run",
                        ])
                    Estimator.assert_not_called()
                    upload.assert_not_called()
                    self.assertIn("DRY RUN", buf.getvalue())
                    self.assertIn(
                        "ml.m5.4xlarge", buf.getvalue())
        finally:
            fixture.cleanup()


class ValidationTest(unittest.TestCase):

    def test_missing_config_dir_errors_before_aws_calls(self):
        module = _load_launcher()
        with self.assertRaises(SystemExit):
            module.main([
                *_base_args("/path/does/not/exist"),
                "--dry-run",
            ])

    def test_invalid_yaml_errors_before_submit(self):
        module = _load_launcher()
        with TemporaryDirectory() as td:
            (Path(td) / "training_config.yaml").write_text(
                "this: is\nnot: valid\nbecause: missing required\n")
            with self.assertRaises(SystemExit):
                module.main(_base_args(td) + ["--dry-run"])


class SubmissionTest(unittest.TestCase):

    def test_submit_creates_estimator_with_defaults(self):
        module = _load_launcher()
        fixture = _Fixture()
        try:
            with patch.object(module, "Estimator") as Estimator:
                instance = Estimator.return_value
                instance.latest_training_job = MagicMock(name="x")
                with patch.object(module, "_upload_config_dir",
                                  return_value="s3://my-staging-bucket/runs/abc/config/"):
                    module.main(_base_args(str(fixture.path)))
                args, kwargs = Estimator.call_args
                self.assertEqual(kwargs["instance_type"], "ml.m5.4xlarge")
                self.assertEqual(kwargs["instance_count"], 1)
                self.assertEqual(kwargs["volume_size"], 200)
                self.assertEqual(kwargs["max_run"], 24 * 3600)
                self.assertIn(
                    "MLFLOW_TRACKING_SERVER",
                    kwargs["environment"],
                )
                self.assertEqual(
                    kwargs["environment"]["MLFLOW_TRACKING_SERVER"],
                    "arn:aws:sagemaker:us-east-1:1:mlflow-app/A",
                )
                instance.fit.assert_called_once()
        finally:
            fixture.cleanup()

    def test_overrides_apply(self):
        module = _load_launcher()
        fixture = _Fixture()
        try:
            with patch.object(module, "Estimator") as Estimator:
                with patch.object(module, "_upload_config_dir",
                                  return_value="s3://b/runs/r/config/"):
                    module.main(_base_args(str(fixture.path)) + [
                        "--instance-type", "ml.c5.9xlarge",
                        "--volume-size-gb", "500",
                        "--max-runtime-hours", "6",
                    ])
                _, kwargs = Estimator.call_args
                self.assertEqual(kwargs["instance_type"], "ml.c5.9xlarge")
                self.assertEqual(kwargs["volume_size"], 500)
                self.assertEqual(kwargs["max_run"], 6 * 3600)
        finally:
            fixture.cleanup()


if __name__ == "__main__":
    unittest.main()
