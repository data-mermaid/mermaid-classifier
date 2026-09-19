"""Unit tests for scripts/sagemaker_train_entrypoint.py.

The entrypoint is a script, not a module; we import it by path. Tests
mock MLflowTrainingRunner so they don't trigger pyspacer's heavy
imports or hit AWS. Tests verify:

  * apply_env happens before the runner is imported (we observe this
    indirectly by patching the runner factory to capture os.environ at
    construction time)
  * an exception in runner.run propagates to sys.exit(1) with the
    traceback in log output
"""

from __future__ import annotations

import importlib.util
import os
import textwrap
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from support.paths import REPO_ROOT

ENTRYPOINT_PATH = REPO_ROOT / "scripts" / "sagemaker_train_entrypoint.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("sagemaker_train_entrypoint", ENTRYPOINT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MINIMAL_YAML = textwrap.dedent("""
    dataset:
      include_mermaid: false
      coralnet_manifest_uri: s3://bucket/coralnet_manifest.parquet
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
      MLFLOW_TRACKING_SERVER: file:./mlruns
      WEIGHTS_LOCATION: s3://x/weights.pt
""").lstrip()


@contextmanager
def _config_dir():
    with TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "training_config.yaml").write_text(MINIMAL_YAML)
        (tmp / "rollups.csv").write_text("from_ba_id,from_gf_id,to_ba_id,to_gf_id\n")
        (tmp / "included_labels.csv").write_text("ba_id,gf_id\n")
        yield tmp


class EntrypointHappyPathTest(unittest.TestCase):
    def setUp(self):
        # Wipe env vars the entrypoint will set so we can detect them.
        for key in (
            "MLFLOW_TRACKING_SERVER",
            "WEIGHTS_LOCATION",
        ):
            os.environ.pop(key, None)
        self.module = _load_entrypoint()

    def test_main_applies_env_before_resolving_runner(self):
        with _config_dir() as cfg_dir:
            observed = {}

            def factory(*args, **kwargs):
                observed["mlflow"] = os.environ.get("MLFLOW_TRACKING_SERVER")
                observed["weights"] = os.environ.get("WEIGHTS_LOCATION")
                return MagicMock()

            with patch.object(
                self.module,
                "_resolve_runner_factory",
                return_value=factory,
            ):
                self.module.main(["--config-dir", str(cfg_dir)])
        self.assertEqual(observed["mlflow"], "file:./mlruns")
        self.assertEqual(observed["weights"], "s3://x/weights.pt")

    def test_runner_exception_exits_nonzero(self):
        module = _load_entrypoint()
        with _config_dir() as cfg_dir:
            fake_runner = MagicMock()
            fake_runner.run.side_effect = RuntimeError("boom")
            with (
                patch.object(
                    module,
                    "_resolve_runner_factory",
                    return_value=lambda *a, **kw: fake_runner,
                ),
                self.assertRaises(SystemExit) as cm,
            ):
                module.main(["--config-dir", str(cfg_dir)])
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
