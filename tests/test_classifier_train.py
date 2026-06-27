"""Unit tests for scripts/classifier_train.py.

classifier_train.py is a script, not a module; we import it by path. The tests
mock the local AWS SSO step and the MLflowTrainingRunner factory so they neither
hit AWS nor run real training, and verify that the local driver:

  * loads a committed config dir, applies its env, builds the three option
    dataclasses, and calls the runner exactly once with them;
  * applies the config's env block before constructing the runner;
  * defaults to the committed coralnet_top108_best config dir.

build_options() itself does no network I/O (it only constructs dataclasses from
the YAML + sibling CSVs), so running it against the committed `example` config
is safe offline.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "classifier_train.py"
EXAMPLE_CONFIG_DIR = REPO_ROOT / "sagemaker" / "configs" / "example"


def _load_module():
    spec = importlib.util.spec_from_file_location("classifier_train", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClassifierTrainMainTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def test_runs_committed_config_through_runner(self):
        """main() loads the config, builds options, and runs the runner once."""
        runner_instance = MagicMock(name="runner_instance")
        runner_class = MagicMock(name="MLflowTrainingRunner", return_value=runner_instance)

        with (
            patch.object(self.module, "_resolve_local_aws_credentials"),
            patch.object(self.module, "_resolve_runner_factory", return_value=runner_class),
        ):
            self.module.main(["--config-dir", str(EXAMPLE_CONFIG_DIR)])

        # Runner constructed exactly once, with the three option dataclasses
        # built from the config, then run() called once.
        runner_class.assert_called_once()
        kwargs = runner_class.call_args.kwargs
        self.assertIn("dataset_options", kwargs)
        self.assertIn("training_options", kwargs)
        self.assertIn("mlflow_options", kwargs)
        runner_instance.run.assert_called_once_with()

    def test_options_reflect_the_chosen_config(self):
        """The DatasetOptions handed to the runner come from the chosen config's YAML."""
        from mermaid_classifier.sagemaker.config import TrainingRunConfig

        expected = TrainingRunConfig.from_yaml_path(
            EXAMPLE_CONFIG_DIR / self.module.CONFIG_FILENAME
        )

        runner_class = MagicMock(return_value=MagicMock())
        with (
            patch.object(self.module, "_resolve_local_aws_credentials"),
            patch.object(self.module, "_resolve_runner_factory", return_value=runner_class),
        ):
            self.module.main(["--config-dir", str(EXAMPLE_CONFIG_DIR)])

        dataset_options = runner_class.call_args.kwargs["dataset_options"]
        # include_mermaid is a stable, simple field carried straight through
        # from the YAML's dataset block into DatasetOptions.
        self.assertEqual(dataset_options.include_mermaid, expected.dataset.include_mermaid)

    def test_default_config_dir_is_coralnet_top108_best(self):
        """With no --config-dir, the default points at the committed best config."""
        self.assertEqual(self.module.DEFAULT_CONFIG_DIR.name, "coralnet_top108_best")
        self.assertTrue(
            self.module.DEFAULT_CONFIG_DIR.is_relative_to(REPO_ROOT),
            msg="default config dir must be in-repo (repo-root-relative)",
        )


if __name__ == "__main__":
    unittest.main()
