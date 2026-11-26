"""Unit tests for scripts/classifier_train.py.

classifier_train.py is a script, not a module; we import it by path. The tests
mock the local AWS SSO step and the MLflowTrainingRunner factory so they neither
hit AWS nor run real training, and verify that the local driver:

  * loads a copy of the committed example config, its feature-set pin
    replaced, and calls the runner once with options built from it;
  * defaults to a config dir that exists inside the repo.

build_options() itself does no network I/O (it only constructs dataclasses from
the YAML + sibling CSVs), so running it against the example config is safe
offline.
"""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from support.paths import REPO_ROOT

SCRIPT_PATH = REPO_ROOT / "scripts" / "classifier_train.py"
EXAMPLE_CONFIG_DIR = REPO_ROOT / "sagemaker" / "configs" / "example"


def _load_module():
    spec = importlib.util.spec_from_file_location("classifier_train", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClassifierTrainMainTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def test_the_chosen_config_reaches_the_runner_and_it_is_run(self):
        """main() wires the chosen config's values into the runner, then runs it.

        The feature-set sha is the assertion rather than a boolean field: a
        hardcoded default could match by accident, but nothing produces this
        sha except the config written here.
        """
        runner_instance = MagicMock(name="runner_instance")
        runner_class = MagicMock(name="MLflowTrainingRunner", return_value=runner_instance)

        with tempfile.TemporaryDirectory() as td:
            config_dir = Path(td)
            text = (EXAMPLE_CONFIG_DIR / "training_config.yaml").read_text()
            (config_dir / "training_config.yaml").write_text(
                text.replace(
                    "c557feffdeb3f89ec82270563b2942d0c4bbe7c976a3568cb92bb91dcfe9935e", "ef" * 32
                )
            )
            for sibling in ("rollups.csv", "included_labels.csv", "groups.csv"):
                shutil.copy(EXAMPLE_CONFIG_DIR / sibling, config_dir / sibling)
            with (
                patch.object(self.module, "_resolve_local_aws_credentials"),
                patch.object(self.module, "_resolve_runner_factory", return_value=runner_class),
            ):
                self.module.main(["--config-dir", str(config_dir)])

        dataset_options = runner_class.call_args.kwargs["dataset_options"]
        self.assertEqual(
            [source.features.manifest_sha256 for source in dataset_options.datasets], ["ef" * 32]
        )
        runner_instance.run.assert_called_once_with()

    def test_default_config_dir_is_in_repo_and_loads(self):
        """The default config dir is repo-root-relative and its config loads.

        Asserts the behavioral invariant (in-repo + a loadable config) rather
        than a magic directory name, so renaming the default config doesn't
        break this test while a broken/missing default still would.
        """
        from mermaid_classifier.sagemaker.config import TrainingRunConfig

        default_dir = self.module.DEFAULT_CONFIG_DIR
        self.assertTrue(
            default_dir.is_relative_to(REPO_ROOT),
            msg="default config dir must be in-repo (repo-root-relative)",
        )
        config_path = default_dir / self.module.CONFIG_FILENAME
        self.assertTrue(config_path.is_file(), msg=f"missing {config_path}")
        # A malformed default config is a real regression.
        TrainingRunConfig.from_yaml_path(config_path)


if __name__ == "__main__":
    unittest.main()
