"""Tests for scripts/select_holdout_groups.py: Beta-source reading and fold-config generation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from support.paths import SCRIPTS_DIR, add_scripts_to_path

add_scripts_to_path()

import select_holdout_groups as shg  # noqa: E402

from mermaid_classifier.sagemaker.config import TrainingRunConfig  # noqa: E402

PARENT_YAML = textwrap.dedent("""
    dataset:
      include_mermaid: true
      coralnet_manifest_uri: s3://bucket/manifest.parquet
      feature_extractor_weights: s3://bucket/efficientnet.pt
      label_rollup_spec_csv: rollups.csv
      included_labels_csv: included_labels.csv
      holdout_groups_csv: holdout_groups.csv
      holdout_fold: test
    training:
      epochs: 5
    mlflow:
      experiment_name: my-experiment
      model_name: my-model
""").lstrip()


class ReadBetaSourcesTest(unittest.TestCase):
    def _read(self, text: str) -> set[str]:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "beta.csv"
            path.write_text(text)
            return shg.read_beta_sources(path)

    def test_reads_ids_as_strings(self):
        self.assertEqual(self._read("id,name\n12,a\n 34 ,b\n"), {"12", "34"})

    def test_missing_id_column_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "id"):
            self._read("source_id,name\n12,a\n")

    def test_empty_id_set_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no ids"):
            self._read("id,name\n")


class WriteFoldConfigsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.parent = self.root / "cfg"
        self.parent.mkdir()
        (self.parent / "training_config.yaml").write_text(PARENT_YAML)
        (self.parent / "rollups.csv").write_text("r\n")
        (self.parent / "included_labels.csv").write_text("i\n")
        self.groups = self.root / "elsewhere.csv"
        self.groups.write_text("site,group_id,fold\ncoralnet,1,0\n")

    def test_each_fold_config_parses_with_its_fold_and_names(self):
        dirs = shg.write_fold_configs(self.parent, self.groups, folds=3)
        self.assertEqual([d.name for d in dirs], ["cfg_fold0", "cfg_fold1", "cfg_fold2"])
        for k, fold_dir in enumerate(dirs):
            self.assertEqual(fold_dir.parent, self.root)
            config = TrainingRunConfig.from_yaml_path(fold_dir / "training_config.yaml")
            options, _, mlflow_options = config.build_options(config_dir=fold_dir)
            self.assertEqual(options.holdout_fold, str(k))
            self.assertEqual(mlflow_options.experiment_name, "my-experiment-cv")
            self.assertEqual(mlflow_options.model_name, f"my-model-cv-fold{k}")
            for path in (
                options.label_rollup_spec_csv,
                options.included_labels_csv,
                options.holdout_groups_csv,
            ):
                self.assertTrue(path and Path(path).parent == fold_dir and Path(path).is_file())
            self.assertEqual(
                Path(options.holdout_groups_csv or "").read_bytes(), self.groups.read_bytes()
            )

    def test_parent_without_mlflow_names_is_an_error(self):
        (self.parent / "training_config.yaml").write_text(
            PARENT_YAML.replace(
                "  experiment_name: my-experiment\n  model_name: my-model\n", "  {}\n"
            )
        )
        with self.assertRaises(ValueError):
            shg.write_fold_configs(self.parent, self.groups, folds=2)


if __name__ == "__main__":
    unittest.main()


class ResolveOutPathTest(unittest.TestCase):
    CONFIG_DIR = Path("/repo/cfg")
    PARENT = CONFIG_DIR / "groups_v2.csv"

    def _resolve(self, out=None, parent=PARENT, write_fold_configs=True):
        return shg.resolve_out_path(out, self.CONFIG_DIR, parent, write_fold_configs)

    def test_default_is_the_parents_referenced_csv(self):
        self.assertEqual(self._resolve(), self.PARENT)

    def test_out_matching_the_parent_csv_is_accepted(self):
        self.assertEqual(self._resolve(out=str(self.PARENT)), self.PARENT)

    def test_out_elsewhere_with_fold_configs_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "holdout_groups_csv"):
            self._resolve(out="/tmp/other.csv")

    def test_out_elsewhere_without_fold_configs_is_used(self):
        got = self._resolve(out="/tmp/other.csv", write_fold_configs=False)
        self.assertEqual(got, Path("/tmp/other.csv").resolve())

    def test_unset_parent_field_defaults_beside_the_config_without_fold_configs(self):
        got = self._resolve(parent=None, write_fold_configs=False)
        self.assertEqual(got, self.CONFIG_DIR / "holdout_groups.csv")

    def test_unset_parent_field_with_fold_configs_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "holdout_groups_csv"):
            self._resolve(parent=None)


class ConfigEnvReachesSettingsTest(unittest.TestCase):
    """The config `env` block must be applied before pyspacer's import-time Settings() is built."""

    def test_env_block_value_reaches_settings_in_a_fresh_interpreter(self):
        yaml_text = PARENT_YAML + "env:\n  CORALNET_TRAIN_DATA_BUCKET: sentinel-bucket\n"
        snippet = textwrap.dedent("""
            import sys
            from pathlib import Path
            import select_holdout_groups as shg

            shg.load_config_then_pyspacer(Path(sys.argv[1]))
            from mermaid_classifier.pyspacer.settings import settings

            print(settings.coralnet_train_data_bucket)
        """)
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / shg.CONFIG_FILENAME).write_text(yaml_text)
            env = {k: v for k, v in os.environ.items() if k != "CORALNET_TRAIN_DATA_BUCKET"}
            env["PYTHONPATH"] = os.pathsep.join([str(SCRIPTS_DIR), env.get("PYTHONPATH", "")])
            result = subprocess.run(
                [sys.executable, "-c", snippet, td],
                capture_output=True,
                text=True,
                env=env,
                cwd=td,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "sentinel-bucket")
