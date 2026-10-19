"""Tests for scripts/select_holdout_groups.py: Beta-source reading and fold-config generation."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from support.paths import add_scripts_to_path

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
