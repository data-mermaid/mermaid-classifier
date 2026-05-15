"""Unit tests for the Pydantic TrainingRunConfig schema.

Schema lives in mermaid_classifier/sagemaker/config.py. These tests
exercise both happy paths (loading a complete YAML) and edge cases
(missing required fields, unknown strategies). They intentionally do
NOT import from mermaid_classifier.pyspacer.* to keep the test fast
and to verify the schema is decoupled from the heavy pyspacer imports.
"""
from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from mermaid_classifier.sagemaker.config import TrainingRunConfig


MINIMAL_YAML = textwrap.dedent("""
    dataset:
      include_mermaid: true
      coralnet_sources_csv: sources.csv
      label_rollup_spec_csv: rollups.csv
      included_labels_csv: included_labels.csv
      subsample:
        strategy: balanced
        total_annotations: 1000
        min_per_class: 10
      weighting:
        strategy: effective_number
        alpha: 0.5
    training:
      epochs: 5
      hidden_layer_sizes: [200, 100]
      learning_rate_init: 0.001
      early_stopping_patience: 3
      random_state: 42
    mlflow:
      experiment_name: test-experiment
      model_name: TestModel
    env:
      MLFLOW_TRACKING_SERVER: file:./mlruns
      WEIGHTS_LOCATION: s3://bucket/weights.pt
""").lstrip()


def _write(tmp: Path, text: str) -> Path:
    p = tmp / "training_config.yaml"
    p.write_text(text)
    return p


class LoadHappyPathTest(unittest.TestCase):

    def test_minimal_yaml_loads(self):
        with TemporaryDirectory() as td:
            path = _write(Path(td), MINIMAL_YAML)
            config = TrainingRunConfig.from_yaml_path(path)
        self.assertTrue(config.dataset.include_mermaid)
        self.assertEqual(config.training.epochs, 5)
        self.assertEqual(config.training.hidden_layer_sizes, (200, 100))
        self.assertEqual(config.mlflow.experiment_name, "test-experiment")
        self.assertEqual(
            config.env["MLFLOW_TRACKING_SERVER"], "file:./mlruns")

    def test_csv_paths_resolve_against_yaml_dir(self):
        with TemporaryDirectory() as td:
            path = _write(Path(td), MINIMAL_YAML)
            config = TrainingRunConfig.from_yaml_path(path)
            self.assertEqual(
                config.dataset.coralnet_sources_csv_path(path.parent),
                Path(td) / "sources.csv",
            )


class SubsampleStrategiesTest(unittest.TestCase):

    def _load(self, subsample_yaml: str) -> TrainingRunConfig:
        import yaml as _yaml
        base = _yaml.safe_load(MINIMAL_YAML)
        patch = _yaml.safe_load(subsample_yaml)
        base["dataset"]["subsample"] = patch.get("subsample")
        with TemporaryDirectory() as td:
            path = _write(Path(td), _yaml.dump(base))
            return TrainingRunConfig.from_yaml_path(path)

    def test_stratified_with_total_annotations_loads(self):
        config = self._load(textwrap.dedent("""\
              subsample:
                strategy: stratified
                total_annotations: 5000
        """))
        self.assertEqual(config.dataset.subsample.strategy, "stratified")
        self.assertEqual(config.dataset.subsample.total_annotations, 5000)

    def test_soft_balanced_requires_balance_alpha(self):
        config = self._load(textwrap.dedent("""\
              subsample:
                strategy: soft_balanced
                total_annotations: 5000
                balance_alpha: 0.5
        """))
        self.assertEqual(config.dataset.subsample.strategy, "soft_balanced")
        self.assertEqual(config.dataset.subsample.balance_alpha, 0.5)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValidationError):
            self._load(textwrap.dedent("""\
                  subsample:
                    strategy: not_a_strategy
                    total_annotations: 1000
            """))


class WeightingTest(unittest.TestCase):

    def test_weighting_yaml_overrides_default(self):
        with TemporaryDirectory() as td:
            path = _write(Path(td), MINIMAL_YAML)
            config = TrainingRunConfig.from_yaml_path(path)
        self.assertTrue(config.dataset.weighting.enabled)
        self.assertEqual(
            config.dataset.weighting.strategy, "effective_number")
        self.assertEqual(config.dataset.weighting.alpha, 0.5)

    def test_weighting_default_strategy_is_tree_balanced(self):
        """Bare WeightingConfig() should use tree_balanced_ba_flat_gf,
        matching the default in mermaid_classifier.training.sample_weighting.options.
        """
        from mermaid_classifier.sagemaker.config import WeightingConfig
        w = WeightingConfig()
        self.assertTrue(w.enabled)
        self.assertEqual(w.strategy, "tree_balanced_ba_flat_gf")
        self.assertEqual(w.alpha, 0.5)
        self.assertIsNone(w.weight_ratio_cap)

    def test_invalid_alpha_rejected(self):
        bad_yaml = MINIMAL_YAML.replace("alpha: 0.5", "alpha: 1.5")
        with TemporaryDirectory() as td:
            path = _write(Path(td), bad_yaml)
            with self.assertRaises(ValidationError):
                TrainingRunConfig.from_yaml_path(path)


class ApplyEnvTest(unittest.TestCase):

    def test_apply_env_writes_to_os_environ(self):
        import os
        from unittest import mock
        with TemporaryDirectory() as td:
            path = _write(Path(td), MINIMAL_YAML)
            config = TrainingRunConfig.from_yaml_path(path)
            with mock.patch.dict(os.environ, {}, clear=False):
                # Pre-existing env vars (if any) stay; we only assert
                # that ours are written.
                os.environ.pop("MLFLOW_TRACKING_SERVER", None)
                os.environ.pop("WEIGHTS_LOCATION", None)
                config.apply_env()
                self.assertEqual(
                    os.environ["MLFLOW_TRACKING_SERVER"], "file:./mlruns")
                self.assertEqual(
                    os.environ["WEIGHTS_LOCATION"],
                    "s3://bucket/weights.pt")

    def test_apply_env_with_empty_block_is_noop(self):
        import os
        from unittest import mock
        yaml_no_env = MINIMAL_YAML.replace(
            "env:\n  MLFLOW_TRACKING_SERVER: file:./mlruns\n"
            "  WEIGHTS_LOCATION: s3://bucket/weights.pt\n",
            "",
        )
        with TemporaryDirectory() as td:
            path = _write(Path(td), yaml_no_env)
            config = TrainingRunConfig.from_yaml_path(path)
            self.assertEqual(config.env, {})
            with mock.patch.dict(os.environ, {}, clear=False):
                # apply_env on empty dict must not raise.
                config.apply_env()


class RequiredFieldsTest(unittest.TestCase):

    def test_missing_dataset_block_rejected(self):
        bad = "training:\n  epochs: 1\nmlflow:\n  experiment_name: x\n"
        with TemporaryDirectory() as td:
            path = _write(Path(td), bad)
            with self.assertRaises(ValidationError):
                TrainingRunConfig.from_yaml_path(path)


class BuildOptionsTest(unittest.TestCase):
    """build_options() lazily imports pyspacer dataclasses and constructs them.

    These tests DO import pyspacer (transitively). Skipped if the
    pyspacer extras aren't installed.
    """

    def test_build_options_produces_three_dataclasses(self):
        try:
            from mermaid_classifier.pyspacer.train import (
                DatasetOptions, MLflowOptions, TrainingOptions,
            )
        except Exception:
            self.skipTest("pyspacer extras not installed")
        with TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "sources.csv").write_text("id\n123\n")
            (tmp / "rollups.csv").write_text(
                "from_ba_id,from_gf_id,to_ba_id,to_gf_id\n")
            (tmp / "included_labels.csv").write_text("ba_id,gf_id\n")
            path = _write(tmp, MINIMAL_YAML)
            config = TrainingRunConfig.from_yaml_path(path)
            dataset, training, mlflow = config.build_options(
                config_dir=tmp)
        self.assertIsInstance(dataset, DatasetOptions)
        self.assertIsInstance(training, TrainingOptions)
        self.assertIsInstance(mlflow, MLflowOptions)
        self.assertEqual(
            dataset.coralnet_sources_csv, str(tmp / "sources.csv"))
        self.assertEqual(training.epochs, 5)
        self.assertEqual(training.hidden_layer_sizes, (200, 100))


class ExampleYamlTest(unittest.TestCase):

    def test_committed_example_loads(self):
        # Resolve relative to the repo root (the tests/ dir is one
        # level deep so the example is at ../sagemaker/configs/example).
        here = Path(__file__).resolve().parent.parent.parent
        example = here / "sagemaker" / "configs" / "example" \
            / "training_config.yaml"
        self.assertTrue(
            example.is_file(),
            f"Example YAML not found at {example}",
        )
        config = TrainingRunConfig.from_yaml_path(example)
        self.assertIsNotNone(config.dataset.subsample)
        self.assertEqual(
            config.dataset.subsample.strategy, "balanced")


if __name__ == "__main__":
    unittest.main()
