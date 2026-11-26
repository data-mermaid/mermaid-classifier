"""Unit tests for the Pydantic TrainingRunConfig schema.

Schema lives in mermaid_classifier/sagemaker/config.py. These tests
exercise both happy paths (loading a complete YAML) and edge cases
(unknown keys, unknown strategies, out-of-charset model names). Only
BuildOptionsTest imports from mermaid_classifier.pyspacer.*, which is
where build_options constructs the option dataclasses.
"""

from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from pydantic import ValidationError
from segdata_spec.manifest import FeatureSetPin, Pin

from mermaid_classifier.sagemaker.config import TrainingRunConfig

MINIMAL_YAML = textwrap.dedent("""
    dataset:
      datasets:
        - name: coralnet-raw
          version: 20261124-5ab21b3
          manifest_sha256: b74e2a9fa32b340e117c0e6f2e0c6fcc64dc1d2df5011778ba0bde76e2568e8b
          features:
            family: efficientnet-v1
            manifest_sha256: abababababababababababababababababababababababababababababababab
          labels: coralnet
          groups_csv: groups.csv
      feature_extractor_weights: s3://bucket/efficientnet.pt
      label_rollup_spec_csv: rollups.csv
      included_labels_csv: included_labels.csv
      subsample:
        strategy: balanced
        total_annotations: 1000
        min_per_class: 10
      weighting:
        enabled: true
        weight_ratio_cap: 5000.0
    training:
      epochs: 5
      early_stopping_patience: 3
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
    def test_csv_paths_resolve_against_yaml_dir(self):
        with TemporaryDirectory() as td:
            path = _write(Path(td), MINIMAL_YAML)
            config = TrainingRunConfig.from_yaml_path(path)
            self.assertEqual(
                config.dataset.label_rollup_spec_csv_path(path.parent),
                Path(td) / "rollups.csv",
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

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValidationError):
            self._load(
                textwrap.dedent("""\
                  subsample:
                    strategy: not_a_strategy
                    total_annotations: 1000
            """)
            )


class WeightingTest(unittest.TestCase):
    def test_invalid_weight_ratio_cap_rejected(self):
        bad_yaml = MINIMAL_YAML.replace("weight_ratio_cap: 5000.0", "weight_ratio_cap: 0.5")
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
                self.assertEqual(os.environ["MLFLOW_TRACKING_SERVER"], "file:./mlruns")
                self.assertEqual(os.environ["WEIGHTS_LOCATION"], "s3://bucket/weights.pt")


class BuildOptionsTest(unittest.TestCase):
    """build_options() lazily imports pyspacer dataclasses and constructs them."""

    def test_build_options_produces_three_dataclasses(self):
        from mermaid_classifier.pyspacer.options import (
            DatasetOptions,
            MLflowOptions,
            TrainingOptions,
        )

        with TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "rollups.csv").write_text("from_ba_id,from_gf_id,to_ba_id,to_gf_id\n")
            (tmp / "included_labels.csv").write_text("ba_id,gf_id\n")
            path = _write(tmp, MINIMAL_YAML)
            config = TrainingRunConfig.from_yaml_path(path)
            dataset, training, mlflow = config.build_options(config_dir=tmp)
        self.assertIsInstance(dataset, DatasetOptions)
        self.assertIsInstance(training, TrainingOptions)
        self.assertIsInstance(mlflow, MLflowOptions)
        [source] = dataset.datasets
        self.assertEqual(
            source.pin,
            Pin(
                name="coralnet-raw",
                version="20261124-5ab21b3",
                manifest_sha256="b74e2a9fa32b340e117c0e6f2e0c6fcc64dc1d2df5011778ba0bde76e2568e8b",
            ),
        )
        self.assertEqual(source.labels, "coralnet")
        self.assertEqual(source.groups_csv, str(tmp / "groups.csv"))
        self.assertIsNone(dataset.taxonomy)
        # The CSVs the YAML names must reach DatasetOptions as resolved paths;
        # dropping them silently trains against every label.
        self.assertEqual(dataset.label_rollup_spec_csv, str(tmp / "rollups.csv"))
        self.assertEqual(dataset.included_labels_csv, str(tmp / "included_labels.csv"))
        self.assertEqual(training.epochs, 5)
        self.assertEqual(training.early_stopping_patience, 3)


class MLflowModelNameTest(unittest.TestCase):
    """The MLflow registered-model regex must be enforced at load time
    so that a typo in `mlflow.model_name` fails the SageMaker job before
    training starts, not at the registration step after hours of compute.
    """

    def _load_with_model_name(self, name: str) -> TrainingRunConfig:
        yaml_text = MINIMAL_YAML.replace("model_name: TestModel", f"model_name: {name}")
        with TemporaryDirectory() as td:
            path = _write(Path(td), yaml_text)
            return TrainingRunConfig.from_yaml_path(path)

    def test_hyphenated_name_accepted(self):
        config = self._load_with_model_name("top108-192best-v1")
        self.assertEqual(config.mlflow.model_name, "top108-192best-v1")

    def test_names_outside_the_mlflow_charset_are_rejected(self):
        """MLflow accepts alphanumerics separated by hyphens. The two anchors
        carry the leading and trailing cases, so a regex that kept the charset
        but dropped them would take "-v1" and "v1-"."""
        for name in ("top108_192best_v1", "-top108-v1", "top108-v1-", "top108.v1"):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                self._load_with_model_name(name)

    def test_over_57_chars_rejected(self):
        with self.assertRaises(ValidationError):
            self._load_with_model_name("a" * 58)

    def test_exactly_57_chars_accepted(self):
        config = self._load_with_model_name("a" * 57)
        self.assertEqual(config.mlflow.model_name, "a" * 57)

    def test_none_accepted(self):
        # Omitting model_name lets MermaidTrainer auto-generate a safe one
        # via _get_model_name(); the validator must not reject None.
        import yaml as _yaml

        base = _yaml.safe_load(MINIMAL_YAML)
        base["mlflow"].pop("model_name")
        with TemporaryDirectory() as td:
            path = _write(Path(td), _yaml.dump(base))
            config = TrainingRunConfig.from_yaml_path(path)
        self.assertIsNone(config.mlflow.model_name)


class StrictSchemaTest(unittest.TestCase):
    """training_config.yaml is hand-edited, so a key the schema does not know
    is a typo or a setting deleted in an earlier refactor. extra="forbid"
    turns it into a load-time failure instead of a silently ignored line."""

    def test_unknown_key_in_the_dataset_block_is_rejected(self):
        import yaml as _yaml

        base = _yaml.safe_load(MINIMAL_YAML)
        base["dataset"]["drop_growthfroms"] = True  # a typo for drop_growthforms
        with TemporaryDirectory() as td:
            path = _write(Path(td), _yaml.dump(base))
            with self.assertRaisesRegex(ValidationError, "drop_growthfroms"):
                TrainingRunConfig.from_yaml_path(path)


def _load_dict(raw: dict) -> TrainingRunConfig:
    with TemporaryDirectory() as td:
        return TrainingRunConfig.from_yaml_path(_write(Path(td), yaml.dump(raw)))


class DatasetPinsTest(unittest.TestCase):
    """The `datasets:` list is hand-edited, so a malformed pin must fail at
    config load, before a SageMaker job spends anything reaching S3."""

    def test_the_feature_set_pin_carries_exactly_its_entrys_dataset_pin(self):
        with TemporaryDirectory() as td:
            path = _write(Path(td), MINIMAL_YAML)
            config = TrainingRunConfig.from_yaml_path(path)
            dataset, _training, _mlflow = config.build_options(config_dir=Path(td))
        [source] = dataset.datasets
        self.assertEqual(
            source.features,
            FeatureSetPin(
                dataset=Pin(
                    name="coralnet-raw",
                    version="20261124-5ab21b3",
                    manifest_sha256=(
                        "b74e2a9fa32b340e117c0e6f2e0c6fcc64dc1d2df5011778ba0bde76e2568e8b"
                    ),
                ),
                family="efficientnet-v1",
                manifest_sha256="ab" * 32,
            ),
        )

    def test_a_mermaid_entry_without_a_taxonomy_pin_is_rejected(self):
        base = yaml.safe_load(MINIMAL_YAML)
        base["dataset"]["datasets"][0]["labels"] = "mermaid"
        with self.assertRaisesRegex(ValidationError, "taxonomy"):
            _load_dict(base)

    def test_a_mermaid_entry_with_a_taxonomy_pin_builds_the_taxonomy_pin(self):
        base = yaml.safe_load(MINIMAL_YAML)
        base["dataset"]["datasets"][0]["labels"] = "mermaid"
        base["dataset"]["taxonomy"] = {"version": "20261124-e6abfad", "manifest_sha256": "cd" * 32}
        with TemporaryDirectory() as td:
            path = _write(Path(td), yaml.dump(base))
            config = TrainingRunConfig.from_yaml_path(path)
            dataset, _training, _mlflow = config.build_options(config_dir=Path(td))
        self.assertEqual(
            dataset.taxonomy,
            Pin(name="mermaid-benthic", version="20261124-e6abfad", manifest_sha256="cd" * 32),
        )

    def test_a_non_hex_sha_is_rejected_naming_its_field(self):
        cases = {
            "features.manifest_sha256": ("features", "not-a-sha"),
            "datasets.0.manifest_sha256": (None, "B" * 64),
        }
        for field, (nested, value) in cases.items():
            base = yaml.safe_load(MINIMAL_YAML)
            entry = base["dataset"]["datasets"][0]
            (entry[nested] if nested else entry)["manifest_sha256"] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, field):
                _load_dict(base)

    def test_retired_dataset_keys_are_rejected(self):
        for key, value in (
            ("coralnet_manifest_uri", "s3://bucket/m.parquet"),
            ("include_mermaid", True),
        ):
            base = yaml.safe_load(MINIMAL_YAML)
            base["dataset"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValidationError, key):
                _load_dict(base)


class ExampleYamlTest(unittest.TestCase):
    def test_committed_example_loads(self):
        # Resolve relative to the repo root (the tests/ dir is one
        # level deep so the example is at ../sagemaker/configs/example).
        here = Path(__file__).resolve().parent.parent.parent
        example = here / "sagemaker" / "configs" / "example" / "training_config.yaml"
        self.assertTrue(
            example.is_file(),
            f"Example YAML not found at {example}",
        )
        TrainingRunConfig.from_yaml_path(example)


if __name__ == "__main__":
    unittest.main()
