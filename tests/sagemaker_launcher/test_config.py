"""Unit tests for the Pydantic TrainingRunConfig schema.

Schema lives in mermaid_classifier/sagemaker/config.py. These tests
exercise both happy paths (loading a complete YAML) and edge cases
(unknown keys, unknown strategies, out-of-charset model names). Only
BuildOptionsTest imports from mermaid_classifier.pyspacer.*, which is
where build_options constructs the option dataclasses.
"""

from __future__ import annotations

import filecmp
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from mermaid_classifier.sagemaker.config import TrainingRunConfig

MINIMAL_YAML = textwrap.dedent("""
    dataset:
      include_mermaid: true
      coralnet_manifest_uri: s3://bucket/coralnet_manifest.parquet
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

    def test_class_target_multipliers_round_trip_to_options(self):
        import yaml as _yaml

        base = _yaml.safe_load(MINIMAL_YAML)
        base["dataset"]["subsample"] = {
            "strategy": "balanced",
            "total_annotations": 1000,
            "class_target_multipliers": {"20090BF4-868E::": 3.0},
        }
        with TemporaryDirectory() as td:
            tmp = Path(td)
            config = TrainingRunConfig.from_yaml_path(_write(tmp, _yaml.dump(base)))
            dataset = config.build_options(config_dir=tmp)[0]
        assert dataset.subsample is not None
        self.assertEqual(dataset.subsample.multipliers_by_class(), {("20090bf4-868e", ""): 3.0})

    def test_bad_multiplier_key_rejected_at_config_load(self):
        with self.assertRaises(ValidationError):
            self._load(
                textwrap.dedent("""\
                  subsample:
                    strategy: balanced
                    total_annotations: 1000
                    class_target_multipliers:
                      not-a-key: 3.0
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
        self.assertEqual(dataset.coralnet_manifest_uri, "s3://bucket/coralnet_manifest.parquet")
        # The CSVs the YAML names must reach DatasetOptions as resolved paths;
        # dropping them silently trains against every label.
        self.assertEqual(dataset.label_rollup_spec_csv, str(tmp / "rollups.csv"))
        self.assertEqual(dataset.included_labels_csv, str(tmp / "included_labels.csv"))
        self.assertEqual(training.epochs, 5)
        self.assertEqual(training.early_stopping_patience, 3)
        self.assertEqual(training.early_stopping_metric, "val_loss")


class EarlyStoppingMetricTest(unittest.TestCase):
    def _load(self, metric: str) -> TrainingRunConfig:
        yaml_text = MINIMAL_YAML.replace(
            "early_stopping_patience: 3",
            f"early_stopping_patience: 3\n  early_stopping_metric: {metric}",
        )
        with TemporaryDirectory() as td:
            return TrainingRunConfig.from_yaml_path(_write(Path(td), yaml_text))

    def test_val_accuracy_round_trips_to_training_options(self):
        config = self._load("val_accuracy")
        with TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "rollups.csv").write_text("from_ba_id,from_gf_id,to_ba_id,to_gf_id\n")
            (tmp / "included_labels.csv").write_text("ba_id,gf_id\n")
            _, training, _ = config.build_options(config_dir=tmp)
        self.assertEqual(training.early_stopping_metric, "val_accuracy")

    def test_invalid_metric_rejected_at_load(self):
        with self.assertRaises(ValidationError):
            self._load("val_f1")


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


class HoldoutOptionsTest(unittest.TestCase):
    """The grouped-holdout fields parse, resolve, and default to today's behaviour."""

    def _build(self, extra_dataset_yaml: str = ""):
        import yaml as _yaml

        base = _yaml.safe_load(MINIMAL_YAML)
        base["dataset"].update(_yaml.safe_load(extra_dataset_yaml) or {})
        with TemporaryDirectory() as td:
            tmp = Path(td)
            path = _write(tmp, _yaml.dump(base))
            config = TrainingRunConfig.from_yaml_path(path)
            return config, config.build_options(config_dir=tmp)[0], tmp

    def test_defaults_reproduce_the_ungrouped_run(self):
        config, dataset, _ = self._build()
        self.assertIsNone(config.dataset.mermaid_project_lookup_uri)
        self.assertIsNone(config.dataset.holdout_groups_csv)
        self.assertEqual(config.dataset.holdout_fold, "test")
        self.assertIsNone(dataset.mermaid_project_lookup_uri)
        self.assertIsNone(dataset.holdout_groups_csv)
        self.assertEqual(dataset.holdout_fold, "test")

    def test_fields_reach_dataset_options(self):
        _, dataset, tmp = self._build(
            textwrap.dedent("""\
                mermaid_project_lookup_uri: s3://private/lookup.parquet
                holdout_groups_csv: holdout_groups.csv
                holdout_fold: "2"
            """)
        )
        self.assertEqual(dataset.mermaid_project_lookup_uri, "s3://private/lookup.parquet")
        self.assertEqual(dataset.holdout_groups_csv, str(tmp / "holdout_groups.csv"))
        self.assertEqual(dataset.holdout_fold, "2")

    def test_split_mode_defaults_to_points_and_reaches_dataset_options(self):
        config, dataset, _ = self._build()
        self.assertEqual(config.dataset.split_mode, "points")
        self.assertEqual(dataset.split_mode, "points")
        _, dataset, _ = self._build("split_mode: images")
        self.assertEqual(dataset.split_mode, "images")

    def test_unknown_split_mode_is_rejected(self):
        with self.assertRaises(ValidationError):
            self._build("split_mode: vectors")

    def test_integer_fold_in_yaml_is_accepted_as_a_string(self):
        config, _, _ = self._build("holdout_fold: 3")
        self.assertEqual(config.dataset.holdout_fold, "3")

    def test_invalid_fold_is_rejected(self):
        with self.assertRaises(ValidationError):
            self._build("holdout_fold: fold-a")


class SourceSpreadOptionsTest(unittest.TestCase):
    """within_class, seed and holdout_max_per_class parse and default to today's behaviour."""

    def _build(self, dataset_patch: dict):
        import yaml as _yaml

        base = _yaml.safe_load(MINIMAL_YAML)
        base["dataset"].update(dataset_patch)
        with TemporaryDirectory() as td:
            tmp = Path(td)
            path = _write(tmp, _yaml.dump(base))
            config = TrainingRunConfig.from_yaml_path(path)
            return config, config.build_options(config_dir=tmp)[0]

    def test_defaults_reproduce_the_ordered_subsample_and_default_holdout_cap(self):
        config, dataset = self._build({})
        assert config.dataset.subsample is not None
        assert dataset.subsample is not None
        self.assertEqual(config.dataset.subsample.within_class, "ordered")
        self.assertEqual(config.dataset.subsample.seed, 0)
        self.assertEqual(dataset.subsample.within_class, "ordered")
        self.assertEqual(dataset.subsample.seed, 0)
        self.assertEqual(config.dataset.holdout_max_per_class, 3000)
        self.assertEqual(dataset.holdout_max_per_class, 3000)

    def test_fields_reach_the_option_dataclasses(self):
        _, dataset = self._build(
            {
                "subsample": {
                    "strategy": "balanced",
                    "total_annotations": 1000,
                    "within_class": "source_spread",
                    "seed": 5,
                },
                "holdout_max_per_class": 500,
            }
        )
        assert dataset.subsample is not None
        self.assertEqual(dataset.subsample.within_class, "source_spread")
        self.assertEqual(dataset.subsample.seed, 5)
        self.assertEqual(dataset.holdout_max_per_class, 500)

    def test_null_holdout_cap_disables_the_cap(self):
        config, dataset = self._build({"holdout_max_per_class": None})
        self.assertIsNone(config.dataset.holdout_max_per_class)
        self.assertIsNone(dataset.holdout_max_per_class)

    def test_unknown_within_class_rejected(self):
        with self.assertRaises(ValidationError):
            self._build(
                {
                    "subsample": {
                        "strategy": "balanced",
                        "total_annotations": 1000,
                        "within_class": "random",
                    }
                }
            )

    def test_non_positive_holdout_cap_rejected(self):
        with self.assertRaises(ValidationError):
            self._build({"holdout_max_per_class": 0})


class ValGroupsOptionsTest(unittest.TestCase):
    """val_groups_csv and val_max_per_class parse, resolve, and pass their cross-field rules."""

    GROUPED_VAL = {
        "val_groups_csv": "val_groups.csv",
        "split_mode": "images",
        "ref_val_ratios": [0.1, 0.0],
    }

    def _build(self, dataset_patch: dict):
        import yaml as _yaml

        base = _yaml.safe_load(MINIMAL_YAML)
        base["dataset"].update(dataset_patch)
        with TemporaryDirectory() as td:
            tmp = Path(td)
            path = _write(tmp, _yaml.dump(base))
            config = TrainingRunConfig.from_yaml_path(path)
            return config.build_options(config_dir=tmp)[0], tmp

    def test_fields_reach_dataset_options(self):
        dataset, tmp = self._build({**self.GROUPED_VAL, "val_max_per_class": 250})
        self.assertEqual(dataset.val_groups_csv, str(tmp / "val_groups.csv"))
        self.assertEqual(dataset.val_max_per_class, 250)
        self.assertEqual(dataset.ref_val_ratios, (0.1, 0.0))

    def test_defaults_leave_val_to_the_split(self):
        dataset, _ = self._build({})
        self.assertIsNone(dataset.val_groups_csv)
        self.assertEqual(dataset.val_max_per_class, 3000)

    def test_inconsistent_val_settings_are_rejected(self):
        cases = {
            "points split with val groups": {**self.GROUPED_VAL, "split_mode": "points"},
            "val share with val groups": {**self.GROUPED_VAL, "ref_val_ratios": [0.1, 0.1]},
            "no val share without val groups": {
                "split_mode": "images",
                "ref_val_ratios": [0.1, 0.0],
            },
            "zero val cap": {**self.GROUPED_VAL, "val_max_per_class": 0},
        }
        for name, patch in cases.items():
            with self.subTest(name), self.assertRaises(ValidationError):
                self._build(patch)


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
        config = TrainingRunConfig.from_yaml_path(example)
        self.assertIsNotNone(config.dataset.subsample)


class GvalTurfCcaConfigTest(unittest.TestCase):
    """The committed gval_turfcca config stays paired with the min-support holdout."""

    REPO = Path(__file__).resolve().parent.parent.parent
    CONFIGS = REPO / "sagemaker" / "configs"
    NEW = CONFIGS / "coralnet_all_plus_mermaid_coarse_grouped_minsupport_gval_turfcca"
    BASE = CONFIGS / "coralnet_all_plus_mermaid_coarse_grouped_minsupport"
    MULTIPLIER_KEYS = (
        "20090bf4-868e-431b-974c-ab9be5bbdb5f",
        "f4df7abd-3d51-42fb-8cab-5102b95fad8e",
    )

    def test_config_loads_and_builds_options(self):
        config = TrainingRunConfig.from_yaml_path(self.NEW / "training_config.yaml")
        config.build_options(config_dir=self.NEW)

    def test_holdout_groups_identical_to_min_support(self):
        self.assertTrue(
            filecmp.cmp(
                self.NEW / "holdout_groups.csv",
                self.BASE / "holdout_groups.csv",
                shallow=False,
            )
        )

    def test_val_groups_disjoint_from_holdout(self):
        from mermaid_classifier.pyspacer.label_specs import HoldoutGroups, ValGroups

        with (
            open(self.NEW / "val_groups.csv") as val_file,
            open(self.NEW / "holdout_groups.csv") as holdout_file,
        ):
            val = ValGroups(val_file)
            holdout = HoldoutGroups(holdout_file)
        val.check_disjoint(holdout, "test")

    def test_multiplier_keys_in_included_labels(self):
        included = (self.NEW / "included_labels.csv").read_text().lower()
        for key in self.MULTIPLIER_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, included)


class GvalCca3Turf1p5ConfigTest(GvalTurfCcaConfigTest):
    """The CCA x3 / Turf x1.5 follow-up shares the gval_turfcca val groups and holdout."""

    NEW = (
        GvalTurfCcaConfigTest.CONFIGS
        / "coralnet_all_plus_mermaid_coarse_grouped_minsupport_gval_cca3_turf1p5"
    )
    PREVIOUS = GvalTurfCcaConfigTest.NEW

    def test_val_groups_identical_to_gval_turfcca(self):
        self.assertTrue(
            filecmp.cmp(
                self.NEW / "val_groups.csv", self.PREVIOUS / "val_groups.csv", shallow=False
            )
        )

    def test_multipliers_are_cca_3_and_turf_1_5(self):
        config = TrainingRunConfig.from_yaml_path(self.NEW / "training_config.yaml")
        assert config.dataset.subsample is not None
        self.assertEqual(
            config.dataset.subsample.class_target_multipliers,
            {
                "20090bf4-868e-431b-974c-ab9be5bbdb5f::": 1.5,
                "f4df7abd-3d51-42fb-8cab-5102b95fad8e::": 3.0,
            },
        )


if __name__ == "__main__":
    unittest.main()


class GvalCca3Turf1p5AccStopConfigTest(GvalTurfCcaConfigTest):
    """The accuracy-stopping rerun keeps the Turf x1.5 data and changes only early stopping."""

    NEW = (
        GvalTurfCcaConfigTest.CONFIGS
        / "coralnet_all_plus_mermaid_coarse_grouped_minsupport_gval_cca3_turf1p5_accstop"
    )
    PREVIOUS = (
        GvalTurfCcaConfigTest.CONFIGS
        / "coralnet_all_plus_mermaid_coarse_grouped_minsupport_gval_cca3_turf1p5"
    )

    def test_data_files_identical_to_turf1p5(self):
        for name in ("val_groups.csv", "excluded_images.csv", "rollups.csv", "included_labels.csv"):
            with self.subTest(name=name):
                self.assertTrue(filecmp.cmp(self.NEW / name, self.PREVIOUS / name, shallow=False))

    def test_early_stopping_is_val_accuracy_with_patience_8(self):
        config = TrainingRunConfig.from_yaml_path(self.NEW / "training_config.yaml")
        _, training_options, _ = config.build_options(config_dir=self.NEW)
        self.assertEqual(training_options.early_stopping_metric, "val_accuracy")
        self.assertEqual(training_options.early_stopping_patience, 8)
