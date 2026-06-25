"""Tests for mermaid_classifier.sagemaker.launcher_config."""
from __future__ import annotations

import unittest

import yaml
from pydantic import ValidationError

from mermaid_classifier.sagemaker.launcher_config import (
    JobConfig, ProcessingConfig, ShardConfig, TrainingConfig,
    parse_run_config,
)


class JobConfigTest(unittest.TestCase):
    def test_required_fields_minimal(self):
        cfg = JobConfig(
            name_prefix="my-run",
            image="mermaid-classifier-jobs:training-latest",
            entrypoint="scripts/sagemaker_train_entrypoint.py",
            instance_type="ml.m5.4xlarge",
            volume_gb=200,
            max_runtime_hours=24,
        )
        self.assertEqual(cfg.instance_count, 1)        # default
        self.assertFalse(cfg.use_spot)                  # default
        self.assertEqual(cfg.env, {})                   # default
        self.assertEqual(cfg.tags, {})                  # default

    def test_missing_required_field_raises(self):
        with self.assertRaises(ValidationError) as ctx:
            JobConfig(
                name_prefix="x",
                # image missing
                entrypoint="x",
                instance_type="x",
                volume_gb=1,
                max_runtime_hours=1,
            )
        self.assertIn("image", str(ctx.exception))

    def test_image_short_form_accepted(self):
        cfg = JobConfig(
            name_prefix="x", image="mermaid-classifier-jobs:training-latest",
            entrypoint="x", instance_type="x",
            volume_gb=1, max_runtime_hours=1)
        self.assertEqual(cfg.image, "mermaid-classifier-jobs:training-latest")

    def test_image_full_uri_accepted(self):
        full = "554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-jobs:training-latest"
        cfg = JobConfig(
            name_prefix="x", image=full,
            entrypoint="x", instance_type="x",
            volume_gb=1, max_runtime_hours=1)
        self.assertEqual(cfg.image, full)


class ShardConfigTest(unittest.TestCase):
    def test_workers_must_be_positive(self):
        with self.assertRaises(ValidationError):
            ShardConfig(items_from="x.csv", workers=0, per_worker_arg="--ids")
        with self.assertRaises(ValidationError):
            ShardConfig(items_from="x.csv", workers=-1, per_worker_arg="--ids")


class ParseRunConfigTest(unittest.TestCase):
    def test_training_only(self):
        y = yaml.safe_dump({
            "job": {
                "name_prefix": "x",
                "image": "y",
                "entrypoint": "z",
                "instance_type": "ml.m5.4xlarge",
                "volume_gb": 200,
                "max_runtime_hours": 24,
            },
            "training": {
                "hyperparameters": {"k": "v"},
            },
        })
        cfg = parse_run_config(y, kind="training")
        self.assertIsNotNone(cfg.training)
        self.assertIsNone(cfg.processing)
        self.assertEqual(cfg.training.hyperparameters, {"k": "v"})

    def test_processing_only(self):
        y = yaml.safe_dump({
            "job": {
                "name_prefix": "x",
                "image": "y",
                "entrypoint": "z",
                "instance_type": "ml.g5.xlarge",
                "volume_gb": 100,
                "max_runtime_hours": 12,
            },
            "processing": {
                "container_args": ["--a=1"],
                "shard": {
                    "items_from": "ids.csv",
                    "workers": 4,
                    "per_worker_arg": "--source-ids",
                },
            },
        })
        cfg = parse_run_config(y, kind="processing")
        self.assertIsNotNone(cfg.processing)
        self.assertEqual(cfg.processing.shard.workers, 4)

    def test_unknown_top_level_key_raises(self):
        y = yaml.safe_dump({
            "job": {
                "name_prefix": "x", "image": "y", "entrypoint": "z",
                "instance_type": "a", "volume_gb": 1, "max_runtime_hours": 1,
            },
            "garbage": {"foo": "bar"},
        })
        with self.assertRaises(ValidationError):
            parse_run_config(y, kind="training")

    def test_classifier_dataset_block_is_ignored(self):
        # The launcher schema must coexist with the existing
        # TrainingRunConfig keys (dataset, training-classifier-specific,
        # mlflow). The launcher ignores those.
        y = yaml.safe_dump({
            "job": {
                "name_prefix": "x", "image": "y", "entrypoint": "z",
                "instance_type": "a", "volume_gb": 1, "max_runtime_hours": 1,
            },
            "dataset": {"include_mermaid": False},
            "mlflow": {"experiment_name": "foo"},
        })
        # No ValidationError; dataset/mlflow are allowed-but-ignored.
        cfg = parse_run_config(y, kind="training", strict=False)
        self.assertIsNone(cfg.training)
