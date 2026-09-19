import unittest
from pathlib import Path

import yaml

from mermaid_classifier.sagemaker.config import TrainingRunConfig

YAML = """
dataset:
  include_mermaid: false
  coralnet_manifest_uri: s3://bucket/coralnet_classifier_manifest_x.parquet
training: {epochs: 1}
mlflow: {}
"""


class ConfigManifestTest(unittest.TestCase):
    def test_manifest_uri_passthrough(self):
        cfg = TrainingRunConfig.model_validate(yaml.safe_load(YAML))
        ds, _, _ = cfg.build_options(config_dir=Path("/tmp"))
        self.assertEqual(
            ds.coralnet_manifest_uri, "s3://bucket/coralnet_classifier_manifest_x.parquet"
        )
