"""Tests for scripts/release_artifact.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

# Allow importing scripts/release_artifact.py (mirrors test_generate_training_config).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import release_artifact as ra  # noqa: E402


class VersionValidationTest(unittest.TestCase):
    def test_accepts_vN(self):
        ra.validate_version('v3')  # no raise
        ra.validate_version('v0')

    def test_rejects_bad_versions(self):
        for bad in ('3', 'v', 'latest', 'v1.0', 'V3', 'v-1', ''):
            with self.assertRaises(ValueError, msg=bad):
                ra.validate_version(bad)


class ParseS3UriTest(unittest.TestCase):
    def test_parses_bucket_and_key(self):
        bucket, key = ra.parse_s3_uri('s3://mermaid-config/classifier/v1/efficientnet_weights.pt')
        self.assertEqual(bucket, 'mermaid-config')
        self.assertEqual(key, 'classifier/v1/efficientnet_weights.pt')

    def test_rejects_non_s3(self):
        for bad in ('https://x/y', '/local/path', 's3://', 'file.pt'):
            with self.assertRaises(ValueError, msg=bad):
                ra.parse_s3_uri(bad)
