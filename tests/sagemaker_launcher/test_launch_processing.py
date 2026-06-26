"""Tests for scripts.launch_processing (mermaid-classifier)."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launch_processing as lp  # type: ignore


class ChunkSourcesTest(unittest.TestCase):
    def test_round_robin_distribution(self):
        chunks = lp.chunk_items(["a", "b", "c", "d", "e"], n_workers=2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(sorted([x for c in chunks for x in c]), ["a", "b", "c", "d", "e"])
        # Round-robin: chunk sizes differ by at most 1.
        self.assertLessEqual(abs(len(chunks[0]) - len(chunks[1])), 1)

    def test_more_workers_than_items_drops_empty(self):
        chunks = lp.chunk_items(["a", "b"], n_workers=5)
        self.assertEqual(len(chunks), 2)  # 3 empty chunks dropped

    def test_single_item(self):
        self.assertEqual(lp.chunk_items(["a"], n_workers=4), [["a"]])

    def test_zero_workers_raises(self):
        with self.assertRaises(ValueError):
            lp.chunk_items(["a"], n_workers=0)


class LoadItemsTest(unittest.TestCase):
    def test_auto_detect_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ids.csv"
            with open(p, "w") as f:
                w = csv.writer(f)
                w.writerow(["source_id", "extra"])
                w.writerow(["1", "x"])
                w.writerow(["2", "y"])
            items = lp.load_items(p, column=None)
            self.assertEqual(items, ["1", "2"])

    def test_explicit_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ids.csv"
            with open(p, "w") as f:
                w = csv.writer(f)
                w.writerow(["a", "b"])
                w.writerow(["1", "2"])
            items = lp.load_items(p, column="b")
            self.assertEqual(items, ["2"])


class BuildProcessingRequestTest(unittest.TestCase):
    @patch("launch_processing.datetime")
    def test_request_shape_no_shard(self, mock_dt):
        mock_dt.now.return_value.strftime.return_value = "20260525T120000Z"
        from mermaid_classifier.sagemaker.launcher_config import parse_run_config
        cfg = parse_run_config(
            """
job:
  name_prefix: mermaid-features
  image: mermaid-classifier-jobs:features-latest
  entrypoint: scripts/build_feature_bucket.py
  instance_type: ml.g5.xlarge
  volume_gb: 100
  max_runtime_hours: 12
processing:
  container_args:
    - --target-bucket=2605-coralnet-public-sources
    - --skip-existing
""",
            kind="processing", strict=True)
        req = lp.build_processing_request(
            cfg=cfg, run_id="mermaid-features-20260525T120000Z",
            worker_idx=0, worker_items=None)
        self.assertEqual(req["ProcessingJobName"], "mermaid-features-20260525T120000Z-0")
        self.assertEqual(req["RoleArn"], "arn:aws:iam::554812291621:role/dev-sm-execution-role")
        self.assertEqual(
            req["AppSpecification"]["ImageUri"],
            "554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-jobs:features-latest")
        self.assertIn("--skip-existing", req["AppSpecification"]["ContainerArguments"])
        self.assertEqual(req["ProcessingResources"]["ClusterConfig"]["InstanceType"], "ml.g5.xlarge")
        self.assertEqual(req["StoppingCondition"]["MaxRuntimeInSeconds"], 12 * 3600)

    @patch("launch_processing.datetime")
    def test_request_shape_with_shard(self, mock_dt):
        mock_dt.now.return_value.strftime.return_value = "20260525T120000Z"
        from mermaid_classifier.sagemaker.launcher_config import parse_run_config
        cfg = parse_run_config(
            """
job:
  name_prefix: mermaid-features
  image: mermaid-classifier-jobs:features-latest
  entrypoint: scripts/build_feature_bucket.py
  instance_type: ml.g5.xlarge
  volume_gb: 100
  max_runtime_hours: 12
processing:
  container_args:
    - --target-bucket=foo
  shard:
    items_from: sources.csv
    workers: 4
    per_worker_arg: --source-ids
""",
            kind="processing", strict=True)
        req = lp.build_processing_request(
            cfg=cfg, run_id="mermaid-features-20260525T120000Z",
            worker_idx=2, worker_items=["1", "5", "9"])
        args = req["AppSpecification"]["ContainerArguments"]
        self.assertIn("--source-ids", args)
        idx = args.index("--source-ids")
        self.assertEqual(args[idx + 1], "1,5,9")
        # The non-shard args are preserved:
        self.assertIn("--target-bucket=foo", args)
