import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd
from spacer.data_classes import DataLocation, ImageLabels
from support.dataset import NoInitDataset
from support.settings import SettingsOverride


class BaseTrainTest(unittest.TestCase):
    def setUp(self):
        self.override = SettingsOverride(aws_anonymous="True")
        self.override.enable()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        self.override.disable()


class LazyLibraryTest(BaseTrainTest):
    """
    The BA and GF libraries hit the MERMAID API in their __init__, so the
    training modules (dataset/runner) reach them through cached accessors
    rather than module-level singletons. Importing one must trigger no network
    call, which is what lets the unit suite run offline.
    """

    def test_importing_training_modules_does_not_call_the_mermaid_api(self):
        module_names = [
            "mermaid_classifier.pyspacer.dataset",
            "mermaid_classifier.pyspacer.runner",
        ]

        def fail(*args, **kwargs):
            raise AssertionError("Importing dataset/runner made a network call to the MERMAID API")

        original_modules = {name: sys.modules.get(name) for name in module_names}
        try:
            with mock.patch("urllib.request.urlopen", side_effect=fail):
                # Force a fresh import so the module body re-executes.
                for name in module_names:
                    sys.modules.pop(name, None)
                for name in module_names:
                    importlib.import_module(name)
        finally:
            # Restore the originally-imported modules for other tests.
            for name, original_module in original_modules.items():
                if original_module is not None:
                    sys.modules[name] = original_module


class AddTrainingSetNamesTest(BaseTrainTest):
    """
    Test add_training_set_names(), which writes the train/ref/val split
    info back into the DuckDB annotations table (as a training_set column)
    for the reporting/stats artifacts.
    """

    def test_training_set_populated_with_filesystem_locations(self):
        """
        prep_annotations_for_pyspacer() builds the PySpacer labels with
        filesystem DataLocations (local download paths), not S3 keys. So
        add_training_set_names() can't read (bucket, feature_vector) off
        the DataLocation directly; it must map the local path back to the
        original S3 location to match the annotations table.

        Otherwise the join matches nothing, every annotation gets a NULL
        training_set, and the stats artifacts wrongly report 100% dropped.
        """
        annotations_df = pd.DataFrame(  # noqa: F841 — referenced by name in DuckDB SQL via Python-scope scanning
            {
                "bucket": ["bucketA", "bucketA", "bucketB"],
                "feature_vector": [
                    "cn/img1.featurevector",
                    "cn/img1.featurevector",
                    "mermaid/img2.featurevector",
                ],
                "row": [10, 30, 50],
                "col": [20, 40, 60],
            }
        )

        dataset = NoInitDataset()
        dataset.duck_conn.execute("CREATE TABLE annotations AS SELECT * FROM annotations_df")

        # The labels carry filesystem DataLocations keyed by local download
        # paths, alongside a mapping back to the original (bucket, key).
        loc1_path = "/tmp/feat/bucketA/cn/img1.featurevector"
        loc2_path = "/tmp/feat/bucketB/mermaid/img2.featurevector"
        dataset._feature_path_to_s3_location = {
            loc1_path: ("bucketA", "cn/img1.featurevector"),
            loc2_path: ("bucketB", "mermaid/img2.featurevector"),
        }

        dataset.labels = SimpleNamespace(
            train=ImageLabels(
                {
                    DataLocation("filesystem", loc1_path): [(10, 20, "BA1")],
                }
            ),
            ref=ImageLabels(
                {
                    DataLocation("filesystem", loc1_path): [(30, 40, "BA1")],
                }
            ),
            val=ImageLabels(
                {
                    DataLocation("filesystem", loc2_path): [(50, 60, "BA2")],
                }
            ),
        )

        dataset.add_training_set_names()

        result = dataset.duck_conn.execute(
            "SELECT bucket, feature_vector, row, col, training_set"
            " FROM annotations ORDER BY bucket, row"
        ).fetchall()

        self.assertListEqual(
            result,
            [
                ("bucketA", "cn/img1.featurevector", 10, 20, "train"),
                ("bucketA", "cn/img1.featurevector", 30, 40, "ref"),
                ("bucketB", "mermaid/img2.featurevector", 50, 60, "val"),
            ],
            msg="Each annotation should be matched to its train/ref/val set",
        )

        dropped = dataset.duck_conn.execute(
            "SELECT count(*) FROM annotations WHERE training_set IS NULL"
        ).fetchone()[0]
        self.assertEqual(dropped, 0, msg="No annotation should be reported as dropped")
