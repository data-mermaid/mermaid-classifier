import importlib
import sys
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pandas as pd
from spacer.data_classes import DataLocation, ImageLabels

from mermaid_classifier.common.benthic_attributes import CoralNetMermaidMapping
from mermaid_classifier.pyspacer.dataset import TrainingDataset
from mermaid_classifier.pyspacer.options import Artifacts, DatasetOptions, Sites
from mermaid_classifier.pyspacer.settings import settings


class SettingsOverride:
    """
    Override the specified Pydantic settings from a call of enable()
    until a call of disable().

    Example usage:
    override = SettingsOverride(aws_anonymous=True, aws_region='ca-central-1')
    override.enable()
    <some code that depends on the above settings>
    override.disable()

    Some parts are from
    https://rednafi.com/python/patch-pydantic-settings-in-pytest/
    """

    def __init__(self, **kwargs):
        self.options = kwargs
        super().__init__()

    def enable(self):
        # Make a copy of the original settings
        self.original_settings = settings.model_copy()

        # Patch the settings with kwargs
        for key, val in self.options.items():
            # Raise an error if kwargs contains a nonexistent setting
            if not hasattr(settings, key):
                raise ValueError(f"Unknown setting: {key}")
            setattr(settings, key, val)

    def disable(self):
        # Restore the original settings
        settings.__dict__.update(self.original_settings.__dict__)


@contextmanager
def override_settings(**kwargs):
    """
    Override the specified Pydantic settings for the duration of the
    context manager.
    """
    override = SettingsOverride(**kwargs)
    override.enable()
    yield
    override.disable()


class NoInitDataset(TrainingDataset):
    """
    init does a lot of stuff in TrainingDataset. When testing, we sometimes
    just want access to the other methods of the class.
    So here we make init barebones.
    """

    def __init__(self):
        self._duck_conn = None
        self.artifacts = Artifacts()
        self._feature_temp_dir = None
        self._feature_dir = "/tmp/mermaid_features_test"


class BaseTrainTest(unittest.TestCase):
    def setUp(self):
        self.override = SettingsOverride(aws_anonymous="True")
        self.override.enable()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        self.override.disable()


def same_char_uuid(char: str):
    """
    If you pass in '0', you get '00000000-0000-0000-0000-000000000000'.
    Passing in 0-9 or a-f should result in a valid uuid4 string.
    """
    return "-".join(
        [
            "".join([char] * 8),
            "".join([char] * 4),
            "".join([char] * 4),
            "".join([char] * 4),
            "".join([char] * 12),
        ]
    )


class ReadCoralNetDataTest(BaseTrainTest):
    """
    Test read_coralnet_manifest().
    """

    def test_gfs_present_and_empty(self):
        """
        When CoralNet annotations have their labels mapped to MERMAID BAs
        and GFs, the case with no GF has the GF read in as null (from JSON).
        However, NULL GFs in DuckDB can be problematic with later data
        manipulations; for example, JOINing on a column with NULLs in it
        can result in rows getting accidentally lost.

        So we check that null GFs become empty strings rather than NULL
        in DuckDB.
        And test alongside with-GF annotations.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        dataset = NoInitDataset()

        source_id = 23

        # CoralNet annotations come from the manifest parquet, one row per
        # annotation point with columns source_id, image_id, row, col,
        # coralnet_id (see mermaid_classifier.coralnet.manifest).
        manifest_table = pa.table(
            {
                "source_id": pa.array([source_id, source_id], pa.int32()),
                "image_id": pa.array(["12345", "67890"], pa.string()),
                "row": pa.array([2000, 1500], pa.int32()),
                "col": pa.array([1200, 2800], pa.int32()),
                "coralnet_id": pa.array([123, 456], pa.int32()),
            }
        )

        # Mock CN-MM mapping to use.
        mapping = [
            {
                "benthic_attribute_id": same_char_uuid("0"),
                "growth_form_id": same_char_uuid("1"),
                "benthic_attribute_name": "BA1",
                "growth_form_name": "GF1",
                "provider_id": "123",
                "provider_label": "Label123",
            },
            {
                "benthic_attribute_id": same_char_uuid("2"),
                # We expect the API's mapping to have null/None
                # for blank growth forms.
                "growth_form_id": None,
                "benthic_attribute_name": "BA2",
                "growth_form_name": None,
                "provider_id": "456",
                "provider_label": "Label456",
            },
        ]

        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            delete_on_close=False,
        ) as manifest_f:
            manifest_f.close()
            pq.write_table(manifest_table, manifest_f.name)
            dataset.options = DatasetOptions(coralnet_manifest_uri=manifest_f.name)

            with mock.patch.object(
                CoralNetMermaidMapping,
                "_download_mapping",
            ) as mock_download_mapping:
                mock_download_mapping.return_value = mapping

                dataset.read_coralnet_manifest()

        result_tuples = dataset.duck_conn.execute(
            "SELECT image_id, row, col, benthic_attribute_id, growth_form_id FROM annotations"
        ).fetchall()
        result_tuples.sort()

        self.assertListEqual(
            list(result_tuples[0]),
            ["12345", 2000, 1200, same_char_uuid("0"), same_char_uuid("1")],
        )
        # Empty GF should show as ''.
        self.assertListEqual(
            list(result_tuples[1]),
            ["67890", 1500, 2800, same_char_uuid("2"), ""],
        )
        # Distinct source IDs are recorded for MLflow logging.
        self.assertEqual(dataset.coralnet_source_ids, [str(source_id)])

    def test_missing_column_raises_friendly_error(self):
        """
        A parquet missing the required 'coralnet_id' column must raise a
        RuntimeError with a message that names the manifest path and the
        missing column, not a raw DuckDB error.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        dataset = NoInitDataset()

        # Build a parquet that is intentionally missing 'coralnet_id'.
        bad_table = pa.table(
            {
                "source_id": pa.array([1], pa.int32()),
                "image_id": pa.array(["abc"], pa.string()),
                "row": pa.array([10], pa.int32()),
                "col": pa.array([11], pa.int32()),
                # coralnet_id is deliberately absent
            }
        )

        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            delete_on_close=False,
        ) as manifest_f:
            manifest_f.close()
            pq.write_table(bad_table, manifest_f.name)
            dataset.options = DatasetOptions(coralnet_manifest_uri=manifest_f.name)

            with self.assertRaises(RuntimeError) as ctx:
                dataset.read_coralnet_manifest()

        msg = str(ctx.exception)
        self.assertIn(manifest_f.name, msg)
        self.assertIn("coralnet_id", msg)


class ReadMermaidDataTest(BaseTrainTest):
    """
    Test read_mermaid_data().
    """

    def test_gfs_present_and_empty(self):
        """
        Test the following regarding no-GF annotations:
        1. CoralNet annotations with no GF don't get dropped during the
        empty-value normalization step. (This was a problem before.)
        2. MERMAID annotations with no GF end up with a GF of '', empty string.

        And test alongside with-GF annotations.
        """
        dataset = NoInitDataset()

        # Write a couple of coralnet annotations to DuckDB.
        # One with growth form, one without (this is represented as the
        # empty string '').
        cn_annotations_df = pd.DataFrame(  # noqa: F841 — referenced by name in DuckDB SQL via Python-scope scanning
            {
                "site": ["coralnet"] * 2,
                "bucket": ["cn-data"] * 2,
                "project_id": ["12", "34"],
                "image_id": ["12345", "67890"],
                "feature_vector": ["s12/i12345.fv", "s34/i67890.fv"],
                "row": [2000, 1500],
                "col": [1200, 2800],
                "label_id": ["123", "456"],
                "benthic_attribute_id": [same_char_uuid("0"), same_char_uuid("1")],
                "growth_form_id": [same_char_uuid("2"), ""],
            }
        )

        dataset.duck_conn.execute("CREATE TABLE annotations AS SELECT * FROM cn_annotations_df")

        with tempfile.NamedTemporaryFile(delete_on_close=False) as parquet_f:
            # Write a couple of MERMAID annotations to a parquet file.
            # One with growth form, one without (this is represented in
            # MERMAID parquet as string 'None').
            mermaid_parquet_df = pd.DataFrame(  # noqa: F841 — referenced by name in DuckDB SQL via Python-scope scanning
                {
                    "image_id": [same_char_uuid("3"), same_char_uuid("4")],
                    "row": [3000, 500],
                    "col": [2200, 1800],
                    "benthic_attribute_id": [same_char_uuid("5"), same_char_uuid("6")],
                    "growth_form_id": [same_char_uuid("7"), "None"],
                }
            )
            dataset.duck_conn.execute(
                "CREATE TABLE mermaid_parquet_input AS SELECT * FROM mermaid_parquet_df"
            )

            # DuckDB will reopen the file, so close it first.
            parquet_f.close()
            dataset.duck_conn.execute(
                f"COPY (SELECT * FROM mermaid_parquet_input) TO '{parquet_f.name}' (FORMAT parquet)"
            )

            with override_settings(
                mermaid_annotations_parquet_pattern=parquet_f.name,
            ):
                dataset.read_mermaid_data()

        result_tuples = dataset.duck_conn.execute(
            "SELECT image_id, row, col, benthic_attribute_id, growth_form_id FROM annotations"
        ).fetchall()
        # We don't really care about the result order. So we'll alphabetize
        # the results to make it easier to assert on them.
        result_tuples.sort()

        self.assertListEqual(
            list(result_tuples[0]),
            ["12345", 2000, 1200, same_char_uuid("0"), same_char_uuid("2")],
        )
        self.assertListEqual(
            list(result_tuples[1]),
            [same_char_uuid("3"), 3000, 2200, same_char_uuid("5"), same_char_uuid("7")],
        )
        # MERMAID 'None' growth form should have become ''.
        self.assertListEqual(
            list(result_tuples[2]),
            [same_char_uuid("4"), 500, 1800, same_char_uuid("6"), ""],
        )
        # coralnet row with '' growth form should not be accidentally
        # dropped.
        self.assertListEqual(
            list(result_tuples[3]),
            ["67890", 1500, 2800, same_char_uuid("1"), ""],
        )


class HandleMissingFeatureVectorsTest(BaseTrainTest):
    @staticmethod
    def annotations_fvs(dataset):
        result_tuples = dataset.duck_conn.execute(
            "SELECT feature_vector FROM annotations"
        ).fetchall()
        feature_vector_files = [tup[0] for tup in result_tuples]
        return sorted(feature_vector_files)

    def test_none_missing(self):
        annotations_df = pd.DataFrame(  # noqa: F841 — referenced by name in DuckDB SQL via Python-scope scanning
            {
                "site": [Sites.MERMAID.value] * 4,
                "bucket": ["my-bucket"] * 4,
                "feature_vector": ["01.fv", "01.fv", "02.fv", "02.fv"],
            }
        )
        s3_paths = {
            "my-bucket/01.fv",
            "my-bucket/02.fv",
        }

        dataset = NoInitDataset()
        dataset.duck_conn.execute("CREATE TABLE annotations AS SELECT * FROM annotations_df")
        # Since there shouldn't be any missing feature vectors, it
        # shouldn't log any warnings.
        with self.assertNoLogs(logger="train", level="WARN"):
            dataset.handle_missing_feature_vectors(s3_paths)

        self.assertListEqual(
            self.annotations_fvs(dataset),
            ["01.fv", "01.fv", "02.fv", "02.fv"],
            msg="No annotations should have been filtered out",
        )

    def test_one_missing(self):
        annotations_df = pd.DataFrame(  # noqa: F841 — referenced by name in DuckDB SQL via Python-scope scanning
            {
                "site": [Sites.MERMAID.value] * 4,
                "bucket": ["my-bucket"] * 4,
                "feature_vector": ["01.fv", "01.fv", "02.fv", "02.fv"],
            }
        )
        # S3 doesn't have 02.
        s3_paths = {
            "my-bucket/01.fv",
            "my-bucket/05.fv",
        }

        dataset = NoInitDataset()
        dataset.duck_conn.execute("CREATE TABLE annotations AS SELECT * FROM annotations_df")
        with (
            self.assertLogs(logger="train", level="WARN") as warn_cm,
            override_settings(training_inputs_percent_missing_allowed=50),
        ):
            dataset.handle_missing_feature_vectors(s3_paths)

        self.assertListEqual(
            self.annotations_fvs(dataset),
            ["01.fv", "01.fv"],
            msg="02.fv should have been filtered out",
        )

        self.assertEqual(
            warn_cm.output[0],
            "WARNING:train:Skipping 1 feature vector(s) because the files"
            " aren't in S3. Example(s):"
            "\nmy-bucket/02.fv",
        )

    def test_over_three_missing(self):
        annotations_df = pd.DataFrame(  # noqa: F841 — referenced by name in DuckDB SQL via Python-scope scanning
            {
                "site": [Sites.MERMAID.value] * 5,
                "bucket": ["my-bucket"] * 5,
                "feature_vector": ["01.fv", "02.fv", "03.fv", "04.fv", "05.fv"],
            }
        )
        # S3 doesn't have 01, 02, 04, 05.
        s3_paths = {
            "my-bucket/03.fv",
            "my-bucket/07.fv",
        }

        dataset = NoInitDataset()
        dataset.duck_conn.execute("CREATE TABLE annotations AS SELECT * FROM annotations_df")
        with (
            self.assertLogs(logger="train", level="WARN") as warn_cm,
            override_settings(training_inputs_percent_missing_allowed=90),
        ):
            dataset.handle_missing_feature_vectors(s3_paths)

        self.assertListEqual(
            self.annotations_fvs(dataset),
            ["03.fv"],
            msg="All but 03.fv should have been filtered out",
        )

        # Should list 3 examples out of the 4.
        message = warn_cm.output[0]
        self.assertIn(
            "WARNING:train:Skipping 4 feature vector(s) because the files"
            " aren't in S3. Example(s):",
            message,
        )
        example_count = sum(
            [
                1 if feature_path in message else 0
                for feature_path in [
                    "my-bucket/01.fv",
                    "my-bucket/02.fv",
                    "my-bucket/04.fv",
                    "my-bucket/05.fv",
                ]
            ]
        )
        self.assertEqual(example_count, 3)

    def test_over_threshold_missing(self):
        annotations_df = pd.DataFrame(  # noqa: F841 — referenced by name in DuckDB SQL via Python-scope scanning
            {
                "site": [Sites.MERMAID.value] * 5,
                "bucket": ["my-bucket"] * 5,
                "feature_vector": ["01.fv", "02.fv", "03.fv", "04.fv", "05.fv"],
            }
        )
        # S3 doesn't have 01, 05 (40% missing).
        # We'll add more extras here to demonstrate that the threshold is
        # out of features in annotations, not features in S3.
        s3_paths = {
            "my-bucket/02.fv",
            "my-bucket/03.fv",
            "my-bucket/04.fv",
            "my-bucket/12.fv",
            "my-bucket/13.fv",
            "my-bucket/14.fv",
        }

        dataset = NoInitDataset()
        dataset.duck_conn.execute("CREATE TABLE annotations AS SELECT * FROM annotations_df")
        with (
            self.assertRaises(RuntimeError) as error_cm,
            override_settings(training_inputs_percent_missing_allowed=39),
        ):
            dataset.handle_missing_feature_vectors(s3_paths)

        message = str(error_cm.exception)
        self.assertIn("Too many feature vectors are missing (2), such as:", message)
        self.assertIn("my-bucket/01.fv", message)
        self.assertIn("my-bucket/05.fv", message)
        self.assertIn("You can configure the tolerance for missing feature vectors", message)

    def test_coralnet_missing_filtered(self):
        """
        CoralNet annotations whose feature vectors are absent from the
        present-paths set must now be filtered out (previously they were
        always kept) and a warning logged.
        """
        annotations_df = pd.DataFrame(  # noqa: F841 — referenced by name in DuckDB SQL via Python-scope scanning
            {
                "site": [Sites.CORALNET.value] * 4,
                "bucket": ["cn-bucket"] * 4,
                "feature_vector": [
                    "s1/features/i01.featurevector",
                    "s1/features/i01.featurevector",
                    "s1/features/i02.featurevector",
                    "s1/features/i02.featurevector",
                ],
            }
        )
        # S3 doesn't have i02.
        s3_paths = {
            "cn-bucket/s1/features/i01.featurevector",
            "cn-bucket/s1/features/i05.featurevector",
        }

        dataset = NoInitDataset()
        dataset.duck_conn.execute("CREATE TABLE annotations AS SELECT * FROM annotations_df")
        with (
            self.assertLogs(logger="train", level="WARN") as warn_cm,
            override_settings(training_inputs_percent_missing_allowed=50),
        ):
            dataset.handle_missing_feature_vectors(s3_paths)

        self.assertListEqual(
            self.annotations_fvs(dataset),
            ["s1/features/i01.featurevector", "s1/features/i01.featurevector"],
            msg="i02 CoralNet feature vector should have been filtered out",
        )

        self.assertEqual(
            warn_cm.output[0],
            "WARNING:train:Skipping 1 feature vector(s) because the files"
            " aren't in S3. Example(s):"
            "\ncn-bucket/s1/features/i02.featurevector",
        )

    def test_mixed_sites_missing_filtered_and_abort(self):
        """
        With both MERMAID and CoralNet annotations missing feature vectors,
        both sites' missing rows are filtered, and the abort triggers when
        the combined missing count exceeds the threshold (counted over the
        union of distinct feature vectors across all sites).
        """
        annotations_df = pd.DataFrame(  # noqa: F841 — referenced by name in DuckDB SQL via Python-scope scanning
            {
                "site": [Sites.MERMAID.value, Sites.MERMAID.value]
                + [Sites.CORALNET.value, Sites.CORALNET.value],
                "bucket": ["mm-bucket", "mm-bucket", "cn-bucket", "cn-bucket"],
                "feature_vector": [
                    "01.fv",
                    "02.fv",
                    "s1/features/i01.featurevector",
                    "s1/features/i02.featurevector",
                ],
            }
        )
        # Present: only one MERMAID and one CoralNet feature vector.
        # Missing: mm 02.fv and cn i02 -> 2 of 4 distinct (50%).
        s3_paths = {
            "mm-bucket/01.fv",
            "cn-bucket/s1/features/i01.featurevector",
        }

        dataset = NoInitDataset()
        dataset.duck_conn.execute("CREATE TABLE annotations AS SELECT * FROM annotations_df")
        # First, with a generous tolerance: both missing rows filtered, no abort.
        with (
            self.assertLogs(logger="train", level="WARN"),
            override_settings(training_inputs_percent_missing_allowed=60),
        ):
            dataset.handle_missing_feature_vectors(s3_paths)

        self.assertListEqual(
            self.annotations_fvs(dataset),
            ["01.fv", "s1/features/i01.featurevector"],
            msg="Both MERMAID and CoralNet missing rows should be filtered out",
        )

        # Now prove the abort triggers when combined missing exceeds threshold.
        dataset2 = NoInitDataset()
        dataset2.duck_conn.execute("CREATE TABLE annotations AS SELECT * FROM annotations_df")
        with (
            self.assertRaises(RuntimeError) as error_cm,
            override_settings(training_inputs_percent_missing_allowed=49),
        ):
            dataset2.handle_missing_feature_vectors(s3_paths)

        message = str(error_cm.exception)
        self.assertIn("Too many feature vectors are missing (2), such as:", message)


class LazyLibraryTest(BaseTrainTest):
    """
    The BA and GF libraries hit the MERMAID API in their __init__. Importing
    the training modules (dataset/runner) must not trigger those network calls
    (it used to, via module-level singletons), so unit tests can run offline.
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
