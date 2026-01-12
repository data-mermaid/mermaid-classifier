from contextlib import contextmanager
import tempfile
import unittest

import pandas as pd

from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.pyspacer.train import Artifacts, Sites, TrainingDataset


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


class BaseTrainTest(unittest.TestCase):

    def setUp(self):
        self.override = SettingsOverride(aws_anonymous='True')
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
    return '-'.join([
        ''.join([char]*8),
        ''.join([char]*4),
        ''.join([char]*4),
        ''.join([char]*4),
        ''.join([char]*12),
    ])


class ReadMermaidDataTest(BaseTrainTest):

    def test_coralnet_null_gfs(self):
        """
        CoralNet data which includes NULL growth form IDs was read in prior
        to read_mermaid_data().

        We've seen that careless DuckDB JOINs can make coralnet rows with
        NULL GF IDs get erased from the annotations, particularly at the
        NULL/'None' normalization step. This test checks that the rows
        are kept.
        """
        dataset = NoInitDataset()

        # Write a couple of coralnet annotations to DuckDB.
        # One with growth form, one without (this is represented as
        # NULL / Python None).
        cn_annotations_df = pd.DataFrame(dict(
            site=['coralnet']*2,
            bucket=['cn-data']*2,
            project_id=['12', '34'],
            image_id=['12345', '67890'],
            feature_vector=['s12/i12345.fv', 's34/i67890.fv'],
            row=[2000, 1500],
            col=[1200, 2800],
            label_id=['123', '456'],
            benthic_attribute_id=[same_char_uuid('0'), same_char_uuid('1')],
            growth_form_id=[same_char_uuid('2'), None],
        ))

        dataset.duck_conn.execute(
            f"CREATE TABLE annotations AS SELECT * FROM cn_annotations_df"
        )

        with tempfile.NamedTemporaryFile(delete_on_close=False) as parquet_f:

            # Write a couple of mermaid annotations to a parquet file.
            # One with growth form, one without (this is represented in
            # mermaid parquet as string 'None').
            mermaid_parquet_df = pd.DataFrame(dict(
                image_id=[same_char_uuid('3'), same_char_uuid('4')],
                row=[3000, 500],
                col=[2200, 1800],
                benthic_attribute_id=[
                    same_char_uuid('5'), same_char_uuid('6')],
                growth_form_id=[same_char_uuid('7'), 'None'],
            ))
            dataset.duck_conn.execute(
                f"CREATE TABLE mermaid_parquet_input AS"
                f" SELECT * FROM mermaid_parquet_df"
            )

            # DuckDB will reopen the file, so close it first.
            parquet_f.close()
            dataset.duck_conn.execute(
                f"COPY (SELECT * FROM mermaid_parquet_input)"
                f" TO '{parquet_f.name}'"
                f" (FORMAT parquet)"
            )

            with override_settings(
                mermaid_annotations_parquet_path=parquet_f.name,
            ):
                dataset.read_mermaid_data()

        result_tuples = dataset.duck_conn.execute(
            "SELECT image_id, row, col, benthic_attribute_id, growth_form_id"
            " FROM annotations").fetchall()
        # We don't really care about the result order. So we'll alphabetize
        # the results to make it easier to assert on them.
        result_tuples.sort()

        self.assertListEqual(
            list(result_tuples[0]),
            ['12345', 2000, 1200,
             same_char_uuid('0'), same_char_uuid('2')],
        )
        self.assertListEqual(
            list(result_tuples[1]),
            [same_char_uuid('3'), 3000, 2200,
             same_char_uuid('5'), same_char_uuid('7')],
        )
        # mermaid 'None' growth form should have become actual NULL / None.
        self.assertListEqual(
            list(result_tuples[2]),
            [same_char_uuid('4'), 500, 1800,
             same_char_uuid('6'), None],
        )
        # coralnet row with None growth form should not be accidentally
        # dropped.
        self.assertListEqual(
            list(result_tuples[3]),
            ['67890', 1500, 2800,
             same_char_uuid('1'), None],
        )


class HandleMissingFeatureVectorsTest(BaseTrainTest):

    @staticmethod
    def annotations_fvs(dataset):
        result_tuples = dataset.duck_conn.execute(
            "SELECT feature_vector FROM annotations").fetchall()
        feature_vector_files = [
            tup[0] for tup in result_tuples]
        return sorted(feature_vector_files)

    def test_none_missing(self):
        annotations_df = pd.DataFrame({
            'site': [Sites.MERMAID.value]*4,
            'bucket': ['my-bucket']*4,
            'feature_vector': ['01.fv', '01.fv', '02.fv', '02.fv'],
        })
        s3_paths = {
            'my-bucket/01.fv',
            'my-bucket/02.fv',
        }

        dataset = NoInitDataset()
        dataset.duck_conn.execute(
            "CREATE TABLE annotations AS SELECT * FROM annotations_df"
        )
        # Since there shouldn't be any missing feature vectors, it
        # shouldn't log any warnings.
        with self.assertNoLogs(logger='train', level='WARN'):
            dataset.handle_missing_feature_vectors(s3_paths)

        self.assertListEqual(
            self.annotations_fvs(dataset),
            ['01.fv', '01.fv', '02.fv', '02.fv'],
            msg="No annotations should have been filtered out",
        )

    def test_one_missing(self):
        annotations_df = pd.DataFrame({
            'site': [Sites.MERMAID.value]*4,
            'bucket': ['my-bucket']*4,
            'feature_vector': ['01.fv', '01.fv', '02.fv', '02.fv'],
        })
        # S3 doesn't have 02.
        s3_paths = {
            'my-bucket/01.fv',
            'my-bucket/05.fv',
        }

        dataset = NoInitDataset()
        dataset.duck_conn.execute(
            "CREATE TABLE annotations AS SELECT * FROM annotations_df"
        )
        with (
            self.assertLogs(logger='train', level='WARN') as warn_cm,
            override_settings(training_inputs_percent_missing_allowed=50),
        ):
            dataset.handle_missing_feature_vectors(s3_paths)

        self.assertListEqual(
            self.annotations_fvs(dataset),
            ['01.fv', '01.fv'],
            msg="02.fv should have been filtered out",
        )

        self.assertEqual(
            warn_cm.output[0],
            "WARNING:train:Skipping 1 feature vector(s) because the files"
            " aren't in S3. Example(s):"
            "\nmy-bucket/02.fv"
        )

    def test_over_three_missing(self):
        annotations_df = pd.DataFrame({
            'site': [Sites.MERMAID.value]*5,
            'bucket': ['my-bucket']*5,
            'feature_vector': ['01.fv', '02.fv', '03.fv', '04.fv', '05.fv'],
        })
        # S3 doesn't have 01, 02, 04, 05.
        s3_paths = {
            'my-bucket/03.fv',
            'my-bucket/07.fv',
        }

        dataset = NoInitDataset()
        dataset.duck_conn.execute(
            "CREATE TABLE annotations AS SELECT * FROM annotations_df"
        )
        with (
            self.assertLogs(logger='train', level='WARN') as warn_cm,
            override_settings(training_inputs_percent_missing_allowed=90),
        ):
            dataset.handle_missing_feature_vectors(s3_paths)

        self.assertListEqual(
            self.annotations_fvs(dataset),
            ['03.fv'],
            msg="All but 03.fv should have been filtered out",
        )

        # Should list 3 examples out of the 4.
        message = warn_cm.output[0]
        self.assertIn(
            "WARNING:train:Skipping 4 feature vector(s) because the files"
            " aren't in S3. Example(s):",
            message,
        )
        example_count = sum([
            1 if feature_path in message else 0
            for feature_path in [
                'my-bucket/01.fv',
                'my-bucket/02.fv',
                'my-bucket/04.fv',
                'my-bucket/05.fv',
            ]
        ])
        self.assertEqual(example_count, 3)

    def test_over_threshold_missing(self):
        annotations_df = pd.DataFrame({
            'site': [Sites.MERMAID.value]*5,
            'bucket': ['my-bucket']*5,
            'feature_vector': ['01.fv', '02.fv', '03.fv', '04.fv', '05.fv'],
        })
        # S3 doesn't have 01, 05 (40% missing).
        # We'll add more extras here to demonstrate that the threshold is
        # out of features in annotations, not features in S3.
        s3_paths = {
            'my-bucket/02.fv',
            'my-bucket/03.fv',
            'my-bucket/04.fv',
            'my-bucket/12.fv',
            'my-bucket/13.fv',
            'my-bucket/14.fv',
        }

        dataset = NoInitDataset()
        dataset.duck_conn.execute(
            "CREATE TABLE annotations AS SELECT * FROM annotations_df"
        )
        with (
            self.assertRaises(RuntimeError) as error_cm,
            override_settings(training_inputs_percent_missing_allowed=39),
        ):
            dataset.handle_missing_feature_vectors(s3_paths)

        message = str(error_cm.exception)
        self.assertIn(
            "Too many feature vectors are missing (2), such as:", message)
        self.assertIn('my-bucket/01.fv', message)
        self.assertIn('my-bucket/05.fv', message)
        self.assertIn(
            "You can configure the tolerance for missing feature vectors",
            message)
