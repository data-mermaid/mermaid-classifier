from contextlib import contextmanager
from io import StringIO
import tempfile
import unittest
from unittest import mock

import pandas as pd
from spacer.data_classes import DataLocation, ImageLabels
from spacer.messages import TrainingTaskLabels

from mermaid_classifier.common.benthic_attributes import CoralNetMermaidMapping
from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.pyspacer.train import (
    Artifacts, CNSourceFilter, Sites, TrainingDataset)


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
        self._feature_temp_dir = tempfile.TemporaryDirectory(prefix='mermaid_features_')
        self._feature_dir = self._feature_temp_dir.name


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


class ReadCoralNetDataTest(BaseTrainTest):
    """
    Test read_coralnet_data().
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
        dataset = NoInitDataset()

        source_id = 23
        dataset.cn_source_filter = CNSourceFilter(StringIO(
            'id\n'
            f'{source_id}\n'
        ))

        # CoralNet annotations are read in as CSV.
        cn_annotations_csv_content = (
            'Image ID,Row,Column,Label ID\n'
            '12345,2000,1200,123\n'
            '67890,1500,2800,456\n'
        )

        # Mock CN-MM mapping to use.
        mapping = [
            dict(
                benthic_attribute_id=same_char_uuid('0'),
                growth_form_id=same_char_uuid('1'),
                benthic_attribute_name='BA1',
                growth_form_name='GF1',
                provider_id='123',
                provider_label='Label123',
            ),
            dict(
                benthic_attribute_id=same_char_uuid('2'),
                # We expect the API's mapping to have null/None
                # for blank growth forms.
                growth_form_id=None,
                benthic_attribute_name='BA2',
                growth_form_name=None,
                provider_id='456',
                provider_label='Label456',
            ),
        ]

        prefix_pattern = 's{source_id}_'
        prefix = prefix_pattern.format(source_id=source_id)

        with tempfile.NamedTemporaryFile(
            prefix=prefix, mode='w', delete_on_close=False,
        ) as csv_f:

            csv_f.write(cn_annotations_csv_content)
            csv_f.close()

            with (
                override_settings(
                    coralnet_annotations_csv_pattern=csv_f.name,
                ),
                mock.patch.object(
                    CoralNetMermaidMapping, '_download_mapping',
                ) as mock_download_mapping,
                # read_coralnet_data uses S3FileSystem.exists to skip
                # sources whose annotations.csv isn't in S3. The test
                # serves its CSV from a local tempfile, so short-circuit
                # that check.
                mock.patch(
                    'mermaid_classifier.pyspacer.train.S3FileSystem',
                ) as mock_s3fs_cls,
            ):
                mock_download_mapping.return_value = mapping
                mock_s3fs_cls.return_value.exists.return_value = True

                dataset.read_coralnet_data()

        result_tuples = dataset.duck_conn.execute(
            "SELECT image_id, row, col, benthic_attribute_id, growth_form_id"
            " FROM annotations").fetchall()
        result_tuples.sort()

        self.assertListEqual(
            list(result_tuples[0]),
            ['12345', 2000, 1200,
             same_char_uuid('0'), same_char_uuid('1')],
        )
        # Empty GF should show as ''.
        self.assertListEqual(
            list(result_tuples[1]),
            ['67890', 1500, 2800,
             same_char_uuid('2'), ''],
        )


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
            growth_form_id=[same_char_uuid('2'), ''],
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
                mermaid_annotations_parquet_pattern=parquet_f.name,
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
        # mermaid 'None' growth form should have become ''.
        self.assertListEqual(
            list(result_tuples[2]),
            [same_char_uuid('4'), 500, 1800,
             same_char_uuid('6'), ''],
        )
        # coralnet row with '' growth form should not be accidentally
        # dropped.
        self.assertListEqual(
            list(result_tuples[3]),
            ['67890', 1500, 2800,
             same_char_uuid('1'), ''],
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


class AddTrainingSetNamesTest(BaseTrainTest):
    """
    add_training_set_names() tags each annotation row with its train/ref/val
    bucket via a LEFT OUTER JOIN. The labels side uses filesystem-backed
    feature_locs (post the pre-download speedup), so the join keys must be
    recovered from feature_loc_to_s3_key -- otherwise every row ends up with
    training_set NULL and the dataset summary reports the entire run as
    "dropped during stratification".
    """

    def test_join_populates_training_set(self):
        dataset = NoInitDataset()
        dataset.feature_loc_to_s3_key = {}

        # Annotations table mirrors what read_coralnet_data produces:
        # S3-style bucket + key, integer row/col.
        annotations_df = pd.DataFrame({
            'bucket': ['cn-data'] * 4,
            'feature_vector': (
                ['s12/i12345.featurevector'] * 2
                + ['s12/i67890.featurevector'] * 2),
            'row': [100, 200, 300, 400],
            'col': [10, 20, 30, 40],
            'site': [Sites.CORALNET.value] * 4,
            'image_id': ['12345', '12345', '67890', '67890'],
            'project_id': ['12'] * 4,
        })
        dataset.duck_conn.execute(
            "CREATE TABLE annotations AS SELECT * FROM annotations_df"
        )

        # The labels side uses filesystem storage with local cache paths --
        # the bucket_name / key on the DataLocation no longer line up with
        # what's in the annotations table, so the join must reach through
        # feature_loc_to_s3_key.
        loc1 = DataLocation(
            storage_type='filesystem',
            key='/tmp/cache/cn-data/s12/i12345.featurevector',
        )
        loc2 = DataLocation(
            storage_type='filesystem',
            key='/tmp/cache/cn-data/s12/i67890.featurevector',
        )
        dataset.feature_loc_to_s3_key[loc1] = (
            'cn-data', 's12/i12345.featurevector')
        dataset.feature_loc_to_s3_key[loc2] = (
            'cn-data', 's12/i67890.featurevector')

        train = ImageLabels()
        train.add_image(loc1, [(100, 10, 'BA1'), (200, 20, 'BA2')])
        ref = ImageLabels()
        ref.add_image(loc2, [(300, 30, 'BA1')])
        val = ImageLabels()
        val.add_image(loc2, [(400, 40, 'BA2')])
        dataset.labels = TrainingTaskLabels(train=train, ref=ref, val=val)

        dataset.add_training_set_names()

        result = sorted(dataset.duck_conn.execute(
            "SELECT feature_vector, row, col, training_set"
            " FROM annotations"
        ).fetchall())
        self.assertEqual(result, [
            ('s12/i12345.featurevector', 100, 10, 'train'),
            ('s12/i12345.featurevector', 200, 20, 'train'),
            ('s12/i67890.featurevector', 300, 30, 'ref'),
            ('s12/i67890.featurevector', 400, 40, 'val'),
        ])
