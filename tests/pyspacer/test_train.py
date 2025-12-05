from contextlib import contextmanager
import unittest

import pandas as pd

from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.pyspacer.train import Sites, TrainingDataset


@contextmanager
def override_settings(**kwargs):
    """
    Override the specified Pydantic settings for the duration of the
    context manager.

    Some parts are from
    https://rednafi.com/python/patch-pydantic-settings-in-pytest/
    """
    # Make a copy of the original settings
    original_settings = settings.model_copy()

    # Patch the settings with kwargs
    for key, val in kwargs.items():
        # Raise an error if kwargs contains a nonexistent setting
        if not hasattr(settings, key):
            raise ValueError(f"Unknown setting: {key}")
        setattr(settings, key, val)

    yield settings

    # Restore the original settings
    settings.__dict__.update(original_settings.__dict__)


class NoInitDataset(TrainingDataset):
    """
    init does a lot of stuff in TrainingDataset. When testing, we sometimes
    just want access to the other methods of the class.
    So here we make init barebones.
    """
    def __init__(self):
        self._duck_conn = None


class HandleMissingFeatureVectorsTest(unittest.TestCase):

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
