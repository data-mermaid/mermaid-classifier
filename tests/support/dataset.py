"""A TrainingDataset that skips the S3 and API work its __init__ does."""

import tempfile
import unittest

from mermaid_classifier.pyspacer.dataset import TrainingDataset
from mermaid_classifier.pyspacer.options import Artifacts, DatasetOptions


class NoInitDataset(TrainingDataset):
    """
    init does a lot of stuff in TrainingDataset. When testing, we sometimes
    just want access to the other methods of the class.
    So here we make init barebones.
    """

    def __init__(self):
        self._duck_conn = None
        self.artifacts = Artifacts()
        # Matches TrainingDataset's own temp-dir shape: unpredictable path,
        # self-cleaning on garbage collection.
        self._feature_temp_dir = tempfile.TemporaryDirectory(
            prefix="mermaid_features_test_", ignore_cleanup_errors=True
        )
        self._feature_dir = self._feature_temp_dir.name


def make_dataset(test_case: unittest.TestCase) -> NoInitDataset:
    """Return a NoInitDataset with all attributes needed by the pipeline methods.

    The temp feature dir is removed via the test case's cleanup so the suite
    stays hermetic and doesn't leak directories.
    """
    dataset = NoInitDataset()
    test_case.addCleanup(dataset.cleanup)
    dataset.profiled_sections = []
    dataset._feature_path_to_s3_location = {}
    dataset.feature_loc_to_source = {}
    dataset.options = DatasetOptions(ref_val_ratios=(0.1, 0.1))
    # artifacts is already set by NoInitDataset.__init__ (Artifacts())
    return dataset
