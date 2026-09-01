import unittest

import numpy as np

from mermaid_classifier.model_review import features
from mermaid_classifier.model_review.sample import ReviewPoint


class _FakeFeatures:
    def __init__(self, value=0.0):
        self.value = value

    def get_array(self, rowcol):
        row, col = rowcol
        return np.full((1, 4), self.value + row + col, dtype=np.float32)


class StackFeaturesTest(unittest.TestCase):
    def test_keys_align_with_matrix_rows(self):
        points = [
            ReviewPoint(
                "coralnet", "A", "109", 10, 20, "x::", "s109/features/iA.featurevector", "b"
            ),
            ReviewPoint(
                "coralnet", "A", "109", 30, 40, "x::", "s109/features/iA.featurevector", "b"
            ),
        ]
        keys, matrix = features.stack_features(points, lambda bucket, key: _FakeFeatures())
        self.assertEqual(keys, [("A", 10, 20), ("A", 30, 40)])
        self.assertEqual(matrix.shape, (2, 4))
        self.assertEqual(matrix.dtype, np.float32)
        # each row is the feature of the key at the same index
        self.assertEqual(matrix[0][0], 30.0)  # row 10 + col 20
        self.assertEqual(matrix[1][0], 70.0)  # row 30 + col 40

    def test_loads_each_feature_file_once(self):
        calls = []

        def counting_loader(bucket, key):
            calls.append(key)
            return _FakeFeatures()

        points = [
            ReviewPoint(
                "coralnet", "A", "109", 10, 20, "x::", "s109/features/iA.featurevector", "b"
            ),
            ReviewPoint(
                "coralnet", "A", "109", 30, 40, "x::", "s109/features/iA.featurevector", "b"
            ),
        ]
        features.stack_features(points, counting_loader)
        self.assertEqual(len(calls), 1)  # both points share one feature file

    def test_points_from_two_sites_load_from_their_own_buckets(self):
        calls = []

        def loader(bucket, key):
            calls.append((bucket, key))
            return _FakeFeatures()

        points = [
            ReviewPoint(
                "coralnet",
                "A",
                "109",
                10,
                20,
                "x::",
                "s109/features/iA.featurevector",
                "2605-coralnet-public-sources",
            ),
            ReviewPoint(
                "mermaid",
                "uuid-1",
                "",
                30,
                40,
                "x::",
                "mermaid/uuid-1_featurevector",
                "coral-reef-training",
            ),
        ]
        keys, matrix = features.stack_features(points, loader)
        self.assertEqual(
            sorted(calls),
            [
                ("2605-coralnet-public-sources", "s109/features/iA.featurevector"),
                ("coral-reef-training", "mermaid/uuid-1_featurevector"),
            ],
        )
        self.assertEqual(sorted(keys), [("A", 10, 20), ("uuid-1", 30, 40)])
        self.assertEqual(matrix.shape, (2, 4))

    def test_no_points_is_rejected(self):
        with self.assertRaises(ValueError):
            features.stack_features([], lambda bucket, key: _FakeFeatures())
