import unittest

import numpy as np

from mermaid_classifier.model_review import v1_infer
from mermaid_classifier.model_review.sample import ReviewPoint


class _FakePredictor:
    classes = ["ba1::gf1", "ba2::", "ba3::gf2"]

    def predict_proba(self, x):
        # one row per input; always argmax to class index 1 ("ba2::")
        n = x.shape[0]
        out = np.zeros((n, 3))
        out[:, 1] = 1.0
        return out


class _FakeFeatures:
    def get_array(self, rowcol):
        return np.zeros((1, 4), dtype=np.float32)


def _fake_loader(bucket, key):
    return _FakeFeatures()


class V1InferTest(unittest.TestCase):
    def test_predicts_argmax_label_per_point(self):
        points = [
            ReviewPoint(
                "coralnet", "A", "109", 10, 20, "ba1::gf1", "s109/features/iA.featurevector", "b"
            ),
            ReviewPoint(
                "coralnet", "A", "109", 30, 40, "ba1::", "s109/features/iA.featurevector", "b"
            ),
        ]
        preds = v1_infer.predict_points(
            points,
            classifier_location="unused",
            load_features=_fake_loader,
            predictor=_FakePredictor(),
        )
        self.assertEqual(preds[("A", 10, 20)], "ba2::")
        self.assertEqual(preds[("A", 30, 40)], "ba2::")

    def test_groups_features_load_once_per_image(self):
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
        v1_infer.predict_points(
            points, "unused", load_features=counting_loader, predictor=_FakePredictor()
        )
        self.assertEqual(len(calls), 1)  # both points share one feature file


class PredictFromFeaturesTest(unittest.TestCase):
    def test_maps_each_key_to_its_argmax_label(self):
        keys = [("A", 10, 20), ("A", 30, 40)]
        matrix = np.zeros((2, 4), dtype=np.float32)
        preds = v1_infer.predict_from_features(keys, matrix, _FakePredictor())
        self.assertEqual(preds, {("A", 10, 20): "ba2::", ("A", 30, 40): "ba2::"})

    def test_row_count_mismatch_is_rejected(self):
        # A silent mismatch would shift every label after the gap onto the wrong point.
        with self.assertRaises(ValueError):
            v1_infer.predict_from_features(
                [("A", 10, 20)], np.zeros((2, 4), dtype=np.float32), _FakePredictor()
            )


class MixedSiteTest(unittest.TestCase):
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
        preds = v1_infer.predict_points(
            points, "unused", load_features=loader, predictor=_FakePredictor()
        )
        self.assertEqual(
            sorted(calls),
            [
                ("2605-coralnet-public-sources", "s109/features/iA.featurevector"),
                ("coral-reef-training", "mermaid/uuid-1_featurevector"),
            ],
        )
        self.assertEqual(preds[("A", 10, 20)], "ba2::")
        self.assertEqual(preds[("uuid-1", 30, 40)], "ba2::")
