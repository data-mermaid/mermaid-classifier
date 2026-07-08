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
            ReviewPoint("A", "109", 10, 20, "ba1::gf1", "s109/features/iA.featurevector", "b"),
            ReviewPoint("A", "109", 30, 40, "ba1::", "s109/features/iA.featurevector", "b"),
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
            ReviewPoint("A", "109", 10, 20, "x::", "s109/features/iA.featurevector", "b"),
            ReviewPoint("A", "109", 30, 40, "x::", "s109/features/iA.featurevector", "b"),
        ]
        v1_infer.predict_points(
            points, "unused", load_features=counting_loader, predictor=_FakePredictor()
        )
        self.assertEqual(len(calls), 1)  # both points share one feature file
