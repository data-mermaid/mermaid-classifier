import unittest

import numpy as np
from spacer.data_classes import ValResults

from mermaid_classifier.pyspacer.metrics import (
    ba_level_accuracy,
    cohens_kappa,
    cover_bias,
    expected_calibration_error,
    matthews_corrcoef,
    top_k_accuracy,
)


def make_val_results(gt, est, scores=None, classes=None):
    """Helper to build a ValResults from lists of class indices."""
    if classes is None:
        n_classes = max(max(gt), max(est)) + 1
        # Use BA::GF style class IDs by default.
        classes = [f"Class{i}::" for i in range(n_classes)]
    if scores is None:
        scores = [0.9] * len(gt)
    return ValResults(scores=scores, gt=gt, est=est, classes=classes)


class CohensKappaTest(unittest.TestCase):

    def test_perfect_agreement(self):
        vr = make_val_results(
            gt=[0, 1, 2, 0, 1, 2],
            est=[0, 1, 2, 0, 1, 2],
        )
        self.assertAlmostEqual(cohens_kappa(vr), 1.0)

    def test_no_agreement(self):
        # Systematically wrong predictions -> negative kappa.
        vr = make_val_results(
            gt=[0, 0, 1, 1, 2, 2],
            est=[1, 2, 0, 2, 0, 1],
        )
        self.assertLess(cohens_kappa(vr), 0.0)

    def test_partial_agreement(self):
        vr = make_val_results(
            gt=[0, 1, 2, 0, 1, 2],
            est=[0, 1, 0, 0, 2, 2],
        )
        k = cohens_kappa(vr)
        self.assertGreater(k, 0.0)
        self.assertLess(k, 1.0)


class MatthewsCorrcoefTest(unittest.TestCase):

    def test_perfect_prediction(self):
        vr = make_val_results(
            gt=[0, 1, 2, 0, 1, 2],
            est=[0, 1, 2, 0, 1, 2],
        )
        self.assertAlmostEqual(matthews_corrcoef(vr), 1.0)

    def test_random_prediction(self):
        # Systematic misprediction -> negative MCC.
        vr = make_val_results(
            gt=[0, 0, 1, 1, 2, 2],
            est=[1, 2, 0, 2, 0, 1],
        )
        self.assertLess(matthews_corrcoef(vr), 0.0)


class BALevelAccuracyTest(unittest.TestCase):

    def test_perfect_ba_accuracy(self):
        # Same BA, different GF -> BA-level accuracy should be 1.0.
        classes = ["Acropora::Branching", "Acropora::Tabular", "Porites::Massive"]
        vr = make_val_results(
            gt=[0, 1, 2],
            est=[1, 0, 2],  # Swapped within Acropora GFs
            classes=classes,
        )
        self.assertAlmostEqual(ba_level_accuracy(vr), 1.0)

    def test_wrong_ba(self):
        classes = ["Acropora::Branching", "Porites::Massive"]
        vr = make_val_results(
            gt=[0, 0, 1, 1],
            est=[1, 1, 0, 0],  # All wrong BA
            classes=classes,
        )
        self.assertAlmostEqual(ba_level_accuracy(vr), 0.0)

    def test_mixed_ba_accuracy(self):
        classes = [
            "Acropora::Branching",
            "Acropora::Tabular",
            "Porites::Massive",
        ]
        vr = make_val_results(
            gt=[0, 1, 2, 2],
            est=[1, 0, 0, 2],  # First two: BA correct (Acropora), third: wrong
            classes=classes,
        )
        # 3 out of 4 correct at BA level.
        self.assertAlmostEqual(ba_level_accuracy(vr), 0.75)

    def test_empty_growth_form(self):
        # Classes with empty growth form (BA + separator only).
        classes = ["HardCoral::", "SoftCoral::"]
        vr = make_val_results(
            gt=[0, 1, 0, 1],
            est=[0, 1, 0, 1],
            classes=classes,
        )
        self.assertAlmostEqual(ba_level_accuracy(vr), 1.0)


class CoverBiasTest(unittest.TestCase):

    def test_perfect_prediction_zero_bias(self):
        vr = make_val_results(
            gt=[0, 1, 2, 0, 1, 2],
            est=[0, 1, 2, 0, 1, 2],
        )
        per_class, mean_abs = cover_bias(vr)
        self.assertAlmostEqual(mean_abs, 0.0)
        for bias in per_class.values():
            self.assertAlmostEqual(bias, 0.0)

    def test_biased_prediction(self):
        # All predictions are class 0.
        vr = make_val_results(
            gt=[0, 1, 2],
            est=[0, 0, 0],
        )
        per_class, mean_abs = cover_bias(vr)
        # Class 0: true 1/3, pred 3/3 -> bias = 2/3
        self.assertAlmostEqual(per_class[0], 2 / 3)
        # Class 1: true 1/3, pred 0/3 -> bias = -1/3
        self.assertAlmostEqual(per_class[1], -1 / 3)
        # Class 2: true 1/3, pred 0/3 -> bias = -1/3
        self.assertAlmostEqual(per_class[2], -1 / 3)
        # Mean abs: (2/3 + 1/3 + 1/3) / 3 = 4/9
        self.assertAlmostEqual(mean_abs, 4 / 9)

    def test_single_class(self):
        vr = make_val_results(
            gt=[0, 0, 0],
            est=[0, 0, 0],
            classes=["OnlyClass::"],
        )
        per_class, mean_abs = cover_bias(vr)
        self.assertAlmostEqual(mean_abs, 0.0)


class ExpectedCalibrationErrorTest(unittest.TestCase):

    def test_perfectly_calibrated(self):
        # All correct with confidence 1.0 -> ECE = 0.
        vr = make_val_results(
            gt=[0, 1, 2, 0, 1, 2],
            est=[0, 1, 2, 0, 1, 2],
            scores=[1.0] * 6,
        )
        self.assertAlmostEqual(expected_calibration_error(vr), 0.0)

    def test_overconfident(self):
        # All wrong with confidence 1.0 -> ECE = 1.0.
        vr = make_val_results(
            gt=[0, 0, 1, 1],
            est=[1, 1, 0, 0],
            scores=[1.0] * 4,
        )
        self.assertAlmostEqual(expected_calibration_error(vr), 1.0)

    def test_ece_between_zero_and_one(self):
        # Mixed correctness and confidence -> ECE in (0, 1).
        vr = make_val_results(
            gt=[0, 1, 0, 1, 0, 1],
            est=[0, 0, 0, 1, 1, 1],
            scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        )
        ece = expected_calibration_error(vr)
        self.assertGreater(ece, 0.0)
        self.assertLess(ece, 1.0)


class TopKAccuracyTest(unittest.TestCase):

    def test_perfect_top1(self):
        vr = make_val_results(gt=[0, 1, 2], est=[0, 1, 2])
        # Probability matrix: perfect confidence.
        proba = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.assertAlmostEqual(top_k_accuracy(vr, proba, k=1), 1.0)
        self.assertAlmostEqual(top_k_accuracy(vr, proba, k=3), 1.0)

    def test_correct_in_top3_but_not_top1(self):
        vr = make_val_results(
            gt=[0, 1],
            est=[1, 0],  # Wrong top-1
            classes=["A::", "B::", "C::", "D::"],
        )
        # True label is second-highest probability.
        proba = np.array([
            [0.3, 0.4, 0.2, 0.1],  # gt=0, top3=[1,0,2] -> 0 is in top3
            [0.35, 0.3, 0.2, 0.15],  # gt=1, top3=[0,1,2] -> 1 is in top3
        ])
        self.assertAlmostEqual(top_k_accuracy(vr, proba, k=1), 0.0)
        self.assertAlmostEqual(top_k_accuracy(vr, proba, k=3), 1.0)

    def test_not_in_top_k(self):
        vr = make_val_results(
            gt=[3],
            est=[0],
            classes=["A::", "B::", "C::", "D::"],
        )
        # True label (3) has lowest probability.
        proba = np.array([
            [0.4, 0.3, 0.2, 0.1],
        ])
        self.assertAlmostEqual(top_k_accuracy(vr, proba, k=3), 0.0)
        self.assertAlmostEqual(top_k_accuracy(vr, proba, k=4), 1.0)


if __name__ == '__main__':
    unittest.main()
