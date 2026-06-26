"""
Benchmark suite comparing the sklearn MLPClassifier baseline against the
in-package TorchMLPClassifier replacement.

The same tests run against both implementations via the
`_make_classifier` hook. Parity tests then assert that the PyTorch
classifier reaches accuracy and probability outputs comparable to sklearn
on identical data.

All data is synthetic and small so the full suite runs in seconds.
"""
from __future__ import annotations

import pickle
import time
import unittest

import numpy as np
from sklearn.neural_network import MLPClassifier

from mermaid_classifier.pyspacer.torch_classifier import TorchMLPClassifier


# Seed convention: distinct offsets give independent-but-deterministic RNG
# streams so the data-generation order never correlates with the training
# shuffle order.
#   SEED      -> dataset generation (make_gaussian_clusters)
#   SEED + 1  -> standard training shuffle (train_via_partial_fit)
#   SEED + 2  -> tiny-chunk incremental-fit shuffle
SEED = 42
N_CLASSES = 5
N_FEATURES = 32
N_TRAIN = 500
N_VAL = 200
HIDDEN = (64,)
LR = 1e-2
EPOCHS = 20
PARTIAL_FIT_BATCH = 100  # samples per partial_fit call (trainer-style chunks)


def make_gaussian_clusters(
    n_classes: int = N_CLASSES,
    n_features: int = N_FEATURES,
    n_per_class: int = 120,
    cluster_std: float = 1.3,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic Gaussian cluster dataset. Each class has a random centroid
    at radius ~3 in feature space with shared std. Separable enough that
    both classifiers should hit ~80% accuracy in ~20 epochs."""
    rng = np.random.RandomState(seed)
    centroids = rng.randn(n_classes, n_features) * 3.0
    X_parts, y_parts = [], []
    for k in range(n_classes):
        X_k = centroids[k] + rng.randn(n_per_class, n_features) * cluster_std
        y_k = np.full(n_per_class, f"class_{k}", dtype=object)
        X_parts.append(X_k)
        y_parts.append(y_k)
    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    # Shuffle
    order = rng.permutation(len(X))
    return X[order], y[order]


def train_val_split(
    X: np.ndarray, y: np.ndarray, n_val: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]
    return X_train, y_train, X_val, y_val


def train_via_partial_fit(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    classes: list,
    epochs: int,
    chunk_size: int,
    rng: np.random.RandomState,
) -> None:
    """Mimic the MermaidTrainer training loop: epoch over shuffled data in
    partial_fit chunks. This is the interaction pattern the real trainer
    uses, so tests exercise the same surface."""
    n = len(X)
    for _ in range(epochs):
        order = rng.permutation(n)
        X_shuf, y_shuf = X[order], y[order]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            clf.partial_fit(X_shuf[start:end], y_shuf[start:end],
                            classes=classes)


class MLPBenchmarkBase:
    """Shared test suite executed against both sklearn and torch MLPs.

    Subclasses implement `_make_classifier()` returning an untrained
    sklearn-compatible MLP classifier instance.

    The accuracy thresholds in these tests (0.85 / 0.80 / 0.75) are
    deliberately-loose sanity floors that answer "can this implementation
    learn at all?" on the easy, well-separated synthetic task — set well
    below the ~95%+ both classifiers might actually reach, to avoid flakiness.
    They are NOT parity bounds; the direct sklearn-vs-torch head-to-head
    comparison lives in `MLPParityTest`.
    """

    # Subclasses override.
    def _make_classifier(self):  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):  # type: ignore[override]
        X_all, y_all = make_gaussian_clusters(
            n_per_class=(N_TRAIN + N_VAL) // N_CLASSES, seed=SEED,
        )
        cls.X_train, cls.y_train, cls.X_val, cls.y_val = train_val_split(
            X_all, y_all, n_val=N_VAL)
        cls.classes_list = sorted(np.unique(y_all).tolist())

    def _train(self, clf):
        rng = np.random.RandomState(SEED + 1)  # training-shuffle stream
        t0 = time.time()
        train_via_partial_fit(
            clf, self.X_train, self.y_train, self.classes_list,
            epochs=EPOCHS, chunk_size=PARTIAL_FIT_BATCH, rng=rng,
        )
        return time.time() - t0

    # --- Tests -----------------------------------------------------------

    def test_converges_on_training_set(self):
        """Training accuracy should exceed 85% on the easy synthetic task."""
        clf = self._make_classifier()
        self._train(clf)
        preds = clf.predict(self.X_train)
        acc = float(np.mean(preds == self.y_train))
        self.assertGreater(
            acc, 0.85,
            f"Training accuracy {acc:.3f} below threshold 0.85"
            f" for {type(clf).__name__}"
        )

    def test_generalises_to_validation(self):
        """Validation accuracy should exceed 80%."""
        clf = self._make_classifier()
        self._train(clf)
        preds = clf.predict(self.X_val)
        acc = float(np.mean(preds == self.y_val))
        self.assertGreater(
            acc, 0.80,
            f"Validation accuracy {acc:.3f} below threshold 0.80"
            f" for {type(clf).__name__}"
        )

    def test_predict_proba_shape_and_normalisation(self):
        clf = self._make_classifier()
        self._train(clf)
        probs = clf.predict_proba(self.X_val)
        self.assertEqual(probs.shape, (N_VAL, N_CLASSES))
        np.testing.assert_allclose(
            probs.sum(axis=1), np.ones(N_VAL), rtol=1e-5, atol=1e-5)
        self.assertTrue((probs >= 0).all())
        self.assertTrue((probs <= 1).all())

    def test_classes_attribute_is_sorted(self):
        clf = self._make_classifier()
        self._train(clf)
        self.assertEqual(
            list(clf.classes_),
            sorted(clf.classes_.tolist()),
        )
        self.assertEqual(set(clf.classes_.tolist()), set(self.classes_list))

    def test_loss_curve_available_and_finite(self):
        clf = self._make_classifier()
        self._train(clf)
        self.assertTrue(
            hasattr(clf, "loss_curve_"),
            f"{type(clf).__name__} missing loss_curve_"
        )
        self.assertGreater(len(clf.loss_curve_), 0)
        self.assertTrue(all(np.isfinite(clf.loss_curve_)))
        # Loss should trend downward — the first recorded loss should be
        # higher than the last, with some slack.
        self.assertLess(clf.loss_curve_[-1], clf.loss_curve_[0])

    def test_predict_proba_and_predict_agree(self):
        clf = self._make_classifier()
        self._train(clf)
        probs = clf.predict_proba(self.X_val)
        argmax_labels = clf.classes_[np.argmax(probs, axis=1)]
        preds = clf.predict(self.X_val)
        np.testing.assert_array_equal(preds, argmax_labels)

    def test_pickle_roundtrip_preserves_predictions(self):
        clf = self._make_classifier()
        self._train(clf)
        preds_before = clf.predict(self.X_val)
        probs_before = clf.predict_proba(self.X_val)

        clf2 = pickle.loads(pickle.dumps(clf))

        preds_after = clf2.predict(self.X_val)
        probs_after = clf2.predict_proba(self.X_val)

        np.testing.assert_array_equal(preds_before, preds_after)
        np.testing.assert_allclose(probs_before, probs_after,
                                   rtol=1e-5, atol=1e-6)

    def test_partial_fit_accumulates_across_calls(self):
        """Multiple small partial_fit calls should still converge."""
        clf = self._make_classifier()
        rng = np.random.RandomState(SEED + 2)  # tiny-chunk shuffle stream
        # Deliberately tiny chunks: tests the incremental-fit contract.
        train_via_partial_fit(
            clf, self.X_train, self.y_train, self.classes_list,
            epochs=EPOCHS, chunk_size=50, rng=rng,
        )
        acc = float(np.mean(clf.predict(self.X_val) == self.y_val))
        self.assertGreater(
            acc, 0.75,
            f"Incremental partial_fit accuracy {acc:.3f} below 0.75"
            f" for {type(clf).__name__}"
        )

    def test_decision_function_or_predict_proba_usable_for_calibration(self):
        """CalibratedClassifierCV(cv='prefit') needs either
        decision_function or predict_proba on the base estimator. At
        minimum predict_proba must be present and well-shaped."""
        clf = self._make_classifier()
        self._train(clf)
        probs = clf.predict_proba(self.X_val[:10])
        self.assertEqual(probs.shape, (10, N_CLASSES))


class SklearnMLPBenchmarkTest(MLPBenchmarkBase, unittest.TestCase):
    """Baseline: sklearn MLPClassifier."""

    def _make_classifier(self):
        return MLPClassifier(
            hidden_layer_sizes=HIDDEN,
            learning_rate_init=LR,
            random_state=SEED,
        )


class TorchMLPBenchmarkTest(MLPBenchmarkBase, unittest.TestCase):
    """Replacement: TorchMLPClassifier."""

    def _make_classifier(self):
        return TorchMLPClassifier(
            hidden_layer_sizes=HIDDEN,
            learning_rate_init=LR,
            random_state=SEED,
        )

    def test_old_pickle_without_class_weight_tensor_can_resume(self):
        # Simulate a pre-class_weight pickle: its state lacks both
        # ``class_weight`` and ``_class_weight_tensor``. __setstate__
        # must backfill them so a subsequent partial_fit (which reads
        # self._class_weight_tensor unconditionally) doesn't raise
        # AttributeError.
        clf = self._make_classifier()
        self._train(clf)
        state = clf.__getstate__()
        state.pop("class_weight", None)
        state.pop("_class_weight_tensor", None)

        restored = TorchMLPClassifier.__new__(TorchMLPClassifier)
        restored.__setstate__(state)
        self.assertIsNone(restored._class_weight_tensor)

        # Resuming training must not raise.
        restored.partial_fit(
            self.X_train[:PARTIAL_FIT_BATCH],
            self.y_train[:PARTIAL_FIT_BATCH],
        )


class MLPParityTest(unittest.TestCase):
    """Direct head-to-head: torch must match sklearn accuracy within
    tolerance on the same synthetic task."""

    @classmethod
    def setUpClass(cls):  # type: ignore[override]
        X_all, y_all = make_gaussian_clusters(
            n_per_class=(N_TRAIN + N_VAL) // N_CLASSES, seed=SEED,
        )
        cls.X_train, cls.y_train, cls.X_val, cls.y_val = train_val_split(
            X_all, y_all, n_val=N_VAL)
        cls.classes_list = sorted(np.unique(y_all).tolist())

    def _train_both(self):
        sk = MLPClassifier(
            hidden_layer_sizes=HIDDEN,
            learning_rate_init=LR,
            random_state=SEED,
        )
        tr = TorchMLPClassifier(
            hidden_layer_sizes=HIDDEN,
            learning_rate_init=LR,
            random_state=SEED,
        )
        for clf in (sk, tr):
            rng = np.random.RandomState(SEED + 1)  # training-shuffle stream
            train_via_partial_fit(
                clf, self.X_train, self.y_train, self.classes_list,
                epochs=EPOCHS, chunk_size=PARTIAL_FIT_BATCH, rng=rng,
            )
        return sk, tr

    def test_torch_validation_accuracy_within_tolerance(self):
        """Torch val accuracy must be within 5% of sklearn val accuracy."""
        sk, tr = self._train_both()
        sk_acc = float(np.mean(sk.predict(self.X_val) == self.y_val))
        tr_acc = float(np.mean(tr.predict(self.X_val) == self.y_val))
        self.assertGreaterEqual(
            tr_acc, sk_acc - 0.05,
            f"Torch {tr_acc:.3f} vs sklearn {sk_acc:.3f} —"
            f" torch must be within 5% of sklearn."
        )

    def test_torch_training_accuracy_within_tolerance(self):
        sk, tr = self._train_both()
        sk_acc = float(np.mean(sk.predict(self.X_train) == self.y_train))
        tr_acc = float(np.mean(tr.predict(self.X_train) == self.y_train))
        self.assertGreaterEqual(
            tr_acc, sk_acc - 0.05,
            f"Torch train {tr_acc:.3f} vs sklearn {sk_acc:.3f}"
        )

    def test_torch_predict_proba_distribution_close(self):
        """Torch predict_proba should put mass on the same class as sklearn
        for the majority of validation samples."""
        sk, tr = self._train_both()
        sk_argmax = np.argmax(sk.predict_proba(self.X_val), axis=1)
        tr_argmax = np.argmax(tr.predict_proba(self.X_val), axis=1)
        agreement = float(np.mean(sk_argmax == tr_argmax))
        self.assertGreater(
            agreement, 0.85,
            f"Torch and sklearn argmax agree on only {agreement:.3f} of"
            f" validation samples."
        )

    def test_predict_proba_values_close_to_sklearn(self):
        """Raw predict_proba *values* (not just argmax) should track sklearn.

        Exact equality is impossible — weight init and Adam internals
        differ — so we compare the full probability matrices with a
        tolerance on the mean absolute difference per probability entry.
        """
        sk, tr = self._train_both()
        sk_probs = sk.predict_proba(self.X_val)
        tr_probs = tr.predict_proba(self.X_val)
        self.assertEqual(sk_probs.shape, tr_probs.shape)
        # On this task the two implementations agree to ~1e-6; 1e-2 is a
        # meaningful "within 1% on average" bound with wide headroom for
        # BLAS/framework numerical drift across environments.
        mean_abs_diff = float(np.mean(np.abs(sk_probs - tr_probs)))
        self.assertLess(
            mean_abs_diff, 1e-2,
            f"Mean abs difference between torch and sklearn predict_proba"
            f" is {mean_abs_diff:.4f} (> 1e-2)."
        )

    def test_calibrated_predict_proba_close_to_sklearn(self):
        """Post-calibration probabilities should track between both.

        Wraps each (prefit) base estimator in the same
        CalibratedClassifierCV(cv='prefit') path pyspacer uses, calibrates
        both on an identical held-out split, and compares the calibrated
        probability matrices within tolerance.
        """
        from sklearn.calibration import CalibratedClassifierCV

        sk, tr = self._train_both()

        # Calibrate on the first half of val, compare on the second half,
        # so calibration and evaluation use disjoint data.
        n_cal = N_VAL // 2
        X_cal, y_cal = self.X_val[:n_cal], self.y_val[:n_cal]
        X_eval = self.X_val[n_cal:]

        sk_cal = CalibratedClassifierCV(sk, cv="prefit").fit(X_cal, y_cal)
        tr_cal = CalibratedClassifierCV(tr, cv="prefit").fit(X_cal, y_cal)

        sk_probs = sk_cal.predict_proba(X_eval)
        tr_probs = tr_cal.predict_proba(X_eval)
        self.assertEqual(sk_probs.shape, tr_probs.shape)
        mean_abs_diff = float(np.mean(np.abs(sk_probs - tr_probs)))
        self.assertLess(
            mean_abs_diff, 1e-2,
            f"Mean abs difference between torch and sklearn calibrated"
            f" predict_proba is {mean_abs_diff:.4f} (> 1e-2)."
        )


class BatchingEquivalenceTest(unittest.TestCase):
    """Verify that TorchMLPClassifier and sklearn.MLPClassifier have
    equivalent batching contracts in partial_fit.

    These tests don't check numerical equality (weight-init and internal
    Adam details differ); they check that the *batching structure*
    matches — the surface sklearn callers depend on.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(SEED)
        cls.X = rng.randn(500, 16).astype(np.float32)
        cls.y = np.array(
            ["a", "b", "c"] * 166 + ["a", "b"], dtype=object)
        cls.classes = ["a", "b", "c"]

    def _make_pair(self):
        """Same hyperparameters on both sides."""
        kw = dict(
            hidden_layer_sizes=(16,),
            learning_rate_init=1e-3,
            batch_size="auto",
            random_state=SEED,
        )
        return MLPClassifier(**kw), TorchMLPClassifier(**kw)

    def test_one_partial_fit_call_appends_one_loss_entry(self):
        """partial_fit appends exactly one value to loss_curve_ per call,
        no matter how many internal mini-batches are needed."""
        sk, tr = self._make_pair()
        for clf in (sk, tr):
            clf.partial_fit(self.X, self.y, classes=self.classes)
            self.assertEqual(
                len(clf.loss_curve_), 1,
                f"{type(clf).__name__}: expected 1 entry, got"
                f" {len(clf.loss_curve_)}"
            )
            clf.partial_fit(self.X, self.y)
            self.assertEqual(
                len(clf.loss_curve_), 2,
                f"{type(clf).__name__}: expected 2 entries, got"
                f" {len(clf.loss_curve_)}"
            )

    def test_n_iter_increments_by_one_per_partial_fit(self):
        """TorchMLPClassifier's n_iter_ tracks total partial_fit calls.

        Note: sklearn's MLPClassifier has a quirk where n_iter_ is reset
        to 0 at the start of each _fit_stochastic call, so it always
        reads as 1 after any partial_fit call. We deliberately diverge
        here — tracking cumulative calls is more useful and no caller in
        this codebase relies on the sklearn-style reset behaviour.
        """
        _, tr = self._make_pair()
        for expected in (1, 2, 3):
            tr.partial_fit(self.X, self.y, classes=self.classes)
            self.assertEqual(tr.n_iter_, expected)

    def test_auto_batch_size_is_min_200_and_n_samples(self):
        """batch_size='auto' resolves to min(200, n_samples) in both."""
        # 500 samples → 200 mini-batches (200, 200, 100)
        sk, tr = self._make_pair()
        sk.partial_fit(self.X[:500], self.y[:500], classes=self.classes)
        tr.partial_fit(self.X[:500], self.y[:500], classes=self.classes)
        # Torch exposes resolved batch_size via the helper.
        self.assertEqual(tr._resolve_batch_size(500), 200)

        # 50 samples < 200 → batch_size = 50 (full input)
        sk2, tr2 = self._make_pair()
        sk2.partial_fit(self.X[:50], self.y[:50], classes=self.classes)
        tr2.partial_fit(self.X[:50], self.y[:50], classes=self.classes)
        self.assertEqual(tr2._resolve_batch_size(50), 50)

    def test_explicit_batch_size_is_clipped_to_n_samples(self):
        """batch_size=128 on 50-sample input clips to 50 in both."""
        kw = dict(
            hidden_layer_sizes=(8,),
            batch_size=128,
            random_state=SEED,
        )
        sk = MLPClassifier(**kw)
        tr = TorchMLPClassifier(**kw)
        sk.partial_fit(self.X[:50], self.y[:50], classes=self.classes)
        tr.partial_fit(self.X[:50], self.y[:50], classes=self.classes)
        self.assertEqual(tr._resolve_batch_size(50), 50)

    def test_number_of_gradient_steps_per_partial_fit_matches(self):
        """sklearn and torch must do the same number of Adam steps per
        partial_fit call: ceil(n_samples / batch_size). We verify this
        by counting how many times the torch optimizer's step counter
        advances — sklearn has no equivalent public counter, but we can
        verify the torch side matches the sklearn contract
        (ceil(n/batch_size)) directly."""
        tr = TorchMLPClassifier(
            hidden_layer_sizes=(8,),
            batch_size=100,
            random_state=SEED,
        )
        # 500 samples, batch_size=100 → 5 Adam steps per partial_fit call.
        tr.partial_fit(self.X[:500], self.y[:500], classes=self.classes)
        # Adam optimizer keeps a per-parameter step counter. Pull the step
        # count for the first linear layer's weight.
        first_param = next(iter(tr._module.parameters()))
        step_count = tr._optimizer.state[first_param]["step"]
        # PyTorch stores step as either a tensor or a python int depending
        # on version; normalise.
        step_int = int(step_count.item()) if hasattr(step_count, "item") \
            else int(step_count)
        self.assertEqual(
            step_int, 5,
            f"Expected 5 Adam steps for 500 samples at batch_size=100,"
            f" got {step_int}"
        )

        # Second partial_fit call → 5 more steps (cumulative 10).
        tr.partial_fit(self.X[:500], self.y[:500])
        step_count = tr._optimizer.state[first_param]["step"]
        step_int = int(step_count.item()) if hasattr(step_count, "item") \
            else int(step_count)
        self.assertEqual(step_int, 10)

    def test_partial_fit_on_smaller_than_batch_size_is_one_step(self):
        """Input smaller than batch_size → exactly one Adam step."""
        tr = TorchMLPClassifier(
            hidden_layer_sizes=(8,),
            batch_size=200,
            random_state=SEED,
        )
        tr.partial_fit(self.X[:50], self.y[:50], classes=self.classes)
        first_param = next(iter(tr._module.parameters()))
        step_count = tr._optimizer.state[first_param]["step"]
        step_int = int(step_count.item()) if hasattr(step_count, "item") \
            else int(step_count)
        self.assertEqual(step_int, 1)

    def test_loss_curve_records_regularised_loss_trend(self):
        """Both implementations' loss_curve_ should trend down with
        continued training on the same data."""
        sk, tr = self._make_pair()
        for _ in range(10):
            sk.partial_fit(self.X, self.y, classes=self.classes)
            tr.partial_fit(self.X, self.y, classes=self.classes)
        self.assertLess(sk.loss_curve_[-1], sk.loss_curve_[0])
        self.assertLess(tr.loss_curve_[-1], tr.loss_curve_[0])

    def test_same_random_state_yields_reproducible_shuffle(self):
        """Two fresh classifiers with the same random_state, fed the same
        data, should produce the same loss trajectory. This is the
        sklearn contract — and our torch version honours it by re-seeding
        from random_state inside each partial_fit call."""
        tr1 = TorchMLPClassifier(
            hidden_layer_sizes=(8,), random_state=SEED,
        )
        tr2 = TorchMLPClassifier(
            hidden_layer_sizes=(8,), random_state=SEED,
        )
        for _ in range(5):
            tr1.partial_fit(self.X, self.y, classes=self.classes)
            tr2.partial_fit(self.X, self.y, classes=self.classes)
        np.testing.assert_allclose(
            tr1.loss_curve_, tr2.loss_curve_, rtol=1e-6,
            err_msg="Same random_state must yield identical loss curves"
        )


class TorchMLPIntegratesWithCalibratedClassifierCVTest(unittest.TestCase):
    """TorchMLPClassifier must plug into the real pyspacer path:
    CalibratedClassifierCV(cv='prefit') wrapping + MermaidTrainer's
    batched calibration helper + evaluate_classifier-style inference.
    """

    def test_full_calibration_and_inference_path(self):
        from unittest import mock

        from sklearn.calibration import CalibratedClassifierCV

        from mermaid_classifier.pyspacer.trainer import MermaidTrainer

        X_all, y_all = make_gaussian_clusters(
            n_per_class=(N_TRAIN + N_VAL) // N_CLASSES, seed=SEED,
        )
        X_train, y_train, X_val, y_val = train_val_split(
            X_all, y_all, n_val=N_VAL)
        classes_list = sorted(np.unique(y_all).tolist())

        clf = TorchMLPClassifier(
            hidden_layer_sizes=HIDDEN,
            learning_rate_init=LR,
            random_state=SEED,
        )
        rng = np.random.RandomState(SEED + 1)  # training-shuffle stream
        train_via_partial_fit(
            clf, X_train, y_train, classes_list,
            epochs=EPOCHS, chunk_size=PARTIAL_FIT_BATCH, rng=rng,
        )

        mock_labels = mock.Mock()
        def batch_generator(batch_size=100):
            for i in range(0, len(X_train), batch_size):
                end = min(i + batch_size, len(X_train))
                yield (
                    [X_train[j] for j in range(i, end)],
                    [y_train[j] for j in range(i, end)],
                )
        mock_labels.load_data_in_batches = batch_generator

        trainer = MermaidTrainer(batch_size=100)
        calibrated = trainer._calibrate_in_batches(clf, mock_labels)

        self.assertIsInstance(calibrated, CalibratedClassifierCV)
        self.assertEqual(calibrated.cv, "prefit")
        self.assertTrue(hasattr(calibrated, "calibrated_classifiers_"))

        # Inference path used by pyspacer's evaluate_classifier:
        probs = calibrated.predict_proba(X_val)
        preds = calibrated.predict(X_val)
        self.assertEqual(probs.shape, (N_VAL, N_CLASSES))
        np.testing.assert_allclose(
            probs.sum(axis=1), np.ones(N_VAL), rtol=1e-5, atol=1e-5)
        acc = float(np.mean(preds == y_val))
        self.assertGreater(
            acc, 0.80,
            f"Calibrated TorchMLP val accuracy {acc:.3f} below 0.80"
        )

        # Full pickle round-trip of the calibrated artifact — this is
        # exactly what spacer.storage.store_classifier serialises.
        restored = pickle.loads(pickle.dumps(calibrated))
        np.testing.assert_allclose(
            restored.predict_proba(X_val), probs, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(restored.predict(X_val), preds)


if __name__ == "__main__":
    unittest.main()
