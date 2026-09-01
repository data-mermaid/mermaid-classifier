import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mermaid_classifier.model_review import beta_infer
from mermaid_classifier.model_review.beta_infer import BetaScoringError

_KEYS = [("A", 10, 20), ("A", 30, 40)]
_MATRIX = np.arange(8, dtype=np.float32).reshape(2, 4)


def _out_path(argv):
    return Path(argv[argv.index("--out") + 1])


def _features_path(argv):
    return Path(argv[argv.index("--features") + 1])


def _writer(preds, version=beta_infer.BETA_SKLEARN, status=0, stderr=""):
    """A fake runner that writes `preds` where the real subprocess would."""

    def run(argv, env):
        if status == 0:
            np.savez_compressed(
                _out_path(argv),
                pred_bagf=np.array(preds, dtype=str),
                sklearn_version=str(version),
            )
        return status, stderr

    return run


class ScoreCommandTest(unittest.TestCase):
    def test_pins_the_interpreter_and_the_calibration_dependencies(self):
        argv = beta_infer.score_command(
            "/usr/bin/uv", Path("/t/f.npz"), Path("/m/classifier.pkl"), Path("/t/o.npz")
        )
        self.assertEqual(argv[:2], ["/usr/bin/uv", "run"])
        self.assertIn("--no-project", argv)
        self.assertEqual(argv[argv.index("--python") + 1], "3.11")
        self.assertIn("scikit-learn==1.1.3", argv)
        self.assertIn("numpy<2", argv)
        # -P keeps the package dir off the child's sys.path
        self.assertIn("-P", argv)
        self.assertIn(str(beta_infer.SCORE_SCRIPT), argv)
        self.assertEqual(argv[argv.index("--classifier") + 1], "/m/classifier.pkl")


class ChildEnvTest(unittest.TestCase):
    def test_drops_the_variables_that_would_leak_this_interpreter(self):
        env = beta_infer.child_env(
            {"VIRTUAL_ENV": "/v", "PYTHONPATH": "/p", "PYTHONHOME": "/h", "AWS_PROFILE": "x"}
        )
        self.assertEqual(env, {"AWS_PROFILE": "x"})


class PredictPointsTest(unittest.TestCase):
    def setUp(self):
        self.pkl = Path(__file__).resolve()  # any existing file; the runner is faked

    def _predict(self, run):
        return beta_infer.predict_points(_KEYS, _MATRIX, str(self.pkl), run=run)

    def test_maps_each_key_to_its_prediction(self):
        preds = self._predict(_writer(["ba1::gf1", "ba2::"]))
        self.assertEqual(preds, {("A", 10, 20): "ba1::gf1", ("A", 30, 40): "ba2::"})

    def test_hands_the_feature_matrix_to_the_subprocess(self):
        seen = {}

        def run(argv, env):
            seen["X"] = np.load(_features_path(argv), allow_pickle=False)["X"]
            return _writer(["ba1::", "ba2::"])(argv, env)

        self._predict(run)
        np.testing.assert_array_equal(seen["X"], _MATRIX)

    def test_prediction_count_mismatch_is_rejected(self):
        with self.assertRaises(BetaScoringError) as ctx:
            self._predict(_writer(["only-one::"]))
        self.assertIn("1 predictions for 2 points", str(ctx.exception))

    def test_unpinned_sklearn_is_rejected(self):
        # A different scikit-learn silently changes the calibration, so the tab would
        # no longer show what the deployed model predicts.
        with self.assertRaises(BetaScoringError) as ctx:
            self._predict(_writer(["ba1::", "ba2::"], version="1.5.2"))
        self.assertIn("1.5.2", str(ctx.exception))
        self.assertIn("1.1.3", str(ctx.exception))

    def test_nonzero_exit_surfaces_the_child_stderr(self):
        with self.assertRaises(BetaScoringError) as ctx:
            self._predict(_writer([], status=3, stderr="boom in the child"))
        self.assertIn("boom in the child", str(ctx.exception))
        self.assertIn("exit 3", str(ctx.exception))

    def test_missing_output_is_distinguished_from_a_crash(self):
        with self.assertRaises(BetaScoringError) as ctx:
            self._predict(lambda argv, env: (0, ""))
        self.assertIn("wrote no predictions", str(ctx.exception))

    def test_temp_dir_is_removed_on_success_and_on_failure(self):
        seen = []

        def run(argv, env):
            seen.append(_features_path(argv).parent)
            return _writer(["ba1::", "ba2::"])(argv, env)

        self._predict(run)
        with self.assertRaises(BetaScoringError):
            self._predict(_writer(["only-one::"]))
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0].exists())

    def test_missing_pickle_names_the_resolved_path(self):
        with self.assertRaises(BetaScoringError) as ctx:
            beta_infer.predict_points(_KEYS, _MATRIX, "no/such/classifier.pkl")
        self.assertIn("classifier.pkl", str(ctx.exception))

    def test_absent_uv_is_reported_before_spawning(self):
        with (
            mock.patch.object(beta_infer.shutil, "which", return_value=None),
            self.assertRaises(BetaScoringError) as ctx,
        ):
            self._predict(_writer(["ba1::", "ba2::"]))
        self.assertIn("uv", str(ctx.exception))


class BetaScoreContractTest(unittest.TestCase):
    """Executes beta_score.py for real, so the npz keys the two modules exchange
    (`X`, `pred_bagf`, `sklearn_version`) cannot drift apart. Substitutes this
    interpreter for the uv prefix, so no scikit-learn 1.1.3 install is needed."""

    def _run_here(self, argv, env):
        # Drop the uv prefix ("uv run --python ... --no-project python -P"), keeping
        # the script path and its arguments.
        script_at = argv.index(str(beta_infer.SCORE_SCRIPT))
        proc = subprocess.run(
            [sys.executable, *argv[script_at:]], capture_output=True, text=True, check=False
        )
        return proc.returncode, proc.stderr

    def _predict(self, classes, tmp):
        """Score `_MATRIX` through the real script, against a genuine sklearn pickle.

        A real estimator is used rather than a stub so the child needs nothing on its
        import path but sklearn itself — which is exactly the isolated env's contract.
        `_MATRIX`'s two rows are the training samples, so each row predicts its own class.
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression().fit(_MATRIX.astype(np.float64), list(classes))
        pkl = Path(tmp) / "classifier.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(clf, f)
        import sklearn

        with mock.patch.object(beta_infer, "BETA_SKLEARN", sklearn.__version__):
            return beta_infer.predict_points(_KEYS, _MATRIX, str(pkl), run=self._run_here)

    def test_round_trips_through_the_real_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            preds = self._predict(["ba1::gf1", "ba2::"], tmp)
        self.assertEqual(preds, {("A", 10, 20): "ba1::gf1", ("A", 30, 40): "ba2::"})

    def test_classes_without_a_growth_form_separator_are_refused(self):
        # models/beta_model/ also holds a bare-BA-id pickle; scoring it would produce
        # labels the review app cannot split into BA + growth form.
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaises(BetaScoringError) as ctx,
        ):
            self._predict(["0ee3a3bb-48aa-41ec", "b76bca12-884b-4404"], tmp)
        self.assertIn("not BA::GF", str(ctx.exception))
