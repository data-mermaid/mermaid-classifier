"""Tests for annotation.py's classifier-artifact resolver: turning an MLflow
model ID, S3 directory, or local directory into local (model.pt, model.json)
paths for load_predictor."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pyspacer._calibrated_model_fixture import make_calibrated_model


class ResolveFilesystemDirectoryTest(unittest.TestCase):
    def test_local_directory_returns_pt_and_json_paths(self):
        from mermaid_classifier.pyspacer.annotation import (
            resolve_classifier_artifact,
        )
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "model.pt").write_bytes(b"pt")
            (Path(d) / "model.json").write_text("{}")

            pt, js = resolve_classifier_artifact(d)

            self.assertEqual(pt, Path(d) / "model.pt")
            self.assertEqual(js, Path(d) / "model.json")

    def test_trailing_slash_is_tolerated(self):
        from mermaid_classifier.pyspacer.annotation import (
            resolve_classifier_artifact,
        )
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "model.pt").write_bytes(b"pt")
            (Path(d) / "model.json").write_text("{}")

            pt, js = resolve_classifier_artifact(d + "/")

            self.assertEqual(pt, Path(d) / "model.pt")
            self.assertEqual(js, Path(d) / "model.json")


class ResolveMlflowModelIdTest(unittest.TestCase):
    def test_round_trip_resolves_and_matches_source(self):
        import mlflow
        from mermaid_classifier.pyspacer.annotation import (
            resolve_classifier_artifact,
        )
        from mermaid_classifier.pyspacer.inference import (
            export_artifact, load_predictor,
        )
        from mermaid_classifier.pyspacer.mlflow_model import log_artifact_model

        model, X = make_calibrated_model()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model_pt, _manifest, _ = export_artifact(model, str(d), X)
            model_json = d / "model.json"

            artifacts_root = d / "artifacts"
            mlflow.set_tracking_uri(f"sqlite:///{d / 'mlflow.db'}")
            exp_id = mlflow.create_experiment(
                "test-resolver",
                artifact_location=artifacts_root.as_uri())
            with mlflow.start_run(experiment_id=exp_id):
                info = log_artifact_model(
                    model_pt, model_json, registered_model_name=None)

            # mlflow_connect() would reset the tracking URI to the production
            # server; patch it to a no-op so resolution stays on sqlite.
            with mock.patch(
                "mermaid_classifier.pyspacer.annotation.mlflow_connect",
                return_value=None,
            ):
                pt, js = resolve_classifier_artifact(info.model_id)

            predictor = load_predictor(pt, js)
            got = predictor.predict_proba(X)

        self.assertLess(
            float(np.max(np.abs(got - model.predict_proba(X)))), 1e-6)


class PyspacerPickleGuardTest(unittest.TestCase):
    """Guard: annotation.py must not reuse the pyspacer classify/pickle path."""

    def _source(self):
        from mermaid_classifier.pyspacer import annotation
        return Path(annotation.__file__).read_text()

    def test_no_classify_image_or_classifyimagemsg(self):
        src = self._source()
        self.assertNotIn("classify_image", src)
        self.assertNotIn("ClassifyImageMsg", src)

    def test_imports_load_predictor(self):
        src = self._source()
        self.assertIn("load_predictor", src)


if __name__ == "__main__":
    unittest.main()
