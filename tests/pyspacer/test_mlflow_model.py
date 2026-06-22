"""Tests for the MLflow pyfunc shim that stores the portable artifact
(model.pt + model.json) and serves it through load_predictor."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyspacer._calibrated_model_fixture import make_calibrated_model


class ArtifactPredictorModelTest(unittest.TestCase):
    def test_load_context_then_predict_matches_source(self):
        from mermaid_classifier.pyspacer.inference import export_artifact
        from mermaid_classifier.pyspacer.mlflow_model import (
            ArtifactPredictorModel,
        )

        model, X = make_calibrated_model()
        with tempfile.TemporaryDirectory() as d:
            model_pt, _manifest, _ = export_artifact(model, d, X)
            model_json = Path(d) / "model.json"

            # A minimal stand-in for mlflow's PythonModelContext: only the
            # .artifacts mapping is used by load_context.
            class _Ctx:
                artifacts = {
                    "model_pt": str(model_pt),
                    "model_json": str(model_json),
                }

            pyfunc_model = ArtifactPredictorModel()
            pyfunc_model.load_context(_Ctx())
            got = pyfunc_model.predict(_Ctx(), X)

        self.assertEqual(got.shape, model.predict_proba(X).shape)
        self.assertLess(
            float(np.max(np.abs(got - model.predict_proba(X)))), 1e-6)


class LogArtifactModelTest(unittest.TestCase):
    def test_logs_pt_and_json_and_no_pickle_and_reloads(self):
        import mlflow
        from mermaid_classifier.pyspacer.inference import export_artifact
        from mermaid_classifier.pyspacer.mlflow_model import log_artifact_model

        model, X = make_calibrated_model()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model_pt, _manifest, _ = export_artifact(model, str(d), X)
            model_json = d / "model.json"

            # Local file-store tracking dir so the test needs no server.
            tracking = d / "mlruns"
            mlflow.set_tracking_uri(f"file://{tracking}")
            mlflow.set_experiment("test-artifact-store")
            with mlflow.start_run():
                info = log_artifact_model(
                    model_pt, model_json,
                    registered_model_name=None)
                loaded = mlflow.pyfunc.load_model(info.model_uri)

            # Stored the deployable files, and NO classifier pickle.
            artifact_files = [p.name for p in tracking.rglob("*")
                              if p.is_file()]
            self.assertIn("model.pt", artifact_files)
            self.assertIn("model.json", artifact_files)
            self.assertNotIn("model.pkl", artifact_files)
            self.assertFalse(
                any(f.endswith(".pkl") and "model" in f
                    for f in artifact_files),
                msg=f"unexpected classifier pickle in {artifact_files}")

            got = loaded.predict(X)
        self.assertLess(
            float(np.max(np.abs(np.asarray(got) - model.predict_proba(X)))),
            1e-6)


if __name__ == "__main__":
    unittest.main()
