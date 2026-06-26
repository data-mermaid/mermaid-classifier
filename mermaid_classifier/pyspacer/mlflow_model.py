"""MLflow pyfunc shim for the portable classifier artifact.

Stores the deployable TorchScript files (model.pt + model.json) as a
registered MLflow model and serves them through the ONE shared loader,
``load_predictor`` — the same loader used at eval time and by the Lambda.
The pyfunc is a registry/packaging shim, not a second implementation.

This module imports ``mlflow`` and therefore lives train-side, NOT in
``mermaid_classifier.pyspacer.inference`` (which must stay mlflow-free so the
``[inference]`` dependency split holds).
"""

from __future__ import annotations

import mlflow

from mermaid_classifier.pyspacer.inference import load_predictor


class ArtifactPredictorModel(mlflow.pyfunc.PythonModel):
    """Loads model.pt + model.json via load_predictor; ``predict`` returns the
    calibrated probability matrix for a feature batch."""

    def load_context(self, context):
        self._predictor = load_predictor(
            context.artifacts["model_pt"],
            context.artifacts["model_json"],
        )

    def predict(self, context, model_input, params=None):
        return self._predictor.predict_proba(model_input)


def log_artifact_model(
    model_pt_path,
    model_json_path,
    *,
    registered_model_name,
    signature=None,
):
    """Log the portable artifact as an MLflow pyfunc model and register it.

    Stores ``model.pt`` + ``model.json`` as the model's artifacts, preserving
    the existing registry + ``get_logged_model().artifact_location``
    resolution that deployment relies on. Returns the MLflow ``ModelInfo``
    (so callers can read ``.model_id``).

    Logs ``ArtifactPredictorModel`` as a standard pyfunc object model.
    MLflow cloudpickles the stateless wrapper instance to
    ``python_model.pkl``; that is intentional and harmless — the spec only
    prohibits the *classifier* pickle (``model.pkl``), not the MLflow
    internal wrapper serialization.
    """
    return mlflow.pyfunc.log_model(
        python_model=ArtifactPredictorModel(),
        artifacts={
            "model_pt": str(model_pt_path),
            "model_json": str(model_json_path),
        },
        registered_model_name=registered_model_name,
        signature=signature,
    )
