"""load_predictor: serve-time loader for the portable artifact, with loud
load-time validation of the graph against its manifest."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mermaid_classifier.pyspacer.inference import SCHEMA_VERSION, ManifestError


class Predictor:
    """A loaded classifier head: feature batch -> calibrated probabilities."""

    def __init__(self, graph, classes: list[str], input_dim: int):
        self._graph = graph
        self.classes = classes
        self.input_dim = input_dim

    def predict_proba(self, features) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.input_dim:
            raise ValueError(
                f"features must be (N, {self.input_dim}); got {arr.shape}."
            )
        with torch.no_grad():
            return self._graph(torch.from_numpy(arr)).numpy().astype(np.float64)


def load_predictor(model_pt_path, model_json_path) -> Predictor:
    """Load model.pt + model.json, validating compatibility loudly.

    Raises ManifestError on schema-version, class-count, or input_dim
    mismatch rather than returning a silently-mispredicting predictor.
    """
    manifest = json.loads(Path(model_json_path).read_text())

    schema_version = manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ManifestError(
            f"model.json schema_version={schema_version!r} is incompatible"
            f" with this loader (expects {SCHEMA_VERSION})."
        )

    classes = manifest["classes"]
    input_dim = int(manifest["input_dim"])

    graph = torch.jit.load(str(model_pt_path), map_location="cpu")
    graph.eval()

    # Probe with a (1, input_dim) batch: catches input_dim mismatch (matmul
    # shape error) and lets us check the output class count.
    try:
        with torch.no_grad():
            probe = graph(torch.zeros(1, input_dim, dtype=torch.float32))
    except Exception as exc:  # noqa: BLE001 - re-raise loudly as ManifestError
        raise ManifestError(
            f"graph rejects input_dim={input_dim} declared in model.json:"
            f" {exc}"
        ) from exc

    if probe.shape[1] != len(classes):
        raise ManifestError(
            f"class-count mismatch: graph outputs {probe.shape[1]} classes"
            f" but model.json declares {len(classes)}."
        )

    return Predictor(graph, list(classes), input_dim)
