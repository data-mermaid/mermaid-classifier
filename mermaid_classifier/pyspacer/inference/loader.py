"""load_predictor: serve-time loader for the portable artifact, with loud
load-time validation of the graph against its manifest."""

from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mermaid_classifier.pyspacer.inference import SCHEMA_VERSION, ManifestError

logger = logging.getLogger(__name__)


class Predictor:
    """A loaded classifier head: feature batch -> calibrated probabilities."""

    def __init__(self, graph: Any, classes: list[str], input_dim: int) -> None:
        self._graph = graph
        self.classes = classes
        self.input_dim = input_dim

    @property
    def classes_(self) -> list[str]:
        """Alias for ``classes`` so a Predictor is a drop-in for the former
        pickled classifier in metrics code that reads ``clf.classes_``."""
        return self.classes

    def predict_proba(self, features: Any) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.input_dim:
            raise ValueError(f"features must be (N, {self.input_dim}); got {arr.shape}.")
        with torch.no_grad():
            return self._graph(torch.from_numpy(arr)).numpy().astype(np.float64)


def load_predictor(model_pt_path: str | Path, model_json_path: str | Path) -> Predictor:
    """Load model.pt + model.json, validating compatibility loudly.

    Raises ManifestError on schema-version, class-count, input_dim, or
    model.pt/manifest digest mismatch, rather than returning a
    silently-mispredicting predictor. A manifest with no ``model_pt_sha256``
    key — an artifact cut before the field existed — is served with a logged
    warning instead of a refusal.
    """
    model_pt_path = Path(model_pt_path)
    manifest = json.loads(Path(model_json_path).read_text())

    schema_version = manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ManifestError(
            f"model.json schema_version={schema_version!r} is incompatible"
            f" with this loader (expects {SCHEMA_VERSION})."
        )

    classes = manifest["classes"]
    input_dim = int(manifest["input_dim"])

    data = model_pt_path.read_bytes()

    recorded_sha256 = manifest.get("model_pt_sha256")
    if recorded_sha256 is None:
        # Stable marker (mirrors classify.py's [classify.unverified_extractor]):
        # a metric filter can count artifacts still serving unverified.
        logger.warning(
            "[loader.unverified_graph] model.json has no 'model_pt_sha256' key;"
            " serving without verifying model.pt against the manifest that names it"
        )
    else:
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != recorded_sha256:
            raise ManifestError(
                f"model.pt sha256={actual_sha256} does not match model.json's"
                f" recorded model_pt_sha256={recorded_sha256}: this model.json"
                " does not describe this model.pt."
            )

    graph = torch.jit.load(io.BytesIO(data), map_location="cpu")
    graph.eval()

    # Probe with a (1, input_dim) batch: catches input_dim mismatch (matmul
    # shape error) and lets us check the output class count.
    try:
        with torch.no_grad():
            probe = graph(torch.zeros(1, input_dim, dtype=torch.float32))
    except Exception as exc:  # noqa: BLE001 - re-raise loudly as ManifestError
        raise ManifestError(
            f"graph rejects input_dim={input_dim} declared in model.json: {exc}"
        ) from exc

    if probe.shape[1] != len(classes):
        raise ManifestError(
            f"class-count mismatch: graph outputs {probe.shape[1]} classes"
            f" but model.json declares {len(classes)}."
        )

    return Predictor(graph, list(classes), input_dim)
