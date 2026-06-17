"""export_artifact: freeze the calibrated head to TorchScript, parity-gate it
against the source model, and write the generated manifest."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sklearn
import torch

from mermaid_classifier.pyspacer.inference import (
    SCHEMA_VERSION, TASK_NAME, ParityError,
)
from mermaid_classifier.pyspacer.inference.head import build_calibrated_head


def export_artifact(
    model,
    output_dir,
    reference_features,
    *,
    config: dict | None = None,
    task: str = TASK_NAME,
    tol: float = 1e-6,
):
    """Build, freeze, parity-gate, and persist the portable artifact.

    Returns (model_pt_path, manifest_dict, max_abs_diff). Raises ParityError
    if the frozen graph diverges from ``model.predict_proba`` beyond ``tol``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    head = build_calibrated_head(model)
    head.eval()
    scripted = torch.jit.script(head)
    frozen = torch.jit.freeze(scripted)

    # Parity gate: frozen graph vs source model on the representative batch.
    ref = np.asarray(reference_features, dtype=np.float32)
    expected = model.predict_proba(ref)
    with torch.no_grad():
        got = frozen(torch.from_numpy(ref)).numpy().astype(np.float64)
    max_diff = float(np.max(np.abs(expected - got)))
    if max_diff > tol:
        raise ParityError(
            f"Frozen graph diverges from source model: max|Δ|={max_diff:.3e}"
            f" exceeds tol={tol:.3e}. Refusing to ship."
        )

    estimator = model.calibrated_classifiers_[0].estimator
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "classes": model.classes_.tolist(),
        "input_dim": int(estimator.n_features_in_),
        "config": config if config is not None else {"patch_size": 224},
        "trained_with": {
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    model_pt = output_dir / "model.pt"
    torch.jit.save(frozen, str(model_pt))
    (output_dir / "model.json").write_text(json.dumps(manifest, indent=2))

    return model_pt, manifest, max_diff
