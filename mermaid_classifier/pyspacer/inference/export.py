"""export_artifact: freeze the calibrated head to TorchScript, parity-gate it
against the source model, and write the generated manifest."""

from __future__ import annotations

import json
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mermaid_classifier.pyspacer.inference import (
    PARITY_PROVEN_SKLEARN,
    SCHEMA_VERSION,
    TASK_NAME,
    ParityError,
    SklearnPinError,
)
from mermaid_classifier.pyspacer.inference.head import build_calibrated_head


def export_artifact(
    model: Any,
    output_dir: str | Path,
    reference_features: Any,
    *,
    config: dict[str, Any] | None = None,
    task: str = TASK_NAME,
    tol: float = 1e-6,
    enforce_sklearn_pin: bool = True,
) -> tuple[Path, dict[str, Any], float]:
    """Build, freeze, parity-gate, and persist the portable artifact.

    Returns (model_pt_path, manifest_dict, max_abs_diff). Raises ParityError
    if the frozen graph diverges from ``model.predict_proba`` beyond ``tol``.
    Raises SklearnPinError if the installed scikit-learn differs from
    PARITY_PROVEN_SKLEARN and enforce_sklearn_pin is True.
    """
    sklearn_version = _pkg_version("scikit-learn")
    if enforce_sklearn_pin and sklearn_version != PARITY_PROVEN_SKLEARN:
        raise SklearnPinError(
            f"scikit-learn {sklearn_version} != parity-proven"
            f" {PARITY_PROVEN_SKLEARN}. Refusing to export: a sklearn change can"
            " silently alter CalibratedClassifierCV calibration semantics."
            " Re-prove parity on real features (live parity test), then update"
            " PARITY_PROVEN_SKLEARN and the pyproject pin together to bump."
        )

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
            # Read via importlib.metadata so importing this module (and thus
            # the inference package) doesn't pull in sklearn — the serve path
            # needs only torch/numpy.
            "sklearn": sklearn_version,
            # Read via importlib.metadata (no heavy import); the serving runtime
            # validates this against its installed pyspacer before scoring.
            "pyspacer": _pkg_version("pyspacer"),
        },
    }

    model_pt = output_dir / "model.pt"
    torch.jit.save(frozen, str(model_pt))
    (output_dir / "model.json").write_text(json.dumps(manifest, indent=2))

    return model_pt, manifest, max_diff
