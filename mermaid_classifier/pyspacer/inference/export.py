"""export_artifact: freeze the calibrated head to TorchScript, parity-gate it
against the source model, and write the generated manifest."""

from __future__ import annotations

import hashlib
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
    ExtractorMismatchError,
    ParityError,
    SklearnPinError,
)
from mermaid_classifier.pyspacer.inference.extractor_spec import MANIFEST_KEY, ExtractorSpec
from mermaid_classifier.pyspacer.inference.head import build_calibrated_head


def export_artifact(
    model: Any,
    output_dir: str | Path,
    reference_features: Any,
    *,
    extractor: ExtractorSpec,
    task: str = TASK_NAME,
    tol: float = 1e-6,
    enforce_sklearn_pin: bool = True,
) -> tuple[Path, dict[str, Any], float]:
    """Build, freeze, parity-gate, and persist the portable artifact.

    ``extractor`` identifies the image -> feature-vector transform the head was
    fitted through, and is the caller's only say over the manifest's
    conditioning record: ``config`` is derived from it rather than passed, so
    a patch size that contradicts the extractor cannot be written.

    Returns (model_pt_path, manifest_dict, max_abs_diff). Raises ParityError
    if the frozen graph diverges from ``model.predict_proba`` beyond ``tol``.
    Raises SklearnPinError if the installed scikit-learn differs from
    PARITY_PROVEN_SKLEARN and enforce_sklearn_pin is True. Raises
    ExtractorMismatchError if the extractor's output width is not the width
    the head takes.
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
    input_dim = int(estimator.n_features_in_)
    # The head takes whatever the extractor emits. Checking it here is what
    # makes the recorded extractor the one the graph can actually be fed by;
    # load_predictor's probe only checks the graph against the manifest.
    if extractor.feature_dim != input_dim:
        raise ExtractorMismatchError(
            f"extractor emits {extractor.feature_dim}-d features but the head"
            f" takes {input_dim}-d. Refusing to record {extractor.describe()}"
            " as this model's feature source."
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "classes": model.classes_.tolist(),
        "input_dim": input_dim,
        "config": {"patch_size": extractor.crop_size},
        MANIFEST_KEY: extractor.to_dict(),
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
    # Hashed after the file lands, so the digest is exactly what
    # load_predictor will recompute from the same bytes.
    manifest["model_pt_sha256"] = hashlib.sha256(model_pt.read_bytes()).hexdigest()
    (output_dir / "model.json").write_text(json.dumps(manifest, indent=2))

    return model_pt, manifest, max_diff
