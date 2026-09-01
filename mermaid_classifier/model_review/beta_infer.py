"""Score the Beta model — the classifier currently deployed to the MERMAID API.

``models/beta_model/classifier.pkl`` is a ``CalibratedClassifierCV`` pickled with
scikit-learn 1.1.3 and unpicklable under the 1.5.2 this project pins (calibration
internals differ), and it has no portable ``model.pt``/``model.json`` form. It is
therefore scored out of process: ``beta_score.py`` runs under a uv-managed ephemeral
environment holding the pinned scikit-learn, reading a feature matrix and writing back
one BA::GF label per point.

Beta shares V1's EfficientNet extractor, so the same pre-extracted feature vectors
score both and the matrix is loaded once (see ``features.stack_features``).
"""

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mermaid_classifier.model_review.features import PointKey

# The version the Beta pickle was created with. Calibration semantics shift between
# scikit-learn releases, so a child that resolved anything else would silently return
# different probabilities; predict_points verifies what actually ran.
BETA_SKLEARN = "1.1.3"
BETA_PYTHON = "3.11"
SCORE_SCRIPT = Path(__file__).resolve().parent / "beta_score.py"

# uv's ephemeral environment must be the only one on the child's import path; an
# inherited VIRTUAL_ENV or PYTHONPATH puts this project's scikit-learn 1.5.2 in front.
_STRIPPED_ENV = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")

Runner = Callable[[list[str], dict[str, str]], tuple[int, str]]


class BetaScoringError(RuntimeError):
    """The Beta scoring subprocess could not run, or produced unusable output."""


def child_env(environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` minus the variables that would leak this project's interpreter."""
    return {k: v for k, v in environ.items() if k not in _STRIPPED_ENV}


def score_command(uv: str, features_npz: Path, classifier_pkl: Path, out_npz: Path) -> list[str]:
    """The uv invocation that scores ``features_npz`` under the pinned scikit-learn."""
    return [
        uv,
        "run",
        "--python",
        BETA_PYTHON,
        "--with",
        f"scikit-learn=={BETA_SKLEARN}",
        "--with",
        "numpy<2",
        "--no-project",
        "python",
        # -P keeps the script's own directory off sys.path, so no sibling module in
        # this package can shadow a stdlib or sklearn import in the child.
        "-P",
        str(SCORE_SCRIPT),
        "--features",
        str(features_npz),
        "--classifier",
        str(classifier_pkl),
        "--out",
        str(out_npz),
    ]


def _run(argv: list[str], env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(argv, env=env, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stderr


def predict_points(
    keys: Sequence[PointKey],
    features: NDArray[np.float32],
    classifier_pkl: str,
    run: Runner = _run,
) -> dict[PointKey, str]:
    """Top-1 BA::GF per point, for rows aligned with ``keys``."""
    pkl = Path(classifier_pkl).resolve()
    if not pkl.is_file():
        raise BetaScoringError(f"Beta classifier pickle not found: {pkl}")
    uv = shutil.which("uv")
    if uv is None:
        raise BetaScoringError(
            "`uv` is not on PATH; it runs the isolated scikit-learn "
            f"{BETA_SKLEARN} environment the Beta pickle needs."
        )

    with tempfile.TemporaryDirectory(prefix="beta-score-") as tmp:
        features_npz = Path(tmp) / "features.npz"
        out_npz = Path(tmp) / "beta_predictions.npz"
        np.savez_compressed(features_npz, X=features.astype(np.float32))

        argv = score_command(uv, features_npz, pkl, out_npz)
        status, stderr = run(argv, child_env(os.environ))
        printable = " ".join(argv)
        if status != 0:
            raise BetaScoringError(
                f"Beta scoring failed (exit {status}).\n  {printable}\n{stderr[-2000:]}"
            )
        if not out_npz.is_file():
            raise BetaScoringError(
                f"Beta scoring wrote no predictions to {out_npz.name}.\n  {printable}"
            )

        loaded = np.load(out_npz, allow_pickle=False)
        preds = [str(p) for p in loaded["pred_bagf"]]
        version = str(loaded["sklearn_version"])

    if version != BETA_SKLEARN:
        raise BetaScoringError(
            f"Beta was scored under scikit-learn {version}, not the pinned "
            f"{BETA_SKLEARN}; its calibration would not match the deployed model."
        )
    if len(preds) != len(keys):
        raise BetaScoringError(f"Beta returned {len(preds)} predictions for {len(keys)} points.")
    return dict(zip(keys, preds, strict=True))
