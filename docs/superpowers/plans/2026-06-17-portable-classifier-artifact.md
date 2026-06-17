# Portable Classifier Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pickled `CalibratedClassifierCV` classifier head with a portable TorchScript graph (`model.pt`) + generated manifest (`model.json`), produced by a parity-gated `export_artifact()` and consumed by a loud-validating `load_predictor()`, all inside `mermaid-classifier[inference]`.

**Architecture:** A single `nn.Module` (`CalibratedHead`) reproduces `CalibratedClassifierCV(TorchMLPClassifier, cv="prefit", method="sigmoid").predict_proba` — MLP logits → softmax → per-class Platt sigmoid → row-normalize → sklearn's overshoot clip. `export_artifact()` builds the head from a fitted in-memory model, freezes it with TorchScript, runs a **parity gate** (frozen-graph probs vs source model within `1e-6`, or the export raises), and writes a generated manifest. `load_predictor()` loads the graph and probes it against the manifest, raising loudly on schema / class-count / input-dim mismatch. The EfficientNet extractor is untouched — only the head becomes a graph.

**Tech Stack:** Python ≥3.10, PyTorch (TorchScript via `torch.jit.script` + `torch.jit.freeze`), scikit-learn (`CalibratedClassifierCV`, `_SigmoidCalibration`) — both already pulled transitively by `pyspacer` in the `[inference]` extra. Tests use `unittest`.

## Branch & PR target

- **Base branch:** `inference_split`. All work branches from `inference_split` and the resulting PR merges **into `inference_split`** (not `main`).
- Create the working branch before Task 1:
  ```bash
  git fetch origin
  git switch inference_split
  git switch -c feature/portable-classifier-artifact
  ```
- Open the PR with `--base inference_split` when the plan is complete.

## Global Constraints

These apply to every task; copied verbatim from issue #47 and its parent PRD (#45) / ADR 0002.

- The portable-artifact code lives in **`mermaid-classifier[inference]`** — modules added here must import only `torch`, `numpy`, `json`, stdlib, `sklearn` (transitive via pyspacer at export time). **No import of `mermaid_classifier.pyspacer.settings`** or any training-only dep (`pydantic-settings`, `psutil`, `mlflow`). The decoupling guard in `tests/pyspacer/test_inference_decoupling.py` must keep passing.
- Parity tolerance is **`1e-6`** (max absolute difference in `predict_proba` over a representative feature batch).
- The EfficientNet extractor is **unchanged** — only the calibrated classifier head becomes a graph.
- `model.json` is **generated, never hand-edited**. Manifest shape (trim as needed, decision-encoding from the design prototype):
  ```json
  {
    "schema_version": 1,
    "task": "pyspacer_mlp_classifier",
    "classes": ["ba_uuid::gf_uuid", "..."],
    "input_dim": 1280,
    "config": { "patch_size": 224 },
    "trained_with": { "torch": "2.8.0", "sklearn": "1.5.2" }
  }
  ```
- `schema_version` is `1` for this round.
- The calibrated classifier in production is `CalibratedClassifierCV(estimator=TorchMLPClassifier, cv="prefit")` with `method="sigmoid"` and a **single** entry in `calibrated_classifiers_` (built by `MermaidTrainer._calibrate_in_batches`). The head reproduces the **multiclass (K > 2)** path; K == 2 is unsupported and must raise.
- Respect ADR 0002 (portable TorchScript artifact).

## Source-of-truth: the calibration math to reproduce

The frozen graph must reproduce this exact computation (verified against the installed sklearn `_CalibratedClassifier.predict_proba` and `_SigmoidCalibration.predict`):

1. `logits = MLP(features)` — the `_MLPModule` Linear→ReLU→…→Linear stack from `torch_classifier.py` (raw logits, no final activation).
2. `p = softmax(logits, dim=1)` — `TorchMLPClassifier` exposes **no** `decision_function`, so sklearn's `_get_response_values` falls back to `predict_proba` (= softmax). The calibrators were therefore fit on softmax probabilities, and the head feeds softmax probabilities into them.
3. Per class `k`: `c_k = sigmoid(-(a_k * p_k + b_k))` where `a_k, b_k` come from `calibrators[k].a_`, `calibrators[k].b_` (`_SigmoidCalibration.predict` returns `expit(-(a*T + b))`). Calibrators align 1:1 with `estimator.classes_`, which equals `model.classes_` (both sorted) — identity column mapping.
4. Normalize: `denom = c.sum(dim=1, keepdim=True)`; `proba = c / denom` where `denom != 0`, else the uniform row `1/K` (sklearn's edge-case fallback).
5. Clip: values in `(1.0, 1.0 + 1e-5]` → `1.0` (sklearn's overshoot clip).

`model.classes_.tolist()` are the `ba_uuid::gf_uuid` strings → manifest `classes`. `estimator.n_features_in_` (== first Linear's `in_features`) → manifest `input_dim`.

## File Structure

New subpackage `mermaid_classifier/pyspacer/inference/` (cohesive new serve/release concern, kept import-light so it stays inside the `[inference]` extra):

- `mermaid_classifier/pyspacer/inference/__init__.py` — public exports + `SCHEMA_VERSION`, `TASK_NAME` constants and the `ParityError` / `ManifestError` exception types.
- `mermaid_classifier/pyspacer/inference/head.py` — `CalibratedHead(nn.Module)` and `build_calibrated_head(model)`.
- `mermaid_classifier/pyspacer/inference/export.py` — `export_artifact(...)` + the parity gate.
- `mermaid_classifier/pyspacer/inference/loader.py` — `load_predictor(...)`, `Predictor`, manifest/graph validation.
- `tests/pyspacer/_calibrated_model_fixture.py` — shared helper to build a small fitted `CalibratedClassifierCV(TorchMLPClassifier)` the same way `MermaidTrainer` does (no network).
- `tests/pyspacer/test_portable_artifact.py` — head parity, export→load round-trip, parity-gate failure, loud-validation failures, optional live-model parity.

---

### Task 1: Inference subpackage scaffold (constants, exceptions, decoupling guard)

Creates the importable subpackage with shared constants/exceptions and proves it does not drag in training-only deps.

**Files:**
- Create: `mermaid_classifier/pyspacer/inference/__init__.py`
- Test: `tests/pyspacer/test_portable_artifact.py` (created here; grows in later tasks)

**Interfaces:**
- Consumes: nothing.
- Produces: `from mermaid_classifier.pyspacer.inference import SCHEMA_VERSION, TASK_NAME, ParityError, ManifestError`. `SCHEMA_VERSION: int = 1`; `TASK_NAME: str = "pyspacer_mlp_classifier"`; `ParityError(Exception)`; `ManifestError(Exception)`.

- [ ] **Step 1: Write the failing test**

Create `tests/pyspacer/test_portable_artifact.py`:

```python
"""Tests for the portable classifier artifact (model.pt + model.json)."""
import subprocess
import sys
import unittest


class ScaffoldTest(unittest.TestCase):
    def test_constants_and_exceptions_exported(self):
        from mermaid_classifier.pyspacer.inference import (
            SCHEMA_VERSION, TASK_NAME, ParityError, ManifestError,
        )
        self.assertEqual(SCHEMA_VERSION, 1)
        self.assertEqual(TASK_NAME, "pyspacer_mlp_classifier")
        self.assertTrue(issubclass(ParityError, Exception))
        self.assertTrue(issubclass(ManifestError, Exception))

    def test_importing_inference_does_not_import_settings(self):
        # The [inference] extra must not pull training-only settings deps.
        child = (
            "import sys\n"
            "import mermaid_classifier.pyspacer.inference  # noqa: F401\n"
            "mod = 'mermaid_classifier.pyspacer.settings'\n"
            "raise SystemExit(1 if mod in sys.modules else 0)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tests && python -m unittest pyspacer.test_portable_artifact.ScaffoldTest -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mermaid_classifier.pyspacer.inference'`

- [ ] **Step 3: Write minimal implementation**

Create `mermaid_classifier/pyspacer/inference/__init__.py`:

```python
"""Portable classifier artifact: TorchScript head export + serve-time loader.

Lives in ``mermaid-classifier[inference]``. Modules in this subpackage import
only torch / numpy / json / stdlib (sklearn is used at export time, pulled
transitively by pyspacer). They must NOT import the training-only settings
layer, so the [inference] dependency split holds.
"""

SCHEMA_VERSION = 1
TASK_NAME = "pyspacer_mlp_classifier"


class ParityError(Exception):
    """Raised when the frozen graph diverges from the source model beyond the
    parity tolerance — fails the export/build."""


class ManifestError(Exception):
    """Raised at load time when model.json is incompatible with the graph
    (schema version, class count, or input_dim mismatch)."""


__all__ = ["SCHEMA_VERSION", "TASK_NAME", "ParityError", "ManifestError"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tests && python -m unittest pyspacer.test_portable_artifact.ScaffoldTest -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the decoupling guard to confirm no regression**

Run: `cd tests && python -m unittest pyspacer.test_inference_decoupling -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mermaid_classifier/pyspacer/inference/__init__.py tests/pyspacer/test_portable_artifact.py
git commit -m "feat(inference): scaffold portable-artifact subpackage with schema constants"
```

---

### Task 2: `CalibratedHead` module + `build_calibrated_head` (eager parity)

Builds the torch module that reproduces the calibrated classifier's `predict_proba`, and a builder that extracts MLP weights + sigmoid `(a, b)` from a fitted model. Parity is proven **eagerly** here (before TorchScript) so any math error surfaces with a clean Python stack trace.

**Files:**
- Create: `mermaid_classifier/pyspacer/inference/head.py`
- Create: `tests/pyspacer/_calibrated_model_fixture.py`
- Modify: `tests/pyspacer/test_portable_artifact.py` (add `HeadParityTest`)

**Interfaces:**
- Consumes: a fitted `CalibratedClassifierCV(estimator=TorchMLPClassifier, cv="prefit")` with one `_CalibratedClassifier` in `calibrated_classifiers_`.
- Produces:
  - `class CalibratedHead(nn.Module)` — `forward(features: torch.Tensor) -> torch.Tensor` returns `(N, K)` calibrated probabilities; constructed from `weights: list[torch.Tensor]`, `biases: list[torch.Tensor]`, `a: torch.Tensor`, `b: torch.Tensor`.
  - `build_calibrated_head(model) -> CalibratedHead`.
  - Fixture: `make_calibrated_model(n_features=8, n_classes=5, n_samples=512, seed=0) -> tuple[CalibratedClassifierCV, np.ndarray]` returning the fitted model and a representative feature batch.

- [ ] **Step 1: Write the shared fixture**

Create `tests/pyspacer/_calibrated_model_fixture.py`:

```python
"""Builds a small fitted CalibratedClassifierCV(TorchMLPClassifier) the same
way MermaidTrainer does, with no network or MLflow. Shared across artifact
tests."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, _fit_calibrator

from mermaid_classifier.pyspacer.torch_classifier import TorchMLPClassifier


def make_calibrated_model(
    n_features: int = 8,
    n_classes: int = 5,
    n_samples: int = 512,
    seed: int = 0,
):
    """Return (fitted CalibratedClassifierCV, representative feature batch).

    Mirrors MermaidTrainer._calibrate_in_batches: train a TorchMLPClassifier,
    then wrap it in CalibratedClassifierCV(cv="prefit") with a single
    sigmoid-calibrated inner classifier fit via _fit_calibrator.
    """
    rng = np.random.default_rng(seed)
    classes = np.array([f"ba{i}::gf{i}" for i in range(n_classes)])

    # Separable-ish features so calibration has signal.
    centers = rng.normal(0, 3, size=(n_classes, n_features)).astype(np.float32)
    y_idx = rng.integers(0, n_classes, size=n_samples)
    X = (centers[y_idx] + rng.normal(0, 1, size=(n_samples, n_features))
         ).astype(np.float32)
    y = classes[y_idx]

    clf = TorchMLPClassifier(hidden_layer_sizes=(16,), random_state=0)
    for _ in range(20):
        clf.partial_fit(X, y, classes=classes.tolist())

    # CalibratedClassifierCV has no decision_function on TorchMLPClassifier,
    # so calibration runs on predict_proba (softmax) outputs.
    predictions = clf.predict_proba(X)
    calibrated_inner = _fit_calibrator(
        clf, predictions, y, clf.classes_, method="sigmoid")
    wrapper = CalibratedClassifierCV(clf, cv="prefit")
    wrapper.calibrated_classifiers_ = [calibrated_inner]
    wrapper.classes_ = clf.classes_

    return wrapper, X
```

- [ ] **Step 2: Write the failing test**

Append to `tests/pyspacer/test_portable_artifact.py`:

```python
import numpy as np
import torch

from pyspacer._calibrated_model_fixture import make_calibrated_model


class HeadParityTest(unittest.TestCase):
    def test_head_matches_source_predict_proba(self):
        from mermaid_classifier.pyspacer.inference.head import (
            build_calibrated_head,
        )
        model, X = make_calibrated_model()
        head = build_calibrated_head(model)
        head.eval()
        with torch.no_grad():
            got = head(torch.from_numpy(X.astype(np.float32))).numpy()
        expected = model.predict_proba(X)
        self.assertEqual(got.shape, expected.shape)
        self.assertLess(float(np.max(np.abs(got - expected))), 1e-6)

    def test_rows_sum_to_one(self):
        from mermaid_classifier.pyspacer.inference.head import (
            build_calibrated_head,
        )
        model, X = make_calibrated_model()
        head = build_calibrated_head(model)
        head.eval()
        with torch.no_grad():
            got = head(torch.from_numpy(X.astype(np.float32))).numpy()
        np.testing.assert_allclose(got.sum(axis=1), 1.0, atol=1e-5)
```

> Note: run unittest **from the `tests/` directory** so the namespace-package import path resolves (matches the repo convention). `tests/pyspacer/` has no `__init__.py` — it is a namespace package, and the fixture is imported as `from pyspacer._calibrated_model_fixture import ...`.

- [ ] **Step 3: Run test to verify it fails**

Run: `cd tests && python -m unittest pyspacer.test_portable_artifact.HeadParityTest -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mermaid_classifier.pyspacer.inference.head'`

- [ ] **Step 4: Write the implementation**

Create `mermaid_classifier/pyspacer/inference/head.py`:

```python
"""CalibratedHead: a TorchScript-friendly module that reproduces
CalibratedClassifierCV(TorchMLPClassifier, cv='prefit', method='sigmoid').predict_proba.

Pipeline (multiclass, K > 2):
  logits = MLP(features)                    # Linear -> ReLU -> ... -> Linear
  p      = softmax(logits)                  # TorchMLPClassifier.predict_proba
  c_k    = sigmoid(-(a_k * p_k + b_k))      # per-class Platt sigmoid
  proba  = c / c.sum(dim=1)                 # row-normalize; uniform if sum == 0
  proba  = where(1 < proba <= 1+1e-5, 1.0)  # sklearn overshoot clip
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibratedHead(nn.Module):
    def __init__(
        self,
        weights: list[torch.Tensor],
        biases: list[torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        super().__init__()
        self.linears = nn.ModuleList()
        for w, bias in zip(weights, biases):
            layer = nn.Linear(int(w.shape[1]), int(w.shape[0]))
            with torch.no_grad():
                layer.weight.copy_(w)
                layer.bias.copy_(bias)
            self.linears.append(layer)
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.n_classes = int(a.shape[0])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features
        n = len(self.linears)
        i = 0
        for linear in self.linears:
            x = linear(x)
            if i < n - 1:
                x = F.relu(x)
            i += 1
        p = F.softmax(x, dim=1)
        c = torch.sigmoid(-(self.a * p + self.b))
        denom = c.sum(dim=1, keepdim=True)
        # Avoid NaN poisoning: both branches of torch.where are evaluated, so
        # use a safe denominator and select the uniform row where denom == 0
        # (sklearn's edge-case fallback).
        nonzero = denom != 0
        safe_denom = torch.where(nonzero, denom, torch.ones_like(denom))
        uniform = torch.full_like(c, 1.0 / float(self.n_classes))
        proba = torch.where(nonzero, c / safe_denom, uniform)
        proba = torch.where(
            (proba > 1.0) & (proba <= 1.0 + 1e-5),
            torch.ones_like(proba),
            proba,
        )
        return proba


def build_calibrated_head(model) -> CalibratedHead:
    """Construct a CalibratedHead from a fitted CalibratedClassifierCV that
    wraps a TorchMLPClassifier with cv='prefit' and method='sigmoid'."""
    calibrated = model.calibrated_classifiers_
    if len(calibrated) != 1:
        raise ValueError(
            f"Expected exactly one calibrated classifier (cv='prefit'), got"
            f" {len(calibrated)}."
        )
    inner = calibrated[0]
    estimator = inner.estimator
    calibrators = inner.calibrators

    if not np.array_equal(estimator.classes_, model.classes_):
        raise ValueError(
            "estimator.classes_ does not match model.classes_; calibrator"
            " column alignment is only valid when they are identical."
        )
    n_classes = len(model.classes_)
    if n_classes <= 2:
        raise ValueError(
            f"CalibratedHead only supports the multiclass (K > 2) path; got"
            f" K={n_classes}. sklearn stores a single calibrator for K == 2."
        )
    if len(calibrators) != n_classes:
        raise ValueError(
            f"Expected {n_classes} per-class calibrators, got"
            f" {len(calibrators)}."
        )

    module = estimator._module
    weights = [lin.weight.detach().clone().float() for lin in module.linears]
    biases = [lin.bias.detach().clone().float() for lin in module.linears]
    a = torch.tensor([float(c.a_) for c in calibrators], dtype=torch.float32)
    b = torch.tensor([float(c.b_) for c in calibrators], dtype=torch.float32)
    return CalibratedHead(weights, biases, a, b)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd tests && python -m unittest pyspacer.test_portable_artifact.HeadParityTest -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add mermaid_classifier/pyspacer/inference/head.py tests/pyspacer/_calibrated_model_fixture.py tests/pyspacer/test_portable_artifact.py
git commit -m "feat(inference): CalibratedHead module reproducing calibrated predict_proba"
```

---

### Task 3: `export_artifact()` — TorchScript freeze + parity gate + manifest

Freezes `CalibratedHead` to `model.pt` with TorchScript, runs the parity gate against the source model, and writes the generated `model.json`. The parity gate failing raises `ParityError` (fails the build).

**Files:**
- Create: `mermaid_classifier/pyspacer/inference/export.py`
- Modify: `mermaid_classifier/pyspacer/inference/__init__.py` (export `export_artifact`)
- Modify: `tests/pyspacer/test_portable_artifact.py` (add `ExportTest`)

**Interfaces:**
- Consumes: `build_calibrated_head`, `SCHEMA_VERSION`, `TASK_NAME`, `ParityError`; a fitted model + representative feature batch from the fixture.
- Produces:
  ```python
  def export_artifact(
      model,                       # fitted CalibratedClassifierCV(TorchMLPClassifier)
      output_dir: str | Path,
      reference_features: np.ndarray,   # (N, input_dim) representative batch
      *,
      config: dict | None = None,       # defaults to {"patch_size": 224}
      task: str = TASK_NAME,
      tol: float = 1e-6,
  ) -> tuple[Path, dict, float]:        # (model_pt_path, manifest_dict, max_abs_diff)
  ```
  Writes `<output_dir>/model.pt` (frozen TorchScript) and `<output_dir>/model.json`.

- [ ] **Step 1: Write the failing test**

Append to `tests/pyspacer/test_portable_artifact.py`:

```python
import json
import tempfile
from pathlib import Path


class ExportTest(unittest.TestCase):
    def test_export_writes_pt_and_manifest_and_passes_parity(self):
        from mermaid_classifier.pyspacer.inference import export_artifact
        model, X = make_calibrated_model()
        with tempfile.TemporaryDirectory() as d:
            model_pt, manifest, max_diff = export_artifact(model, d, X)
            self.assertTrue(Path(model_pt).is_file())
            self.assertTrue((Path(d) / "model.json").is_file())
            self.assertLess(max_diff, 1e-6)

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["task"], "pyspacer_mlp_classifier")
            self.assertEqual(manifest["classes"], model.classes_.tolist())
            self.assertEqual(manifest["input_dim"], X.shape[1])
            self.assertEqual(manifest["config"], {"patch_size": 224})
            self.assertIn("torch", manifest["trained_with"])
            self.assertIn("sklearn", manifest["trained_with"])

            on_disk = json.loads((Path(d) / "model.json").read_text())
            self.assertEqual(on_disk, manifest)

    def test_frozen_graph_reloads_and_matches_source(self):
        import torch
        from mermaid_classifier.pyspacer.inference import export_artifact
        model, X = make_calibrated_model()
        with tempfile.TemporaryDirectory() as d:
            model_pt, _, _ = export_artifact(model, d, X)
            graph = torch.jit.load(str(model_pt))
            graph.eval()
            with torch.no_grad():
                got = graph(torch.from_numpy(X.astype(np.float32))).numpy()
        self.assertLess(float(np.max(np.abs(got - model.predict_proba(X)))), 1e-6)

    def test_parity_gate_raises_when_graph_diverges(self):
        from mermaid_classifier.pyspacer.inference import (
            export_artifact, ParityError,
        )
        model, X = make_calibrated_model()
        # Corrupt a calibrator so the source model no longer matches a head
        # rebuilt from the (now-tampered) parameters via an impossible tol.
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ParityError):
                export_artifact(model, d, X, tol=-1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tests && python -m unittest pyspacer.test_portable_artifact.ExportTest -v`
Expected: FAIL with `ImportError: cannot import name 'export_artifact'`

- [ ] **Step 3: Write the implementation**

Create `mermaid_classifier/pyspacer/inference/export.py`:

```python
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
```

Add to `mermaid_classifier/pyspacer/inference/__init__.py`'s end (after the exception classes), and extend `__all__`:

```python
from mermaid_classifier.pyspacer.inference.export import export_artifact  # noqa: E402

__all__ = [
    "SCHEMA_VERSION", "TASK_NAME", "ParityError", "ManifestError",
    "export_artifact",
]
```

> Note: the `export_artifact` import sits at the bottom of `__init__.py` (after the constants/exceptions it depends on are defined) to avoid a circular import — `export.py` imports those names from the package.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tests && python -m unittest pyspacer.test_portable_artifact.ExportTest -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full artifact + decoupling suites**

Run: `cd tests && python -m unittest pyspacer.test_portable_artifact pyspacer.test_inference_decoupling -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add mermaid_classifier/pyspacer/inference/export.py mermaid_classifier/pyspacer/inference/__init__.py tests/pyspacer/test_portable_artifact.py
git commit -m "feat(inference): export_artifact with TorchScript freeze and parity gate"
```

---

### Task 4: `load_predictor()` + loud validation + `Predictor`

Loads the frozen graph, validates it against the manifest with loud failures (schema version, class count, input_dim), and returns a `Predictor` that classifies feature batches.

**Files:**
- Create: `mermaid_classifier/pyspacer/inference/loader.py`
- Modify: `mermaid_classifier/pyspacer/inference/__init__.py` (export `load_predictor`, `Predictor`)
- Modify: `tests/pyspacer/test_portable_artifact.py` (add `LoadValidationTest`)

**Interfaces:**
- Consumes: `model.pt` + `model.json` produced by `export_artifact`; `SCHEMA_VERSION`, `ManifestError`.
- Produces:
  ```python
  class Predictor:
      classes: list[str]
      input_dim: int
      def predict_proba(self, features: np.ndarray) -> np.ndarray: ...   # (N, K)

  def load_predictor(model_pt_path, model_json_path) -> Predictor: ...
  ```
  Raises `ManifestError` on schema-version mismatch, class-count mismatch (graph output width ≠ `len(classes)`), or input_dim mismatch (graph rejects a `(1, input_dim)` probe).

- [ ] **Step 1: Write the failing test**

Append to `tests/pyspacer/test_portable_artifact.py`:

```python
class LoadValidationTest(unittest.TestCase):
    def _export(self, d):
        from mermaid_classifier.pyspacer.inference import export_artifact
        model, X = make_calibrated_model()
        model_pt, manifest, _ = export_artifact(model, d, X)
        return model_pt, Path(d) / "model.json", model, X

    def test_load_predictor_round_trip_matches_source(self):
        from mermaid_classifier.pyspacer.inference import load_predictor
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, model, X = self._export(d)
            predictor = load_predictor(model_pt, model_json)
            self.assertEqual(predictor.classes, model.classes_.tolist())
            self.assertEqual(predictor.input_dim, X.shape[1])
            got = predictor.predict_proba(X)
            self.assertLess(
                float(np.max(np.abs(got - model.predict_proba(X)))), 1e-6)

    def test_schema_version_mismatch_raises(self):
        from mermaid_classifier.pyspacer.inference import (
            load_predictor, ManifestError,
        )
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, _, _ = self._export(d)
            manifest = json.loads(Path(model_json).read_text())
            manifest["schema_version"] = 999
            Path(model_json).write_text(json.dumps(manifest))
            with self.assertRaises(ManifestError):
                load_predictor(model_pt, model_json)

    def test_class_count_mismatch_raises(self):
        from mermaid_classifier.pyspacer.inference import (
            load_predictor, ManifestError,
        )
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, _, _ = self._export(d)
            manifest = json.loads(Path(model_json).read_text())
            manifest["classes"] = manifest["classes"][:-1]  # drop one class
            Path(model_json).write_text(json.dumps(manifest))
            with self.assertRaises(ManifestError):
                load_predictor(model_pt, model_json)

    def test_input_dim_mismatch_raises(self):
        from mermaid_classifier.pyspacer.inference import (
            load_predictor, ManifestError,
        )
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, _, _ = self._export(d)
            manifest = json.loads(Path(model_json).read_text())
            manifest["input_dim"] = manifest["input_dim"] + 7  # wrong dim
            Path(model_json).write_text(json.dumps(manifest))
            with self.assertRaises(ManifestError):
                load_predictor(model_pt, model_json)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tests && python -m unittest pyspacer.test_portable_artifact.LoadValidationTest -v`
Expected: FAIL with `ImportError: cannot import name 'load_predictor'`

- [ ] **Step 3: Write the implementation**

Create `mermaid_classifier/pyspacer/inference/loader.py`:

```python
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
```

Add to `mermaid_classifier/pyspacer/inference/__init__.py` (after the `export_artifact` import) and extend `__all__`:

```python
from mermaid_classifier.pyspacer.inference.loader import (  # noqa: E402
    Predictor, load_predictor,
)

__all__ = [
    "SCHEMA_VERSION", "TASK_NAME", "ParityError", "ManifestError",
    "export_artifact", "Predictor", "load_predictor",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tests && python -m unittest pyspacer.test_portable_artifact.LoadValidationTest -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add mermaid_classifier/pyspacer/inference/loader.py mermaid_classifier/pyspacer/inference/__init__.py tests/pyspacer/test_portable_artifact.py
git commit -m "feat(inference): load_predictor with loud manifest validation"
```

---

### Task 5: Live-model parity test (optional, env-gated) + suite green

Adds a test that proves parity on the **current live model version** when one is reachable, and skips cleanly when it is not (so CI without S3/MLflow credentials stays green). Closes the final acceptance criterion.

**Files:**
- Modify: `tests/pyspacer/test_portable_artifact.py` (add `LiveModelParityTest`)

**Interfaces:**
- Consumes: a pickled `CalibratedClassifierCV` reachable at the path/URI in env var `PORTABLE_ARTIFACT_LIVE_MODEL` (local path or S3 URI), loaded via spacer's storage (already an `[inference]` dep). Optional feature batch via `PORTABLE_ARTIFACT_LIVE_FEATURES` (`.npy`); falls back to a seeded random batch of the model's `input_dim`.
- Produces: nothing consumed downstream.

- [ ] **Step 1: Write the test**

Append to `tests/pyspacer/test_portable_artifact.py`:

```python
import os


class LiveModelParityTest(unittest.TestCase):
    def _load_live_model(self):
        from spacer.storage import storage_factory
        from spacer.data_classes import DataLocation
        from urllib.parse import urlparse
        import pickle

        loc = os.environ["PORTABLE_ARTIFACT_LIVE_MODEL"]
        uri = urlparse(loc)
        if uri.scheme == "s3":
            data_loc = DataLocation(
                "s3", bucket_name=uri.netloc, key=uri.path.strip("/"))
        else:
            data_loc = DataLocation("filesystem", key=loc)
        storage = storage_factory(data_loc.storage_type, data_loc.bucket_name)
        with storage.load(data_loc.key) as stream:
            return pickle.load(stream)

    @unittest.skipUnless(
        os.environ.get("PORTABLE_ARTIFACT_LIVE_MODEL"),
        "set PORTABLE_ARTIFACT_LIVE_MODEL to run live-model parity",
    )
    def test_live_model_export_round_trip_within_tolerance(self):
        from mermaid_classifier.pyspacer.inference import (
            export_artifact, load_predictor,
        )
        model = self._load_live_model()
        input_dim = int(model.calibrated_classifiers_[0].estimator.n_features_in_)

        feats_path = os.environ.get("PORTABLE_ARTIFACT_LIVE_FEATURES")
        if feats_path:
            X = np.load(feats_path).astype(np.float32)
        else:
            rng = np.random.default_rng(0)
            X = rng.normal(0, 1, size=(256, input_dim)).astype(np.float32)

        with tempfile.TemporaryDirectory() as d:
            model_pt, manifest, max_diff = export_artifact(model, d, X)
            self.assertLess(max_diff, 1e-6)
            self.assertEqual(manifest["input_dim"], input_dim)
            self.assertEqual(manifest["classes"], model.classes_.tolist())
            predictor = load_predictor(model_pt, Path(d) / "model.json")
            got = predictor.predict_proba(X)
        self.assertLess(
            float(np.max(np.abs(got - model.predict_proba(X)))), 1e-6)
```

- [ ] **Step 2: Run the test (skips without env var)**

Run: `cd tests && python -m unittest pyspacer.test_portable_artifact.LiveModelParityTest -v`
Expected: `OK (skipped=1)` — the test is `skipUnless` the env var is set.

- [ ] **Step 3: Run against the live model (when credentials/model are available)**

Run:
```bash
PORTABLE_ARTIFACT_LIVE_MODEL=s3://<bucket>/<path>/model.pkl \
  cd tests && python -m unittest pyspacer.test_portable_artifact.LiveModelParityTest -v
```
Expected: PASS — frozen graph reproduces the live model's `predict_proba` within `1e-6`.

> If credentials/model are unavailable in your environment, record in the PR description that this manual gate is pending and have a reviewer with access run it before merge.

- [ ] **Step 4: Run the entire artifact + decoupling suite**

Run: `cd tests && python -m unittest pyspacer.test_portable_artifact pyspacer.test_inference_decoupling -v`
Expected: PASS (live test skipped unless env var set)

- [ ] **Step 5: Commit**

```bash
git add tests/pyspacer/test_portable_artifact.py
git commit -m "test(inference): env-gated live-model parity for portable artifact"
```

- [ ] **Step 6: Open the PR into `inference_split`**

```bash
git push -u origin feature/portable-classifier-artifact
gh pr create --base inference_split \
  --title "Portable classifier artifact: TorchScript head export + parity gate" \
  --body "Implements #47. Adds mermaid_classifier/pyspacer/inference/ (CalibratedHead, export_artifact, load_predictor) with a 1e-6 parity gate and loud load-time validation. Live-model parity test is env-gated (PORTABLE_ARTIFACT_LIVE_MODEL)."
```

---

## Self-Review

**1. Spec coverage** (issue #47 acceptance criteria):
- *Calibrated-head torch module reproduces predict_proba within 1e-6* → Task 2 (`HeadParityTest`).
- *`export_artifact()` emits `model.pt` + generated `model.json`; manifest never hand-edited* → Task 3 (`ExportTest`; manifest built in code, written from the dict).
- *`load_predictor()` errors loudly on class-count / input_dim / schema-version mismatch* → Task 4 (`LoadValidationTest`, three failure tests + round-trip).
- *Parity gate fails the build beyond 1e-6* → Task 3 (`test_parity_gate_raises_when_graph_diverges`, raises `ParityError`).
- *Tests prove parity on the current live model version + assert loud-validation failures; prior art `test_inference_decoupling.py`* → Task 5 (`LiveModelParityTest`, env-gated) + Task 4 failures; decoupling guard re-run each task.
- *All in `mermaid-classifier[inference]`* → new subpackage imports only torch/numpy/json/stdlib/sklearn(+pyspacer); Task 1 guard test asserts settings is not imported.
- *EfficientNet extractor unchanged* → no extractor files touched; only the head is graphed.

**2. Placeholder scan:** No `TBD`/`handle edge cases`/"write tests for the above" — every step carries concrete code or an exact command + expected output.

**3. Type consistency:** `CalibratedHead(weights, biases, a, b)` and `build_calibrated_head(model)` (Task 2) are consumed verbatim by `export_artifact` (Task 3). `export_artifact(...) -> (Path, dict, float)` matches its callers in Tasks 3 and 5. `load_predictor(model_pt, model_json) -> Predictor` and `Predictor.predict_proba`/`.classes`/`.input_dim` (Task 4) match usage in Tasks 4 and 5. `SCHEMA_VERSION`/`TASK_NAME`/`ParityError`/`ManifestError` (Task 1) are imported consistently throughout. Manifest keys (`schema_version`, `task`, `classes`, `input_dim`, `config`, `trained_with`) are identical in `export.py` and the load/validation tests.

## Notes for the implementer

- **Run unittest from the `tests/` directory** (`cd tests && python -m unittest pyspacer.test_portable_artifact -v`), matching the existing repo convention (e.g. `pyspacer.test_inference_decoupling`). `tests/` and `tests/pyspacer/` have **no** `__init__.py` — they are namespace packages — so do **not** add `__init__.py` files, and import the fixture as `from pyspacer._calibrated_model_fixture import make_calibrated_model` (a `pyspacer.`-rooted import, not `tests.pyspacer.`).
- **Install:** `pip install -e .[inference]` is sufficient to build, export, load, and test (torch + sklearn + pyspacer come via the extra). Per the root CLAUDE.md, prefer `uv run` if the environment is uv-managed.
- **TorchScript scripting** of `CalibratedHead` relies on the `ModuleList` loop with a manual index counter (scriptable) and buffer-backed `a`/`b`. If `torch.jit.script` ever rejects the module on a torch upgrade, the parity gate in Task 3 is the safety net — it will fail loudly rather than ship a divergent graph.
- **Why eager parity in Task 2 then frozen parity in Task 3:** the eager test isolates math errors (clean Python traceback); the frozen test catches scripting/freezing divergence. Both assert `1e-6`.
