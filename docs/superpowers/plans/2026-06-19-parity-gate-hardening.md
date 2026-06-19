# Parity-Gate Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two PR #56 review gaps on the portable-artifact parity gate: (1) prove parity on **real EfficientNet features**, never seeded-random vectors, and (2) **pin scikit-learn** so a version bump cannot silently change calibration semantics without re-proving parity and failing CI.

**Architecture:** A single `PARITY_PROVEN_SKLEARN` constant in the inference package is the source of truth for the sklearn version parity was proven against. `export_artifact()` refuses to ship if the installed sklearn differs from it (`SklearnPinError`); a standalone guard test asserts the same so it **fails CI** on drift; scikit-learn is pinned to that exact version in `pyproject.toml` + `uv.lock`. The env-gated live parity test stops falling back to `rng.normal` random features — it now **requires** real features, and a new operator script stacks real pyspacer feature vectors (`.featurevector` → `ImageFeatures.point_features[*].data`) into the `(N, 1280)` float32 `.npy` the gate consumes.

**Tech Stack:** Python ≥3.10, scikit-learn 1.5.2 (pinned), PyTorch/TorchScript, `unittest`, `uv` (with `--frozen` lockfile in CI), pyspacer `ImageFeatures`. CI is GitHub Actions (`.github/workflows/tests.yml`, inherited from `main`).

## Branch & PR target

- Working branch: `feature/portable-classifier-artifact` (PR #56), already checked out in worktree `/Users/gregn/Documents/wcs/mermaid-classifier/.claude/worktrees/portable-classifier-artifact`.
- PR base stays as currently configured (`inference_split` per the original plan). **Task 1 merges `origin/main` into the branch** to pull in the CI workflow and `uv.lock` that now live on `main`, per the reviewer note.

## Global Constraints

These apply to every task; copied verbatim from PR #56 review and the original portable-artifact plan.

- **Parity-proven sklearn version is `1.5.2`** — matches the transitive pin from `pyspacer==0.14.0` (`pyspacer/pyproject.toml: scikit-learn==1.5.2`). This is the value of `PARITY_PROVEN_SKLEARN` and the explicit `pyproject.toml` pin.
- **Parity tolerance is `1e-6`** (max absolute difference in `predict_proba` over the feature batch). Unchanged.
- The inference subpackage stays **import-light**: modules under `mermaid_classifier/pyspacer/inference/` import only `torch` / `numpy` / `json` / stdlib (sklearn used at export time only, via `importlib.metadata`, never imported). The decoupling guard (`tests/pyspacer/test_inference_decoupling.py` if present) must keep passing. **The new constant `PARITY_PROVEN_SKLEARN` is a plain string — it adds no import.** The reference-features extractor script lives **outside** the package (it pulls heavy pyspacer extractor deps) and must not be imported by the package.
- `model.json` is **generated, never hand-edited**. `trained_with.sklearn` must equal the installed (== proven) version.
- CI runs `uv sync --frozen ...` then `cd tests && uv run --no-sync python -m unittest -v`. `--frozen` fails CI if `uv.lock` is out of sync with `pyproject.toml`, so **any `pyproject.toml` dependency change requires re-running `uv lock` and committing `uv.lock`** in the same task.

---

### Task 1: Integrate `main` (CI workflow + lockfile) and reconcile extras

`main` now carries `.github/workflows/tests.yml` (uv-based, runs the `unittest` suite on every PR) and a `uv.lock`. This branch needs both so the guard test added later literally fails CI. `main` exposes a `pyspacer` optional-extra; this branch renamed the dependency split to `inference` / `training`. Reconcile so CI installs an extra that exists on the branch and is sufficient to run the full suite.

**Files:**
- Merge: `origin/main` into `feature/portable-classifier-artifact`
- Modify (conflict resolution): `pyproject.toml` — keep the branch's `inference` / `training` extras
- Modify: `.github/workflows/tests.yml:` install step — change `--extra pyspacer` → `--extra training`
- Add (from merge): `uv.lock`

**Interfaces:**
- Produces: a green `unittest` suite on the branch with `uv sync --frozen --extra training`; a `.github/workflows/tests.yml` present on the branch so PR #56 gets CI runs.

- [ ] **Step 1: Merge `origin/main`**

```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier/.claude/worktrees/portable-classifier-artifact
git fetch origin
git merge origin/main
```

Expect conflicts in `pyproject.toml` (extras renamed) and possibly `uv.lock`.

- [ ] **Step 2: Resolve `pyproject.toml`**

Keep this branch's `inference` and `training` optional-dependency groups (the deliberate split). Discard `main`'s single `pyspacer` extra. Keep the explicit unpinned `scikit-learn` line under `training` for now (Task 2 pins it). After editing, the `[project.optional-dependencies]` section must contain `inference` and `training` (no `pyspacer` key).

- [ ] **Step 3: Point CI at an extra that exists on this branch**

In `.github/workflows/tests.yml`, the install step reads `run: uv sync --frozen --extra pyspacer`. Change it to:

```yaml
        run: uv sync --frozen --extra training
```

`training` includes `inference` (`"mermaid-classifier[inference]"`), so the full suite's deps install.

- [ ] **Step 4: Re-lock and finish the merge**

```bash
uv lock
git add pyproject.toml .github/workflows/tests.yml uv.lock
git commit --no-edit
```

- [ ] **Step 5: Run the full suite locally to confirm the merge is clean**

Run:
```bash
uv sync --frozen --extra training
cd tests && uv run --no-sync python -m unittest -v
```
Expected: all tests PASS (including the existing `test_portable_artifact.py` classes). If `--frozen` complains the lock is stale, re-run `uv lock` and re-commit.

---

### Task 2: Pin scikit-learn and add the parity-proven-version constant

Add the single source of truth (`PARITY_PROVEN_SKLEARN`) and the `SklearnPinError` type to the import-light package, and pin `scikit-learn==1.5.2` explicitly so a `pyspacer` bump (or any resolver change) can't silently move it.

**Files:**
- Modify: `mermaid_classifier/pyspacer/inference/__init__.py`
- Modify: `pyproject.toml` (`inference` and `training` extras)
- Modify: `uv.lock` (via `uv lock`)

**Interfaces:**
- Produces: `PARITY_PROVEN_SKLEARN: str` (`"1.5.2"`) and `SklearnPinError(Exception)`, both importable from `mermaid_classifier.pyspacer.inference`. Consumed by Tasks 3 and 4.

- [ ] **Step 1: Add the constant and exception to `__init__.py`**

In `mermaid_classifier/pyspacer/inference/__init__.py`, below `TASK_NAME = "pyspacer_mlp_classifier"` add:

```python
# The scikit-learn version the TorchScript parity gate was proven against.
# CalibratedClassifierCV / _SigmoidCalibration semantics can change between
# sklearn releases, so a bump must NOT pass silently: export refuses to ship
# (SklearnPinError) and a guard test fails CI until parity is re-proven on real
# features and this constant + the pin are updated together.
PARITY_PROVEN_SKLEARN = "1.5.2"
```

After the existing `ManifestError` class add:

```python
class SklearnPinError(Exception):
    """Raised at export when the installed scikit-learn differs from
    PARITY_PROVEN_SKLEARN — the version the parity gate was proven against."""
```

Add both names to `__all__`:

```python
__all__ = [
    "SCHEMA_VERSION", "TASK_NAME", "PARITY_PROVEN_SKLEARN",
    "ParityError", "ManifestError", "SklearnPinError",
    "export_artifact", "Predictor", "load_predictor",
]
```

- [ ] **Step 2: Pin the dependency in both extras**

In `pyproject.toml`, under `inference`, add an explicit pin alongside the pyspacer pin:

```toml
inference = [
    # Pinned to Release 0.14.0 (git hash b52acdb)
    "pyspacer==0.14.0",
    # Parity gate was proven against this sklearn; see PARITY_PROVEN_SKLEARN.
    # A bump can silently change CalibratedClassifierCV calibration semantics.
    "scikit-learn==1.5.2",
]
```

Under `training`, replace the bare `"scikit-learn",` line with the pinned form:

```toml
    # pyspacer uses scikit-learn, but we also use it directly for metrics.
    # Pinned in lockstep with PARITY_PROVEN_SKLEARN (see inference extra).
    "scikit-learn==1.5.2",
```

- [ ] **Step 3: Re-lock**

Run: `uv lock`
Expected: lock resolves with `scikit-learn==1.5.2` (already the resolved version via pyspacer, so no real version change — the pin just makes drift explicit).

- [ ] **Step 4: Verify the constant imports without pulling sklearn**

Run:
```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier/.claude/worktrees/portable-classifier-artifact
uv run --no-sync python -c "import sys, mermaid_classifier.pyspacer.inference as m; print(m.PARITY_PROVEN_SKLEARN); assert 'sklearn' not in sys.modules, 'inference import pulled sklearn'"
```
Expected: prints `1.5.2`, no assertion error.

- [ ] **Step 5: Commit**

```bash
git add mermaid_classifier/pyspacer/inference/__init__.py pyproject.toml uv.lock
git commit -m "feat(inference): pin sklearn==1.5.2 and add PARITY_PROVEN_SKLEARN constant"
```

---

### Task 3: Enforce the sklearn pin at export

`export_artifact` must refuse to build an artifact when the installed sklearn differs from `PARITY_PROVEN_SKLEARN`, and stamp that same version into the manifest. This is the "pin sklearn at export" half of the review.

**Files:**
- Modify: `mermaid_classifier/pyspacer/inference/export.py`
- Test: `tests/pyspacer/test_portable_artifact.py` (`ExportTest`)

**Interfaces:**
- Consumes: `PARITY_PROVEN_SKLEARN`, `SklearnPinError` from `mermaid_classifier.pyspacer.inference` (Task 2).
- Produces: `export_artifact(..., enforce_sklearn_pin: bool = True)` raises `SklearnPinError` when `importlib.metadata.version("scikit-learn") != PARITY_PROVEN_SKLEARN` and the flag is set; manifest `trained_with.sklearn` equals the installed version.

- [ ] **Step 1: Write the failing test**

Add to `ExportTest` in `tests/pyspacer/test_portable_artifact.py` (add `from unittest import mock` to the imports near the top of the file if not present):

```python
    def test_export_raises_when_sklearn_unpinned(self):
        from mermaid_classifier.pyspacer.inference import (
            export_artifact, SklearnPinError,
        )
        model, X = make_calibrated_model()
        with tempfile.TemporaryDirectory() as d:
            # Patch the proven version to something the runner can't have, so
            # the installed sklearn is guaranteed to differ.
            with mock.patch(
                "mermaid_classifier.pyspacer.inference.export"
                ".PARITY_PROVEN_SKLEARN", "0.0.0-never",
            ):
                with self.assertRaises(SklearnPinError):
                    export_artifact(model, d, X)

    def test_export_manifest_records_proven_sklearn(self):
        from mermaid_classifier.pyspacer.inference import (
            export_artifact, PARITY_PROVEN_SKLEARN,
        )
        model, X = make_calibrated_model()
        with tempfile.TemporaryDirectory() as d:
            _, manifest, _ = export_artifact(model, d, X)
        self.assertEqual(manifest["trained_with"]["sklearn"],
                         PARITY_PROVEN_SKLEARN)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tests && uv run --no-sync python -m unittest -v pyspacer.test_portable_artifact.ExportTest.test_export_raises_when_sklearn_unpinned`
Expected: FAIL — `SklearnPinError` not raised (export currently has no pin check) / `ImportError` for `SklearnPinError` if Task 2 names drifted.

- [ ] **Step 3: Implement the export-time check**

In `mermaid_classifier/pyspacer/inference/export.py`, extend the import block:

```python
from mermaid_classifier.pyspacer.inference import (
    SCHEMA_VERSION, TASK_NAME, PARITY_PROVEN_SKLEARN, ParityError, SklearnPinError,
)
```

Add an `enforce_sklearn_pin: bool = True` keyword parameter to `export_artifact` (after `tol`), and at the **top of the function body** (before building the head), insert:

```python
    sklearn_version = _pkg_version("scikit-learn")
    if enforce_sklearn_pin and sklearn_version != PARITY_PROVEN_SKLEARN:
        raise SklearnPinError(
            f"scikit-learn {sklearn_version} != parity-proven"
            f" {PARITY_PROVEN_SKLEARN}. Refusing to export: a sklearn change can"
            " silently alter CalibratedClassifierCV calibration semantics."
            " Re-prove parity on real features (live parity test), then update"
            " PARITY_PROVEN_SKLEARN and the pyproject pin together to bump."
        )
```

Then reuse `sklearn_version` in the manifest instead of calling `_pkg_version` again — change the `trained_with` block:

```python
        "trained_with": {
            "torch": torch.__version__,
            # Read via importlib.metadata so importing this module (and thus
            # the inference package) doesn't pull in sklearn — the serve path
            # needs only torch/numpy.
            "sklearn": sklearn_version,
        },
```

- [ ] **Step 4: Run the new and existing export tests**

Run: `cd tests && uv run --no-sync python -m unittest -v pyspacer.test_portable_artifact.ExportTest`
Expected: PASS — all `ExportTest` methods, including the two new ones. (The pre-existing tests pass because the runner's sklearn is the pinned `1.5.2 == PARITY_PROVEN_SKLEARN`.)

- [ ] **Step 5: Commit**

```bash
git add mermaid_classifier/pyspacer/inference/export.py tests/pyspacer/test_portable_artifact.py
git commit -m "feat(inference): refuse export when sklearn diverges from PARITY_PROVEN_SKLEARN"
```

---

### Task 4: Standalone CI guard test for sklearn drift

A focused test that fails the suite (and therefore CI, present since Task 1) the moment the installed sklearn drifts from the proven version — even if no export runs in a given CI job. Its failure message documents the sanctioned re-prove workflow.

**Files:**
- Create: `tests/pyspacer/test_sklearn_pin.py`

**Interfaces:**
- Consumes: `PARITY_PROVEN_SKLEARN` from `mermaid_classifier.pyspacer.inference`.

- [ ] **Step 1: Write the test**

Create `tests/pyspacer/test_sklearn_pin.py`:

```python
"""CI guard: the installed scikit-learn must match the version the portable
artifact's parity gate was proven against. A silent sklearn bump can change
CalibratedClassifierCV / _SigmoidCalibration semantics, so this test fails the
build until parity is re-proven and the pin + constant are updated together."""
import unittest
from importlib.metadata import version

from mermaid_classifier.pyspacer.inference import PARITY_PROVEN_SKLEARN


class SklearnPinTest(unittest.TestCase):
    def test_installed_sklearn_matches_parity_proven(self):
        installed = version("scikit-learn")
        self.assertEqual(
            installed,
            PARITY_PROVEN_SKLEARN,
            msg=(
                f"scikit-learn {installed} != parity-proven"
                f" {PARITY_PROVEN_SKLEARN}. To bump: (1) update the"
                " scikit-learn pin in pyproject.toml and run `uv lock`,"
                " (2) set PARITY_PROVEN_SKLEARN to the new version,"
                " (3) re-run the live-feature parity gate"
                " (PORTABLE_ARTIFACT_LIVE_MODEL + PORTABLE_ARTIFACT_LIVE_FEATURES)"
                " and confirm max|Δ| < 1e-6 on REAL EfficientNet features."
            ),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it passes on the pinned version**

Run: `cd tests && uv run --no-sync python -m unittest -v pyspacer.test_sklearn_pin`
Expected: PASS (runner sklearn is `1.5.2`).

- [ ] **Step 3: Sanity-check it would fail on drift**

Run:
```bash
cd tests && uv run --no-sync python -c "
from mermaid_classifier.pyspacer.inference import PARITY_PROVEN_SKLEARN
from importlib.metadata import version
print('installed', version('scikit-learn'), 'proven', PARITY_PROVEN_SKLEARN)
assert version('scikit-learn') == PARITY_PROVEN_SKLEARN
print('guard holds')
"
```
Expected: prints matching versions and `guard holds`. (No code change — this only confirms the guard's premise.)

- [ ] **Step 4: Commit**

```bash
git add tests/pyspacer/test_sklearn_pin.py
git commit -m "test(inference): CI guard fails on scikit-learn drift from parity-proven pin"
```

---

### Task 5: Real-features extractor helper

Give operators a repeatable way to produce the `(N, 1280)` float32 `.npy` the live parity gate consumes, stacked from **real** pyspacer feature vectors (`ImageFeatures.point_features[*].data`) rather than synthetic ones. Lives outside the import-light package because it pulls pyspacer's extractor/storage stack.

**Files:**
- Create: `scripts/extract_reference_features.py`

**Interfaces:**
- Produces: a CLI that reads one or more pyspacer `.featurevector` files (local or `s3://`) and writes a `(total_points, feature_dim)` float32 `.npy`. Output is the value of `PORTABLE_ARTIFACT_LIVE_FEATURES` for Task 6.

- [ ] **Step 1: Write the script**

Create `scripts/extract_reference_features.py`:

```python
"""Stack REAL EfficientNet feature vectors from pyspacer feature files into an
(N, feature_dim) float32 .npy for the portable-artifact live parity gate.

The live gate must run on real features (random vectors sit in flat softmax
regions and under-exercise the per-class calibration tails). pyspacer's
production pipeline already emits ImageFeatures (.featurevector) files; this
tool concatenates their per-point vectors.

Usage (local or s3):
  uv run python scripts/extract_reference_features.py \\
      --out reference_features.npy \\
      s3://bucket/key1.featurevector s3://bucket/key2.featurevector
  uv run python scripts/extract_reference_features.py \\
      --out reference_features.npy /path/to/*.featurevector

Then run the live gate:
  PORTABLE_ARTIFACT_LIVE_MODEL=s3://bucket/model.pkl \\
  PORTABLE_ARTIFACT_LIVE_FEATURES=reference_features.npy \\
  cd tests && uv run --no-sync python -m unittest -v \\
      pyspacer.test_portable_artifact.LiveModelParityTest
"""
from __future__ import annotations

import argparse
from urllib.parse import urlparse

import numpy as np
from spacer.data_classes import DataLocation, ImageFeatures


def _data_location(loc: str) -> DataLocation:
    uri = urlparse(loc)
    if uri.scheme == "s3":
        return DataLocation(
            "s3", bucket_name=uri.netloc, key=uri.path.strip("/"))
    return DataLocation("filesystem", key=loc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument(
        "features", nargs="+",
        help="pyspacer feature files (.featurevector), local paths or s3:// URIs",
    )
    args = ap.parse_args()

    vectors: list[list[float]] = []
    for loc in args.features:
        feats = ImageFeatures.load(_data_location(loc))
        for pf in feats.point_features:
            vectors.append(pf.data)

    X = np.asarray(vectors, dtype=np.float32)
    if X.ndim != 2:
        raise SystemExit(f"expected a 2-D feature matrix; got shape {X.shape}")
    np.save(args.out, X)
    print(f"wrote {X.shape[0]} real feature vectors "
          f"(dim {X.shape[1]}) to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the script with a synthetic feature file (no S3/network)**

Run:
```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier/.claude/worktrees/portable-classifier-artifact
uv run --no-sync python -c "
import tempfile, os, numpy as np
from spacer.data_classes import DataLocation, ImageFeatures, PointFeatures
d = tempfile.mkdtemp()
feat = os.path.join(d, 'a.featurevector'); out = os.path.join(d, 'X.npy')
pfs = [PointFeatures(row=r, col=r, data=[float(r)]*1280) for r in range(3)]
ImageFeatures(point_features=pfs, valid_rowcol=True, feature_dim=1280, npoints=3).store(DataLocation('filesystem', key=feat))
import subprocess, sys
subprocess.run([sys.executable, 'scripts/extract_reference_features.py', '--out', out, feat], check=True)
arr = np.load(out); print('shape', arr.shape, 'dtype', arr.dtype)
assert arr.shape == (3, 1280) and arr.dtype == np.float32
print('ok')
"
```
Expected: prints `wrote 3 real feature vectors (dim 1280) to ...`, then `shape (3, 1280) dtype float32`, then `ok`. (If `ImageFeatures.store` differs, use `.save`; confirm against `spacer/data_classes.py`.)

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_reference_features.py
git commit -m "feat(inference): script to stack real pyspacer features for the live parity gate"
```

---

### Task 6: Require real features in the live parity test

Delete the `rng.normal` fallback so the live parity test can never pass on synthetic vectors. When the live **model** is provided but real **features** are not, fail loudly with instructions pointing at the Task 5 script.

**Files:**
- Modify: `tests/pyspacer/test_portable_artifact.py` (`LiveModelParityTest`)

**Interfaces:**
- Consumes: env vars `PORTABLE_ARTIFACT_LIVE_MODEL` (gates the test) and `PORTABLE_ARTIFACT_LIVE_FEATURES` (path to the Task 5 `.npy`).

- [ ] **Step 1: Replace the random-fallback block**

In `LiveModelParityTest.test_live_model_export_round_trip_within_tolerance`, replace:

```python
        feats_path = os.environ.get("PORTABLE_ARTIFACT_LIVE_FEATURES")
        if feats_path:
            X = np.load(feats_path).astype(np.float32)
        else:
            rng = np.random.default_rng(0)
            X = rng.normal(0, 1, size=(256, input_dim)).astype(np.float32)
```

with:

```python
        # Parity must be proven on REAL EfficientNet features. Random vectors
        # sit in flat softmax regions and under-exercise the per-class
        # calibration tails, where the frozen graph diverges most — so we
        # refuse to "prove" parity on them. Build the .npy with
        # scripts/extract_reference_features.py.
        feats_path = os.environ.get("PORTABLE_ARTIFACT_LIVE_FEATURES")
        if not feats_path:
            self.fail(
                "PORTABLE_ARTIFACT_LIVE_MODEL is set but"
                " PORTABLE_ARTIFACT_LIVE_FEATURES is not. The live parity gate"
                " requires REAL EfficientNet features (no random fallback)."
                " Generate them with"
                " `python scripts/extract_reference_features.py --out X.npy"
                " <.featurevector files>` and set PORTABLE_ARTIFACT_LIVE_FEATURES=X.npy."
            )
        X = np.load(feats_path).astype(np.float32)
        if X.ndim != 2 or X.shape[1] != input_dim:
            self.fail(
                f"real features must be (N, {input_dim}) to match the live"
                f" model; got {X.shape}."
            )
```

- [ ] **Step 2: Confirm the test still skips cleanly with no live model set**

Run: `cd tests && uv run --no-sync python -m unittest -v pyspacer.test_portable_artifact.LiveModelParityTest`
Expected: `skipped` (because `PORTABLE_ARTIFACT_LIVE_MODEL` is unset) — the `@unittest.skipUnless` gate is untouched, so the default CI run is unaffected.

- [ ] **Step 3: Confirm it fails loudly when model is set but features are missing**

Run:
```bash
cd tests && PORTABLE_ARTIFACT_LIVE_MODEL=/nonexistent/model.pkl \
  uv run --no-sync python -m unittest -v \
  pyspacer.test_portable_artifact.LiveModelParityTest 2>&1 | head -30
```
Expected: the test runs (not skipped) and FAILS — either at `_load_live_model` (model path invalid) or, once a real model path is given without features, at the `PORTABLE_ARTIFACT_LIVE_FEATURES is not` `self.fail`. The key assertion: it does **not** silently pass on random features. (A full green run requires a real model + real-features `.npy`; that is the operator's pre-release/pre-bump gate, not a default CI job.)

- [ ] **Step 4: Commit**

```bash
git add tests/pyspacer/test_portable_artifact.py
git commit -m "test(inference): require real EfficientNet features for live parity gate"
```

---

## Operator runbook (record in PR description)

The full real-feature parity proof — run before merge and before any sklearn bump:

```bash
# 1. Stack real features from production .featurevector files
uv run python scripts/extract_reference_features.py \
    --out reference_features.npy s3://<bucket>/<key>.featurevector ...

# 2. Run the live gate on the real model + real features
cd tests && \
PORTABLE_ARTIFACT_LIVE_MODEL=s3://<bucket>/<model>.pkl \
PORTABLE_ARTIFACT_LIVE_FEATURES=$(pwd)/../reference_features.npy \
  uv run --no-sync python -m unittest -v \
  pyspacer.test_portable_artifact.LiveModelParityTest
# Expect: 1 test passed, max|Δ| < 1e-6 on real features.
```

To bump scikit-learn later: update the pin in `pyproject.toml` + `uv lock`, set `PARITY_PROVEN_SKLEARN`, then re-run the runbook above and confirm `< 1e-6` before merging.

## Self-Review

- **Review item 1 (real features):** Task 5 produces real-feature `.npy`; Task 6 removes the random fallback and requires it. ✓
- **Review item 2 (pin sklearn at export + fail CI on drift without re-proving):** Task 2 pins the version + constant; Task 3 fails export on mismatch; Task 4 fails CI on drift; Task 1 ensures CI exists on the branch. ✓
- **No placeholders:** every code/test step shows complete code; the one verification caveat (`ImageFeatures.store` vs `.save`) is flagged with the file to confirm against. ✓
- **Type/name consistency:** `PARITY_PROVEN_SKLEARN` and `SklearnPinError` defined in Task 2, imported with identical names in Tasks 3–4; `enforce_sklearn_pin` keyword consistent in Task 3. ✓
- **Lockfile discipline:** every `pyproject.toml` edit (Tasks 1, 2) is followed by `uv lock` + commit, satisfying CI's `--frozen`. ✓
