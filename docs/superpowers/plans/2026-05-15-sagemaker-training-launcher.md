# SageMaker Training Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch the mermaid-classifier training pipeline as a SageMaker TrainingJob, driven by a YAML config, while leaving `scripts/classifier_train.py` untouched as the notebook-style example.

**Architecture:** A Pydantic schema (`mermaid_classifier/sagemaker/config.py`) describes the YAML. The container entrypoint (`scripts/sagemaker_train_entrypoint.py`) reads the YAML, sets env vars *before* importing pyspacer (whose `Settings()` reads env at import time), then constructs `DatasetOptions` / `TrainingOptions` / `MLflowOptions` and calls `MLflowTrainingRunner.run()`. The launcher (`scripts/launch_training_sagemaker.py`) uses the SageMaker SDK's `Estimator` to submit the job with the config dir mounted as a single input channel.

**Tech Stack:** Python 3.10+, Pydantic v2 (already via `pydantic-settings`), PyYAML (new dep), SageMaker SDK (new optional dep), Docker, AWS S3 + ECR + SageMaker, MLflow (existing). Tests use `unittest` (project convention; NOT pytest).

**Reference design spec:** `docs/superpowers/specs/2026-05-15-sagemaker-training-launcher-design.md`. **Branch:** `sagemaker_training_launcher` (already created off `rollup`).

**Test runner reminder:** all commands assume `cwd = mermaid-classifier/`. Tests are run as `cd tests && python -m unittest <module.path>` per `CLAUDE.md`.

---

## File structure

| File | Created in task | Responsibility |
|------|-----------------|----------------|
| `mermaid_classifier/sagemaker/__init__.py` | 1 | Subpackage marker. |
| `mermaid_classifier/sagemaker/config.py` | 1 | Pydantic `TrainingRunConfig` + sub-models + `from_yaml_path()` + `build_options()`. The only file that maps YAML ↔ pyspacer dataclasses. |
| `tests/sagemaker_launcher/__init__.py` | 1 | Test package marker. |
| `tests/sagemaker_launcher/test_config.py` | 1 | Unit tests for the schema. |
| `sagemaker/configs/example/training_config.yaml` | 2 | Reference YAML, fully populated, with comments. |
| `sagemaker/configs/example/sources.csv` | 2 | 2-source fixture for smoke tests. |
| `sagemaker/configs/example/rollups.csv` | 2 | Empty header-only fixture. |
| `sagemaker/configs/example/included_labels.csv` | 2 | Empty header-only fixture. |
| `scripts/sagemaker_train_entrypoint.py` | 3 | Container entrypoint: stage logging, env dump, YAML → options → `runner.run()`. |
| `tests/sagemaker_launcher/test_entrypoint.py` | 3 | Unit tests for the entrypoint (mocked runner). |
| `docker/training/Dockerfile` | 4 | CPU-only `python:3.10-slim` image; CPU torch installed before `.[pyspacer]`. |
| `docker/training/entrypoint.sh` | 4 | `exec python -u scripts/sagemaker_train_entrypoint.py "$@"`. |
| `docker/training/local_smoke.sh` | 5 | Builds the image, runs the example config locally with `-v` mounts, asserts exit 0. |
| `scripts/launch_training_sagemaker.py` | 6 | SageMaker SDK launcher: argparse, validate, S3 sync, `Estimator.fit(wait=True, logs="All")`. |
| `tests/sagemaker_launcher/test_launcher.py` | 6 | Unit tests for the launcher (mocked boto3/SDK). |
| `docs/training_at_scale.md` | 7 | One-time setup runbook. |
| `pyproject.toml` (modified) | 1, 6 | Adds `pyyaml` to `[pyspacer]`; adds new `[sagemaker]` extra with the SageMaker SDK. |

---

## Task 1: Pydantic config schema

**Files:**
- Create: `mermaid_classifier/sagemaker/__init__.py`
- Create: `mermaid_classifier/sagemaker/config.py`
- Create: `tests/sagemaker_launcher/__init__.py`
- Create: `tests/sagemaker_launcher/test_config.py`
- Modify: `pyproject.toml` (add `pyyaml` to `[pyspacer]` extras)

### Why this comes first
The schema is the architectural keystone (per the spec). Everything else either produces or consumes a YAML file conforming to it. Building it first means the example YAML and entrypoint can be written against a concrete, tested contract.

### Critical design constraint
`config.py` must NOT import from `mermaid_classifier.pyspacer.*` at module load. The pyspacer package has import-time side effects (`set_env_vars_for_packages()` in `pyspacer/__init__.py`) that read `Settings()` from env vars. The entrypoint must be able to import `config.py`, parse the YAML, **set env vars from the YAML's `env` block**, and only *then* import pyspacer. So `build_options()` does the heavy imports lazily inside the method.

### Steps

- [ ] **Step 1.1:** Create empty subpackage markers.

```bash
mkdir -p mermaid_classifier/sagemaker tests/sagemaker
touch mermaid_classifier/sagemaker/__init__.py
touch tests/sagemaker_launcher/__init__.py
```

- [ ] **Step 1.2:** Add `pyyaml` to `[pyspacer]` extras in `pyproject.toml`.

Edit `pyproject.toml`. Inside the `pyspacer = [...]` list, add `"pyyaml",` near `"pydantic-settings",` (alphabetical placement). Resulting list should include:

```toml
    "pydantic-settings",
    "pyspacer==0.14.0",
    "pyyaml",
    "s3fs",
```

Reinstall locally:

```bash
pip install -e .[pyspacer]
```

- [ ] **Step 1.3:** Write the failing test file.

Create `tests/sagemaker_launcher/test_config.py`:

```python
"""Unit tests for the Pydantic TrainingRunConfig schema.

Schema lives in mermaid_classifier/sagemaker/config.py. These tests
exercise both happy paths (loading a complete YAML) and edge cases
(missing required fields, unknown strategies). They intentionally do
NOT import from mermaid_classifier.pyspacer.* to keep the test fast
and to verify the schema is decoupled from the heavy pyspacer imports.
"""
from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from mermaid_classifier.sagemaker.config import TrainingRunConfig


MINIMAL_YAML = textwrap.dedent("""
    dataset:
      include_mermaid: true
      coralnet_sources_csv: sources.csv
      label_rollup_spec_csv: rollups.csv
      included_labels_csv: included_labels.csv
      subsample:
        strategy: balanced
        total_annotations: 1000
        min_per_class: 10
      weighting:
        strategy: effective_number
        alpha: 0.5
    training:
      epochs: 5
      hidden_layer_sizes: [200, 100]
      learning_rate_init: 0.001
      early_stopping_patience: 3
      random_state: 42
    mlflow:
      experiment_name: test-experiment
      model_name: TestModel
    env:
      MLFLOW_TRACKING_SERVER: file:./mlruns
      WEIGHTS_LOCATION: s3://bucket/weights.pt
""").lstrip()


def _write(tmp: Path, text: str) -> Path:
    p = tmp / "training_config.yaml"
    p.write_text(text)
    return p


class LoadHappyPathTest(unittest.TestCase):

    def test_minimal_yaml_loads(self):
        with TemporaryDirectory() as td:
            path = _write(Path(td), MINIMAL_YAML)
            config = TrainingRunConfig.from_yaml_path(path)
        self.assertTrue(config.dataset.include_mermaid)
        self.assertEqual(config.training.epochs, 5)
        self.assertEqual(config.training.hidden_layer_sizes, (200, 100))
        self.assertEqual(config.mlflow.experiment_name, "test-experiment")
        self.assertEqual(
            config.env["MLFLOW_TRACKING_SERVER"], "file:./mlruns")

    def test_csv_paths_resolve_against_yaml_dir(self):
        with TemporaryDirectory() as td:
            path = _write(Path(td), MINIMAL_YAML)
            config = TrainingRunConfig.from_yaml_path(path)
            self.assertEqual(
                config.dataset.coralnet_sources_csv_path(path.parent),
                Path(td) / "sources.csv",
            )


class SubsampleStrategiesTest(unittest.TestCase):

    def _load(self, subsample_yaml: str) -> TrainingRunConfig:
        yaml_text = MINIMAL_YAML.replace(
            textwrap.dedent("""\
              subsample:
                strategy: balanced
                total_annotations: 1000
                min_per_class: 10
            """).rstrip(),
            subsample_yaml.rstrip(),
        )
        with TemporaryDirectory() as td:
            path = _write(Path(td), yaml_text)
            return TrainingRunConfig.from_yaml_path(path)

    def test_stratified_requires_total_annotations(self):
        config = self._load(textwrap.dedent("""\
              subsample:
                strategy: stratified
                total_annotations: 5000
        """))
        self.assertEqual(config.dataset.subsample.strategy, "stratified")
        self.assertEqual(config.dataset.subsample.total_annotations, 5000)

    def test_soft_balanced_requires_balance_alpha(self):
        config = self._load(textwrap.dedent("""\
              subsample:
                strategy: soft_balanced
                total_annotations: 5000
                balance_alpha: 0.5
        """))
        self.assertEqual(config.dataset.subsample.strategy, "soft_balanced")
        self.assertEqual(config.dataset.subsample.balance_alpha, 0.5)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValidationError):
            self._load(textwrap.dedent("""\
                  subsample:
                    strategy: not_a_strategy
                    total_annotations: 1000
            """))


class WeightingTest(unittest.TestCase):

    def test_default_weighting_strategy(self):
        with TemporaryDirectory() as td:
            path = _write(Path(td), MINIMAL_YAML)
            config = TrainingRunConfig.from_yaml_path(path)
        self.assertTrue(config.dataset.weighting.enabled)
        self.assertEqual(
            config.dataset.weighting.strategy, "effective_number")
        self.assertEqual(config.dataset.weighting.alpha, 0.5)

    def test_invalid_alpha_rejected(self):
        bad_yaml = MINIMAL_YAML.replace("alpha: 0.5", "alpha: 1.5")
        with TemporaryDirectory() as td:
            path = _write(Path(td), bad_yaml)
            with self.assertRaises(ValidationError):
                TrainingRunConfig.from_yaml_path(path)


class RequiredFieldsTest(unittest.TestCase):

    def test_missing_dataset_block_rejected(self):
        bad = "training:\n  epochs: 1\nmlflow:\n  experiment_name: x\n"
        with TemporaryDirectory() as td:
            path = _write(Path(td), bad)
            with self.assertRaises(ValidationError):
                TrainingRunConfig.from_yaml_path(path)


class BuildOptionsTest(unittest.TestCase):
    """build_options() lazily imports pyspacer dataclasses and constructs them.

    These tests DO import pyspacer (transitively). Skipped if the
    pyspacer extras aren't installed.
    """

    def test_build_options_produces_three_dataclasses(self):
        try:
            from mermaid_classifier.pyspacer.train import (
                DatasetOptions, MLflowOptions, TrainingOptions,
            )
        except ImportError:
            self.skipTest("pyspacer extras not installed")
        with TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "sources.csv").write_text("id\n123\n")
            (tmp / "rollups.csv").write_text(
                "from_ba_id,from_gf_id,to_ba_id,to_gf_id\n")
            (tmp / "included_labels.csv").write_text("ba_id,gf_id\n")
            path = _write(tmp, MINIMAL_YAML)
            config = TrainingRunConfig.from_yaml_path(path)
            dataset, training, mlflow = config.build_options(
                config_dir=tmp)
        self.assertIsInstance(dataset, DatasetOptions)
        self.assertIsInstance(training, TrainingOptions)
        self.assertIsInstance(mlflow, MLflowOptions)
        self.assertEqual(
            dataset.coralnet_sources_csv, str(tmp / "sources.csv"))
        self.assertEqual(training.epochs, 5)
        self.assertEqual(training.hidden_layer_sizes, (200, 100))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 1.4:** Run the test to verify it fails.

```bash
cd tests && python -m unittest sagemaker_launcher.test_config -v
```

Expected: `ModuleNotFoundError: No module named 'mermaid_classifier.sagemaker.config'` or `ImportError: cannot import name 'TrainingRunConfig'`. Failure is expected — `config.py` doesn't exist yet.

- [ ] **Step 1.5:** Implement `mermaid_classifier/sagemaker/config.py`.

Create `mermaid_classifier/sagemaker/config.py`:

```python
"""Pydantic schema for the SageMaker training YAML config.

This module is the single point where the YAML structure maps to the
pyspacer training option dataclasses. Keep it self-contained: do NOT
import from mermaid_classifier.pyspacer.* at module load time. The
pyspacer package has import-time side effects that read env vars via
Settings(), and the SageMaker container entrypoint depends on being
able to load this schema, apply env vars from the YAML's `env` block,
and ONLY THEN import pyspacer.

`build_options()` performs the heavy imports lazily inside the method
so callers can sequence env-var application correctly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Mirror SUBSAMPLE_STRATEGIES from mermaid_classifier.training.subsample.options.
# Duplicated here to avoid importing the pyspacer subtree at module load.
# Keep in sync when adding a new strategy in pyspacer.
_SUBSAMPLE_STRATEGIES = ("stratified", "balanced", "soft_balanced")


class SubsampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["stratified", "balanced", "soft_balanced"]
    total_annotations: int | None = None
    min_per_class: int = 0
    target_per_class: int | None = None
    balance_alpha: float | None = None
    seed: int = 0

    @field_validator("total_annotations")
    @classmethod
    def _positive_total(cls, v):
        if v is not None and v <= 0:
            raise ValueError("total_annotations must be > 0 or None")
        return v

    @field_validator("min_per_class")
    @classmethod
    def _non_negative_floor(cls, v):
        if v < 0:
            raise ValueError("min_per_class must be >= 0")
        return v


class WeightingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    strategy: str = "tree_balanced_ba_flat_gf"
    alpha: float = 0.5
    weight_ratio_cap: float | None = None

    @field_validator("alpha")
    @classmethod
    def _alpha_in_unit_interval(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        return v

    @field_validator("weight_ratio_cap")
    @classmethod
    def _cap_at_least_one(cls, v):
        if v is not None and v < 1.0:
            raise ValueError("weight_ratio_cap must be None or >= 1.0")
        return v


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_mermaid: bool = True
    # Relative paths are resolved as siblings of the YAML file by
    # *_path() helpers. The launcher uploads the YAML and CSVs together
    # to the same S3 prefix so the same resolution works in the
    # container.
    coralnet_sources_csv: str | None = None
    drop_growthforms: bool = False
    label_rollup_spec_csv: str | None = None
    included_labels_csv: str | None = None
    excluded_labels_csv: str | None = None
    ref_val_ratios: tuple[float, float] = (0.1, 0.1)
    subsample: SubsampleConfig | None = None
    weighting: WeightingConfig | None = None

    def coralnet_sources_csv_path(self, base: Path) -> Path | None:
        return None if self.coralnet_sources_csv is None \
            else base / self.coralnet_sources_csv

    def label_rollup_spec_csv_path(self, base: Path) -> Path | None:
        return None if self.label_rollup_spec_csv is None \
            else base / self.label_rollup_spec_csv

    def included_labels_csv_path(self, base: Path) -> Path | None:
        return None if self.included_labels_csv is None \
            else base / self.included_labels_csv

    def excluded_labels_csv_path(self, base: Path) -> Path | None:
        return None if self.excluded_labels_csv is None \
            else base / self.excluded_labels_csv


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = 10
    hidden_layer_sizes: tuple[int, ...] | None = None
    learning_rate_init: float | None = None
    early_stopping_patience: int | None = None
    random_state: int = 0


class MLflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str | None = None
    model_name: str | None = None
    annotations_to_log: str | None = None


class TrainingRunConfig(BaseModel):
    """Top-level schema for `training_config.yaml`."""
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig
    training: TrainingConfig
    mlflow: MLflowConfig
    # Env vars to apply *before* importing pyspacer. Used for
    # MLFLOW_TRACKING_SERVER, WEIGHTS_LOCATION, bucket overrides, etc.
    env: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> "TrainingRunConfig":
        text = Path(path).read_text()
        raw = yaml.safe_load(text) or {}
        return cls.model_validate(raw)

    def apply_env(self) -> None:
        """Apply the `env` block to os.environ.

        Call this BEFORE importing mermaid_classifier.pyspacer.*.
        """
        import os
        for key, value in self.env.items():
            os.environ[key] = str(value)

    def build_options(self, config_dir: Path):
        """Translate this config into the three pyspacer option dataclasses.

        Heavy imports happen here. Call apply_env() first if any env
        vars in this config affect pyspacer's Settings().

        Returns
        -------
        (DatasetOptions, TrainingOptions, MLflowOptions)
        """
        from mermaid_classifier.pyspacer.train import (
            DatasetOptions, MLflowOptions, TrainingOptions,
        )
        from mermaid_classifier.training.subsample import SubsampleOptions
        from mermaid_classifier.training.sample_weighting import (
            SampleWeightingOptions,
        )

        d = self.dataset

        subsample = None
        if d.subsample is not None:
            subsample = SubsampleOptions(
                strategy=d.subsample.strategy,
                total_annotations=d.subsample.total_annotations,
                min_per_class=d.subsample.min_per_class,
                target_per_class=d.subsample.target_per_class,
                balance_alpha=d.subsample.balance_alpha,
                seed=d.subsample.seed,
            )

        weighting = None
        if d.weighting is not None:
            weighting = SampleWeightingOptions(
                enabled=d.weighting.enabled,
                strategy=d.weighting.strategy,
                alpha=d.weighting.alpha,
                weight_ratio_cap=d.weighting.weight_ratio_cap,
            )

        def _resolve(p):
            return None if p is None else str(p)

        dataset_options = DatasetOptions(
            include_mermaid=d.include_mermaid,
            coralnet_sources_csv=_resolve(
                d.coralnet_sources_csv_path(config_dir)),
            drop_growthforms=d.drop_growthforms,
            label_rollup_spec_csv=_resolve(
                d.label_rollup_spec_csv_path(config_dir)),
            included_labels_csv=_resolve(
                d.included_labels_csv_path(config_dir)),
            excluded_labels_csv=_resolve(
                d.excluded_labels_csv_path(config_dir)),
            ref_val_ratios=tuple(d.ref_val_ratios),
            subsample=subsample,
            weighting=weighting,
        )

        t = self.training
        training_options = TrainingOptions(
            epochs=t.epochs,
            hidden_layer_sizes=(
                tuple(t.hidden_layer_sizes)
                if t.hidden_layer_sizes is not None
                else None
            ),
            learning_rate_init=t.learning_rate_init,
            early_stopping_patience=t.early_stopping_patience,
            random_state=t.random_state,
        )

        m = self.mlflow
        mlflow_options = MLflowOptions(
            experiment_name=m.experiment_name,
            model_name=m.model_name,
            annotations_to_log=m.annotations_to_log,
        )

        return dataset_options, training_options, mlflow_options
```

- [ ] **Step 1.6:** Run the tests to verify they pass.

```bash
cd tests && python -m unittest sagemaker_launcher.test_config -v
```

Expected: all tests pass. `BuildOptionsTest.test_build_options_produces_three_dataclasses` runs if pyspacer extras are installed; otherwise it's skipped.

- [ ] **Step 1.7:** Commit.

```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier
git add pyproject.toml \
    mermaid_classifier/sagemaker/__init__.py \
    mermaid_classifier/sagemaker/config.py \
    tests/sagemaker_launcher/__init__.py \
    tests/sagemaker_launcher/test_config.py
git commit -m "$(cat <<'EOF'
feat(sagemaker): add Pydantic TrainingRunConfig schema

Single source of truth for the YAML <-> pyspacer dataclass mapping.
build_options() defers the pyspacer imports until after env vars
from the YAML's env block can be applied. Adds pyyaml to [pyspacer]
extras. Tests cover happy path, all three subsample strategies,
weighting validation, and required-field rejection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Example config dir

**Files:**
- Create: `sagemaker/configs/example/training_config.yaml`
- Create: `sagemaker/configs/example/sources.csv`
- Create: `sagemaker/configs/example/rollups.csv`
- Create: `sagemaker/configs/example/included_labels.csv`
- Modify: `tests/sagemaker_launcher/test_config.py` (add example-loads-OK test)

### Why this is task 2
The example serves three purposes: (1) committed template users copy when starting a new run, (2) fixture for the local Docker smoke test in Task 5, (3) sanity test that the schema and a real YAML stay in sync.

### Steps

- [ ] **Step 2.1:** Create the example config YAML.

Create `sagemaker/configs/example/training_config.yaml`:

```yaml
# Example training config for the SageMaker launcher.
#
# Copy this directory to a new name (e.g. sagemaker/configs/my-run/),
# edit values, then run:
#   python scripts/launch_training_sagemaker.py \
#     --config-dir sagemaker/configs/my-run \
#     --mlflow-tracking-uri <ARN> \
#     --role-arn <ARN> \
#     --ecr-image-uri <URI> \
#     --staging-bucket <BUCKET>
#
# CSV paths are resolved as siblings of THIS file. Bare filenames only.

dataset:
  include_mermaid: false
  # 2-source fixture; replace with your sources.csv for real runs.
  coralnet_sources_csv: sources.csv
  drop_growthforms: false
  label_rollup_spec_csv: rollups.csv
  included_labels_csv: included_labels.csv
  ref_val_ratios: [0.1, 0.1]
  # Subsample. Strategies: 'stratified', 'balanced', 'soft_balanced'.
  # Keep this section out (set to null) to disable subsampling entirely.
  subsample:
    strategy: balanced
    total_annotations: 1000
    min_per_class: 10
  # Sample weighting. Remove the block (or set enabled: false) to
  # disable.
  weighting:
    enabled: true
    strategy: effective_number
    alpha: 0.5
    weight_ratio_cap: 5000.0

training:
  epochs: 1
  # PyTorch MLP head architecture. None falls back to MermaidTrainer's
  # label-count heuristic.
  hidden_layer_sizes: [200, 100]
  learning_rate_init: 0.0001
  early_stopping_patience: 3
  random_state: 0

mlflow:
  experiment_name: example-smoke-test
  model_name: ExampleModel
  # annotations_to_log: all  # Uncomment to log all annotations as an artifact.

# Env vars applied BEFORE importing mermaid_classifier. Keep secrets
# out of this file; commit only non-sensitive values. The launcher
# overrides MLFLOW_TRACKING_SERVER from its --mlflow-tracking-uri
# flag, so the value here is just a placeholder for local smoke runs.
env:
  MLFLOW_TRACKING_SERVER: file:./mlruns
  WEIGHTS_LOCATION: s3://mermaid-config/classifier/v1/efficientnet_weights.pt
  CORALNET_TRAIN_DATA_BUCKET: coral-reef-training
  MERMAID_TRAIN_DATA_BUCKET: coral-reef-training
  MLFLOW_HTTP_REQUEST_MAX_RETRIES: "2"
```

- [ ] **Step 2.2:** Create the CSV fixtures.

Create `sagemaker/configs/example/sources.csv`:

```
id
1645
1776
```

Create `sagemaker/configs/example/rollups.csv`:

```
from_ba_id,from_gf_id,to_ba_id,to_gf_id
```

Create `sagemaker/configs/example/included_labels.csv`:

```
ba_id,gf_id
```

Header-only files are valid — pyspacer's CSV specs accept empty bodies.

- [ ] **Step 2.3:** Add a test that loads the committed example.

Edit `tests/sagemaker_launcher/test_config.py`. After the `BuildOptionsTest` class and before the `if __name__ == "__main__":` block, append:

```python
class ExampleYamlTest(unittest.TestCase):

    def test_committed_example_loads(self):
        # Resolve relative to the repo root (the tests/ dir is one
        # level deep so the example is at ../sagemaker/configs/example).
        here = Path(__file__).resolve().parent.parent.parent
        example = here / "sagemaker" / "configs" / "example" \
            / "training_config.yaml"
        self.assertTrue(
            example.is_file(),
            f"Example YAML not found at {example}",
        )
        config = TrainingRunConfig.from_yaml_path(example)
        self.assertIsNotNone(config.dataset.subsample)
        self.assertEqual(
            config.dataset.subsample.strategy, "balanced")
```

- [ ] **Step 2.4:** Run the new test.

```bash
cd tests && python -m unittest sagemaker_launcher.test_config.ExampleYamlTest -v
```

Expected: passes.

- [ ] **Step 2.5:** Commit.

```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier
git add sagemaker/configs/example/ tests/sagemaker_launcher/test_config.py
git commit -m "$(cat <<'EOF'
feat(sagemaker): add example config dir + schema-conformance test

Reference YAML plus minimal CSV fixtures, sized for the local Docker
smoke test in a later task. Adds a test that re-validates the
committed example against the schema so the two stay in sync.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Container entrypoint script

**Files:**
- Create: `scripts/sagemaker_train_entrypoint.py`
- Create: `tests/sagemaker_launcher/test_entrypoint.py`

### Why this comes before Docker
The entrypoint can be unit-tested with a mocked runner without ever building the image. Getting the contract right here means Task 4 is just packaging.

### Critical sequencing
The entrypoint must:

1. Set up logging (basic, with `python -u` already buffering-disabled by the shell wrapper).
2. Read the YAML config.
3. Apply `env` block to `os.environ`.
4. **Now** import `MLflowTrainingRunner` (this triggers `Settings()` reading env).
5. Build options.
6. Run.

Stage markers around each phase. First-line dump (Python/pkg versions + env + dir listing) before anything else.

### Steps

- [ ] **Step 3.1:** Write the failing test file.

Create `tests/sagemaker_launcher/test_entrypoint.py`:

```python
"""Unit tests for scripts/sagemaker_train_entrypoint.py.

The entrypoint is a script, not a module; we import it by path. Tests
mock MLflowTrainingRunner so they don't trigger pyspacer's heavy
imports or hit AWS. Tests verify:

  * apply_env happens before the runner is imported (we observe this
    indirectly by patching the runner factory to capture os.environ at
    construction time)
  * runner is called once with options built from the YAML
  * stage markers appear in log output in the right order
  * an exception in runner.run propagates to sys.exit(1) with the
    traceback in log output
"""
from __future__ import annotations

import importlib.util
import io
import logging
import os
import sys
import textwrap
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENTRYPOINT_PATH = REPO_ROOT / "scripts" / "sagemaker_train_entrypoint.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "sagemaker_train_entrypoint", ENTRYPOINT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MINIMAL_YAML = textwrap.dedent("""
    dataset:
      include_mermaid: false
      coralnet_sources_csv: sources.csv
      label_rollup_spec_csv: rollups.csv
      included_labels_csv: included_labels.csv
      subsample:
        strategy: balanced
        total_annotations: 100
    training:
      epochs: 1
    mlflow:
      experiment_name: test
      model_name: T
    env:
      MLFLOW_TRACKING_SERVER: file:./mlruns
      WEIGHTS_LOCATION: s3://x/weights.pt
""").lstrip()


@contextmanager
def _config_dir():
    with TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "training_config.yaml").write_text(MINIMAL_YAML)
        (tmp / "sources.csv").write_text("id\n123\n")
        (tmp / "rollups.csv").write_text(
            "from_ba_id,from_gf_id,to_ba_id,to_gf_id\n")
        (tmp / "included_labels.csv").write_text("ba_id,gf_id\n")
        yield tmp


class EntrypointHappyPathTest(unittest.TestCase):

    def setUp(self):
        # Wipe env vars the entrypoint will set so we can detect them.
        for key in (
            "MLFLOW_TRACKING_SERVER", "WEIGHTS_LOCATION",
        ):
            os.environ.pop(key, None)
        self.module = _load_entrypoint()

    def test_main_runs_runner_once_with_built_options(self):
        with _config_dir() as cfg_dir:
            with patch.object(
                self.module, "_resolve_runner_factory"
            ) as get_factory:
                fake_runner = MagicMock()
                factory = MagicMock(return_value=fake_runner)
                get_factory.return_value = factory
                self.module.main([
                    "--config-dir", str(cfg_dir),
                ])
        factory.assert_called_once()
        fake_runner.run.assert_called_once_with()

    def test_main_applies_env_before_resolving_runner(self):
        with _config_dir() as cfg_dir:
            observed = {}

            def factory(*args, **kwargs):
                observed["mlflow"] = os.environ.get(
                    "MLFLOW_TRACKING_SERVER")
                observed["weights"] = os.environ.get("WEIGHTS_LOCATION")
                return MagicMock()

            with patch.object(
                self.module, "_resolve_runner_factory",
                return_value=factory,
            ):
                self.module.main(["--config-dir", str(cfg_dir)])
        self.assertEqual(observed["mlflow"], "file:./mlruns")
        self.assertEqual(observed["weights"], "s3://x/weights.pt")

    def test_main_logs_stage_markers_in_order(self):
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        try:
            with _config_dir() as cfg_dir:
                with patch.object(
                    self.module, "_resolve_runner_factory",
                    return_value=lambda *a, **kw: MagicMock(),
                ):
                    self.module.main(["--config-dir", str(cfg_dir)])
        finally:
            root.removeHandler(handler)
        log_text = buf.getvalue()
        # We expect ENTER markers in this exact order. EXIT markers
        # appear too; we only assert ENTER ordering here.
        markers = [
            line for line in log_text.splitlines()
            if "ENTER" in line
        ]
        self.assertTrue(any("load_config" in m for m in markers))
        self.assertTrue(any("apply_env" in m for m in markers))
        self.assertTrue(any("build_options" in m for m in markers))
        self.assertTrue(any("runner_run" in m for m in markers))


class EntrypointFailureTest(unittest.TestCase):

    def test_runner_exception_exits_nonzero(self):
        module = _load_entrypoint()
        with _config_dir() as cfg_dir:
            fake_runner = MagicMock()
            fake_runner.run.side_effect = RuntimeError("boom")
            with patch.object(
                module, "_resolve_runner_factory",
                return_value=lambda *a, **kw: fake_runner,
            ):
                with self.assertRaises(SystemExit) as cm:
                    module.main(["--config-dir", str(cfg_dir)])
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3.2:** Run the test to verify it fails.

```bash
cd tests && python -m unittest sagemaker_launcher.test_entrypoint -v
```

Expected: failure — `scripts/sagemaker_train_entrypoint.py` does not exist.

- [ ] **Step 3.3:** Implement the entrypoint.

Create `scripts/sagemaker_train_entrypoint.py`:

```python
"""SageMaker container entrypoint: YAML -> MLflowTrainingRunner.

Lifecycle inside the container:

  1. Parse --config-dir (default: /opt/ml/input/data/config).
  2. Load training_config.yaml.
  3. Log first-line dump: python/pkg versions, env vars (redacted),
     /opt/ml/input/data/ listing.
  4. Apply YAML's env block to os.environ.
  5. Import MLflowTrainingRunner (heavy import; reads Settings()).
  6. Build options.
  7. Run.

If anything raises, the traceback is logged and the process exits 1
so SageMaker marks the job Failed.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path


DEFAULT_CONFIG_DIR = "/opt/ml/input/data/config"
CONFIG_FILENAME = "training_config.yaml"

# Env vars whose values are sensitive enough to redact in the
# startup dump. Substring match on the key (uppercased).
_SECRET_KEY_FRAGMENTS = ("SECRET", "TOKEN", "PASSWORD", "KEY")


log = logging.getLogger("sagemaker_train_entrypoint")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


@contextmanager
def _stage(name: str):
    log.info("[stage:%s] ENTER", name)
    try:
        yield
    except Exception:
        log.info("[stage:%s] FAIL", name)
        raise
    else:
        log.info("[stage:%s] EXIT", name)


def _redact_env(env: dict) -> dict:
    out = {}
    for key, value in env.items():
        upper = key.upper()
        if any(frag in upper for frag in _SECRET_KEY_FRAGMENTS):
            out[key] = "***"
        else:
            out[key] = value
    return out


def _first_line_dump(config_dir: Path) -> None:
    """Log everything a future debugger will want to see first."""
    log.info("python: %s", sys.version.replace("\n", " "))
    try:
        from importlib.metadata import version
        for pkg in ("pyspacer", "mlflow", "duckdb", "torch",
                    "pydantic", "pyyaml"):
            try:
                log.info("pkg %s: %s", pkg, version(pkg))
            except Exception:
                log.info("pkg %s: <not installed>", pkg)
    except Exception:
        log.warning("could not read package versions")

    log.info("config_dir: %s", config_dir)
    try:
        if config_dir.is_dir():
            for entry in sorted(config_dir.iterdir()):
                log.info(
                    "config_dir entry: %s (%d bytes)",
                    entry.name,
                    entry.stat().st_size if entry.is_file() else -1,
                )
        else:
            log.warning("config_dir does not exist: %s", config_dir)
    except Exception as e:
        log.warning("config_dir listing failed: %s", e)

    redacted = _redact_env(dict(os.environ))
    for k in sorted(redacted):
        log.info("env %s=%s", k, redacted[k])


def _resolve_runner_factory():
    """Return the MLflowTrainingRunner class (heavy import).

    Factored into a function so tests can patch it without triggering
    the pyspacer import side effects.
    """
    from mermaid_classifier.pyspacer.train import MLflowTrainingRunner
    return MLflowTrainingRunner


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        default=os.environ.get(
            "SAGEMAKER_CONFIG_DIR", DEFAULT_CONFIG_DIR),
        help=(
            "Directory containing training_config.yaml and sibling "
            "CSVs. Default: /opt/ml/input/data/config (where SageMaker "
            "mounts the 'config' input channel)."
        ),
    )
    args = parser.parse_args(argv)
    config_dir = Path(args.config_dir).resolve()

    try:
        _first_line_dump(config_dir)

        with _stage("load_config"):
            from mermaid_classifier.sagemaker.config import (
                TrainingRunConfig,
            )
            config = TrainingRunConfig.from_yaml_path(
                config_dir / CONFIG_FILENAME)

        with _stage("apply_env"):
            config.apply_env()

        with _stage("build_options"):
            dataset_options, training_options, mlflow_options = (
                config.build_options(config_dir=config_dir))
            log.info("dataset_options: %s", dataset_options)
            log.info("training_options: %s", training_options)
            log.info("mlflow_options: %s", mlflow_options)

        with _stage("runner_run"):
            RunnerClass = _resolve_runner_factory()
            runner = RunnerClass(
                dataset_options=dataset_options,
                training_options=training_options,
                mlflow_options=mlflow_options,
            )
            runner.run()

    except Exception:
        log.error(
            "Training run failed; full traceback follows.\n%s",
            traceback.format_exc(),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.4:** Run the entrypoint tests.

```bash
cd tests && python -m unittest sagemaker_launcher.test_entrypoint -v
```

Expected: all four tests pass.

- [ ] **Step 3.5:** Commit.

```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier
git add scripts/sagemaker_train_entrypoint.py tests/sagemaker_launcher/test_entrypoint.py
git commit -m "$(cat <<'EOF'
feat(sagemaker): add container entrypoint that drives MLflowTrainingRunner

Reads /opt/ml/input/data/config/training_config.yaml, applies the env
block before importing pyspacer (so Settings() picks up the right
values), then builds options and runs. Stage markers (ENTER/EXIT/FAIL)
around each phase and a first-line env+version dump make CloudWatch
debugging tractable. Tests cover sequencing, logging, and failure exit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Dockerfile + entrypoint shim

**Files:**
- Create: `docker/training/Dockerfile`
- Create: `docker/training/entrypoint.sh`

### Why image-baked code
Per the design spec: the image holds the code, not the SageMaker `source_dir` upload. Image tag pins the running code, so rolling back a bad change is `docker tag` + push.

### Critical detail: install CPU torch first
The pyspacer dep transitively pulls torch. If we let it resolve naturally, pip picks the default CUDA wheel (~2 GB). Installing CPU torch *before* `pip install -e .[pyspacer]` makes the second install see the torch requirement as satisfied and skip the CUDA wheel.

### Steps

- [ ] **Step 4.1:** Write the Dockerfile.

Create `docker/training/Dockerfile`:

```dockerfile
# CPU-only image for the mermaid-classifier SageMaker TrainingJob.
# Build from the mermaid-classifier/ directory:
#   docker buildx build --platform linux/amd64 \
#       -t <ECR_URI>:<tag> -f docker/training/Dockerfile .
#
# See docs/training_at_scale.md for the full build/push runbook.

FROM python:3.10-slim

# Build deps for pyarrow, duckdb, and the rest of the pyspacer chain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Don't write .pyc; don't buffer stdout (SageMaker tails it to CloudWatch).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/ml/code

# 1. Install CPU-only PyTorch FIRST. Without this, pyspacer's transitive
#    resolution pulls the default ~2GB CUDA wheel even though we never
#    use GPU at training time.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# 2. Copy the project metadata and install the package with training
#    extras. torch is already satisfied so this is fast.
COPY pyproject.toml ./
COPY mermaid_classifier/ ./mermaid_classifier/
RUN pip install -e .[pyspacer]

# 3. Copy in the entrypoint and SageMaker scripts.
COPY scripts/sagemaker_train_entrypoint.py ./scripts/
COPY docker/training/entrypoint.sh ./docker/training/
RUN chmod +x ./docker/training/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/docker/training/entrypoint.sh"]
```

- [ ] **Step 4.2:** Write the entrypoint shim.

Create `docker/training/entrypoint.sh`:

```bash
#!/usr/bin/env bash
# SageMaker passes arbitrary args through to ENTRYPOINT. Use `exec` so
# signals (SageMaker sends SIGTERM on stop) reach Python directly.
set -euo pipefail
cd /opt/ml/code
exec python -u scripts/sagemaker_train_entrypoint.py "$@"
```

Make it executable locally too (so git tracks the bit):

```bash
chmod +x /Users/gregn/Documents/wcs/mermaid-classifier/docker/training/entrypoint.sh
```

- [ ] **Step 4.3:** Verify the image builds locally.

This is a manual verification step since CI won't run Docker. From `mermaid-classifier/`:

```bash
docker buildx build --platform linux/amd64 \
    -t mermaid-classifier-training:dev \
    -f docker/training/Dockerfile .
```

Expected: build succeeds. First build is slow (~5 min) due to torch + pyspacer install. Note the final image size with `docker images mermaid-classifier-training:dev`. Should be ~1.5 GB; if it's >3 GB the CUDA-torch slipped through and step 1 of the Dockerfile needs investigating.

If you don't have Docker locally, skip this step and rely on the smoke test in Task 5.

- [ ] **Step 4.4:** Commit.

```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier
git add docker/training/Dockerfile docker/training/entrypoint.sh
git commit -m "$(cat <<'EOF'
feat(sagemaker): add CPU-only Dockerfile for training image

python:3.10-slim base with CPU torch installed before the pyspacer
extras so we avoid the ~2GB CUDA wheel. ENTRYPOINT routes through a
shell shim that execs the Python entrypoint with -u for unbuffered
CloudWatch logging.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Local Docker smoke test script

**Files:**
- Create: `docker/training/local_smoke.sh`

### Why this is its own task
Validates the image + entrypoint + example config work end-to-end without paying for SageMaker. Catches packaging issues (wrong paths in Dockerfile, env vars not flowing, etc.) before the first cloud submission. Used both by the implementer for verification and by future users following the runbook.

### Constraints
The smoke run cannot actually download training data (would need real AWS creds and S3 access) — it exercises *startup* (config load, env apply, runner construction). It deliberately fails at the data-download phase, so we treat "log shows ENTER for build_options" as success criteria, not a 0 exit code. This documents the boundary between what the smoke test *can* prove and what only a real run does.

### Steps

- [ ] **Step 5.1:** Write the smoke script.

Create `docker/training/local_smoke.sh`:

```bash
#!/usr/bin/env bash
# Local Docker smoke test for the mermaid-classifier training image.
#
# Goal: prove that the image starts, reads the example config, applies
# env vars, and gets as far as constructing MLflowTrainingRunner.
# Without real AWS credentials and a real MLflow server the run cannot
# complete; we treat "made it past build_options stage" as success.
#
# Usage (from mermaid-classifier/):
#   bash docker/training/local_smoke.sh
#
# Pass --with-aws to mount your local ~/.aws and attempt a fuller run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE_TAG=mermaid-classifier-training:smoke
EXAMPLE_DIR="${REPO_ROOT}/sagemaker/configs/example"

WITH_AWS=0
for arg in "$@"; do
    case "$arg" in
        --with-aws) WITH_AWS=1 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [[ ! -d "$EXAMPLE_DIR" ]]; then
    echo "Missing example config at $EXAMPLE_DIR" >&2
    exit 1
fi

echo "==> Building image $IMAGE_TAG"
docker buildx build --platform linux/amd64 \
    -t "$IMAGE_TAG" \
    -f "${REPO_ROOT}/docker/training/Dockerfile" \
    "$REPO_ROOT"

echo "==> Running container with example config mounted"
DOCKER_ARGS=(
    --rm
    --platform linux/amd64
    -v "${EXAMPLE_DIR}:/opt/ml/input/data/config:ro"
)
if [[ "$WITH_AWS" -eq 1 ]]; then
    DOCKER_ARGS+=( -v "${HOME}/.aws:/root/.aws:ro" )
fi

LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT

# Capture both stdout and stderr.
set +e
docker run "${DOCKER_ARGS[@]}" "$IMAGE_TAG" 2>&1 | tee "$LOG_FILE"
RC=$?
set -e

echo
echo "==> Smoke verification"
if grep -q '\[stage:build_options\] ENTER' "$LOG_FILE"; then
    echo "OK: reached build_options stage"
else
    echo "FAIL: build_options stage marker missing in container log"
    exit 1
fi

if grep -q '\[stage:apply_env\] EXIT' "$LOG_FILE"; then
    echo "OK: applied env vars"
else
    echo "FAIL: apply_env did not complete"
    exit 1
fi

echo "==> Smoke test passed (container exit code was $RC; non-zero is"
echo "    expected without real AWS creds and an MLflow server)."
```

Make it executable:

```bash
chmod +x /Users/gregn/Documents/wcs/mermaid-classifier/docker/training/local_smoke.sh
```

- [ ] **Step 5.2:** Run the smoke test (manual verification).

```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier
bash docker/training/local_smoke.sh
```

Expected: build succeeds (slow first run); container starts; log shows `[stage:load_config] ENTER ... EXIT`, `[stage:apply_env] ENTER ... EXIT`, `[stage:build_options] ENTER`. Then a failure inside `build_options` or `runner_run` because there's no real MLflow server or S3 access. The script prints `Smoke test passed` and exits 0 once both markers are seen.

If you don't have Docker locally, skip this step. The Pydantic + entrypoint unit tests cover the same code paths in-process.

- [ ] **Step 5.3:** Commit.

```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier
git add docker/training/local_smoke.sh
git commit -m "$(cat <<'EOF'
feat(sagemaker): add local Docker smoke test script

Builds the training image, runs the example config with the config dir
bind-mounted, and verifies stage markers in the container log. Covers
the integration gap between unit tests (no Docker) and SageMaker
(real AWS). Treats reaching build_options as success since the run
cannot complete without real AWS creds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Launcher script

**Files:**
- Create: `scripts/launch_training_sagemaker.py`
- Create: `tests/sagemaker_launcher/test_launcher.py`
- Modify: `pyproject.toml` (add `[sagemaker]` extras with the SDK)

### Why this comes last
Everything the launcher submits must already work in the container. By this point the schema, entrypoint, image, and example are all proven.

### Steps

- [ ] **Step 6.1:** Add the SageMaker SDK as a new optional extra.

Edit `pyproject.toml`. After the `[project.optional-dependencies]` block already containing `pyspacer` and `jupyterlab`, append:

```toml
sagemaker = [
    "sagemaker",
]
```

The launcher also needs `pyyaml` (already in `[pyspacer]`) and `boto3` (already in base `dependencies`). Install:

```bash
pip install -e .[sagemaker,pyspacer]
```

- [ ] **Step 6.2:** Write the failing test file.

Create `tests/sagemaker_launcher/test_launcher.py`:

```python
"""Unit tests for scripts/launch_training_sagemaker.py.

The launcher is a script; tests import it by path. SageMaker SDK and
boto3 are mocked so tests don't hit AWS.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LAUNCHER_PATH = REPO_ROOT / "scripts" / "launch_training_sagemaker.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "launch_training_sagemaker", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MINIMAL_YAML = textwrap.dedent("""
    dataset:
      include_mermaid: false
      coralnet_sources_csv: sources.csv
      label_rollup_spec_csv: rollups.csv
      included_labels_csv: included_labels.csv
      subsample:
        strategy: balanced
        total_annotations: 100
    training:
      epochs: 1
    mlflow:
      experiment_name: test
      model_name: T
    env:
      WEIGHTS_LOCATION: s3://x/weights.pt
""").lstrip()


class _Fixture:
    """Lazy fixture: a TemporaryDirectory with a valid config dir."""
    def __init__(self):
        self._td = TemporaryDirectory()
        self.path = Path(self._td.name)
        (self.path / "training_config.yaml").write_text(MINIMAL_YAML)
        (self.path / "sources.csv").write_text("id\n1\n")
        (self.path / "rollups.csv").write_text(
            "from_ba_id,from_gf_id,to_ba_id,to_gf_id\n")
        (self.path / "included_labels.csv").write_text("ba_id,gf_id\n")

    def cleanup(self):
        self._td.cleanup()


def _base_args(cfg_dir: str) -> list[str]:
    return [
        "--config-dir", cfg_dir,
        "--mlflow-tracking-uri", "arn:aws:sagemaker:us-east-1:1:mlflow-app/A",
        "--role-arn", "arn:aws:iam::1:role/MermaidTrainer",
        "--ecr-image-uri", "1.dkr.ecr.us-east-1.amazonaws.com/training:latest",
        "--staging-bucket", "my-staging-bucket",
    ]


class DryRunTest(unittest.TestCase):

    def test_dry_run_does_not_call_sdk(self):
        module = _load_launcher()
        fixture = _Fixture()
        try:
            with patch.object(module, "Estimator") as Estimator:
                with patch.object(module, "_upload_config_dir") as upload:
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        module.main(_base_args(str(fixture.path)) + [
                            "--dry-run",
                        ])
                    Estimator.assert_not_called()
                    upload.assert_not_called()
                    self.assertIn("DRY RUN", buf.getvalue())
                    self.assertIn(
                        "ml.m5.4xlarge", buf.getvalue())
        finally:
            fixture.cleanup()


class ValidationTest(unittest.TestCase):

    def test_missing_config_dir_errors_before_aws_calls(self):
        module = _load_launcher()
        with self.assertRaises(SystemExit):
            module.main([
                *_base_args("/path/does/not/exist"),
                "--dry-run",
            ])

    def test_invalid_yaml_errors_before_submit(self):
        module = _load_launcher()
        with TemporaryDirectory() as td:
            (Path(td) / "training_config.yaml").write_text(
                "this: is\nnot: valid\nbecause: missing required\n")
            with self.assertRaises(SystemExit):
                module.main(_base_args(td) + ["--dry-run"])


class SubmissionTest(unittest.TestCase):

    def test_submit_creates_estimator_with_defaults(self):
        module = _load_launcher()
        fixture = _Fixture()
        try:
            with patch.object(module, "Estimator") as Estimator:
                instance = Estimator.return_value
                instance.latest_training_job = MagicMock(name="x")
                with patch.object(module, "_upload_config_dir",
                                  return_value="s3://my-staging-bucket/runs/abc/config/"):
                    module.main(_base_args(str(fixture.path)))
                args, kwargs = Estimator.call_args
                self.assertEqual(kwargs["instance_type"], "ml.m5.4xlarge")
                self.assertEqual(kwargs["instance_count"], 1)
                self.assertEqual(kwargs["volume_size"], 200)
                self.assertEqual(kwargs["max_run"], 24 * 3600)
                self.assertIn(
                    "MLFLOW_TRACKING_SERVER",
                    kwargs["environment"],
                )
                self.assertEqual(
                    kwargs["environment"]["MLFLOW_TRACKING_SERVER"],
                    "arn:aws:sagemaker:us-east-1:1:mlflow-app/A",
                )
                instance.fit.assert_called_once()
        finally:
            fixture.cleanup()

    def test_overrides_apply(self):
        module = _load_launcher()
        fixture = _Fixture()
        try:
            with patch.object(module, "Estimator") as Estimator:
                with patch.object(module, "_upload_config_dir",
                                  return_value="s3://b/runs/r/config/"):
                    module.main(_base_args(str(fixture.path)) + [
                        "--instance-type", "ml.c5.9xlarge",
                        "--volume-size-gb", "500",
                        "--max-runtime-hours", "6",
                    ])
                _, kwargs = Estimator.call_args
                self.assertEqual(kwargs["instance_type"], "ml.c5.9xlarge")
                self.assertEqual(kwargs["volume_size"], 500)
                self.assertEqual(kwargs["max_run"], 6 * 3600)
        finally:
            fixture.cleanup()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6.3:** Run the tests to verify they fail.

```bash
cd tests && python -m unittest sagemaker_launcher.test_launcher -v
```

Expected: failure — `scripts/launch_training_sagemaker.py` does not exist.

- [ ] **Step 6.4:** Implement the launcher.

Create `scripts/launch_training_sagemaker.py`:

```python
"""Launch a mermaid-classifier training run as a SageMaker TrainingJob.

Wraps the SageMaker Python SDK's Estimator. The user supplies a local
config directory containing training_config.yaml plus sibling CSVs.
The launcher:

  1. Validates the YAML against the Pydantic schema.
  2. Uploads the directory to a run-scoped S3 prefix.
  3. Builds an Estimator pointing at the ECR image.
  4. Calls fit(wait=True, logs="All") so CloudWatch logs stream live.

Example
-------
python scripts/launch_training_sagemaker.py \\
    --config-dir sagemaker/configs/my-run \\
    --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:ACCT:mlflow-app/APP \\
    --role-arn arn:aws:iam::ACCT:role/MermaidTrainer \\
    --ecr-image-uri ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-training:latest \\
    --staging-bucket my-staging-bucket
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# These imports are heavy; they happen at module load. The launcher
# always needs them (unlike the entrypoint, which sequences imports).
import boto3
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.session import Session


log = logging.getLogger("launch_training_sagemaker")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _validate_config_dir(config_dir: Path) -> None:
    if not config_dir.is_dir():
        log.error("Config dir does not exist: %s", config_dir)
        sys.exit(2)
    yaml_path = config_dir / "training_config.yaml"
    if not yaml_path.is_file():
        log.error("Missing training_config.yaml in %s", config_dir)
        sys.exit(2)

    # Schema validation. Don't suppress the Pydantic ValidationError --
    # its message includes the field path, which is exactly what the
    # user needs.
    from mermaid_classifier.sagemaker.config import TrainingRunConfig
    config = TrainingRunConfig.from_yaml_path(yaml_path)

    # Verify referenced CSVs exist next to the YAML.
    referenced = [
        config.dataset.coralnet_sources_csv,
        config.dataset.label_rollup_spec_csv,
        config.dataset.included_labels_csv,
        config.dataset.excluded_labels_csv,
    ]
    for filename in referenced:
        if filename is None:
            continue
        candidate = config_dir / filename
        if not candidate.is_file():
            log.error(
                "Referenced CSV missing: %s (expected at %s)",
                filename, candidate,
            )
            sys.exit(2)


def _make_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _upload_config_dir(
    *,
    config_dir: Path,
    staging_bucket: str,
    run_id: str,
    session: Session,
) -> str:
    """Upload the config dir to s3://<bucket>/runs/<run_id>/config/.

    Returns the S3 URI prefix (with trailing slash) the Estimator should
    mount as its 'config' channel.
    """
    key_prefix = f"runs/{run_id}/config"
    log.info(
        "Uploading %s to s3://%s/%s/",
        config_dir, staging_bucket, key_prefix,
    )
    session.upload_data(
        path=str(config_dir),
        bucket=staging_bucket,
        key_prefix=key_prefix,
    )
    return f"s3://{staging_bucket}/{key_prefix}/"


def _build_environment(args) -> dict:
    return {
        "MLFLOW_TRACKING_SERVER": args.mlflow_tracking_uri,
        "AWS_DEFAULT_REGION": args.region,
        "FEATURE_CACHE_DIR": "/tmp/feature_cache",
        "SPACER_EXTRACTORS_CACHE_DIR": "/tmp/spacer_extractors",
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
    }


def _print_dry_run(args, environment) -> None:
    print("=" * 60)
    print("DRY RUN -- not submitting")
    print("=" * 60)
    print(f"config_dir:           {args.config_dir}")
    print(f"staging_bucket:       {args.staging_bucket}")
    print(f"role_arn:             {args.role_arn}")
    print(f"ecr_image_uri:        {args.ecr_image_uri}")
    print(f"region:               {args.region}")
    print(f"instance_type:        {args.instance_type}")
    print(f"instance_count:       {args.instance_count}")
    print(f"volume_size_gb:       {args.volume_size_gb}")
    print(f"max_runtime_hours:    {args.max_runtime_hours}")
    print(f"use_spot:             {args.use_spot}")
    print(f"job_name_prefix:      {args.job_name_prefix}")
    print("environment:")
    for k, v in sorted(environment.items()):
        print(f"  {k}={v}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--ecr-image-uri", required=True)
    parser.add_argument("--staging-bucket", required=True,
                        help="S3 bucket for run-scoped config uploads "
                             "and SageMaker output artifacts.")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="ml.m5.4xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--volume-size-gb", type=int, default=200)
    parser.add_argument("--max-runtime-hours", type=int, default=24)
    parser.add_argument("--use-spot", action="store_true")
    parser.add_argument("--job-name-prefix",
                        default="mermaid-classifier")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate, print the planned config, "
                             "skip upload/submit.")
    args = parser.parse_args(argv)

    args.config_dir = args.config_dir.resolve()
    _validate_config_dir(args.config_dir)

    environment = _build_environment(args)

    if args.dry_run:
        _print_dry_run(args, environment)
        return

    boto_session = boto3.Session(region_name=args.region)
    sm_session = Session(boto_session=boto_session)

    run_id = _make_run_id(args.job_name_prefix)
    config_s3 = _upload_config_dir(
        config_dir=args.config_dir,
        staging_bucket=args.staging_bucket,
        run_id=run_id,
        session=sm_session,
    )

    output_path = f"s3://{args.staging_bucket}/runs/{run_id}/output/"
    cw_url = (
        f"https://{args.region}.console.aws.amazon.com/cloudwatch/home"
        f"?region={args.region}#logsV2:log-groups/log-group/"
        f"$252Faws$252Fsagemaker$252FTrainingJobs"
        f"/log-events/{run_id}"
    )
    log.info("Run ID:          %s", run_id)
    log.info("Output S3:       %s", output_path)
    log.info("CloudWatch:      %s", cw_url)

    estimator_kwargs = dict(
        image_uri=args.ecr_image_uri,
        role=args.role_arn,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        volume_size=args.volume_size_gb,
        max_run=args.max_runtime_hours * 3600,
        output_path=output_path,
        environment=environment,
        sagemaker_session=sm_session,
        base_job_name=args.job_name_prefix,
    )
    if args.use_spot:
        estimator_kwargs["use_spot_instances"] = True
        estimator_kwargs["max_wait"] = (
            args.max_runtime_hours * 3600 + 3600)

    estimator = Estimator(**estimator_kwargs)

    inputs = {
        "config": TrainingInput(
            s3_data=config_s3,
            input_mode="File",
        ),
    }

    log.info("Submitting TrainingJob...")
    estimator.fit(inputs=inputs, wait=True, logs="All",
                  job_name=run_id)
    log.info("Job %s reached terminal state.", run_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.5:** Run the tests to verify they pass.

```bash
cd tests && python -m unittest sagemaker_launcher.test_launcher -v
```

Expected: all four tests pass.

- [ ] **Step 6.6:** Manual dry-run sanity check.

```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier
python scripts/launch_training_sagemaker.py \
    --config-dir sagemaker/configs/example \
    --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:000000000000:mlflow-app/PLACEHOLDER \
    --role-arn arn:aws:iam::000000000000:role/PLACEHOLDER \
    --ecr-image-uri 000000000000.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-training:latest \
    --staging-bucket placeholder-bucket \
    --dry-run
```

Expected: prints `DRY RUN -- not submitting`, the resolved defaults (`ml.m5.4xlarge`, volume 200, max 24h, spot off), the full environment dict, exits 0. No AWS calls were made (verifiable since the role ARN is fake).

- [ ] **Step 6.7:** Commit.

```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier
git add pyproject.toml scripts/launch_training_sagemaker.py tests/sagemaker_launcher/test_launcher.py
git commit -m "$(cat <<'EOF'
feat(sagemaker): add training launcher using SageMaker SDK Estimator

Validates the local config dir against the schema, uploads to a
run-scoped S3 prefix, builds an Estimator with the agreed CPU-PyTorch
defaults (ml.m5.4xlarge / 200GB / 24h / on-demand), and streams
CloudWatch logs via fit(wait=True, logs="All"). --dry-run validates
without submitting. Adds the sagemaker SDK as a new optional extra
so it's only pulled when launching, not when running tests in CI.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Setup runbook

**Files:**
- Create: `docs/training_at_scale.md`

### Steps

- [ ] **Step 7.1:** Write the runbook.

Create `docs/training_at_scale.md`:

```markdown
# Training at Scale (SageMaker TrainingJob)

Run the mermaid-classifier training pipeline as a SageMaker
TrainingJob. Use this for runs that don't fit on a laptop -- e.g.,
the 192-source production run.

For day-to-day local iteration use `scripts/classifier_train.py`. The
SageMaker path is for production / large runs only.

## One-time setup

You need to do these once per AWS account.

### 1. ECR repo

Create the container registry that will hold the training image:

    aws ecr create-repository \
        --repository-name mermaid-classifier-training \
        --region us-east-1 \
        --profile wcs

### 2. IAM role for training jobs

Create an IAM role (e.g. `MermaidClassifierTrainerRole`) that SageMaker
will assume during the job. Trust policy:

    {
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }

Inline permissions:

- `s3:GetObject` + `s3:ListBucket` on the training-data bucket
  (`coral-reef-training` and any others the YAML's `env` overrides).
- `s3:GetObject` + `s3:PutObject` + `s3:ListBucket` on the staging
  bucket you pass to `--staging-bucket` (for config upload + output).
- `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`,
  `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer` on `*` (image pull).
- `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`
  on `/aws/sagemaker/TrainingJobs*`.
- `sagemaker-mlflow:*` on the MLflow tracking server ARN, OR the
  equivalent IAM allow for whatever MLflow auth you use.

### 3. Build and push the image

From `mermaid-classifier/`:

    ACCOUNT_ID=<your-account-id>
    REGION=us-east-1
    IMAGE=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/mermaid-classifier-training:latest

    aws ecr get-login-password --region ${REGION} --profile wcs \
        | docker login --username AWS --password-stdin \
            ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

    docker buildx build --platform linux/amd64 \
        -t ${IMAGE} \
        -f docker/training/Dockerfile .

    docker push ${IMAGE}

Re-run this every time you change code. The image tag is the unit of
versioning -- promote a build by tagging it (`:prod`, `:2026-05-15`,
etc.) and pushing.

### 4. Verify locally first

Before paying SageMaker:

    bash docker/training/local_smoke.sh

This builds the image, runs the example config with `-v` mounts, and
asserts that the container reaches the `build_options` stage. Without
real AWS creds it cannot complete training, but it catches packaging
and entrypoint mistakes.

## Launching a training run

### 5. Author a config dir

Copy `sagemaker/configs/example/` to a new name:

    cp -r sagemaker/configs/example sagemaker/configs/my-run

Edit `training_config.yaml` to point at your sources/rollups/included-labels
CSVs (paths are resolved as siblings of the YAML, so just `sources.csv`
not `/abs/path/sources.csv`). Replace the placeholder CSVs with the
real ones for your experiment.

### 6. SageMaker smoke run

Before the production run, do a 1-source / 1-epoch run on the
cheapest instance to prove the whole pipeline works end-to-end:

    python scripts/launch_training_sagemaker.py \
        --config-dir sagemaker/configs/example \
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:ACCT:mlflow-app/APP \
        --role-arn arn:aws:iam::ACCT:role/MermaidClassifierTrainerRole \
        --ecr-image-uri ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-training:latest \
        --staging-bucket my-staging-bucket \
        --instance-type ml.m5.2xlarge

Expect: ~10-20 minutes wall-clock, ~$0.15, an MLflow run with metrics
+ artifacts, CloudWatch logs streaming live to your terminal.

### 7. Production run

For the full 192-source run, the defaults are tuned to be a sensible
starting point:

    python scripts/launch_training_sagemaker.py \
        --config-dir sagemaker/configs/my-run \
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:ACCT:mlflow-app/APP \
        --role-arn arn:aws:iam::ACCT:role/MermaidClassifierTrainerRole \
        --ecr-image-uri ACCT.dkr.ecr.us-east-1.amazonaws.com/mermaid-classifier-training:latest \
        --staging-bucket my-staging-bucket

This uses:

- `ml.m5.4xlarge` -- 16 vCPU, 64 GB RAM, ~$0.92/hr. PyTorch on CPU
  parallelizes BLAS across all cores; 64 GB is comfortable for the
  ~3-4M subsampled annotations at 192 sources.
- 200 GB volume.
- 24h max runtime (hard kill).

`--dry-run` prints the planned Estimator config and exits without
submitting.

## Tuning for cost vs speed

| Instance | vCPU | RAM | $/hr | When to use |
|----------|------|-----|------|-------------|
| `ml.m5.2xlarge` | 8 | 32 GB | ~$0.46 | 20-source runs, smoke tests |
| `ml.m5.4xlarge` (default) | 16 | 64 GB | ~$0.92 | 192-source runs |
| `ml.c5.9xlarge` | 36 | 72 GB | ~$1.94 | "Fast mode" -- roughly halves wall-clock at ~2x hourly. Often net-cheaper per run if the job otherwise hits the 24h limit. |

Pricing is approximate (us-east-1 on-demand, 2026). Check the AWS
SageMaker pricing page for current numbers.

`--use-spot` enables spot pricing (~70% discount) but the trainer has
no checkpointing, so a spot interruption restarts from epoch zero.
Use only for runs you're OK retrying.

## Debugging a failed job

1. Open the CloudWatch URL printed by the launcher. The container's
   first ~50 lines dump Python version, package versions, the loaded
   YAML, the resolved env vars (secrets redacted), and the
   `/opt/ml/input/data/` listing.
2. Scan for `[stage:XX] FAIL` -- this tells you which phase failed
   (`load_config`, `apply_env`, `build_options`, `runner_run`).
3. The full traceback is logged immediately after the FAIL marker.
4. The launcher writes the exact config that ran to
   `s3://<staging-bucket>/runs/<run-id>/config/`, so you can re-run
   the smoke test against that exact bundle:

       aws s3 sync s3://<staging-bucket>/runs/<run-id>/config/ /tmp/repro/
       SAGEMAKER_CONFIG_DIR=/tmp/repro python scripts/sagemaker_train_entrypoint.py

   (This skips Docker; runs the entrypoint directly against your
   local pyspacer install.)

## MLflow

The container needs to reach the MLflow tracking server. The server
URI flows through the launcher's `--mlflow-tracking-uri` flag into the
`MLFLOW_TRACKING_SERVER` env var on the Estimator. The training run
appears under whatever `experiment_name` is set in the YAML.

If MLflow auth is via the SageMaker IAM integration (server URI is an
`arn:aws:sagemaker:...:mlflow-app/...`), the training role needs
`sagemaker-mlflow:*` on that ARN. Without it the runner will hang or
fail at the first `mlflow.log_param` call.
```

- [ ] **Step 7.2:** Commit.

```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier
git add docs/training_at_scale.md
git commit -m "$(cat <<'EOF'
docs: add SageMaker training launcher runbook

One-time AWS setup (ECR + IAM role + bucket policies), build/push
recipe, smoke-run sequence (local Docker -> 1-source SageMaker ->
full production), instance type tradeoff table for cost-vs-speed
tuning, and a debugging guide that exploits the per-stage logging
and run-scoped config archive.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Verification (full plan)

After all tasks complete, run end-to-end:

```bash
cd /Users/gregn/Documents/wcs/mermaid-classifier

# 1. All unit tests green
cd tests && python -m unittest sagemaker -v && cd ..

# 2. Launcher dry-run prints sensible defaults and exits 0
python scripts/launch_training_sagemaker.py \
    --config-dir sagemaker/configs/example \
    --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:000000000000:mlflow-app/X \
    --role-arn arn:aws:iam::000000000000:role/X \
    --ecr-image-uri 000000000000.dkr.ecr.us-east-1.amazonaws.com/X:latest \
    --staging-bucket placeholder \
    --dry-run

# 3. Local Docker smoke (requires Docker)
bash docker/training/local_smoke.sh

# 4. Manual: SageMaker smoke + production runs per docs/training_at_scale.md
```

## Self-review notes

This plan was reviewed against the spec at
`docs/superpowers/specs/2026-05-15-sagemaker-training-launcher-design.md`.
All design-spec components are covered by a task:

- Spec §Components → Tasks 1-7 (every file listed has an owning task).
- Spec §Data flow → Task 1 (schema + `env` block) + Task 3 (entrypoint
  applies env before importing pyspacer) + Task 6 (launcher mounts a
  single `config` channel).
- Spec §Resource defaults → Task 6 argparse defaults match the table.
- Spec §PyTorch-specific notes → Task 4 Dockerfile installs CPU torch
  first; runbook (Task 7) documents the choice and the alternatives.
- Spec §Error handling → Task 3 (stage markers, first-line dump,
  try/except around `runner.run`) + Task 6 (pre-submit validation,
  `--dry-run`).
- Spec §Testing → Tasks 1, 3, 6 carry unit tests; Task 5 is the local
  Docker smoke; runbook (Task 7) documents the SageMaker smoke recipe.

The three "Open questions" in the spec are addressed in the plan:

- Pydantic representation of subsample / weighting → Task 1 uses
  `Literal` for subsample's strategy discriminator and plain string
  for weighting's strategy (validated downstream by the
  weighting registry, as the existing dataclass does).
- Example CSVs → Task 2 commits header-only CSVs as fixtures.
- Upload manifest → not implemented; out of scope. The launcher
  already prints the S3 prefix, which is enough to inspect manually.
