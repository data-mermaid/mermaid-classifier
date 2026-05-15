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
