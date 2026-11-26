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

import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from segdata_spec.manifest import FeatureSetPin, Pin, Version

# Mirrors the MLflow registered-model name regex enforced by the
# SageMaker MLflow registry. Validating at config load fails the job
# cheaply (before any training work) instead of at the very end after
# `mlflow.sklearn.log_model(registered_model_name=...)` has spent hours
# of compute.
_MLFLOW_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,56}$")


class SubsampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Field shape only — all validation (allowed strategies, numeric
    # bounds, cross-field rules) is delegated to SubsampleOptions so there
    # is a single source of truth. SubsampleOptions is imported lazily in
    # the validator to keep this module free of pyspacer-subtree imports at
    # load time.
    strategy: str = "stratified"
    total_annotations: int | None = None
    min_per_class: int = 0

    @model_validator(mode="after")
    def _validate_via_options(self) -> SubsampleConfig:
        from mermaid_classifier.training.subsample.options import SubsampleOptions

        # Constructing SubsampleOptions runs its __post_init__ validation.
        # Re-raise any ValueError so pydantic surfaces it as a ValidationError.
        SubsampleOptions(
            strategy=self.strategy,
            total_annotations=self.total_annotations,
            min_per_class=self.min_per_class,
        )
        return self


class WeightingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    weight_ratio_cap: float | None = None

    @model_validator(mode="after")
    def _validate_via_options(self) -> WeightingConfig:
        from mermaid_classifier.training.sample_weighting import SampleWeightingOptions

        SampleWeightingOptions(enabled=self.enabled, weight_ratio_cap=self.weight_ratio_cap)
        return self


# A manifest pin is the hex sha256 of the published manifest.json bytes.
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

# The taxonomy every `labels: mermaid` entry resolves growth-form names against.
TAXONOMY_NAME = "mermaid-benthic"


class FeatureSetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str
    manifest_sha256: Sha256Hex


class DatasetPinConfig(BaseModel):
    """One pinned dataset version, the feature set read for it, and how its labels map."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: Version
    manifest_sha256: Sha256Hex
    features: FeatureSetConfig
    labels: Literal["coralnet", "mermaid"]
    # One dataset group per line; absent selects every group of the version.
    groups_csv: str | None = None

    def pin(self) -> Pin:
        return Pin(name=self.name, version=self.version, manifest_sha256=self.manifest_sha256)

    def feature_set_pin(self) -> FeatureSetPin:
        return FeatureSetPin(
            dataset=self.pin(),
            family=self.features.family,
            manifest_sha256=self.features.manifest_sha256,
        )

    def groups_csv_path(self, base: Path) -> Path | None:
        return None if self.groups_csv is None else base / self.groups_csv


class TaxonomyPinConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Version
    manifest_sha256: Sha256Hex

    def pin(self) -> Pin:
        return Pin(name=TAXONOMY_NAME, version=self.version, manifest_sha256=self.manifest_sha256)


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: list[DatasetPinConfig] = Field(min_length=1)
    taxonomy: TaxonomyPinConfig | None = None
    # Relative paths are resolved as siblings of the YAML file by
    # *_path() helpers. The launcher uploads the YAML and CSVs together
    # to the same S3 prefix so the same resolution works in the
    # container.
    # Required: a run must say which extractor produced the feature vectors
    # it trains on. Absent, the released artifact could not record it either.
    feature_extractor_weights: str
    drop_growthforms: bool = False
    label_rollup_spec_csv: str | None = None
    included_labels_csv: str | None = None
    excluded_labels_csv: str | None = None
    excluded_images_csv: str | None = None
    ref_val_ratios: tuple[float, float] = (0.1, 0.1)
    subsample: SubsampleConfig | None = None
    weighting: WeightingConfig | None = None

    @model_validator(mode="after")
    def _check_datasets(self) -> DatasetConfig:
        names = [entry.name for entry in self.datasets]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"datasets names a dataset more than once: {repeated}")
        for kind in ("coralnet", "mermaid"):
            of_kind = [entry.name for entry in self.datasets if entry.labels == kind]
            if len(of_kind) > 1:
                raise ValueError(
                    f"datasets has more than one `labels: {kind}` entry ({of_kind}); "
                    "per-source reports key on the label kind, so two datasets of one "
                    "kind would merge"
                )
        if self.taxonomy is None and any(entry.labels == "mermaid" for entry in self.datasets):
            raise ValueError(
                "a `labels: mermaid` dataset needs a `taxonomy:` pin to resolve growth forms"
            )
        return self

    def label_rollup_spec_csv_path(self, base: Path) -> Path | None:
        return None if self.label_rollup_spec_csv is None else base / self.label_rollup_spec_csv

    def included_labels_csv_path(self, base: Path) -> Path | None:
        return None if self.included_labels_csv is None else base / self.included_labels_csv

    def excluded_labels_csv_path(self, base: Path) -> Path | None:
        return None if self.excluded_labels_csv is None else base / self.excluded_labels_csv

    def excluded_images_csv_path(self, base: Path) -> Path | None:
        return None if self.excluded_images_csv is None else base / self.excluded_images_csv


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The MLP architecture and learning rate are fixed at the production
    # values baked into MermaidTrainer (hidden_layer_sizes=(500, 300,
    # 100) @ learning_rate_init=1e-4; see docs/research/hidden-layer-experiments.md),
    # so they are intentionally not exposed here.
    epochs: int = 10
    early_stopping_patience: int | None = None


class MLflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str | None = None
    model_name: str | None = None
    annotations_to_log: str | None = None

    @field_validator("model_name")
    @classmethod
    def _model_name_matches_mlflow_regex(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not _MLFLOW_MODEL_NAME_RE.fullmatch(v):
            bad = sorted({c for c in v if not (c.isalnum() or c == "-")})
            disallowed = f" Disallowed characters in value: {bad!r}." if bad else ""
            raise ValueError(
                f"model_name {v!r} is not a legal MLflow registered-model "
                f"name. Required pattern: "
                f"{_MLFLOW_MODEL_NAME_RE.pattern} (alphanumerics and "
                f"hyphens only; must start and end with an alphanumeric; "
                f"max 57 characters)."
                f"{disallowed} "
                f"Tip: replace underscores or dots with hyphens."
            )
        return v


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
    def from_yaml_path(cls, path: str | Path) -> TrainingRunConfig:
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
        from mermaid_classifier.pyspacer.options import (
            DatasetOptions,
            MLflowOptions,
            PinnedDataset,
            TrainingOptions,
        )
        from mermaid_classifier.training.sample_weighting import (
            SampleWeightingOptions,
        )
        from mermaid_classifier.training.subsample import SubsampleOptions

        d = self.dataset

        subsample = None
        if d.subsample is not None:
            subsample = SubsampleOptions(
                strategy=d.subsample.strategy,
                total_annotations=d.subsample.total_annotations,
                min_per_class=d.subsample.min_per_class,
            )

        weighting = None
        if d.weighting is not None:
            weighting = SampleWeightingOptions(
                enabled=d.weighting.enabled,
                weight_ratio_cap=d.weighting.weight_ratio_cap,
            )

        def _resolve(p: Path | None) -> str | None:
            return None if p is None else str(p)

        datasets = [
            PinnedDataset(
                pin=entry.pin(),
                features=entry.feature_set_pin(),
                labels=entry.labels,
                groups_csv=_resolve(entry.groups_csv_path(config_dir)),
            )
            for entry in d.datasets
        ]

        dataset_options = DatasetOptions(
            datasets=datasets,
            taxonomy=None if d.taxonomy is None else d.taxonomy.pin(),
            feature_extractor_weights=d.feature_extractor_weights,
            drop_growthforms=d.drop_growthforms,
            label_rollup_spec_csv=_resolve(d.label_rollup_spec_csv_path(config_dir)),  # pyright: ignore[reportArgumentType]  # DatasetOptions accepts str|None
            included_labels_csv=_resolve(d.included_labels_csv_path(config_dir)),  # pyright: ignore[reportArgumentType]  # DatasetOptions accepts str|None
            excluded_labels_csv=_resolve(d.excluded_labels_csv_path(config_dir)),  # pyright: ignore[reportArgumentType]  # DatasetOptions accepts str|None
            excluded_images_csv=_resolve(d.excluded_images_csv_path(config_dir)),  # pyright: ignore[reportArgumentType]  # DatasetOptions accepts str|None
            ref_val_ratios=tuple(d.ref_val_ratios),  # pyright: ignore[reportArgumentType]  # runtime tuple[float,float] matches DatasetOptions
            subsample=subsample,
            weighting=weighting,
        )

        t = self.training
        training_options = TrainingOptions(
            epochs=t.epochs,
            early_stopping_patience=t.early_stopping_patience,
        )

        m = self.mlflow
        mlflow_options = MLflowOptions(
            experiment_name=m.experiment_name,
            model_name=m.model_name,
            extra_annotations_to_log=m.annotations_to_log,
        )

        return dataset_options, training_options, mlflow_options
