"""Pydantic schema for the SageMaker launcher YAML.

This module is the launcher-side counterpart to `config.py`. The
launcher cares about the `job:` block plus an optional `processing:`
or `training:` block. The container entrypoint cares about everything
else (`dataset:`, `mlflow:`, the in-container `training:` block in
config.py). Both schemas read the same YAML file independently with
`extra="ignore"` on the non-overlapping keys.

See `mermaid-api/iac/sagemaker-launcher-convention.md` for the schema
authority.
"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class JobConfig(BaseModel):
    """Required `job:` block. Every run YAML must include this."""

    model_config = ConfigDict(extra="forbid")

    name_prefix: str
    image: str
    entrypoint: str
    instance_type: str
    instance_count: int = 1
    volume_gb: int = Field(gt=0)
    max_runtime_hours: int = Field(gt=0)
    use_spot: bool = False
    env: dict[str, str] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)


class ShardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items_from: str
    items_column: str | None = None
    workers: int = Field(gt=0)
    per_worker_arg: str


class ProcessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container_args: list[str] = Field(default_factory=list)
    shard: ShardConfig | None = None


class TrainingChannel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    s3_uri: str
    input_mode: Literal["File", "FastFile", "Pipe"] = "File"


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hyperparameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    channels: dict[str, TrainingChannel] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """Top-level launcher schema.

    `strict=True` mode (default) forbids unknown top-level keys.
    `strict=False` mode allows them (so a YAML can also carry the
    classifier-specific `dataset:` / `mlflow:` blocks for the container
    entrypoint to consume).
    """

    model_config = ConfigDict(extra="forbid")

    job: JobConfig
    processing: ProcessingConfig | None = None
    training: TrainingConfig | None = None


class _LooseRunConfig(BaseModel):
    """Permissive top-level schema used in `strict=False` mode."""

    model_config = ConfigDict(extra="ignore")

    job: JobConfig
    processing: ProcessingConfig | None = None
    training: TrainingConfig | None = None


def parse_run_config(
    yaml_text: str,
    *,
    kind: Literal["training", "processing"],
    strict: bool = True,
) -> RunConfig | _LooseRunConfig:
    """Parse a run YAML and validate it for the named kind.

    `kind` is informational only; we don't reject a `processing:` block
    in a training YAML or vice-versa here. The launcher script enforces
    that (each script consults `cfg.training` or `cfg.processing` only).
    """
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        from pydantic import ValidationError

        raise ValidationError.from_exception_data(
            "RunConfig", [{"type": "model_type", "loc": (), "input": data}]
        )
    model = RunConfig if strict else _LooseRunConfig
    return model.model_validate(data)
