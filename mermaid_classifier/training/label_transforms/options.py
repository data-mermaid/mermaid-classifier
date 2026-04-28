"""LabelTransformsOptions dataclass with eager validation.

Mirrors ``SampleWeightingOptions`` in shape: a simple dataclass with
``__post_init__`` validation and a ``to_log_dict`` method for MLflow.

The pipeline is a *list* of stage specs because transforms compose:
e.g. drop very-rare first, then merge moderately-rare into BA parents.
List ordering is meaningful — earlier stages run first, and later
stages see the transformed counts.
"""
from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class TransformSpec:
    """One stage in a label-transforms pipeline.

    ``name`` looks up a ``Transform`` subclass in ``TRANSFORM_REGISTRY``;
    ``params`` is forwarded as kwargs to that class's constructor.
    """

    name: str
    params: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"TransformSpec.name must be a non-empty string,"
                f" got {self.name!r}"
            )
        if not isinstance(self.params, dict):
            raise ValueError(
                f"TransformSpec.params must be a dict,"
                f" got {type(self.params).__name__}"
            )


@dataclasses.dataclass
class LabelTransformsOptions:
    """Configuration for the label-transforms pipeline.

    Fields:
        enabled  -- master switch. When False, the pipeline is skipped
                    entirely (no-op) regardless of ``pipeline``.
        pipeline -- ordered list of TransformSpec. Each spec's ``name``
                    must resolve in ``TRANSFORM_REGISTRY`` (validated
                    lazily on first use, since importing the registry
                    here would cycle).
    """

    enabled: bool = False
    pipeline: list[TransformSpec] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.enabled and not self.pipeline:
            raise ValueError(
                "LabelTransformsOptions.enabled=True requires at least one"
                " TransformSpec in pipeline."
            )
        # Coerce dict-shaped pipeline entries (common when the options
        # are constructed from CLI parsing or YAML) into TransformSpec.
        coerced: list[TransformSpec] = []
        for i, entry in enumerate(self.pipeline):
            if isinstance(entry, TransformSpec):
                coerced.append(entry)
            elif isinstance(entry, dict):
                coerced.append(TransformSpec(**entry))
            elif (
                isinstance(entry, tuple)
                and len(entry) == 2
                and isinstance(entry[0], str)
                and isinstance(entry[1], dict)
            ):
                coerced.append(TransformSpec(name=entry[0], params=entry[1]))
            else:
                raise ValueError(
                    f"pipeline[{i}] must be a TransformSpec, dict, or"
                    f" (name, params) tuple, got {type(entry).__name__}"
                )
        self.pipeline = coerced

    def to_log_dict(self) -> dict[str, object]:
        """Flat dict suitable for ``mlflow.log_params``.

        Emits one ``label_transforms/N/...`` group per pipeline stage,
        plus ``label_transforms/enabled`` and ``label_transforms/n_stages``.
        Per-stage params are flattened with their parameter names.
        """
        out: dict[str, object] = {
            "label_transforms/enabled": self.enabled,
            "label_transforms/n_stages": len(self.pipeline),
        }
        for i, spec in enumerate(self.pipeline):
            out[f"label_transforms/{i}/name"] = spec.name
            for k, v in spec.params.items():
                out[f"label_transforms/{i}/{k}"] = v
        return out
