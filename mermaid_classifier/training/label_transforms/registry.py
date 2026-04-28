"""Transform registry and pipeline entry point.

Registry pattern mirrors ``sample_weighting.registry``: each concrete
``Transform`` module imports ``register_transform`` and decorates its
class. Module-level imports of the concrete transforms live at the
bottom of this file so the decorators run on first import of the
registry.
"""
from __future__ import annotations

from typing import Type

from spacer.task_utils import TrainingTaskLabels

from mermaid_classifier.training.label_transforms.base import (
    BALibrary,
    GFLibrary,
    Transform,
    TransformPlan,
)
from mermaid_classifier.training.label_transforms.options import (
    LabelTransformsOptions,
)


TRANSFORM_REGISTRY: dict[str, Type[Transform]] = {}


def register_transform(name: str):
    """Decorator that registers a Transform subclass under a string name."""

    def _decorator(cls: Type[Transform]) -> Type[Transform]:
        if name in TRANSFORM_REGISTRY:
            raise ValueError(
                f"Transform name {name!r} is already registered to"
                f" {TRANSFORM_REGISTRY[name].__name__}."
            )
        cls.name = name
        TRANSFORM_REGISTRY[name] = cls
        return cls

    return _decorator


# Imports at the bottom to avoid circular dependency: each transform
# module imports ``register_transform`` from this module.
from mermaid_classifier.training.label_transforms import (  # noqa: E402,F401
    drop_rare,
    merge_rare,
)


def apply_label_transforms(
    labels: TrainingTaskLabels,
    options: LabelTransformsOptions,
    ba_library: BALibrary,
    gf_library: GFLibrary,
) -> tuple[TrainingTaskLabels, list[TransformPlan]]:
    """Run the configured transform pipeline against ``labels``.

    Each stage sees the *current* training-split counts (after prior
    stages have rewritten or dropped classes), so a pipeline like
    ``[drop_rare, merge_rare]`` evaluates merge against the post-drop
    label space.

    Returns the rewritten ``TrainingTaskLabels`` and one
    ``TransformPlan`` per stage. Returns the input unchanged (and
    empty plan list) when ``options.enabled`` is False.
    """
    if not options.enabled or not options.pipeline:
        return labels, []

    plans: list[TransformPlan] = []
    for spec in options.pipeline:
        if spec.name not in TRANSFORM_REGISTRY:
            available = sorted(TRANSFORM_REGISTRY)
            raise ValueError(
                f"Unknown label transform {spec.name!r}."
                f" Available: {available!r}"
            )
        cls = TRANSFORM_REGISTRY[spec.name]
        transform = cls(**spec.params)

        train_counts = dict(labels.train.label_count_per_class)
        plan = transform.plan(train_counts, ba_library, gf_library)
        labels = transform.apply(labels, plan)
        plans.append(plan)

    return labels, plans
