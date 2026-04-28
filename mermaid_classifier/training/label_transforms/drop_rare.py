"""DropRareClasses — drop training classes whose count is below a threshold.

The data-pipeline replacement for the old ``rare_policy='drop'`` setting
on ``SampleWeightingOptions``. Where the old behaviour zeroed the loss
weight on rare classes (leaving them in the validation set with recall
mathematically pinned to 0), this transform removes the rare classes
from the data entirely — train, ref, and val. The trained classifier
never sees those classes, so its ``classes_`` array doesn't include
them, and metrics aren't dragged down by zero-recall ghost classes.
"""
from __future__ import annotations

from mermaid_classifier.training.label_transforms.base import (
    DROP,
    BALibrary,
    GFLibrary,
    Transform,
    TransformPlan,
)
from mermaid_classifier.training.label_transforms.registry import (
    register_transform,
)


@register_transform("drop_rare")
class DropRareClasses(Transform):
    """Drop classes with training count < ``min_count``."""

    def __init__(self, min_count: int = 10):
        if int(min_count) < 1:
            raise ValueError(
                f"min_count must be >= 1, got {min_count!r}"
            )
        self.min_count = int(min_count)

    def plan(
        self,
        train_class_counts: dict[str, int],
        ba_library: BALibrary,
        gf_library: GFLibrary,
    ) -> TransformPlan:
        actions: dict[str, str] = {}
        for bagf, count in train_class_counts.items():
            if count < self.min_count:
                actions[bagf] = DROP
        return TransformPlan(
            transform_name=self.name,
            actions=actions,
            train_counts=dict(train_class_counts),
        )
