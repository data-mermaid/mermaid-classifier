"""
Label transformations for training data.

A label transformation rewrites or drops per-class entries in the
``ImageLabels`` (train/ref/val) before they reach the classifier and
the metrics layer. Because train, ref, and val all flow through the
same pipeline, applying the transform once here keeps the three splits
consistent — and the trained classifier's ``classes_`` array reflects
exactly the post-transform label space, so inference produces only
valid post-transform labels with no extra plumbing.

Public API:
    Transform                    -- abstract base class
    TransformPlan                -- per-class decisions (returned by plan())
    LabelTransformsOptions       -- configuration dataclass
    TRANSFORM_REGISTRY           -- name -> class lookup
    apply_label_transforms       -- entry point used by TrainingDataset
"""

from mermaid_classifier.training.label_transforms.base import (
    Transform,
    TransformPlan,
)
from mermaid_classifier.training.label_transforms.options import (
    LabelTransformsOptions,
    TransformSpec,
)
from mermaid_classifier.training.label_transforms.registry import (
    TRANSFORM_REGISTRY,
    apply_label_transforms,
    register_transform,
)

__all__ = [
    "Transform",
    "TransformPlan",
    "TransformSpec",
    "LabelTransformsOptions",
    "TRANSFORM_REGISTRY",
    "apply_label_transforms",
    "register_transform",
]
