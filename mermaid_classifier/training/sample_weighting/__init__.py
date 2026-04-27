"""
Sample weighting for the TorchMLP classifier.

Provides hierarchy- and growth-form-aware per-class weight tensors that
can be passed to ``nn.CrossEntropyLoss(weight=...)`` to mitigate label
imbalance at training time.

Public API:
    Strategy                     -- abstract base class
    SampleWeightingOptions       -- configuration dataclass
    STRATEGY_REGISTRY            -- name -> class lookup
    compute_class_weights        -- convenience entry point used by trainer
    rare_action_for_class        -- inspector for MLflow logging
"""

from mermaid_classifier.training.sample_weighting.base import (
    Strategy,
    rare_action_for_class,
)
from mermaid_classifier.training.sample_weighting.options import (
    SampleWeightingOptions,
)
from mermaid_classifier.training.sample_weighting.registry import (
    STRATEGY_REGISTRY,
    compute_class_weights,
)

__all__ = [
    "Strategy",
    "SampleWeightingOptions",
    "STRATEGY_REGISTRY",
    "compute_class_weights",
    "rare_action_for_class",
]
