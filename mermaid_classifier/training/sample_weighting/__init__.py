"""
Sample weighting for the TorchMLP classifier.

Produces per-class weights (BA+GF combo string -> float) for
``nn.CrossEntropyLoss(weight=...)`` to mitigate label imbalance at
training time, using the effective-number-of-samples formulation
(Cui et al. 2019).

Public API:
    SampleWeightingOptions   -- configuration dataclass
    compute_class_weights    -- entry point used by the trainer
"""

from mermaid_classifier.training.sample_weighting.effective_number import (
    compute_class_weights,
)
from mermaid_classifier.training.sample_weighting.options import (
    SampleWeightingOptions,
)

__all__ = [
    "SampleWeightingOptions",
    "compute_class_weights",
]
