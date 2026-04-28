"""SampleWeightingOptions dataclass with eager validation."""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class SampleWeightingOptions:
    """Configuration for the sample-weighting layer.

    Fields:
        enabled       -- master switch. When False, the trainer skips
                         weight computation entirely (equivalent to
                         passing weighting=None).
        strategy      -- registry key, e.g. ``"tree_balanced_ba_flat_gf"``.
                         The corresponding Strategy class is instantiated
                         with ``alpha`` and the weighting settings.
        alpha         -- smoothing exponent in [0, 1].
                         alpha=0 uniform, alpha=1 inverse-frequency.
                         Each strategy's docstring documents how alpha is
                         applied for that specific algorithm.
        weight_ratio_cap
                      -- optional cap on the max:min ratio of the
                         per-class weights produced by the strategy.
                         When set, any weight above
                         ``min_weight * weight_ratio_cap`` is clamped
                         down to that ceiling, which bounds how much a
                         single class can dominate the loss. ``None``
                         (default) disables capping. Must be ``>= 1.0``;
                         a cap of 1.0 forces all weights equal.

    Note: rare-class drop/merge handling lives in
    ``mermaid_classifier.training.label_transforms`` and is configured
    via ``DatasetOptions.label_transforms_options``. This module is
    purely about loss-weight computation over whatever class set the
    label-transforms pipeline produces.
    """

    enabled: bool = True
    strategy: str = "tree_balanced_ba_flat_gf"
    alpha: float = 0.5
    weight_ratio_cap: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(
                f"alpha must be in [0, 1], got {self.alpha!r}"
            )
        if self.weight_ratio_cap is not None and self.weight_ratio_cap < 1.0:
            raise ValueError(
                f"weight_ratio_cap must be None or >= 1.0,"
                f" got {self.weight_ratio_cap!r}"
            )
        # Strategy name is validated by the registry on lookup; we don't
        # import the registry here to avoid a circular import at module
        # import time.

    def to_log_dict(self) -> dict[str, object]:
        """Flat dict suitable for ``mlflow.log_params``."""
        return {
            "weighting/enabled": self.enabled,
            "weighting/strategy": self.strategy,
            "weighting/alpha": self.alpha,
            "weighting/weight_ratio_cap": self.weight_ratio_cap,
        }
