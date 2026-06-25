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
        weight_ratio_cap
                      -- optional cap on the max:min ratio of the
                         per-class weights. When set, any weight above
                         ``min_weight * weight_ratio_cap`` is clamped
                         down to that ceiling, which bounds how much a
                         single class can dominate the loss. ``None``
                         (default) disables capping. Must be ``>= 1.0``;
                         a cap of 1.0 forces all weights equal.
    """

    enabled: bool = True
    weight_ratio_cap: float | None = None

    def __post_init__(self) -> None:
        if self.weight_ratio_cap is not None and self.weight_ratio_cap < 1.0:
            raise ValueError(
                f"weight_ratio_cap must be None or >= 1.0,"
                f" got {self.weight_ratio_cap!r}"
            )

    def to_log_dict(self) -> dict[str, object]:
        """Flat dict suitable for ``mlflow.log_params``."""
        return {
            "weighting/enabled": self.enabled,
            "weighting/weight_ratio_cap": self.weight_ratio_cap,
        }
