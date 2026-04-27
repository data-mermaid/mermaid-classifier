"""SampleWeightingOptions dataclass with eager validation."""
from __future__ import annotations

import dataclasses
from typing import Literal


RarePolicy = Literal["drop", "keep", "merge"]
_VALID_POLICIES: tuple[str, ...] = ("drop", "keep", "merge")


@dataclasses.dataclass
class SampleWeightingOptions:
    """Configuration for the sample-weighting layer.

    Fields:
        enabled       -- master switch. When False, the trainer skips
                         weight computation entirely (equivalent to
                         passing weighting=None).
        strategy      -- registry key, e.g. ``"tree_balanced_ba_flat_gf"``.
                         The corresponding Strategy class is instantiated
                         with ``alpha`` and the rare-class settings.
        alpha         -- smoothing exponent in [0, 1].
                         alpha=0 uniform, alpha=1 inverse-frequency.
                         Each strategy's docstring documents how alpha is
                         applied for that specific algorithm.
        min_count     -- minimum sample count for a class to be kept
                         (subject to rare_policy). A class with fewer
                         training samples is treated as "rare".
        rare_policy   -- "drop"  zero out weights for classes below
                                  min_count (their loss contribution is
                                  zero, equivalent to dropping them
                                  from the gradient signal).
                          "keep" leave rare classes in the weighting,
                                  computed normally.
                          "merge" not implemented: merging would require
                                  relabelling samples up the BA tree
                                  inside the data pipeline, which is a
                                  larger change. Selecting "merge" raises
                                  NotImplementedError at strategy time.
    """

    enabled: bool = True
    strategy: str = "tree_balanced_ba_flat_gf"
    alpha: float = 0.5
    min_count: int = 10
    rare_policy: RarePolicy = "drop"

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(
                f"alpha must be in [0, 1], got {self.alpha!r}"
            )
        if self.min_count < 1:
            raise ValueError(
                f"min_count must be >= 1, got {self.min_count!r}"
            )
        if self.rare_policy not in _VALID_POLICIES:
            raise ValueError(
                f"rare_policy must be one of {_VALID_POLICIES},"
                f" got {self.rare_policy!r}"
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
            "weighting/min_count": self.min_count,
            "weighting/rare_policy": self.rare_policy,
        }
