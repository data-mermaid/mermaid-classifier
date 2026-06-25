"""SubsampleOptions dataclass with eager validation.

Configures deterministic per-class subsampling of the annotation set.
Applied after rollup + included-labels filter, before the train/ref/val
split.

Two strategies are wired in:

* ``'stratified'`` -- proportional sampling. Each class's row count is
  scaled by ``total_annotations / N`` so the resulting subset preserves
  the original class distribution within rounding error.

* ``'balanced'`` -- equalize counts across classes. Each class is capped
  at ``total_annotations / num_classes``. This is the production recipe
  (see ``docs/balancing-experiments.md``): at full-data scale most
  classes fall below the per-class target and are kept in full, so the
  allocator mostly just caps the dominant classes, with ``min_per_class``
  flooring the rare ones.

Adding a new strategy is a two-step process:

  1. Append the strategy name to ``SUBSAMPLE_STRATEGIES`` in this module.
  2. Add an allocator function in ``registry.py`` and register it in the
     ``_ALLOCATORS`` dict.

The pipeline (``TrainingDataset._apply_subsample``) doesn't change.
"""
from __future__ import annotations

import dataclasses


# Authoritative list of strategy names. Kept in this module so it can
# be imported without touching the registry (avoids circular imports
# during strategy unit tests). When you add a new strategy, append its
# name here AND register an allocator in ``registry._ALLOCATORS``.
SUBSAMPLE_STRATEGIES: tuple[str, ...] = (
    "stratified",
    "balanced",
)


@dataclasses.dataclass
class SubsampleOptions:
    """Configuration for per-class subsampling.

    Fields:
        strategy           -- registry key. ``'stratified'`` (proportional)
                              or ``'balanced'`` (equalized). New strategies
                              are added by extending SUBSAMPLE_STRATEGIES
                              and registering an allocator in registry.py.
        total_annotations  -- target number of rows after subsampling.
                              Required for both strategies. For
                              ``'balanced'`` the budget is split equally
                              across classes.
        min_per_class      -- floor on the per-class budget. Keeps rare
                              classes alive when proportional rounding
                              would round them down to 0. ``0`` (the
                              default) lets a class drop out of the
                              subsample if its proportional share rounds
                              to 0.

    Determinism: the built-in allocators carry no random state.
    Determinism is a property of the SQL applied in
    TrainingDataset._apply_subsample (deterministic ORDER BY on a
    primary-key tuple, not RANDOM()).
    """

    strategy: str = "stratified"
    total_annotations: int | None = None
    min_per_class: int = 0

    def __post_init__(self) -> None:
        if self.strategy not in SUBSAMPLE_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {SUBSAMPLE_STRATEGIES},"
                f" got {self.strategy!r}"
            )
        if (self.total_annotations is not None
                and self.total_annotations <= 0):
            raise ValueError(
                f"total_annotations must be > 0 or None,"
                f" got {self.total_annotations!r}"
            )
        if self.min_per_class < 0:
            raise ValueError(
                f"min_per_class must be >= 0,"
                f" got {self.min_per_class!r}"
            )

        if self.strategy == "stratified":
            if self.total_annotations is None:
                raise ValueError(
                    "strategy='stratified' requires total_annotations"
                )
        elif self.strategy == "balanced":
            if self.total_annotations is None:
                raise ValueError(
                    "strategy='balanced' requires total_annotations"
                    " (split equally across classes)"
                )

    def to_log_dict(self) -> dict[str, object]:
        """Flat dict suitable for ``mlflow.log_params``.

        Mirrors the shape of SampleWeightingOptions.to_log_dict() so the
        MLflow UI groups subsample params together under one prefix.
        """
        return {
            "subsample/enabled": True,
            "subsample/strategy": self.strategy,
            "subsample/total_annotations": self.total_annotations,
            "subsample/min_per_class": self.min_per_class,
        }
