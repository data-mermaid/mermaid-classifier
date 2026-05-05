"""SubsampleOptions dataclass with eager validation.

Configures deterministic per-class subsampling of the annotation set.
Applied after rollup + included-labels filter, before the train/ref/val
split.

Two strategies are wired in today:

* ``'stratified'`` -- proportional sampling. Each class's row count is
  scaled by ``total_annotations / N`` so the resulting subset preserves
  the original class distribution within rounding error. Used today by
  the screen stage of ``classifier_train_sweep.py`` to fix the
  non-deterministic ``annotation_limit`` bug.

* ``'balanced'`` -- equalize counts across classes. Each class is
  capped at either ``total_annotations / num_classes`` or an explicit
  ``target_per_class``, whichever is set. Useful for class-balanced
  experiments where you want the model to see roughly the same number
  of examples per class.

Adding a new strategy (e.g. ``'effective_number'``,
``'log_balanced'``) is a two-step process:

  1. Append the strategy name to ``SUBSAMPLE_STRATEGIES`` in
     ``options.py`` (this module).
  2. Add an allocator function in ``registry.py`` and register it in
     the ``_ALLOCATORS`` dict.

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
    "soft_balanced",
)


@dataclasses.dataclass
class SubsampleOptions:
    """Configuration for per-class subsampling.

    Fields:
        strategy           -- registry key. ``'stratified'`` (proportional),
                              ``'balanced'`` (equalized), or
                              ``'soft_balanced'`` (alpha-interpolated
                              between the two: ``target_c ~ n_c^(1-alpha)``).
                              New strategies are added by extending
                              SUBSAMPLE_STRATEGIES and registering an
                              allocator in registry.py.
        total_annotations  -- target number of rows after subsampling.
                              Required for ``'stratified'`` and
                              ``'soft_balanced'``. For ``'balanced'``,
                              optional: when set, the budget is split
                              equally across classes.
        min_per_class      -- floor on the per-class budget. Useful in
                              all strategies to keep rare classes alive
                              when proportional rounding (or a small
                              ``target_per_class``) would round them
                              down to 0. ``0`` (the default) lets a
                              class drop out of the subsample if its
                              proportional share rounds to 0.
        target_per_class   -- explicit per-class budget. Only meaningful
                              for ``strategy='balanced'``; rejected
                              otherwise. Caps at the available class
                              count (no oversampling).
        balance_alpha      -- only meaningful for
                              ``strategy='soft_balanced'``. Interpolation
                              exponent in [0, 1]: 0 collapses to
                              ``'stratified'`` (target_c ~ n_c), 1
                              collapses to ``'balanced'`` (target_c ~ 1),
                              0.5 produces square-root sampling
                              (target_c ~ sqrt(n_c)). Required for
                              ``'soft_balanced'``; rejected otherwise.
        seed               -- Random seed. **Currently unused by the
                              built-in allocators**, which are fully
                              deterministic via SQL ordering on the
                              primary-key tuple ``(site, project_id,
                              image_id, row, col)`` -- changing the seed
                              does not change which rows are selected.
                              The field exists so that (a) the seed
                              shows up in MLflow params for visibility,
                              and (b) future stochastic strategies
                              (e.g. bootstrap oversampling) can read it
                              without having to extend this dataclass
                              again. Default ``0``.

    Determinism: today's built-in allocators carry no random state.
    Determinism is a property of the SQL applied in
    TrainingDataset._apply_subsample (deterministic ORDER BY on a
    primary-key tuple, not RANDOM()). The ``seed`` field is reserved
    for future stochastic strategies; see field docs above.

    Extension hooks (deliberately small today):
      * ``oversample: bool`` for bootstrap upsampling of rare classes.
        Not implemented. Would require a separate UNION-ALL SQL pass.
      * ``stratification_level: str`` to partition by top-level BA
        instead of BA+GF. Trivial to add: change the PARTITION BY in
        the apply method and surface the column name here.
      * Custom target distribution: replace the strategy switch with
        a callable that takes counts and returns targets. Not needed
        until a use case demands it; the strategy-string-plus-registry
        shape covers all foreseeable cases.
    """

    strategy: str = "stratified"
    total_annotations: int | None = None
    min_per_class: int = 0
    target_per_class: int | None = None
    balance_alpha: float | None = None
    seed: int = 0

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
        if (self.target_per_class is not None
                and self.target_per_class <= 0):
            raise ValueError(
                f"target_per_class must be > 0 or None,"
                f" got {self.target_per_class!r}"
            )

        if self.strategy == "stratified":
            if self.total_annotations is None:
                raise ValueError(
                    "strategy='stratified' requires total_annotations"
                )
            if self.target_per_class is not None:
                raise ValueError(
                    "target_per_class is only meaningful for"
                    " strategy='balanced'"
                )
            if self.balance_alpha is not None:
                raise ValueError(
                    "balance_alpha is only meaningful for"
                    " strategy='soft_balanced'"
                )
        elif self.strategy == "balanced":
            if (self.total_annotations is None
                    and self.target_per_class is None):
                raise ValueError(
                    "strategy='balanced' requires either"
                    " total_annotations (split equally across classes)"
                    " or target_per_class (explicit per-class budget)"
                )
            if (self.total_annotations is not None
                    and self.target_per_class is not None):
                raise ValueError(
                    "strategy='balanced' takes either total_annotations"
                    " OR target_per_class, not both"
                )
            if self.balance_alpha is not None:
                raise ValueError(
                    "balance_alpha is only meaningful for"
                    " strategy='soft_balanced'"
                )
        elif self.strategy == "soft_balanced":
            if self.total_annotations is None:
                raise ValueError(
                    "strategy='soft_balanced' requires total_annotations"
                )
            if self.target_per_class is not None:
                raise ValueError(
                    "target_per_class is only meaningful for"
                    " strategy='balanced'"
                )
            if self.balance_alpha is None:
                raise ValueError(
                    "strategy='soft_balanced' requires balance_alpha"
                    " in [0, 1]"
                )
            if not (0.0 <= self.balance_alpha <= 1.0):
                raise ValueError(
                    f"balance_alpha must be in [0, 1],"
                    f" got {self.balance_alpha!r}"
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
            "subsample/target_per_class": self.target_per_class,
            "subsample/balance_alpha": self.balance_alpha,
            "subsample/seed": self.seed,
        }
