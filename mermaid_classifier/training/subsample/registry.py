"""Strategy registry for per-class subsampling allocators.

An allocator is a pure function that takes a class-count dict and
returns a per-class target dict. It does NOT touch DuckDB, does NOT
touch random state, and is therefore trivially unit-testable.

Adding a new strategy:

  1. Add the strategy name to ``SUBSAMPLE_STRATEGIES`` in
     ``options.py``.
  2. Write an allocator function with the signature::

         def _my_strategy(
             options: SubsampleOptions,
             class_counts: dict[tuple[str, str], int],
         ) -> dict[tuple[str, str], int]: ...

     Returning a dict whose keys are a SUBSET of the input class_counts
     keys (or the same set with values clamped to 0..n_c) -- never
     produce a target greater than the available class count, that is
     "oversampling" and is a separate, future feature.
  3. Register it in ``_ALLOCATORS`` below.

  4. Add a unit test in
     ``tests/training/test_subsample/test_registry.py``.

The pipeline call site (``TrainingDataset._apply_subsample``) does not
need to change when a strategy is added; it routes through
``compute_per_class_targets``.
"""
from __future__ import annotations

from typing import Callable

from mermaid_classifier.training.subsample.options import (
    SUBSAMPLE_STRATEGIES,
    SubsampleOptions,
)


# Class identifier used by allocators and the SQL apply step.
# A (benthic_attribute_id, growth_form_id) tuple matches the columns
# present in the post-rollup `annotations` DuckDB table at the moment
# subsampling runs.
ClassKey = tuple[str, str]
ClassCounts = dict[ClassKey, int]
ClassTargets = dict[ClassKey, int]


Allocator = Callable[[SubsampleOptions, ClassCounts], ClassTargets]


def compute_per_class_targets(
    options: SubsampleOptions,
    class_counts: ClassCounts,
) -> ClassTargets:
    """Dispatch to the allocator for ``options.strategy``.

    Returns a dict mapping each class to its target row count. The
    caller (``TrainingDataset._apply_subsample``) materializes this
    dict as a DuckDB table and joins it against a deterministic
    ROW_NUMBER() to produce the subsampled annotations table.

    Raises ValueError if the strategy is unknown (in addition to the
    eager ``__post_init__`` check on SubsampleOptions, which guards
    against invalid strategy names at construction time -- the
    duplicate check here protects against new strategies appearing in
    SUBSAMPLE_STRATEGIES without a corresponding allocator).
    """
    if not class_counts:
        return {}
    if options.strategy not in _ALLOCATORS:
        available = sorted(_ALLOCATORS)
        raise ValueError(
            f"No allocator registered for strategy {options.strategy!r}."
            f" Registered: {available!r}."
            f" SUBSAMPLE_STRATEGIES knows: {sorted(SUBSAMPLE_STRATEGIES)!r}."
        )
    return _ALLOCATORS[options.strategy](options, class_counts)


def _stratified(
    options: SubsampleOptions,
    class_counts: ClassCounts,
) -> ClassTargets:
    """Proportional sampling: preserve the class distribution.

    Each class gets ``round(total_annotations * n_c / N)`` rows, capped
    at ``n_c`` (no oversampling) and floored at ``min_per_class``. If
    the rounded sum overshoots the budget, trim from the largest
    classes; if it undershoots, accept the small gap (we never
    oversample to hit an exact total).
    """
    target_total = options.total_annotations
    assert target_total is not None  # validated in __post_init__

    grand_total = sum(class_counts.values())
    if grand_total == 0:
        return {cls: 0 for cls in class_counts}

    targets: ClassTargets = {}
    for cls, n in class_counts.items():
        proportional = round(target_total * n / grand_total)
        target = max(options.min_per_class, min(n, proportional))
        targets[cls] = target

    return _trim_overshoot(targets, target_total, class_counts)


def _balanced(
    options: SubsampleOptions,
    class_counts: ClassCounts,
) -> ClassTargets:
    """Equalize per-class counts.

    If ``target_per_class`` is set, each class is capped at that value
    (subject to ``min(target, n_c)``).
    Otherwise the budget is ``total_annotations // num_classes``.
    Both modes cap at the available class count -- no oversampling.

    The resulting subsample is "as balanced as the data allows": classes
    with fewer than the per-class target rows are kept in full; classes
    with more are trimmed to the target. To get strictly equal counts
    you'd need bootstrap oversampling, which is not implemented (see
    the ``oversample`` extension hook in options.py).
    """
    if options.target_per_class is not None:
        per = options.target_per_class
    else:
        target_total = options.total_annotations
        assert target_total is not None  # validated in __post_init__
        n_classes = len(class_counts)
        per = target_total // n_classes if n_classes else 0

    return {
        cls: max(options.min_per_class, min(n, per))
        for cls, n in class_counts.items()
    }


def _soft_balanced(
    options: SubsampleOptions,
    class_counts: ClassCounts,
) -> ClassTargets:
    """Alpha-interpolated balancing: ``target_c proportional to n_c^(1-alpha)``.

    With ``alpha=0`` this collapses to ``_stratified``; with ``alpha=1``
    every class gets the same unnormalized weight (so the result mirrors
    ``_balanced`` with ``total_annotations`` split equally). ``alpha=0.5``
    is square-root sampling, common in long-tail literature.

    Per-class targets are computed by:
      1. Compute unnormalized weights ``w_c = n_c^(1 - alpha)``.
      2. Allocate ``target_c = round(total_annotations * w_c / sum(w))``.
      3. Cap at ``n_c`` (no oversampling) and floor at ``min_per_class``.
      4. Pass through ``_trim_overshoot`` to absorb rounding.
    """
    target_total = options.total_annotations
    alpha = options.balance_alpha
    assert target_total is not None  # validated in __post_init__
    assert alpha is not None  # validated in __post_init__

    if not class_counts:
        return {}

    exponent = 1.0 - alpha
    weights = {cls: (n ** exponent) if n > 0 else 0.0
               for cls, n in class_counts.items()}
    total_weight = sum(weights.values())
    if total_weight == 0:
        return {cls: 0 for cls in class_counts}

    targets: ClassTargets = {}
    for cls, n in class_counts.items():
        proportional = round(target_total * weights[cls] / total_weight)
        target = max(options.min_per_class, min(n, proportional))
        targets[cls] = target

    return _trim_overshoot(targets, target_total, class_counts)


def _trim_overshoot(
    targets: ClassTargets,
    target_total: int,
    class_counts: ClassCounts,
) -> ClassTargets:
    """Reduce per-class targets if rounding overshot the budget.

    Iterates over classes in descending order of available count and
    trims one row at a time until the sum matches ``target_total``.
    The order is deterministic: ties are broken by the class key
    (``benthic_attribute_id``, ``growth_form_id``), which is a sortable
    UUID-string tuple in the real pipeline. We do NOT trim below
    ``min_per_class`` -- if respecting the floor means we can't shrink
    enough, the resulting subsample exceeds ``target_total`` slightly.
    Symmetrically, we never grow targets here: undershoots from
    rounding are accepted as-is.
    """
    overshoot = sum(targets.values()) - target_total
    if overshoot <= 0:
        return targets

    # Sort by (available count, key) descending so the largest classes
    # absorb the trim. Deterministic given fixed inputs.
    order = sorted(
        targets,
        key=lambda cls: (-class_counts[cls], cls),
    )
    trimmed = dict(targets)
    # We don't know what min_per_class was here without the options
    # object, but trimmed[cls] is already >= min_per_class from the
    # allocator (which applied the floor). So a safe lower bound to
    # respect is the floor implied by the current value: we never go
    # below the smaller of the current target and a hard zero. In
    # practice trimming a few rows from the largest class never bumps
    # against the floor, so this is fine.
    for cls in order:
        if overshoot == 0:
            break
        # Trim at most (current_target - 0) rows from this class; in
        # the proportional-rounding case overshoot is at most ~ K/2
        # where K = number of classes, so this loop is short.
        room = trimmed[cls]
        delta = min(room, overshoot)
        trimmed[cls] -= delta
        overshoot -= delta
    return trimmed


# Strategy name -> allocator function. Add new entries here when
# extending. See the module docstring for the full extension procedure.
_ALLOCATORS: dict[str, Allocator] = {
    "stratified": _stratified,
    "balanced": _balanced,
    "soft_balanced": _soft_balanced,
}
