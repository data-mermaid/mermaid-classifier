"""Per-class subsampling of the annotation set.

Replaces the old non-deterministic ``DatasetOptions.annotation_limit``
(``LIMIT`` without ``ORDER BY``, unstable under DuckDB parallel scans)
with a deterministic, strategy-dispatched subsampler that runs after
rollup + included-labels filter and before the train/ref/val split.

Public API:
    SubsampleOptions             -- configuration dataclass (lives on
                                    ``DatasetOptions.subsample``).
    SUBSAMPLE_STRATEGIES         -- tuple of allowed strategy names.
    compute_per_class_targets    -- dispatch entry point used by the
                                    pipeline.

Extension points (see options.py and registry.py for details):

* New strategies (e.g. ``'effective_number'``, ``'log_balanced'``) are
  added by writing a new allocator function and registering it in
  ``SUBSAMPLE_STRATEGIES``. No call-site changes required -- the
  pipeline reads ``opts.strategy`` and dispatches.
* Class-balanced sampling (every class gets the same N rows, with
  rare classes capped at their available count) is already implemented
  as ``strategy='balanced'`` -- pass ``total_annotations`` (the budget
  is split equally across classes).
* Bootstrap oversampling for rare classes is intentionally NOT
  supported here. It needs a UNION-ALL pass over the source rows;
  add an ``oversample`` field to ``SubsampleOptions`` and a second
  SQL pass when that becomes a requirement.
"""
from mermaid_classifier.training.subsample.options import SubsampleOptions
from mermaid_classifier.training.subsample.registry import (
    SUBSAMPLE_STRATEGIES,
    compute_per_class_targets,
)

__all__ = [
    "SubsampleOptions",
    "SUBSAMPLE_STRATEGIES",
    "compute_per_class_targets",
]
