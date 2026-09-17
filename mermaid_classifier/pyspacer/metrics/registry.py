"""Declarative registry of metric groups.

Adding a metric group is a one-line edit here — the coordinator iterates
this list and no longer needs editing. Order is significant and preserved.
Each group declares the context inputs it needs; the coordinator skips
groups whose inputs are unavailable.
"""

from __future__ import annotations

import dataclasses
import typing

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import MetricGroupResult
from mermaid_classifier.pyspacer.metrics.calibration import compute_calibration
from mermaid_classifier.pyspacer.metrics.classification import (
    compute_balanced_accuracy_mcc,
    compute_confusion_matrices,
    compute_precision_recall_f1,
)
from mermaid_classifier.pyspacer.metrics.cover import compute_cover
from mermaid_classifier.pyspacer.metrics.per_source import compute_per_source
from mermaid_classifier.pyspacer.metrics.probability import compute_probability
from mermaid_classifier.pyspacer.metrics.ranking import compute_ranking
from mermaid_classifier.pyspacer.metrics.taxonomic import compute_taxonomic

MetricGroupFunc = typing.Callable[[MetricsContext], MetricGroupResult]


@dataclasses.dataclass(frozen=True)
class MetricGroupSpec:
    name: str
    func: MetricGroupFunc
    requires_dataset: bool = False
    requires_val_proba: bool = False


# Order is significant — mirrors the historical coordinator ordering.
METRIC_GROUPS: list[MetricGroupSpec] = [
    MetricGroupSpec("confusion_matrices", compute_confusion_matrices),
    MetricGroupSpec("precision_recall_f1", compute_precision_recall_f1),
    MetricGroupSpec("balanced_accuracy_mcc", compute_balanced_accuracy_mcc),
    MetricGroupSpec("taxonomic", compute_taxonomic),
    MetricGroupSpec("calibration", compute_calibration),
    MetricGroupSpec("cover", compute_cover, requires_dataset=True),
    MetricGroupSpec("per_source", compute_per_source, requires_dataset=True),
    MetricGroupSpec("probability", compute_probability, requires_val_proba=True),
    MetricGroupSpec("ranking", compute_ranking, requires_val_proba=True),
]


def applicable_metric_groups(ctx: MetricsContext) -> list[tuple[str, MetricGroupFunc]]:
    """Ordered (name, func) for groups whose required ctx inputs are present."""
    groups: list[tuple[str, MetricGroupFunc]] = []
    for spec in METRIC_GROUPS:
        if spec.requires_dataset and ctx.dataset is None:
            continue
        if spec.requires_val_proba and ctx.val_proba is None:
            continue
        groups.append((spec.name, spec.func))
    return groups
