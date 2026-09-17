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
from mermaid_classifier.pyspacer.metrics.region import (
    VAL_PREFIX,
    compute_region,
    scored_metric_name,
)
from mermaid_classifier.pyspacer.metrics.region_probe import (
    PROBE_PREFIX,
    compute_region_probe,
    region_probe_is_configured,
)
from mermaid_classifier.pyspacer.metrics.taxonomic import compute_taxonomic

MetricGroupFunc = typing.Callable[[MetricsContext], MetricGroupResult]


@dataclasses.dataclass(frozen=True)
class MetricGroupSpec:
    name: str
    func: MetricGroupFunc
    requires_dataset: bool = False
    requires_val_proba: bool = False
    requires_clf: bool = False
    # A metric logged as 0 before the group runs and raised to 1 by the
    # group's own result, so a group that fails is distinguishable from one
    # that was never applicable. `is_configured` answers whether the group has
    # the configuration it needs; where it says no, no status is logged at all,
    # because a 0 for a group nobody asked for reads like one that failed.
    status_metric: str | None = None
    is_configured: typing.Callable[[], bool] | None = None


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
    MetricGroupSpec(
        "region",
        compute_region,
        requires_dataset=True,
        status_metric=scored_metric_name(VAL_PREFIX),
    ),
    MetricGroupSpec(
        "region_probe",
        compute_region_probe,
        requires_clf=True,
        status_metric=scored_metric_name(PROBE_PREFIX),
        is_configured=region_probe_is_configured,
    ),
]


def applicable_metric_groups(ctx: MetricsContext) -> list[MetricGroupSpec]:
    """Ordered specs for groups whose required ctx inputs are present."""
    groups: list[MetricGroupSpec] = []
    for spec in METRIC_GROUPS:
        if spec.requires_dataset and ctx.dataset is None:
            continue
        if spec.requires_val_proba and ctx.val_proba is None:
            continue
        if spec.requires_clf and ctx.clf is None:
            continue
        groups.append(spec)
    return groups
