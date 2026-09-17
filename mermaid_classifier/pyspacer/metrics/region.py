"""Region-mismatch metrics over the validation split.

How often the model predicts a label from a region the image did not come
from, read against the same rate on the ground truth: the floor of label
noise and region-polygon error that no model can go below. A rate quoted
without that floor is unreadable, so both travel together.

The split is composed by the training config rather than fixed, so these
numbers move with the config and are not comparable across runs that drew
different data. `region_probe` measures the same behaviour on a fixed image
set; the two prefixes never merge.

Requires dataset (TrainingDataset) in MetricsContext, which provides the
`feature_loc_to_region` mapping populated during data loading. Iteration
order matches `compute_per_source` and `compute_cover`: walk
`dataset.labels.val.keys()` and treat each image's points as contiguous in
`val_results.gt`/`val_results.est`.

CoralNet images carry no MERMAID region and are dropped before any rate is
read -- the region predicates raise on an unrecorded region rather than
guess -- and the dropped count is logged as `n_points_unrecorded_region`, so
the exclusion is a number a reader can see. A run with no MERMAID data in it
has every point dropped and the group is a no-op.

`settings.region_val_n_resamples` sizes every confidence interval here. Each
rate in each table draws that many cluster resamples, so the cost scales with
the split, which can reach millions of points; the default trades interval
precision for wall clock.
"""

import math

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import (
    DataFrameResult,
    MetricGroupResult,
    ScalarMetric,
)
from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.region_eval.metrics import (
    RateEstimate,
    RegionMetricsOptions,
    RegionMismatchMetrics,
    compute_region_metrics,
    prepare_scored_points,
)

VAL_PREFIX = "region_val"


def scored_metric_name(prefix: str) -> str:
    """The status scalar published under a region metric prefix."""
    return f"{prefix}/scored"


# The tables that explain a rate, in the order a reader works through them:
# which region, which label, which direction, which confusion.
TABLE_NAMES = ("per_region", "per_label", "per_direction", "confusion")


def compute_region(ctx: MetricsContext) -> MetricGroupResult:
    """Compute region-mismatch rates over the validation split."""
    dataset = ctx.dataset
    feature_loc_to_region = getattr(dataset, "feature_loc_to_region", None)
    if not feature_loc_to_region:
        # Older dataset instances (e.g. from re-evaluation paths) may not
        # have the per-image region map.
        return MetricGroupResult()

    assert dataset is not None  # guarded above: the map is an attribute of it
    val_results = ctx.val_results
    n_val_points = len(val_results.gt)
    image_ids: list[str] = []
    image_region_ids: list[str] = []
    region_names: dict[str, str] = {}
    for feature_loc in dataset.labels.val.keys():  # noqa: SIM118 — ImageLabels.keys() is not a plain dict; __iter__ differs
        region_id, region_name = feature_loc_to_region[feature_loc]
        n_points = len(dataset.labels.val[feature_loc])
        if len(image_region_ids) + n_points > n_val_points:
            # The running count is checked per image so the drift names the
            # image it first appeared at, rather than only a wrong total.
            raise ValueError(
                f"Per-region index ran past val_results at image"
                f" {feature_loc.key!r}: {len(image_region_ids) + n_points} points"
                f" indexed over {n_val_points} results."
                " dataset.labels.val iteration order may have diverged from"
                " evaluate_classifier."
            )
        image_ids.extend([str(feature_loc.key)] * n_points)
        image_region_ids.extend([region_id] * n_points)
        if region_id:
            region_names[region_id] = region_name

    if len(image_region_ids) != n_val_points:
        # Defensive: order or counts drifted from evaluate_classifier.
        # Don't poison the run with a silently-wrong breakdown.
        raise ValueError(
            f"Per-region index count ({len(image_region_ids)}) does not"
            f" match val_results length ({len(val_results.gt)})."
            " dataset.labels.val iteration order may have diverged from"
            " evaluate_classifier."
        )

    if not any(image_region_ids):
        # Every image is CoralNet: nothing carries a region to score against.
        return MetricGroupResult()

    # val_results.classes is list[LabelId] (int|str); MERMAID always uses str.
    classes: list[str] = val_results.classes  # pyright: ignore[reportAssignmentType]
    points = prepare_scored_points(
        image_ids=image_ids,
        image_region_ids=image_region_ids,
        gt_labels=[classes[index] for index in val_results.gt],
        pred_labels=[classes[index] for index in val_results.est],
        region_ids_by_attribute=ctx.ba_library.region_ids_by_id,
        model_classes=classes,
    )
    metrics = compute_region_metrics(
        points,
        options=RegionMetricsOptions(n_resamples=settings.region_val_n_resamples),
        label_names={
            label: ctx.ba_library.bagf_id_to_name(label, ctx.gf_library) for label in classes
        },
        region_names=region_names,
    )
    return emit_region_metrics(VAL_PREFIX, metrics)


def emit_region_metrics(prefix: str, metrics: RegionMismatchMetrics) -> MetricGroupResult:
    """Scalars and tables for one scored population, under one metric prefix.

    The prefix is the caller's: two populations with incomparable denominators
    must not share a metric name, and a shared emitter is how they stay
    identical in shape while staying distinct in name.

    `{prefix}/scored` reaches 1 here and nowhere else: the coordinator logs a
    0 before the group runs, so rates that never arrive read as a failure
    rather than as a model with no incidents.
    """
    rates = metrics.overall
    result = MetricGroupResult()
    result.scalars.extend(
        [
            ScalarMetric(name=scored_metric_name(prefix), value=1.0),
            ScalarMetric(name=f"{prefix}/n_points", value=float(metrics.n_points)),
            ScalarMetric(name=f"{prefix}/n_images", value=float(metrics.n_images)),
            ScalarMetric(name=f"{prefix}/n_regions", value=float(len(metrics.observed_region_ids))),
            ScalarMetric(
                name=f"{prefix}/n_points_unrecorded_region",
                value=float(metrics.n_unrecorded_region_excluded),
            ),
        ]
    )

    for name, estimate in (
        ("oor_rate", rates.oor_rate),
        ("oor_rate_disc", rates.oor_rate_disc),
        ("oor_rate_disc_gt", rates.oor_rate_disc_gt),
        ("gt_oor_rate", rates.gt_oor_rate),
        ("image_affected_rate", rates.image_affected_rate),
        ("accuracy", rates.accuracy),
    ):
        result.scalars.extend(_rate_scalars(f"{prefix}/{name}", estimate))

    result.scalars.extend(
        _value_with_bounds(
            f"{prefix}/excess", rates.excess.value, rates.excess.ci_low, rates.excess.ci_high
        )
    )
    result.scalars.extend(
        _value_with_bounds(
            f"{prefix}/ratio_to_gt",
            rates.ratio_to_gt.value,
            rates.ratio_to_gt.ci_low,
            rates.ratio_to_gt.ci_high,
        )
    )

    for name in TABLE_NAMES:
        result.dataframes.append(
            DataFrameResult(df=getattr(metrics, name), artifact_path=f"{prefix}/{name}")
        )
    return result


def _rate_scalars(name: str, estimate: RateEstimate) -> list[ScalarMetric]:
    """A rate, the k and n it was read from, and its interval bounds.

    An empty denominator reads as a rate of zero beside an n of zero rather
    than as the NaN the estimate carries: MLflow logging skips NaN, which
    would drop the cell out of the run without a trace.
    """
    rate = estimate.rate if math.isfinite(estimate.rate) else 0.0
    scalars = [
        ScalarMetric(name=name, value=rate),
        ScalarMetric(name=f"{name}_k", value=float(estimate.k)),
        ScalarMetric(name=f"{name}_n", value=float(estimate.n)),
    ]
    scalars.extend(_bounds(name, estimate.ci_low, estimate.ci_high))
    return scalars


def _value_with_bounds(name: str, value: float, low: float, high: float) -> list[ScalarMetric]:
    """A difference or ratio and its paired interval, both dropped when
    non-finite: a thin ground-truth floor leaves an unbounded multiple, which
    is a state rather than a number."""
    if not math.isfinite(value):
        return []
    return [ScalarMetric(name=name, value=value), *_bounds(name, low, high)]


def _bounds(name: str, low: float, high: float) -> list[ScalarMetric]:
    """The interval bounds as their own metrics, so MLflow renders and diffs
    them across runs beside the rate they belong to."""
    if not (math.isfinite(low) and math.isfinite(high)):
        return []
    return [
        ScalarMetric(name=f"{name}_lo95", value=low),
        ScalarMetric(name=f"{name}_hi95", value=high),
    ]
