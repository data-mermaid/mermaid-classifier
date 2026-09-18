"""Region-mismatch rates over a scored slice of annotation points.

Three questions the headline numbers have to keep apart.

*Which denominator?* A rate of out-of-region predictions is diluted by the
third or so of the label space that is globally distributed and can never be
out of region, so `oor_rate` is never presented alone. `oor_rate_disc` divides
the same incidents by the predictions that could have been wrong, which moves
with the model's label space. `oor_rate_disc_gt` restricts both count and
denominator to the points whose ground truth could have been wrong, a
population fixed by the data alone and therefore comparable across model
versions. Every estimate carries its own numerator and denominator so a
number cannot travel without them.

*Compared with what?* The same formula on the ground truth is the floor: label
noise plus region-polygon error, which no model can go below. `excess` and
`ratio_to_gt` read a model rate against that floor.

*How certain?* The cluster is the image, not the point. Annotation points
inside one image are strongly correlated, so every interval resamples whole
images; a point-level interval understates the width several-fold. The naive
Wilson interval and the implied design effect travel alongside so a reader can
see by how much. A cell with no events reports a rule-of-three upper bound
rather than a bare zero, read over the images its denominator spans rather
than over its points: the point-level bound understates by the design effect
and can land below the ground-truth floor. A bootstrap interval of zero width
measures no clustering, so the design effect is NaN there and the sample-size
check falls back to the nominal cluster size. A resample whose ground-truth
floor holds no event divides by zero, so `ratio_to_gt` reads its interval over
the draws that do and reports how many did not: an unbounded upper end where
the tail is unbounded, rather than a number those draws cannot support.

*Which direction?* Incidents break out per ordered (image region, excluded
region) pair and are never pooled: opposite directions differ in available
points by two orders of magnitude, so their rates are not comparable. A
prediction permitted in several regions counts once under each of them, so the
direction matrix and `per_direction.n_out_of_region` sum to more than the
incidents they cover; `per_direction.n_out_of_region_events` carries the
unduplicated count.

Rates are computed over all scored points and again over the held-out subset.
Where a sample cannot measure its rate to the caller's target margin the
estimate is flagged `imprecise` rather than dressed up as a comparison.

*Which name?* Every table keys on ids and renders the caller's frozen display
names beside them. An id with no name renders as the id: an empty cell reads
as "this thing has no name", where the id reads as "unresolved", which is what
actually happened.

`prepare_scored_points` is the only way in. It drops points whose image region
is unrecorded and reports how many, which is what keeps the region predicates
from raising inside a metric group that a caller has wrapped in a broad
`except Exception`.
"""

import dataclasses
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_fscore_support

from mermaid_classifier.common.benthic_attributes import split_ba_gf
from mermaid_classifier.region_eval.region_rules import (
    cluster_bootstrap_ci,
    cluster_bootstrap_draws,
    design_effect,
    is_out_of_region,
    is_region_discriminating,
    paired_cluster_bootstrap_diff,
    partition_by_recorded_region,
    rule_of_three,
    wilson_ci,
)

ALL_POINTS = "all"
HELD_OUT = "held_out"


@dataclasses.dataclass(frozen=True)
class RegionMetricsOptions:
    """Knobs shared by every estimate in one run.

    `n_resamples` dominates the cost: each rate in every table draws that many
    cluster resamples, so a full-corpus run trades it against wall clock.

    `target_margin` is the half-width each cell's confidence interval is
    checked against, and `RateEstimate.imprecise` carries the verdict.
    """

    alpha: float = 0.05
    n_resamples: int = 2000
    seed: int = 0
    target_margin: float | None = None


@dataclasses.dataclass(frozen=True)
class RateEstimate:
    """A rate with the denominator it was read from and its uncertainty.

    `rate` is `k / n`; `k` is always a count over a subset of the `n`
    denominator, so it is a proportion in [0, 1] and the Wilson interval and
    design effect are defined whenever `n` is positive. An empty denominator
    reports `rate`, both intervals and `design_effect` as NaN; `upper_bound`
    is `None` and `upper_bound_n_images` is 0, since there is no image left
    to read a bound over; `imprecise` is `None`, the same value it carries
    whenever the caller supplied no target margin.

    `upper_bound` holds the rule-of-three 95% bound when a nonempty
    denominator saw no event, so a zero cell reads as "none in n, at most
    this" rather than "0%". It is read over the `upper_bound_n_images`
    images the denominator spans, not over the points, which travels with it
    so the bound cannot be mistaken for a point-level one.

    `design_effect` is NaN where the bootstrap interval has zero width, which
    measures no clustering rather than perfect independence. `imprecise` is
    None unless the caller supplied a target margin, and otherwise says the
    interval half-width exceeds it.
    """

    k: int
    n: int
    rate: float
    ci_low: float
    ci_high: float
    wilson_low: float
    wilson_high: float
    design_effect: float
    upper_bound: float | None
    upper_bound_n_images: int
    imprecise: bool | None


@dataclasses.dataclass(frozen=True)
class DiffEstimate:
    """A difference of two rates measured on the same points.

    The interval is paired: one draw of images feeds both rates, so variation
    shared between them cancels instead of inflating the difference.
    """

    value: float
    ci_low: float
    ci_high: float


@dataclasses.dataclass(frozen=True)
class RatioEstimate:
    """One rate as a multiple of another measured on the same points.

    The interval is paired like `DiffEstimate`'s: one draw of images feeds
    both rates. It widens without bound as the denominator rate approaches
    zero, so a thin ground-truth floor reads as an unbounded multiple rather
    than a precise one.

    A resample whose denominator rate is zero has no multiple to report and
    contributes an infinite draw. `n_nonfinite_draws` of `n_draws` counts
    those, and the interval is read over the remaining draws alone: it is the
    multiple conditional on a non-empty floor, and a reader needs the two
    counts to know how conditional that is.
    """

    value: float
    ci_low: float
    ci_high: float
    n_draws: int
    n_nonfinite_draws: int


@dataclasses.dataclass(frozen=True)
class MacroEstimate:
    """An unweighted mean of per-region rates, and how many regions it spans.

    Unweighted so the number does not move when the corpus mix moves. A
    resample that happens to draw no image from a region averages the regions
    it did draw.
    """

    value: float
    ci_low: float
    ci_high: float
    n_regions: int


@dataclasses.dataclass(frozen=True)
class RegionMacroRates:
    oor_rate: MacroEstimate
    oor_rate_disc: MacroEstimate
    oor_rate_disc_gt: MacroEstimate
    gt_oor_rate: MacroEstimate


@dataclasses.dataclass(frozen=True)
class PopulationRates:
    """Every deployment-weighted rate for one population, plus its macro mean."""

    name: str
    n_points: int
    n_images: int
    n_pred_region_unknown: int
    n_gt_region_unknown: int
    n_gt_outside_model_classes: int
    n_accuracy_points: int
    accuracy: RateEstimate
    n_f1_disc_points: int
    precision_macro_disc: float
    recall_macro_disc: float
    f1_macro_disc: float
    oor_rate: RateEstimate
    oor_rate_disc: RateEstimate
    oor_rate_disc_gt: RateEstimate
    gt_oor_rate: RateEstimate
    image_affected_rate: RateEstimate
    excess: DiffEstimate
    ratio_to_gt: RatioEstimate
    region_macro: RegionMacroRates


@dataclasses.dataclass(frozen=True)
class ScoredPoints:
    """Points with a recorded image region, and everything derived from them.

    Built by `prepare_scored_points`, which is what guarantees no empty region
    reaches the predicates. `observed_region_ids` is the set the scored slice
    actually contains and is what "region-discriminating" is judged against,
    so it stays fixed across the nested populations rather than being
    recomputed on a subset.
    """

    image_ids: tuple[str, ...]
    image_region_ids: tuple[str, ...]
    gt_labels: tuple[str, ...]
    pred_labels: tuple[str, ...]
    gt_attribute_ids: tuple[str, ...]
    pred_attribute_ids: tuple[str, ...]
    gt_allowed: tuple[frozenset[str], ...]
    pred_allowed: tuple[frozenset[str], ...]
    held_out: NDArray[np.bool_]
    pred_out_of_region: NDArray[np.bool_]
    gt_out_of_region: NDArray[np.bool_]
    pred_discriminating: NDArray[np.bool_]
    gt_discriminating: NDArray[np.bool_]
    pred_region_unknown: NDArray[np.bool_]
    gt_region_unknown: NDArray[np.bool_]
    gt_in_model_classes: NDArray[np.bool_]
    correct: NDArray[np.bool_]
    observed_region_ids: frozenset[str]
    unmapped_attribute_ids: frozenset[str]
    n_unrecorded_region_excluded: int

    @property
    def n_points(self) -> int:
        return len(self.image_ids)

    @property
    def n_images(self) -> int:
        return len(set(self.image_ids))


@dataclasses.dataclass(frozen=True)
class RegionMismatchMetrics:
    """Rates for both populations, plus the tables that explain them."""

    n_points: int
    n_images: int
    n_unrecorded_region_excluded: int
    observed_region_ids: tuple[str, ...]
    unmapped_attribute_ids: tuple[str, ...]
    overall: PopulationRates
    held_out: PopulationRates | None
    per_region: pd.DataFrame
    per_label: pd.DataFrame
    direction_matrix: pd.DataFrame
    per_direction: pd.DataFrame
    confusion: pd.DataFrame


def prepare_scored_points(
    *,
    image_ids: Sequence[str],
    image_region_ids: Sequence[str],
    gt_labels: Sequence[str],
    pred_labels: Sequence[str],
    region_ids_by_attribute: Mapping[str, frozenset[str]],
    model_classes: Sequence[str],
    held_out: Sequence[bool] | None = None,
) -> ScoredPoints:
    """Filter to points with a recorded image region and derive every flag.

    Labels are BA-GF strings; `region_ids_by_attribute` is a frozen snapshot
    keyed by benthic-attribute id. An attribute the snapshot does not know is
    treated as region-unknown and named in `unmapped_attribute_ids`, so a
    stale snapshot shows up as a reported set rather than as a silent zero.

    `model_classes` decides which ground truths are inside the model's label
    space. Points whose truth falls outside it stay in the region statistics,
    which need only the image region and the prediction, and drop out of
    accuracy.
    """
    lengths = {len(image_ids), len(image_region_ids), len(gt_labels), len(pred_labels)}
    if held_out is not None:
        lengths.add(len(held_out))
    if len(lengths) > 1:
        raise ValueError(f"inputs must be the same length, got lengths {sorted(lengths)}")

    scorable, unscorable = partition_by_recorded_region(image_region_ids)

    scored_image_ids = tuple(image_ids[position] for position in scorable)
    scored_regions = tuple(image_region_ids[position] for position in scorable)
    scored_gt = tuple(gt_labels[position] for position in scorable)
    scored_pred = tuple(pred_labels[position] for position in scorable)
    scored_held_out = np.asarray(
        [False if held_out is None else bool(held_out[position]) for position in scorable],
        dtype=bool,
    )

    gt_attribute_ids = tuple(split_ba_gf(label)[0] for label in scored_gt)
    pred_attribute_ids = tuple(split_ba_gf(label)[0] for label in scored_pred)
    unmapped = frozenset(
        attribute_id
        for attribute_id in set(gt_attribute_ids) | set(pred_attribute_ids)
        if attribute_id not in region_ids_by_attribute
    )
    gt_allowed = tuple(
        region_ids_by_attribute.get(attribute_id, frozenset()) for attribute_id in gt_attribute_ids
    )
    pred_allowed = tuple(
        region_ids_by_attribute.get(attribute_id, frozenset())
        for attribute_id in pred_attribute_ids
    )

    observed = frozenset(scored_regions)
    class_set = set(model_classes)

    return ScoredPoints(
        image_ids=scored_image_ids,
        image_region_ids=scored_regions,
        gt_labels=scored_gt,
        pred_labels=scored_pred,
        gt_attribute_ids=gt_attribute_ids,
        pred_attribute_ids=pred_attribute_ids,
        gt_allowed=gt_allowed,
        pred_allowed=pred_allowed,
        held_out=scored_held_out,
        pred_out_of_region=_mask(
            is_out_of_region(allowed, region)
            for allowed, region in zip(pred_allowed, scored_regions, strict=True)
        ),
        gt_out_of_region=_mask(
            is_out_of_region(allowed, region)
            for allowed, region in zip(gt_allowed, scored_regions, strict=True)
        ),
        pred_discriminating=_mask(
            is_region_discriminating(allowed, observed) for allowed in pred_allowed
        ),
        gt_discriminating=_mask(
            is_region_discriminating(allowed, observed) for allowed in gt_allowed
        ),
        pred_region_unknown=_mask(not allowed for allowed in pred_allowed),
        gt_region_unknown=_mask(not allowed for allowed in gt_allowed),
        gt_in_model_classes=_mask(label in class_set for label in scored_gt),
        correct=_mask(
            predicted == truth for predicted, truth in zip(scored_pred, scored_gt, strict=True)
        ),
        observed_region_ids=observed,
        unmapped_attribute_ids=unmapped,
        n_unrecorded_region_excluded=len(unscorable),
    )


def compute_region_metrics(
    points: ScoredPoints,
    *,
    options: RegionMetricsOptions | None = None,
    label_names: Mapping[str, str] | None = None,
    region_names: Mapping[str, str] | None = None,
) -> RegionMismatchMetrics:
    """Every region-mismatch rate and table for one scored slice.

    `label_names` and `region_names` supply the display names the tables
    render; an id either mapping omits renders as the id itself. The tables
    that inventory incidents (per label, direction matrix, confusion) cover all
    scored points, because splitting counts that already sit in the tens across
    two populations reads noise.
    """
    resolved = RegionMetricsOptions() if options is None else options
    names: Mapping[str, str] = {} if label_names is None else label_names
    regions: Mapping[str, str] = {} if region_names is None else region_names

    overall = _population_rates(points, ALL_POINTS, resolved)
    held_out_points = (
        _restrict(points, np.flatnonzero(points.held_out)) if bool(points.held_out.any()) else None
    )
    held_out = (
        None if held_out_points is None else _population_rates(held_out_points, HELD_OUT, resolved)
    )

    populations: list[tuple[str, ScoredPoints]] = [(ALL_POINTS, points)]
    if held_out_points is not None:
        populations.append((HELD_OUT, held_out_points))

    direction_columns = _direction_columns(points)
    per_region_rows: list[dict[str, object]] = []
    per_direction_rows: list[dict[str, object]] = []
    for name, population in populations:
        per_region_rows.extend(_per_region_rows(population, name, resolved, regions))
        per_direction_rows.extend(
            _per_direction_rows(population, name, direction_columns, resolved, regions)
        )

    return RegionMismatchMetrics(
        n_points=points.n_points,
        n_images=points.n_images,
        n_unrecorded_region_excluded=points.n_unrecorded_region_excluded,
        observed_region_ids=tuple(sorted(points.observed_region_ids)),
        unmapped_attribute_ids=tuple(sorted(points.unmapped_attribute_ids)),
        overall=overall,
        held_out=held_out,
        per_region=columns_frame(per_region_rows, _per_region_columns()),
        per_label=_per_label_table(points, names, regions, resolved),
        direction_matrix=_direction_matrix(points, direction_columns, regions),
        per_direction=columns_frame(per_direction_rows, PER_DIRECTION_COLUMNS),
        confusion=_confusion_table(points, names, regions),
    )


def required_n_for_margin(
    rate: float,
    target_margin: float,
    *,
    alpha: float,
    design_effect: float = 1.0,
) -> int:
    """Points needed to measure a rate to a half-width of `target_margin`.

    The normal approximation z^2 p(1-p) deff / margin^2, rounded up. This is
    precision, not power: two estimates whose half-widths are each
    `target_margin` still have overlapping intervals at a true difference of
    that size, so it answers "how tight is this estimate" rather than "could
    two versions be told apart".

    A rate of zero or one carries no usable variance estimate, so p = 0.5
    stands in, which is the value that demands the largest sample.
    """
    if target_margin <= 0.0:
        raise ValueError(f"target_margin must be positive, got {target_margin}")
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    variance = z * z * _variance_factor(rate) * design_effect
    return math.ceil(variance / (target_margin * target_margin))


def _variance_factor(rate: float) -> float:
    """The p(1-p) term of a sample-size formula, at p = 0.5 where the rate is unusable.

    A rate of zero, one or NaN carries no variance estimate; 0.5 is the value
    that demands the largest sample.
    """
    proportion = rate if math.isfinite(rate) and 0.0 < rate < 1.0 else 0.5
    return proportion * (1.0 - proportion)


def _mask(values: Iterable[bool]) -> NDArray[np.bool_]:
    """A boolean array from an iterable, empty-safe."""
    return np.asarray(list(values), dtype=bool)


def _restrict(points: ScoredPoints, positions: NDArray[np.intp]) -> ScoredPoints:
    """The same points narrowed to `positions`, keeping the observed regions.

    The region set and the excluded count describe the whole scored slice, so
    a subset inherits them: discriminating labels must mean the same thing in
    both populations for their rates to be comparable.
    """
    index = [int(position) for position in positions]
    return ScoredPoints(
        image_ids=tuple(points.image_ids[position] for position in index),
        image_region_ids=tuple(points.image_region_ids[position] for position in index),
        gt_labels=tuple(points.gt_labels[position] for position in index),
        pred_labels=tuple(points.pred_labels[position] for position in index),
        gt_attribute_ids=tuple(points.gt_attribute_ids[position] for position in index),
        pred_attribute_ids=tuple(points.pred_attribute_ids[position] for position in index),
        gt_allowed=tuple(points.gt_allowed[position] for position in index),
        pred_allowed=tuple(points.pred_allowed[position] for position in index),
        held_out=points.held_out[positions],
        pred_out_of_region=points.pred_out_of_region[positions],
        gt_out_of_region=points.gt_out_of_region[positions],
        pred_discriminating=points.pred_discriminating[positions],
        gt_discriminating=points.gt_discriminating[positions],
        pred_region_unknown=points.pred_region_unknown[positions],
        gt_region_unknown=points.gt_region_unknown[positions],
        gt_in_model_classes=points.gt_in_model_classes[positions],
        correct=points.correct[positions],
        observed_region_ids=points.observed_region_ids,
        unmapped_attribute_ids=points.unmapped_attribute_ids,
        n_unrecorded_region_excluded=points.n_unrecorded_region_excluded,
    )


def _ratio_statistic(
    numerator: NDArray[np.bool_],
    denominator: NDArray[np.bool_],
    fallback: float,
) -> Callable[[NDArray[np.intp]], float]:
    """An estimator of numerator/denominator over a set of drawn positions.

    A resample holding no denominator point carries no information about the
    rate, so it contributes `fallback` -- the observed value -- rather than
    shifting the interval toward zero.
    """

    def statistic(index: NDArray[np.intp]) -> float:
        drawn = int(np.count_nonzero(denominator[index]))
        if drawn == 0:
            return fallback
        return float(np.count_nonzero(numerator[index]) / drawn)

    return statistic


def _blank_estimate(k: int, n: int) -> RateEstimate:
    """An estimate whose denominator is empty: no image contributes to it."""
    return RateEstimate(
        k=k,
        n=n,
        rate=math.nan,
        ci_low=math.nan,
        ci_high=math.nan,
        wilson_low=math.nan,
        wilson_high=math.nan,
        design_effect=math.nan,
        upper_bound=None,
        upper_bound_n_images=0,
        imprecise=None,
    )


def _estimate(
    cluster_ids: Sequence[str],
    numerator: NDArray[np.bool_],
    denominator: NDArray[np.bool_],
    options: RegionMetricsOptions,
) -> RateEstimate:
    """A rate, its cluster-bootstrap interval, and the naive interval beside it."""
    k = int(np.count_nonzero(numerator))
    n = int(np.count_nonzero(denominator))
    if n == 0 or len(cluster_ids) == 0:
        return _blank_estimate(k, n)

    rate = k / n
    n_images = _denominator_clusters(cluster_ids, denominator)
    ci_low, ci_high = cluster_bootstrap_ci(
        cluster_ids,
        _ratio_statistic(numerator, denominator, rate),
        n_resamples=options.n_resamples,
        alpha=options.alpha,
        seed=options.seed,
    )
    wilson_low, wilson_high = wilson_ci(k, n, options.alpha)
    # A zero-width bootstrap interval measures no clustering at all, which is
    # not a design effect of zero: a value below 1 reads as better than
    # independent, a state that does not exist.
    effect = (
        design_effect((ci_low, ci_high), k, n, alpha=options.alpha)
        if ci_high > ci_low
        else math.nan
    )
    return RateEstimate(
        k=k,
        n=n,
        rate=rate,
        ci_low=ci_low,
        ci_high=ci_high,
        wilson_low=wilson_low,
        wilson_high=wilson_high,
        design_effect=effect,
        upper_bound=rule_of_three(n_images) if k == 0 else None,
        upper_bound_n_images=n_images,
        imprecise=_imprecise(rate, n, n_images, effect, options),
    )


def _denominator_clusters(cluster_ids: Sequence[str], denominator: NDArray[np.bool_]) -> int:
    """How many distinct clusters contribute a point to the denominator.

    The rule-of-three bound counts independent trials, and the image is the
    trial: a point-level count understates the bound by the design effect.
    """
    return len(
        {
            cluster
            for cluster, counted in zip(cluster_ids, denominator, strict=True)
            if bool(counted)
        }
    )


def _imprecise(
    rate: float, n: int, n_images: int, effect: float, options: RegionMetricsOptions
) -> bool | None:
    """Whether this cell's interval half-width exceeds the caller's target margin.

    A degenerate bootstrap measures no design effect, so the inflation falls
    back to the nominal cluster size: the bound at an intra-image correlation
    of 1, where a whole image carries the information of a single point.
    """
    if options.target_margin is None:
        return None
    if math.isfinite(effect) and effect > 0.0:
        # A measured effect below 1 would read as more precise than
        # independent sampling, a state clustering cannot produce.
        inflation = max(effect, 1.0)
    elif n_images > 0:
        inflation = n / n_images
    else:
        inflation = 1.0
    required = required_n_for_margin(
        rate,
        options.target_margin,
        alpha=options.alpha,
        design_effect=inflation,
    )
    return n < required


def _image_affected_estimate(points: ScoredPoints, options: RegionMetricsOptions) -> RateEstimate:
    """The share of images carrying at least one out-of-region point.

    One row per image, so the cluster and the observation coincide.
    """
    order = list(dict.fromkeys(points.image_ids))
    if not order:
        return _blank_estimate(0, 0)
    affected: dict[str, bool] = dict.fromkeys(order, False)
    for image_id, out_of_region in zip(points.image_ids, points.pred_out_of_region, strict=True):
        if bool(out_of_region):
            affected[image_id] = True
    indicator = _mask(affected[image_id] for image_id in order)
    return _estimate(tuple(order), indicator, np.ones(len(order), dtype=bool), options)


def _region_masks(points: ScoredPoints) -> dict[str, NDArray[np.bool_]]:
    regions = sorted(set(points.image_region_ids))
    return {
        region: _mask(point_region == region for point_region in points.image_region_ids)
        for region in regions
    }


def _macro_estimate(
    points: ScoredPoints,
    masks: Mapping[str, NDArray[np.bool_]],
    numerator: NDArray[np.bool_],
    denominator: NDArray[np.bool_],
    options: RegionMetricsOptions,
) -> MacroEstimate:
    def macro(index: NDArray[np.intp]) -> float:
        rates: list[float] = []
        drawn_numerator = numerator[index]
        drawn_denominator = denominator[index]
        for mask in masks.values():
            selected = mask[index]
            n = int(np.count_nonzero(drawn_denominator & selected))
            if n == 0:
                continue
            rates.append(float(np.count_nonzero(drawn_numerator & selected) / n))
        if not rates:
            return math.nan
        return sum(rates) / len(rates)

    all_positions = np.arange(points.n_points, dtype=np.intp)
    value = macro(all_positions) if points.n_points else math.nan
    n_regions = sum(1 for mask in masks.values() if int(np.count_nonzero(denominator & mask)) > 0)
    if n_regions == 0:
        return MacroEstimate(value=math.nan, ci_low=math.nan, ci_high=math.nan, n_regions=0)

    def statistic(index: NDArray[np.intp]) -> float:
        drawn = macro(index)
        return value if math.isnan(drawn) else drawn

    ci_low, ci_high = cluster_bootstrap_ci(
        points.image_ids,
        statistic,
        n_resamples=options.n_resamples,
        alpha=options.alpha,
        seed=options.seed,
    )
    return MacroEstimate(value=value, ci_low=ci_low, ci_high=ci_high, n_regions=n_regions)


@dataclasses.dataclass(frozen=True)
class _CoreRates:
    """The four out-of-region rates and the accuracy of one slice of points."""

    oor_rate: RateEstimate
    oor_rate_disc: RateEstimate
    oor_rate_disc_gt: RateEstimate
    gt_oor_rate: RateEstimate
    accuracy: RateEstimate


def _core_rates(points: ScoredPoints, options: RegionMetricsOptions) -> _CoreRates:
    """The out-of-region rates, one per denominator, plus the accuracy floor.

    `oor_rate` and `oor_rate_disc` share a numerator: every out-of-region
    prediction, read against evaluable and region-discriminating predictions
    respectively. `oor_rate_disc_gt` restricts that numerator to points whose
    ground truth also discriminates, so it is a proportion of a denominator
    that depends only on the ground truth and the region map, and is
    therefore comparable across model versions.

    Accuracy runs through the same estimator as the rates, so it reaches a
    reader with the image-clustered interval they carry rather than as a bare
    proportion a reporting layer has to widen for itself.
    """
    return _CoreRates(
        oor_rate=_estimate(
            points.image_ids,
            points.pred_out_of_region,
            ~points.pred_region_unknown,
            options,
        ),
        oor_rate_disc=_estimate(
            points.image_ids, points.pred_out_of_region, points.pred_discriminating, options
        ),
        oor_rate_disc_gt=_estimate(
            points.image_ids,
            points.pred_out_of_region & points.gt_discriminating,
            points.gt_discriminating,
            options,
        ),
        gt_oor_rate=_estimate(
            points.image_ids, points.gt_out_of_region, ~points.gt_region_unknown, options
        ),
        accuracy=_estimate(
            points.image_ids,
            points.correct & points.gt_in_model_classes,
            points.gt_in_model_classes,
            options,
        ),
    )


def _discriminating_classes(points: ScoredPoints) -> list[str]:
    """Every label in this population's ground truth or predictions that
    region-discriminates, so a class predicted but never true still earns a
    place in `_macro_f1_disc`'s label list rather than dropping out of it.
    """
    allowed_by_label: dict[str, frozenset[str]] = {}
    for label, allowed in zip(points.gt_labels, points.gt_allowed, strict=True):
        allowed_by_label.setdefault(label, allowed)
    for label, allowed in zip(points.pred_labels, points.pred_allowed, strict=True):
        allowed_by_label.setdefault(label, allowed)
    return sorted(
        label
        for label, allowed in allowed_by_label.items()
        if is_region_discriminating(allowed, points.observed_region_ids)
    )


def _macro_f1_disc(points: ScoredPoints) -> tuple[int, float, float, float]:
    """Point count, macro precision, recall and F1 over ground-truth-discriminating points.

    `labels` fixes the class list to every region-discriminating label this
    population carries, so a class absent from the masked subset (predicted
    but never true here) still contributes a zero rather than shrinking the
    average's denominator.
    """
    mask = points.gt_discriminating
    n = int(np.count_nonzero(mask))
    if n == 0:
        return n, math.nan, math.nan, math.nan
    labels = _discriminating_classes(points)
    y_true = [label for label, flag in zip(points.gt_labels, mask, strict=True) if flag]
    y_pred = [label for label, flag in zip(points.pred_labels, mask, strict=True) if flag]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,  # pyright: ignore[reportArgumentType]  # sklearn stubs type as str only
    )
    return n, float(precision), float(recall), float(f1)


def _population_rates(
    points: ScoredPoints, name: str, options: RegionMetricsOptions
) -> PopulationRates:
    pred_evaluable = ~points.pred_region_unknown
    gt_evaluable = ~points.gt_region_unknown

    core = _core_rates(points, options)
    oor = core.oor_rate
    gt_oor = core.gt_oor_rate
    n_f1_disc_points, precision_macro_disc, recall_macro_disc, f1_macro_disc = _macro_f1_disc(
        points
    )

    masks = _region_masks(points)
    macro = RegionMacroRates(
        oor_rate=_macro_estimate(points, masks, points.pred_out_of_region, pred_evaluable, options),
        oor_rate_disc=_macro_estimate(
            points, masks, points.pred_out_of_region, points.pred_discriminating, options
        ),
        oor_rate_disc_gt=_macro_estimate(
            points,
            masks,
            points.pred_out_of_region & points.gt_discriminating,
            points.gt_discriminating,
            options,
        ),
        gt_oor_rate=_macro_estimate(points, masks, points.gt_out_of_region, gt_evaluable, options),
    )

    return PopulationRates(
        name=name,
        n_points=points.n_points,
        n_images=points.n_images,
        n_pred_region_unknown=int(np.count_nonzero(points.pred_region_unknown)),
        n_gt_region_unknown=int(np.count_nonzero(points.gt_region_unknown)),
        n_gt_outside_model_classes=int(np.count_nonzero(~points.gt_in_model_classes)),
        n_accuracy_points=core.accuracy.n,
        accuracy=core.accuracy,
        n_f1_disc_points=n_f1_disc_points,
        precision_macro_disc=precision_macro_disc,
        recall_macro_disc=recall_macro_disc,
        f1_macro_disc=f1_macro_disc,
        oor_rate=oor,
        oor_rate_disc=core.oor_rate_disc,
        oor_rate_disc_gt=core.oor_rate_disc_gt,
        gt_oor_rate=gt_oor,
        image_affected_rate=_image_affected_estimate(points, options),
        excess=_excess(points, oor, gt_oor, options),
        ratio_to_gt=_ratio_to_gt(points, oor, gt_oor, options),
        region_macro=macro,
    )


def _excess(
    points: ScoredPoints,
    oor: RateEstimate,
    gt_oor: RateEstimate,
    options: RegionMetricsOptions,
) -> DiffEstimate:
    value = oor.rate - gt_oor.rate
    if oor.n == 0 or gt_oor.n == 0 or points.n_points == 0:
        return DiffEstimate(value=value, ci_low=math.nan, ci_high=math.nan)
    ci_low, ci_high = paired_cluster_bootstrap_diff(
        points.image_ids,
        _ratio_statistic(points.pred_out_of_region, ~points.pred_region_unknown, oor.rate),
        _ratio_statistic(points.gt_out_of_region, ~points.gt_region_unknown, gt_oor.rate),
        n_resamples=options.n_resamples,
        alpha=options.alpha,
        seed=options.seed,
    )
    return DiffEstimate(value=value, ci_low=ci_low, ci_high=ci_high)


def _ratio_to_gt(
    points: ScoredPoints,
    oor: RateEstimate,
    gt_oor: RateEstimate,
    options: RegionMetricsOptions,
) -> RatioEstimate:
    """The model rate as a multiple of the ground-truth floor, with an interval.

    Paired like `_excess`: one draw of images feeds both rates, so the
    variation they share cancels instead of inflating the multiple. A draw
    whose floor holds no event has no multiple to report and contributes an
    infinite one, which is how a floor too thin to divide by reads.
    """
    value = oor.rate / gt_oor.rate if gt_oor.rate > 0.0 else math.nan
    if math.isnan(value) or oor.n == 0 or gt_oor.n == 0 or points.n_points == 0:
        return RatioEstimate(
            value=value, ci_low=math.nan, ci_high=math.nan, n_draws=0, n_nonfinite_draws=0
        )

    model_rate = _ratio_statistic(points.pred_out_of_region, ~points.pred_region_unknown, oor.rate)
    floor_rate = _ratio_statistic(points.gt_out_of_region, ~points.gt_region_unknown, gt_oor.rate)

    def ratio(index: NDArray[np.intp]) -> float:
        floor = floor_rate(index)
        return math.inf if floor <= 0.0 else model_rate(index) / floor

    draws = cluster_bootstrap_draws(
        points.image_ids,
        ratio,
        n_resamples=options.n_resamples,
        seed=options.seed,
    )
    ci_low, ci_high = _conditional_percentile_interval(draws, options.alpha)
    return RatioEstimate(
        value=value,
        ci_low=ci_low,
        ci_high=ci_high,
        n_draws=int(draws.size),
        n_nonfinite_draws=int(np.count_nonzero(~np.isfinite(draws))),
    )


def _conditional_percentile_interval(
    draws: NDArray[np.float64], alpha: float
) -> tuple[float, float]:
    """A percentile interval over the finite draws, unbounded where the tail is not.

    Percentiling the whole array would interpolate between infinite draws,
    which arrives as NaN -- an interval that reads "not computed" where the
    draws say "unbounded" -- so the infinities are counted out first.

    Each bound is infinite when the infinite draws, which sort above every
    finite one, reach it: the upper bound sits alpha/2 from the top, the lower
    bound 1 - alpha/2.
    """
    finite = draws[np.isfinite(draws)]
    nonfinite_share = 1.0 - finite.size / draws.size
    return (
        math.inf
        if nonfinite_share > 1.0 - alpha / 2.0
        else float(np.percentile(finite, 100.0 * alpha / 2.0)),
        math.inf
        if nonfinite_share > alpha / 2.0
        else float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0))),
    )


# The tables below render these rates into DataFrames a report reads. Row
# builders call back into the same rate estimator used above (`_estimate`,
# `_core_rates`) rather than recomputing a rate, so a table's number cannot
# drift from the population-level estimate it reflects.

CONFUSION_ROW_LIMIT = 50

PER_REGION_BASE_COLUMNS = (
    "population",
    "region_id",
    "region_name",
    "n_images",
    "n_points",
    "n_accuracy_points",
    "accuracy",
)
RATE_PREFIXES = ("oor_rate", "oor_rate_disc", "oor_rate_disc_gt", "gt_oor_rate")
PER_LABEL_COLUMNS = (
    "label",
    "label_name",
    "allowed_region_ids",
    "allowed_region_names",
    "n_predicted",
    "n_out_of_region",
    "rate",
    "wilson_low",
    "wilson_high",
    "upper_bound",
    "upper_bound_n_images",
    "n_ground_truth",
    "cumulative_share",
)
CONFUSION_COLUMNS = (
    "image_region_id",
    "image_region_name",
    "gt_label",
    "gt_label_name",
    "pred_label",
    "pred_label_name",
    "n",
)

# Every RateEstimate field but the rate itself, in declaration order -- the
# suffixes _prefixed and _per_region_columns key each per-region column under.
ESTIMATE_SUFFIXES: tuple[str, ...] = tuple(
    field.name for field in dataclasses.fields(RateEstimate) if field.name != "rate"
)

PER_DIRECTION_COLUMNS = (
    "population",
    "image_region_id",
    "image_region_name",
    "excluded_region_id",
    "excluded_region_name",
    "n_out_of_region",
    "n_out_of_region_events",
    "n_points",
    "rate",
    *(suffix for suffix in ESTIMATE_SUFFIXES if suffix not in ("k", "n")),
)


def _prefixed(prefix: str, estimate: RateEstimate) -> dict[str, object]:
    return {
        prefix: estimate.rate,
        f"{prefix}_k": estimate.k,
        f"{prefix}_n": estimate.n,
        f"{prefix}_ci_low": estimate.ci_low,
        f"{prefix}_ci_high": estimate.ci_high,
        f"{prefix}_wilson_low": estimate.wilson_low,
        f"{prefix}_wilson_high": estimate.wilson_high,
        f"{prefix}_design_effect": estimate.design_effect,
        f"{prefix}_upper_bound": estimate.upper_bound,
        f"{prefix}_upper_bound_n_images": estimate.upper_bound_n_images,
        f"{prefix}_imprecise": estimate.imprecise,
    }


def _per_region_columns() -> tuple[str, ...]:
    columns: list[str] = list(PER_REGION_BASE_COLUMNS)
    for prefix in RATE_PREFIXES:
        columns.append(prefix)
        columns.extend(f"{prefix}_{suffix}" for suffix in ESTIMATE_SUFFIXES)
    return tuple(columns)


def _per_region_rows(
    points: ScoredPoints,
    population: str,
    options: RegionMetricsOptions,
    region_names: Mapping[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for region, mask in _region_masks(points).items():
        slice_ = _restrict(points, np.flatnonzero(mask))
        core = _core_rates(slice_, options)
        row: dict[str, object] = {
            "population": population,
            "region_id": region,
            "region_name": display_name(region, region_names),
            "n_images": slice_.n_images,
            "n_points": slice_.n_points,
            "n_accuracy_points": core.accuracy.n,
            "accuracy": core.accuracy.rate,
        }
        for prefix in RATE_PREFIXES:
            estimate: RateEstimate = getattr(core, prefix)
            row.update(_prefixed(prefix, estimate))
        rows.append(row)
    return rows


def _direction_columns(points: ScoredPoints) -> tuple[str, ...]:
    """Regions a direction can point at: every observed one, plus every region
    an out-of-region label does belong to."""
    regions = set(points.observed_region_ids)
    for allowed, out_of_region in zip(points.pred_allowed, points.pred_out_of_region, strict=True):
        if bool(out_of_region):
            regions |= allowed
    return tuple(sorted(regions))


def _direction_matrix(
    points: ScoredPoints, columns: Sequence[str], region_names: Mapping[str, str]
) -> pd.DataFrame:
    """Out-of-region counts by image region and the region the label belongs to.

    A label permitted in several regions counts once under each of them, so a
    row sums to at least the number of events it covers;
    `per_direction.n_out_of_region_events` carries the unduplicated count.

    Both axes are labelled by region name, falling back to the region id, which
    keeps every cell numeric. `per_direction` is the long form of this same
    table and carries each pair's ids beside its names.
    """
    rows = sorted(points.observed_region_ids)
    counts: Counter[tuple[str, str]] = Counter()
    for region, allowed, out_of_region in zip(
        points.image_region_ids, points.pred_allowed, points.pred_out_of_region, strict=True
    ):
        if not bool(out_of_region):
            continue
        for target in allowed:
            counts[(region, target)] += 1
    return pd.DataFrame(
        [[counts[(row, column)] for column in columns] for row in rows],
        index=pd.Index([display_name(row, region_names) for row in rows]),
        columns=pd.Index([display_name(column, region_names) for column in columns]),
        dtype=int,
    )


def _direction_rate_fields(estimate: RateEstimate) -> dict[str, object]:
    """A rate's fields under `PER_DIRECTION_COLUMNS`' own names.

    `n_out_of_region`/`n_points` are this table's names for `k`/`n`; every
    other field keeps `RateEstimate`'s own name.
    """
    fields: dict[str, object] = {
        "n_out_of_region": estimate.k,
        "n_points": estimate.n,
        "rate": estimate.rate,
    }
    fields.update(
        (suffix, getattr(estimate, suffix))
        for suffix in ESTIMATE_SUFFIXES
        if suffix not in ("k", "n")
    )
    return fields


def _per_direction_rows(
    points: ScoredPoints,
    population: str,
    columns: Sequence[str],
    options: RegionMetricsOptions,
    region_names: Mapping[str, str],
) -> list[dict[str, object]]:
    """One row per ordered (image region, excluded region) pair.

    The denominator is the exposure of the image region, never pooled across
    directions: opposite directions differ in available points by two orders
    of magnitude, so their rates are not comparable and must not be averaged.

    A prediction permitted in several regions is out of region towards each of
    them, so `n_out_of_region` counts it under each and summing a region's
    directions overstates the incidents it carries.
    `n_out_of_region_events` is that unduplicated count -- distinct
    out-of-region points in the image region -- and repeats on each of the
    region's rows.
    """
    rows: list[dict[str, object]] = []
    for region, mask in _region_masks(points).items():
        slice_ = _restrict(points, np.flatnonzero(mask))
        evaluable = ~slice_.pred_region_unknown
        n_events = int(np.count_nonzero(slice_.pred_out_of_region))
        for excluded in columns:
            if excluded == region:
                continue
            belongs = _mask(excluded in allowed for allowed in slice_.pred_allowed)
            estimate = _estimate(
                slice_.image_ids, slice_.pred_out_of_region & belongs, evaluable, options
            )
            rows.append(
                {
                    "population": population,
                    "image_region_id": region,
                    "image_region_name": display_name(region, region_names),
                    "excluded_region_id": excluded,
                    "excluded_region_name": display_name(excluded, region_names),
                    "n_out_of_region_events": n_events,
                    **_direction_rate_fields(estimate),
                }
            )
    return rows


def _per_label_table(
    points: ScoredPoints,
    label_names: Mapping[str, str],
    region_names: Mapping[str, str],
    options: RegionMetricsOptions,
) -> pd.DataFrame:
    """Region-discriminating classes, heaviest incident count first.

    `cumulative_share` runs over the out-of-region counts, so a reader can
    read off how few labels carry most of the incidents. The zero-cell upper
    bound counts the images the label was predicted in, not the points, since
    the rule of three counts independent trials.
    """
    predicted: Counter[str] = Counter(points.pred_labels)
    ground_truth: Counter[str] = Counter(points.gt_labels)
    out_of_region: Counter[str] = Counter(
        label
        for label, flag in zip(points.pred_labels, points.pred_out_of_region, strict=True)
        if bool(flag)
    )
    allowed_by_label = dict(zip(points.pred_labels, points.pred_allowed, strict=True))
    images_by_label: dict[str, set[str]] = {}
    for label, image_id in zip(points.pred_labels, points.image_ids, strict=True):
        images_by_label.setdefault(label, set()).add(image_id)

    ranked: list[tuple[int, int, str]] = []
    for label, n_predicted in predicted.items():
        allowed = allowed_by_label[label]
        if not is_region_discriminating(allowed, points.observed_region_ids):
            continue
        ranked.append((out_of_region[label], n_predicted, label))
    ranked.sort(key=lambda entry: (-entry[0], -entry[1], entry[2]))

    total = sum(entry[0] for entry in ranked)
    rows: list[dict[str, object]] = []
    running = 0
    for k, n_predicted, label in ranked:
        running += k
        wilson_low, wilson_high = wilson_ci(k, n_predicted, options.alpha)
        n_images = len(images_by_label[label])
        allowed_ids = tuple(sorted(allowed_by_label[label]))
        rows.append(
            {
                "label": label,
                "label_name": display_name(label, label_names),
                "allowed_region_ids": allowed_ids,
                "allowed_region_names": tuple(
                    display_name(region_id, region_names) for region_id in allowed_ids
                ),
                "n_predicted": n_predicted,
                "n_out_of_region": k,
                "rate": k / n_predicted,
                "wilson_low": wilson_low,
                "wilson_high": wilson_high,
                "upper_bound": rule_of_three(n_images) if k == 0 else None,
                "upper_bound_n_images": n_images,
                "n_ground_truth": ground_truth[label],
                "cumulative_share": running / total if total else math.nan,
            }
        )
    return columns_frame(rows, PER_LABEL_COLUMNS)


def _confusion_table(
    points: ScoredPoints, label_names: Mapping[str, str], region_names: Mapping[str, str]
) -> pd.DataFrame:
    """The commonest (image region, truth, prediction) triples among incidents."""
    counts: Counter[tuple[str, str, str]] = Counter()
    for region, truth, prediction, out_of_region in zip(
        points.image_region_ids,
        points.gt_labels,
        points.pred_labels,
        points.pred_out_of_region,
        strict=True,
    ):
        if bool(out_of_region):
            counts[(region, truth, prediction)] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    rows: list[dict[str, object]] = [
        {
            "image_region_id": region,
            "image_region_name": display_name(region, region_names),
            "gt_label": truth,
            "gt_label_name": display_name(truth, label_names),
            "pred_label": prediction,
            "pred_label_name": display_name(prediction, label_names),
            "n": count,
        }
        for (region, truth, prediction), count in ordered[:CONFUSION_ROW_LIMIT]
    ]
    return columns_frame(rows, CONFUSION_COLUMNS)


def display_name(identifier: str, names: Mapping[str, str]) -> str:
    """The display name for an id, or the id where no name resolves.

    An empty cell reads as "this thing has no name"; the id reads as
    "unresolved", which is the state a partial name snapshot leaves behind.
    """
    return names.get(identifier) or identifier


def columns_frame(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    """A DataFrame that keeps its columns even with no rows."""
    if not rows:
        return pd.DataFrame(columns=pd.Index(columns))
    return pd.DataFrame(list(rows), columns=pd.Index(columns))
