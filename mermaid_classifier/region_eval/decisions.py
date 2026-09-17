"""Region-mismatch statistics that choose between mitigations.

`metrics` says how bad the problem is. These say what to do about it, and each
one is built to separate two mitigations that a rate alone cannot:

`region_blind_baseline` -- the ratio of the measured out-of-region rate to the
rate the same predictions produce once region is shuffled between images. The
label space is asymmetric, so a raw rate is uninterpretable: a model drawing
labels at random with the right marginal frequencies still lands some of them
outside the region. A ratio near 1 says the model reads nothing from the image
about where it was taken, which is the case for a hard constraint. A ratio well
below 1 says region is already implicit in the features and only a tail is
leaking, which is the case for reweighting, regional training data, or dropping
the handful of labels responsible.

`masking_counterfactual` -- what a masking layer at inference would buy and
cost, priced before anyone builds one. Its accuracy delta pools two directions
that a scientist values very differently, so the predictions masking corrects
and the predictions it ruins are counted apart, and the confidence it has to
overturn travels alongside as a margin.

`within_branch_share` -- how much of the out-of-region traffic is the right
taxon in the wrong ocean rather than ordinary confusion. Masking cleans up the
first and merely reshuffles the second.

`confidence_stratification` -- whether a confidence threshold could suppress
these predictions at all. An AUROC near 0.5 settles that cheaply, before
anyone tunes a cutoff that cannot work.

Pure formulas over arrays. Nothing here reaches S3, MLflow, a model file or a
plotting backend, and nothing here reads the live benthic-attribute library:
the region map arrives from the caller as a frozen snapshot, so a curation
change upstream cannot move a model's score for reasons unrelated to the model.
Taxonomic ancestry arrives the same way. `pyspacer.metrics._taxonomy_helpers`
holds the equivalent lookups, but importing it pulls boto3, duckdb, matplotlib
and mlflow in through its package `__init__`, which this module stays clear of.

Every entry point filters points whose image region is unrecorded and reports
how many. The region predicates raise on that sentinel and a metrics
orchestrator swallows what a metric group raises, so an unfiltered point would
erase the whole result rather than skew it.
"""

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from mermaid_classifier.common.benthic_attributes import split_ba_gf
from mermaid_classifier.region_eval.region_rules import (
    cluster_bootstrap_ci,
    is_out_of_region,
    is_region_discriminating,
    partition_by_recorded_region,
    permutation_baseline,
)

DEFAULT_N_PERMUTATIONS = 1000
DEFAULT_N_RESAMPLES = 2000
DEFAULT_CONFIDENCE_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclasses.dataclass(frozen=True)
class RegionBlindBaseline:
    """A measured out-of-region rate against the rate region-blindness implies.

    `observed_rate` and the permuted rates share the denominator of
    region-discriminating predictions, which permutation leaves fixed: whether
    a label could ever be out of region depends on the set of regions the data
    spans, and shuffling regions between images does not change that set. The
    denominator therefore cancels out of `ratio`, which is the count ratio
    whichever denominator a reader prefers.

    `baseline_sd` and the percentile interval describe the permutation
    distribution itself, not the uncertainty of a sample: a ratio of 0.8 means
    one thing against a baseline spread of 0.01 and nothing at all against a
    spread of 0.3.

    `ratio` is NaN where the baseline rate is zero, which is a corpus in which
    no assignment of regions could have placed a prediction outside one.
    """

    n_points: int
    n_images: int
    n_unrecorded_region_excluded: int
    n_region_unknown: int
    n_discriminating: int
    n_out_of_region: int
    observed_rate: float
    baseline_rate: float
    baseline_sd: float
    baseline_ci_low: float
    baseline_ci_high: float
    ratio: float
    n_permutations: int


def region_blind_baseline(
    *,
    image_ids: Sequence[str],
    image_region_ids: Sequence[str],
    pred_labels: Sequence[str],
    region_ids_by_attribute: Mapping[str, frozenset[str]],
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    alpha: float = 0.05,
    seed: int = 0,
) -> RegionBlindBaseline:
    """Score the out-of-region rate against a region-blind null.

    Regions are permuted between images rather than between points, since an
    image's region is a property of the image; a within-image shuffle would
    leave every point where it was. `region_ids_by_attribute` is a frozen
    snapshot keyed by benthic-attribute id, and an attribute it omits counts
    as unrecorded.

    Seeded, and deterministic for a given seed and input.
    """
    _check_lengths(
        image_ids=len(image_ids),
        image_region_ids=len(image_region_ids),
        pred_labels=len(pred_labels),
    )
    scorable, n_excluded = _partition(image_region_ids)
    images = [image_ids[position] for position in scorable]
    regions = [image_region_ids[position] for position in scorable]
    allowed = [
        _allowed_regions(pred_labels[position], region_ids_by_attribute) for position in scorable
    ]

    observed_regions = frozenset(regions)
    n_region_unknown = sum(1 for allowed_ids in allowed if not allowed_ids)
    n_discriminating = sum(
        1 for allowed_ids in allowed if is_region_discriminating(allowed_ids, observed_regions)
    )
    n_out_of_region = sum(
        1
        for allowed_ids, region in zip(allowed, regions, strict=True)
        if is_out_of_region(allowed_ids, region)
    )

    if n_discriminating == 0:
        return RegionBlindBaseline(
            n_points=len(scorable),
            n_images=len(set(images)),
            n_unrecorded_region_excluded=n_excluded,
            n_region_unknown=n_region_unknown,
            n_discriminating=0,
            n_out_of_region=n_out_of_region,
            observed_rate=math.nan,
            baseline_rate=math.nan,
            baseline_sd=math.nan,
            baseline_ci_low=math.nan,
            baseline_ci_high=math.nan,
            ratio=math.nan,
            n_permutations=0,
        )

    def rate(permuted_regions: NDArray[np.object_]) -> float:
        events = sum(
            1
            for allowed_ids, region in zip(allowed, permuted_regions, strict=True)
            if is_out_of_region(allowed_ids, str(region))
        )
        return events / n_discriminating

    recording, per_permutation = _recording(rate)
    baseline_rate = permutation_baseline(
        images, regions, recording, n_permutations=n_permutations, seed=seed
    )
    distribution = np.asarray(per_permutation, dtype=np.float64)
    observed_rate = n_out_of_region / n_discriminating

    return RegionBlindBaseline(
        n_points=len(scorable),
        n_images=len(set(images)),
        n_unrecorded_region_excluded=n_excluded,
        n_region_unknown=n_region_unknown,
        n_discriminating=n_discriminating,
        n_out_of_region=n_out_of_region,
        observed_rate=observed_rate,
        baseline_rate=baseline_rate,
        baseline_sd=float(np.std(distribution)),
        baseline_ci_low=float(np.percentile(distribution, 100.0 * alpha / 2.0)),
        baseline_ci_high=float(np.percentile(distribution, 100.0 * (1.0 - alpha / 2.0))),
        ratio=observed_rate / baseline_rate if baseline_rate > 0.0 else math.nan,
        n_permutations=n_permutations,
    )


@dataclasses.dataclass(frozen=True)
class MaskingCounterfactual:
    """What a hard region constraint at inference would cost and buy.

    An **upper bound** on masking's benefit. It assumes every image's recorded
    region is correct, so each prediction masking "fixes" is a fix only where
    the region assignment behind it holds; a wrong region turns the same
    machinery into a source of error this number cannot see.

    `n_fixed` and `n_broken` are the two directions `accuracy_delta` pools. A
    delta of zero is one fix against one break as readily as it is no change at
    all, and the two cost a scientist very different things.

    `margin_*` summarise `p_top - max(p over in-region classes)` over the
    predictions whose top-1 is out of region. A median near zero means the
    in-region runner-up was already close and masking is nearly free; a large
    median means masking overturns predictions the model was confident in.

    Accuracy spans the points whose ground truth is inside `model_classes`,
    since no model carrying those classes can be right about the others.
    `changed_share` spans every scored point, needing no ground truth.
    """

    n_points: int
    n_images: int
    n_unrecorded_region_excluded: int
    n_region_unknown_classes: int
    n_no_permitted_class: int
    n_accuracy_points: int
    accuracy_unmasked: float
    accuracy_masked: float
    accuracy_delta: float
    accuracy_delta_ci_low: float
    accuracy_delta_ci_high: float
    n_changed: int
    changed_share: float
    n_fixed: int
    n_broken: int
    margin_n: int
    margin_median: float
    margin_p90: float
    margin_max: float


def masking_counterfactual(
    *,
    image_ids: Sequence[str],
    image_region_ids: Sequence[str],
    gt_labels: Sequence[str],
    probabilities: NDArray[np.float64],
    model_classes: Sequence[str],
    region_ids_by_attribute: Mapping[str, frozenset[str]],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> MaskingCounterfactual:
    """Re-score every prediction with the out-of-region classes zeroed.

    `probabilities` is one row per point over `model_classes` in that column
    order. Classes incompatible with the image's region are zeroed and the
    remaining row is re-argmaxed; renormalising first would change the scale
    but not the argmax (when probabilities are non-negative), so it is skipped.
    A class with no recorded regions is never zeroed: unrecorded is not
    permitted-nowhere, and masking one out would charge the model for a gap in
    the region lists.

    Where an image's region permits no class at all the prediction is left
    exactly as the model made it and counted in `n_no_permitted_class`, since
    a fully zeroed row has no argmax to take.

    The interval on the accuracy delta resamples images: annotation points
    cluster around 25 to an image and a point-level interval understates its
    width several-fold. A resample drawing no scorable ground truth reads as no
    difference.
    """
    probability_matrix = np.asarray(probabilities, dtype=np.float64)
    _check_lengths(
        image_ids=len(image_ids),
        image_region_ids=len(image_region_ids),
        gt_labels=len(gt_labels),
        probabilities=len(probability_matrix),
    )
    if probability_matrix.ndim != 2 or probability_matrix.shape[1] != len(model_classes):
        raise ValueError(
            f"probabilities must be (n_points, {len(model_classes)}),"
            f" got {probability_matrix.shape}"
        )

    scorable, n_excluded = _partition(image_region_ids)
    images = [image_ids[position] for position in scorable]
    regions = [image_region_ids[position] for position in scorable]
    truths = [gt_labels[position] for position in scorable]
    point_probabilities = probability_matrix[scorable]

    class_allowed = [_allowed_regions(label, region_ids_by_attribute) for label in model_classes]
    unique_regions, region_of_point = np.unique(np.asarray(regions), return_inverse=True)
    permitted_by_unique_region = np.array(
        [
            [not is_out_of_region(allowed_ids, region) for allowed_ids in class_allowed]
            for region in unique_regions
        ],
        dtype=bool,
    ).reshape(len(unique_regions), len(model_classes))
    permitted_matrix = permitted_by_unique_region[region_of_point]
    class_index = {label: index for index, label in enumerate(model_classes)}

    n_points = len(scorable)
    point_positions = np.arange(n_points)
    unmasked_index = np.argmax(point_probabilities, axis=1)
    any_permitted = permitted_matrix.any(axis=1)
    masked_probs = np.where(permitted_matrix, point_probabilities, 0.0)
    masked_index = np.where(any_permitted, np.argmax(masked_probs, axis=1), unmasked_index)
    n_no_permitted_class = int(np.count_nonzero(~any_permitted))

    truth_index = np.array([class_index.get(truth, -1) for truth in truths], dtype=np.intp)
    scorable_truth = truth_index >= 0
    unmasked_correct = truth_index == unmasked_index
    masked_correct = truth_index == masked_index
    changed = masked_index != unmasked_index

    # The margin covers a top-1 pick the region excludes (unpermitted) where the
    # region still permits something else; a region with no permitted class has
    # nothing for the pick to be measured against.
    unmasked_permitted = permitted_matrix[point_positions, unmasked_index]
    margin_eligible = any_permitted & ~unmasked_permitted
    best_permitted = masked_probs.max(axis=1)
    unmasked_confidence = point_probabilities[point_positions, unmasked_index]
    margin_values = (unmasked_confidence - best_permitted)[margin_eligible]

    n_accuracy_points = int(scorable_truth.sum())
    accuracy_unmasked = _ratio(int(unmasked_correct.sum()), n_accuracy_points)
    accuracy_masked = _ratio(int(masked_correct.sum()), n_accuracy_points)

    def delta(index: NDArray[np.intp]) -> float:
        denominator = int(scorable_truth[index].sum())
        if denominator == 0:
            return 0.0
        return float(masked_correct[index].sum() - unmasked_correct[index].sum()) / denominator

    if n_accuracy_points == 0:
        delta_low, delta_high = math.nan, math.nan
    else:
        delta_low, delta_high = cluster_bootstrap_ci(
            images, delta, n_resamples=n_resamples, alpha=alpha, seed=seed
        )

    return MaskingCounterfactual(
        n_points=n_points,
        n_images=len(set(images)),
        n_unrecorded_region_excluded=n_excluded,
        n_region_unknown_classes=sum(1 for allowed_ids in class_allowed if not allowed_ids),
        n_no_permitted_class=n_no_permitted_class,
        n_accuracy_points=n_accuracy_points,
        accuracy_unmasked=accuracy_unmasked,
        accuracy_masked=accuracy_masked,
        accuracy_delta=accuracy_masked - accuracy_unmasked,
        accuracy_delta_ci_low=delta_low,
        accuracy_delta_ci_high=delta_high,
        n_changed=int(changed.sum()),
        changed_share=_ratio(int(changed.sum()), n_points),
        n_fixed=int((masked_correct & ~unmasked_correct).sum()),
        n_broken=int((unmasked_correct & ~masked_correct).sum()),
        margin_n=int(margin_values.size),
        margin_median=_quantile(margin_values, 50.0),
        margin_p90=_quantile(margin_values, 90.0),
        margin_max=_quantile(margin_values, 100.0),
    )


@dataclasses.dataclass(frozen=True)
class WithinBranchShare:
    """How much of the out-of-region traffic is the right group, wrong ocean.

    A share near 1 says the model has the broad taxon right and only the ocean
    wrong, which a region mask corrects cleanly. A share near 0 says these are
    ordinary misclassifications that happen to cross a region line, and masking
    them away swaps one wrong label for another.

    `n_ancestry_unknown` holds the events whose predicted or true attribute the
    ancestry snapshot does not place. They leave both halves of the share, on
    the same grounds as an attribute with no recorded regions: not placeable is
    not the same as placed apart.
    """

    n_points: int
    n_unrecorded_region_excluded: int
    n_region_unknown: int
    n_out_of_region: int
    n_ancestry_unknown: int
    n_evaluable: int
    n_within_branch: int
    share: float


def within_branch_share(
    *,
    image_region_ids: Sequence[str],
    gt_labels: Sequence[str],
    pred_labels: Sequence[str],
    region_ids_by_attribute: Mapping[str, frozenset[str]],
    ancestry_by_attribute: Mapping[str, Sequence[str]],
) -> WithinBranchShare:
    """The share of out-of-region predictions sharing an ancestor with the truth.

    `ancestry_by_attribute` maps a benthic-attribute id to its root-to-leaf
    path, root first and ending in the attribute itself. Two attributes share
    an ancestor exactly when their paths agree at the root, which is the
    condition under which a lowest common ancestor exists.

    The ancestry is a caller-supplied snapshot for the same reason the region
    map is: a taxonomy edit upstream must not move a model's score.
    """
    _check_lengths(
        image_region_ids=len(image_region_ids),
        gt_labels=len(gt_labels),
        pred_labels=len(pred_labels),
    )
    scorable, n_excluded = _partition(image_region_ids)

    n_region_unknown = 0
    n_out_of_region = 0
    n_ancestry_unknown = 0
    n_within_branch = 0
    for position in scorable:
        pred_attribute = split_ba_gf(pred_labels[position])[0]
        allowed = region_ids_by_attribute.get(pred_attribute, frozenset())
        if not allowed:
            n_region_unknown += 1
            continue
        if not is_out_of_region(allowed, image_region_ids[position]):
            continue

        n_out_of_region += 1
        gt_attribute = split_ba_gf(gt_labels[position])[0]
        pred_path = ancestry_by_attribute.get(pred_attribute)
        gt_path = ancestry_by_attribute.get(gt_attribute)
        if pred_path is None or gt_path is None:
            n_ancestry_unknown += 1
        elif _shares_ancestor(gt_path, pred_path):
            n_within_branch += 1

    n_evaluable = n_out_of_region - n_ancestry_unknown
    return WithinBranchShare(
        n_points=len(scorable),
        n_unrecorded_region_excluded=n_excluded,
        n_region_unknown=n_region_unknown,
        n_out_of_region=n_out_of_region,
        n_ancestry_unknown=n_ancestry_unknown,
        n_evaluable=n_evaluable,
        n_within_branch=n_within_branch,
        share=_ratio(n_within_branch, n_evaluable),
    )


@dataclasses.dataclass(frozen=True)
class ConfidenceBin:
    """One confidence band and how much of it went out of region.

    The band is `[lower, upper)`, except the last, which closes on `upper` so
    a prediction at full confidence lands somewhere. `rate` is NaN for a band
    no prediction fell into.
    """

    lower: float
    upper: float
    n: int
    n_out_of_region: int
    rate: float


@dataclasses.dataclass(frozen=True)
class ConfidenceStratification:
    """Whether a confidence threshold could suppress out-of-region predictions.

    `auroc` is the probability that a randomly drawn out-of-region prediction
    carries more confidence than a randomly drawn in-region one, over the
    region-discriminating predictions alone. A value near 0.5 says confidence
    does not separate the two and rules a threshold out as a mitigation; the
    bins say where on the scale any separation sits.

    Predictions permitted everywhere the data goes could never have been out of
    region, so they carry no evidence either way and leave both the bins and
    the AUROC. Predictions whose attribute has no recorded regions leave for
    the separate reason that the question cannot be answered for them, and are
    counted in `n_region_unknown`.

    `auroc` is NaN where either class is empty, which is no evidence rather
    than the 0.5 that means evidence of nothing.
    """

    n_points: int
    n_unrecorded_region_excluded: int
    n_region_unknown: int
    n_discriminating: int
    n_out_of_region: int
    auroc: float
    bins: tuple[ConfidenceBin, ...]


def confidence_stratification(
    *,
    image_region_ids: Sequence[str],
    pred_labels: Sequence[str],
    pred_confidences: Sequence[float],
    region_ids_by_attribute: Mapping[str, frozenset[str]],
    bin_edges: Sequence[float] = DEFAULT_CONFIDENCE_BIN_EDGES,
) -> ConfidenceStratification:
    """Bin the out-of-region rate by confidence, and rank one against the other.

    `pred_confidences[i]` is the probability the model gave the label it
    predicted. Every confidence must fall inside `bin_edges`, which are
    strictly increasing: a score outside them would vanish from the bins while
    still counting in the totals.
    """
    _check_lengths(
        image_region_ids=len(image_region_ids),
        pred_labels=len(pred_labels),
        pred_confidences=len(pred_confidences),
    )
    if len(bin_edges) < 2:
        raise ValueError(f"bin_edges must hold at least two edges, got {list(bin_edges)}")
    if any(upper <= lower for lower, upper in zip(bin_edges, bin_edges[1:], strict=False)):
        raise ValueError(f"bin_edges must increase strictly, got {list(bin_edges)}")

    scorable, n_excluded = _partition(image_region_ids)
    regions = [image_region_ids[position] for position in scorable]
    observed_regions = frozenset(regions)

    n_region_unknown = 0
    scores: list[float] = []
    events: list[bool] = []
    for position in scorable:
        confidence = float(pred_confidences[position])
        if not bin_edges[0] <= confidence <= bin_edges[-1]:
            raise ValueError(
                f"confidence {confidence} falls outside the bins [{bin_edges[0]}, {bin_edges[-1]}]"
            )
        allowed = _allowed_regions(pred_labels[position], region_ids_by_attribute)
        if not allowed:
            n_region_unknown += 1
            continue
        if not is_region_discriminating(allowed, observed_regions):
            continue
        scores.append(confidence)
        events.append(is_out_of_region(allowed, image_region_ids[position]))

    score_values = np.asarray(scores, dtype=np.float64)
    event_values = np.asarray(events, dtype=bool)

    bins: list[ConfidenceBin] = []
    for index, (lower, upper) in enumerate(zip(bin_edges, bin_edges[1:], strict=False)):
        last = index == len(bin_edges) - 2
        in_bin = (score_values >= lower) & (score_values <= upper if last else score_values < upper)
        n_in_bin = int(in_bin.sum())
        n_events = int(event_values[in_bin].sum())
        bins.append(
            ConfidenceBin(
                lower=float(lower),
                upper=float(upper),
                n=n_in_bin,
                n_out_of_region=n_events,
                rate=_ratio(n_events, n_in_bin),
            )
        )

    return ConfidenceStratification(
        n_points=len(scorable),
        n_unrecorded_region_excluded=n_excluded,
        n_region_unknown=n_region_unknown,
        n_discriminating=len(scores),
        n_out_of_region=int(event_values.sum()),
        auroc=_auroc(score_values, event_values),
        bins=tuple(bins),
    )


def _recording(
    statistic: Callable[[NDArray[np.object_]], float],
) -> tuple[Callable[[NDArray[np.object_]], float], list[float]]:
    """A statistic that also appends each value it returns to a list.

    `permutation_baseline` calls its statistic exactly once per permutation and
    returns only the mean, so the list it fills is the permutation distribution
    the dispersion is read from.
    """
    values: list[float] = []

    def recording(groups: NDArray[np.object_]) -> float:
        value = statistic(groups)
        values.append(value)
        return value

    return recording, values


def _allowed_regions(
    label: str, region_ids_by_attribute: Mapping[str, frozenset[str]]
) -> frozenset[str]:
    """The regions a BA-GF label's benthic attribute is recorded in."""
    return region_ids_by_attribute.get(split_ba_gf(label)[0], frozenset())


def _partition(image_region_ids: Sequence[str]) -> tuple[list[int], int]:
    """Positions with a recorded image region, and how many lack one."""
    scorable, unscorable = partition_by_recorded_region(image_region_ids)
    return scorable, len(unscorable)


def _check_lengths(**lengths: int) -> None:
    """Reject ragged parallel inputs, which would silently misalign points."""
    if len(set(lengths.values())) > 1:
        raise ValueError(f"inputs must be the same length, got {lengths}")


def _ratio(numerator: int, denominator: int) -> float:
    """A proportion, or NaN where nothing was counted."""
    return numerator / denominator if denominator else math.nan


def _quantile(values: NDArray[np.float64], percentile: float) -> float:
    """A percentile of a distribution, or NaN where it holds no values."""
    return float(np.percentile(values, percentile)) if values.size else math.nan


def _shares_ancestor(path_a: Sequence[str], path_b: Sequence[str]) -> bool:
    """Whether two root-to-leaf paths have a lowest common ancestor.

    Paths that agree at the root agree on at least that node; paths that
    differ there share nothing, which is the None case of an LCA walk.
    """
    return bool(path_a) and bool(path_b) and path_a[0] == path_b[0]


def _auroc(scores: NDArray[np.float64], positive: NDArray[np.bool_]) -> float:
    """Area under the ROC curve of `scores` separating positives from the rest.

    Ties resolve to their average rank internally, which sends an all-tied
    input to exactly 0.5 rather than to whichever extreme the sort order
    implied. NaN where either class is empty, a case `roc_auc_score` raises on
    rather than answers.
    """
    n_positive = int(positive.sum())
    n_negative = int(positive.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return math.nan
    return float(roc_auc_score(positive, scores))
