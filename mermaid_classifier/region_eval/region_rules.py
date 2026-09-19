"""Region predicates and the statistics core for region-mismatch metrics.

Pure functions: no I/O, no network, no settings. Two groups.

The predicates decide whether a predicted label belongs in the region an image
came from. A benthic attribute with no recorded regions is *unrecorded*, never
*permitted nowhere*, so it stays out of both the numerator and the denominator
of every rate built on these. The same epistemic state on the image side is not
answerable at all: an image whose region is unrecorded raises rather than
returning a verdict, which leaves exclusion a caller decision.

The statistics helpers quantify how certain a measured rate is. Annotation
points are clustered within images and are strongly correlated inside a
cluster, so the bootstrap helpers resample whole clusters. Resampling
individual points instead yields intervals that are far too narrow.
"""

import math
from collections.abc import Callable, Hashable, Sequence
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray


def is_out_of_region(label_region_ids: frozenset[str], image_region_id: str) -> bool:
    """Whether a label is applied to an image from a region it does not cover.

    `label_region_ids` are the regions the label's benthic attribute is
    recorded in; an empty set means unrecorded and yields False.

    An empty `image_region_id` means the image's region is unrecorded, which
    is neither a mismatch nor a match, and raises ValueError. Every CoralNet
    annotation carries one, so answering False would understate the rate and
    answering True would inflate it; the caller filters those rows first.
    """
    if not image_region_id:
        raise ValueError("image region is unrecorded; filter these rows before scoring")
    if not label_region_ids:
        return False
    return image_region_id not in label_region_ids


def is_region_discriminating(
    label_region_ids: frozenset[str], observed_region_ids: frozenset[str]
) -> bool:
    """Whether a label could ever be out of region within the scored data.

    A label discriminates when it is not permitted in every region the
    evaluation set actually contains, so `observed_region_ids` is derived from
    that data rather than from a fixed list of MERMAID regions. Membership in a
    region no image came from carries no weight. An unrecorded (empty)
    `label_region_ids` yields False.

    An `observed_region_ids` containing "" raises ValueError, on the same
    grounds as `is_out_of_region`: an unrecorded region is not a region, and
    admitting one here would make labels discriminate against a region no image
    can be matched to.

    Holds `is_out_of_region(label, region) implies
    is_region_discriminating(label, observed)` for any `observed` containing
    `region`.
    """
    if "" in observed_region_ids:
        raise ValueError(
            "observed regions include an unrecorded region; filter these rows before scoring"
        )
    if not label_region_ids:
        return False
    return not observed_region_ids <= label_region_ids


def partition_by_recorded_region(image_region_ids: Sequence[str]) -> tuple[list[int], list[int]]:
    """Positions with a recorded image region, and positions without one.

    This is how a caller honours the "" contract `is_out_of_region` and
    `is_region_discriminating` enforce by raising: derive both the scored rows
    and any parallel group vector (for `permutation_baseline`, say) from the
    same two lists, so a caller cannot filter one and forget the other.

    An empty string means the image's region is unrecorded, the sentinel used
    throughout this module.
    """
    scorable: list[int] = []
    unscorable: list[int] = []
    for position, region_id in enumerate(image_region_ids):
        (scorable if region_id else unscorable).append(position)
    return scorable, unscorable


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for k events in n independent trials.

    Assumes independence, so it understates the width whenever the
    observations are clustered; `design_effect` measures by how much. With no
    trials the interval is the whole unit interval.
    """
    if n < 0:
        raise ValueError(f"n must not be negative, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"k must lie in [0, {n}], got {k}")
    if n == 0:
        return 0.0, 1.0

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_squared = z * z
    denominator = n + z_squared
    centre = (k + z_squared / 2.0) / denominator
    half_width = z / denominator * math.sqrt(k * (n - k) / n + z_squared / 4.0)
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def rule_of_three(n: int) -> float:
    """The 95% upper bound on a rate when zero events were seen in n trials.

    Lets a zero cell report "0 events in 253 images; 95% upper bound 1.2%"
    rather than a bare 0%. Capped at 1.0, since 3/n exceeds 1 below three
    trials.
    """
    if n < 0:
        raise ValueError(f"n must not be negative, got {n}")
    if n == 0:
        return 1.0
    return min(1.0, 3.0 / n)


def cluster_bootstrap_draws(
    cluster_ids: Sequence[Hashable],
    statistic: Callable[[NDArray[np.intp]], float],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> NDArray[np.float64]:
    """One estimate per resample, drawn by resampling whole clusters.

    `cluster_ids[i]` is the cluster (typically the image) that observation i
    belongs to. Each resample draws len(clusters) clusters with replacement and
    calls `statistic` with the positions of every observation in the drawn
    clusters, repeats included, so the caller supplies its own estimator over
    whichever columns it holds.

    The draws themselves, for a statistic whose distribution a caller must
    inspect before summarizing it -- an unbounded ratio, say, whose infinite
    draws no percentile can interpolate between.

    Seeded, and deterministic for a given seed and input.
    """
    positions = _cluster_positions(cluster_ids)
    if not positions:
        raise ValueError("cluster_ids must contain at least one cluster")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")

    rng = np.random.default_rng(seed)
    n_clusters = len(positions)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for resample in range(n_resamples):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        estimates[resample] = statistic(_gather(positions, drawn))
    return estimates


def cluster_bootstrap_ci(
    cluster_ids: Sequence[Hashable],
    statistic: Callable[[NDArray[np.intp]], float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval that resamples whole clusters.

    The two-sided percentile interval of `cluster_bootstrap_draws`, for a
    statistic bounded enough that every draw is finite.

    Seeded, and deterministic for a given seed and input.
    """
    return _percentile_interval(
        cluster_bootstrap_draws(cluster_ids, statistic, n_resamples=n_resamples, seed=seed),
        alpha,
    )


def paired_cluster_bootstrap_diff(
    cluster_ids: Sequence[Hashable],
    statistic_a: Callable[[NDArray[np.intp]], float],
    statistic_b: Callable[[NDArray[np.intp]], float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for statistic_a - statistic_b.

    For two models scored on the same points. Each resample draws one set of
    clusters and feeds it to both statistics, which keeps the comparison
    paired: the shared variation between clusters cancels out of the
    difference instead of inflating it.

    Seeded, and deterministic for a given seed and input.
    """
    return _percentile_interval(
        cluster_bootstrap_draws(
            cluster_ids,
            lambda index: statistic_a(index) - statistic_b(index),
            n_resamples=n_resamples,
            seed=seed,
        ),
        alpha,
    )


def design_effect(bootstrap_ci: tuple[float, float], k: int, n: int, *, alpha: float) -> float:
    """How much wider the clustered interval is than the naive one, squared.

    The ratio of the cluster-bootstrap half-width to the Wilson half-width for
    the same k of n, squared. A value of 4 means the clustering costs three
    quarters of the apparent sample size; 1 means the points behave as if
    independent.

    `alpha` is required because both halves of the ratio must be read at the
    same confidence level: a 99% bootstrap interval measured against a 95%
    Wilson interval overstates the design effect by about 73% at realistic
    sample sizes.
    """
    bootstrap_half_width = (bootstrap_ci[1] - bootstrap_ci[0]) / 2.0
    wilson_lower, wilson_upper = wilson_ci(k, n, alpha)
    wilson_half_width = (wilson_upper - wilson_lower) / 2.0
    return (bootstrap_half_width / wilson_half_width) ** 2


def permutation_baseline(
    cluster_ids: Sequence[Hashable],
    groups: Sequence[Hashable],
    statistic: Callable[[NDArray[np.object_]], float],
    *,
    n_permutations: int = 1000,
    seed: int = 0,
) -> float:
    """Mean of a statistic under group assignments shuffled between clusters.

    `groups[i]` is the group of observation i and must be constant within a
    cluster, since a group is a property of the cluster (an image's region, for
    instance) rather than of a point. Each permutation reshuffles the
    per-cluster groups, broadcasts them back over the observations, and calls
    `statistic` with the resulting per-observation group vector.

    The result is what the caller's statistic would read if group membership
    carried no information, which is the baseline a measured value is judged
    against. Domain-free: the caller's statistic holds the labels and
    predictions.

    "" is reserved for "unrecorded" throughout this module and is rejected
    here, on the same grounds as `is_region_discriminating`: an unrecorded
    region is not a region, so shuffling it in as if it were a real group
    would baseline the measured rate against a differently-filtered set of
    rows. Use `partition_by_recorded_region` to filter groups (and cluster_ids)
    to recorded regions before calling.

    Seeded, and deterministic for a given seed and input.
    """
    if len(groups) != len(cluster_ids):
        raise ValueError(
            f"groups and cluster_ids must be the same length,"
            f" got {len(groups)} and {len(cluster_ids)}"
        )
    if "" in groups:
        raise ValueError("groups include an unrecorded region; filter these rows before scoring")
    positions = _cluster_positions(cluster_ids)
    if not positions:
        raise ValueError("cluster_ids must contain at least one cluster")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be at least 1, got {n_permutations}")

    cluster_groups: NDArray[np.object_] = np.empty(len(positions), dtype=object)
    cluster_of_observation = np.empty(len(cluster_ids), dtype=np.intp)
    for cluster, member_positions in enumerate(positions):
        distinct = {groups[position] for position in member_positions}
        if len(distinct) > 1:
            raise ValueError(
                f"cluster {cluster_ids[member_positions[0]]!r} spans"
                f" {len(distinct)} groups; a group is a property of the cluster"
            )
        cluster_groups[cluster] = groups[member_positions[0]]
        cluster_of_observation[member_positions] = cluster

    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_permutations):
        permuted: NDArray[np.object_] = rng.permutation(cluster_groups)
        total += statistic(permuted[cluster_of_observation])
    return total / n_permutations


def _cluster_positions(cluster_ids: Sequence[Hashable]) -> list[NDArray[np.intp]]:
    """Observation positions grouped by cluster, in first-appearance order.

    The order is fixed so that a seeded resample is reproducible.
    """
    positions_by_cluster: dict[Hashable, list[int]] = {}
    for position, cluster_id in enumerate(cluster_ids):
        positions_by_cluster.setdefault(cluster_id, []).append(position)
    return [np.asarray(positions, dtype=np.intp) for positions in positions_by_cluster.values()]


def _gather(positions: list[NDArray[np.intp]], drawn: NDArray[np.int64]) -> NDArray[np.intp]:
    """Observation positions of the drawn clusters, in draw order with repeats."""
    return np.concatenate([positions[cluster] for cluster in drawn])


def _percentile_interval(estimates: NDArray[np.float64], alpha: float) -> tuple[float, float]:
    """Two-sided percentile interval of a bootstrap distribution."""
    lower = float(np.percentile(estimates, 100.0 * alpha / 2.0))
    upper = float(np.percentile(estimates, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper
