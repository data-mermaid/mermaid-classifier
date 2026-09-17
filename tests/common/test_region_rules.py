"""Unit tests for common/region_rules.py.

Two families of behaviour are pinned here.

The region predicates encode rules that silently corrupt every downstream rate
if they drift: an unrecorded (empty) region set is never a mismatch, a label is
region-discriminating only relative to the regions the scored data actually
contains, and an unrecorded region on the image side is refused rather than
answered.

The statistics helpers are checked against literals derived by hand from the
underlying formulas, never from the functions under test. The load-bearing case
is ClusterBootstrapTest.test_resamples_clusters_not_points -- resampling
individual observations instead of clusters passes every other test in this
file while producing intervals that are far too narrow.
"""

import itertools
import unittest
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from mermaid_classifier.common.region_rules import (
    cluster_bootstrap_ci,
    design_effect,
    is_out_of_region,
    is_region_discriminating,
    paired_cluster_bootstrap_diff,
    partition_by_recorded_region,
    permutation_baseline,
    rule_of_three,
    wilson_ci,
)

# MERMAID region UUIDs. Fixed literals so every expectation below is readable.
TROPICAL_ATLANTIC = "f0f0f0f0-0000-4000-8000-000000000001"
CENTRAL_INDO_PACIFIC = "f0f0f0f0-0000-4000-8000-000000000002"
WESTERN_INDO_PACIFIC = "f0f0f0f0-0000-4000-8000-000000000003"
EASTERN_INDO_PACIFIC = "f0f0f0f0-0000-4000-8000-000000000004"


def _clustered(
    cluster_values: Sequence[float],
    points_per_cluster: int,
) -> tuple[list[str], NDArray[np.float64]]:
    """Build data with perfect intra-cluster correlation.

    Every point inside a cluster carries its cluster's value, so the only
    variation in the data is between clusters. A cluster-level resample of the
    mean can then take only the values k/len(cluster_values); a point-level
    resample concentrates tightly around the overall mean.
    """
    cluster_ids: list[str] = []
    values: list[float] = []
    for index, value in enumerate(cluster_values):
        cluster_ids.extend([f"image-{index}"] * points_per_cluster)
        values.extend([value] * points_per_cluster)
    return cluster_ids, np.asarray(values, dtype=np.float64)


class IsOutOfRegionTest(unittest.TestCase):
    """A label is out of region when the image's region is not one of its own."""

    def test_label_not_permitted_in_the_image_region_is_out_of_region(self):
        label_regions = frozenset({TROPICAL_ATLANTIC})
        self.assertTrue(is_out_of_region(label_regions, CENTRAL_INDO_PACIFIC))

    def test_label_permitted_in_the_image_region_is_not_out_of_region(self):
        label_regions = frozenset({TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC})
        self.assertFalse(is_out_of_region(label_regions, CENTRAL_INDO_PACIFIC))

    def test_empty_region_set_is_never_out_of_region(self):
        """An empty set means the regions are unrecorded, not 'allowed nowhere'.

        11 of 810 MERMAID benthic attributes carry regions: []. Treating those
        as permitted nowhere makes every one of their predictions a mismatch.
        """
        self.assertFalse(is_out_of_region(frozenset(), CENTRAL_INDO_PACIFIC))

    def test_unrecorded_image_region_is_refused(self):
        """An unrecorded image region has no answer, so neither may be returned.

        Every CoralNet annotation carries region_id == "" and those rows
        dominate. Returning True calls each one a mismatch, which reports a
        true 5.0% rate as 90.5%; returning False buries them in the
        denominator. Refusing makes the caller filter them.
        """
        for label_regions in (frozenset({TROPICAL_ATLANTIC}), frozenset()):
            with self.subTest(label=sorted(label_regions)), self.assertRaises(ValueError):
                is_out_of_region(label_regions, "")


class IsRegionDiscriminatingTest(unittest.TestCase):
    """A label discriminates when it is not permitted in every observed region."""

    def test_label_permitted_in_only_some_observed_regions_discriminates(self):
        """The subtle case: membership outside the observed regions is irrelevant.

        The label is allowed in Tropical Atlantic and Eastern Indo-Pacific, but
        the scored data covers Tropical Atlantic, Central and Western
        Indo-Pacific. Among the regions actually observed the label is
        exclusive to Tropical Atlantic, so it discriminates; the Eastern
        Indo-Pacific membership carries no weight because no data was observed
        there.
        """
        label_regions = frozenset({TROPICAL_ATLANTIC, EASTERN_INDO_PACIFIC})
        observed = frozenset({TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC, WESTERN_INDO_PACIFIC})
        self.assertTrue(is_region_discriminating(label_regions, observed))

    def test_label_permitted_in_every_observed_region_does_not_discriminate(self):
        label_regions = frozenset(
            {
                TROPICAL_ATLANTIC,
                CENTRAL_INDO_PACIFIC,
                WESTERN_INDO_PACIFIC,
                EASTERN_INDO_PACIFIC,
            }
        )
        observed = frozenset({TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC, WESTERN_INDO_PACIFIC})
        self.assertFalse(is_region_discriminating(label_regions, observed))

    def test_empty_region_set_never_discriminates(self):
        """Unrecorded regions must not enter a numerator or a denominator."""
        observed = frozenset({TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC})
        self.assertFalse(is_region_discriminating(frozenset(), observed))

    def test_unrecorded_region_among_the_observed_is_refused(self):
        """An unrecorded region is not a region and cannot make a label discriminate.

        Admitting it puts a region in the denominator that no image can be
        matched against, which breaks the implication the next test pins.
        """
        observed = frozenset({TROPICAL_ATLANTIC, ""})
        for label_regions in (frozenset({TROPICAL_ATLANTIC}), frozenset()):
            with self.subTest(label=sorted(label_regions)), self.assertRaises(ValueError):
                is_region_discriminating(label_regions, observed)

    def test_out_of_region_implies_region_discriminating(self):
        """Every out-of-region label must also be region-discriminating.

        The denominator of the mismatch rate is the discriminating labels, so
        an out-of-region label that fails to discriminate would be a numerator
        event with no matching denominator.
        """
        observed = frozenset({TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC, WESTERN_INDO_PACIFIC})
        all_regions = [
            TROPICAL_ATLANTIC,
            CENTRAL_INDO_PACIFIC,
            WESTERN_INDO_PACIFIC,
            EASTERN_INDO_PACIFIC,
        ]
        out_of_region_cases = 0
        for size in range(len(all_regions) + 1):
            for combination in itertools.combinations(all_regions, size):
                label_regions = frozenset(combination)
                for image_region in sorted(observed):
                    if not is_out_of_region(label_regions, image_region):
                        continue
                    out_of_region_cases += 1
                    with self.subTest(label=sorted(label_regions), image=image_region):
                        self.assertTrue(
                            is_region_discriminating(label_regions, observed),
                            msg="out-of-region label is not region-discriminating",
                        )
        self.assertGreater(
            out_of_region_cases, 0, msg="no out-of-region case reached, test is vacuous"
        )


class PartitionByRecordedRegionTest(unittest.TestCase):
    """Splits image region ids into recorded and unrecorded positions."""

    def test_splits_recorded_and_unrecorded_positions(self):
        region_ids = [TROPICAL_ATLANTIC, "", CENTRAL_INDO_PACIFIC, ""]
        scorable, unscorable = partition_by_recorded_region(region_ids)
        self.assertEqual(scorable, [0, 2])
        self.assertEqual(unscorable, [1, 3])

    def test_all_recorded_leaves_unscorable_empty(self):
        scorable, unscorable = partition_by_recorded_region(
            [TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC]
        )
        self.assertEqual(scorable, [0, 1])
        self.assertEqual(unscorable, [])

    def test_all_unrecorded_leaves_scorable_empty(self):
        scorable, unscorable = partition_by_recorded_region(["", ""])
        self.assertEqual(scorable, [])
        self.assertEqual(unscorable, [0, 1])


class WilsonCiTest(unittest.TestCase):
    """Wilson score interval, against bounds derived by hand from the formula."""

    def test_zero_events_in_one_hundred_trials(self):
        """With k=0 the centre and half-width are both z^2 / (2(n + z^2)).

        So the lower bound is exactly 0 and the upper bound is
        z^2 / (n + z^2) = 3.8414588 / 103.8414588 = 0.0369935.
        """
        lower, upper = wilson_ci(0, 100)
        self.assertAlmostEqual(lower, 0.0000, places=4)
        self.assertAlmostEqual(upper, 0.0370, places=4)

    def test_five_events_in_twenty_trials(self):
        """centre = (5 + z^2/2)/(20 + z^2) = 0.2902813,
        half = z/(20 + z^2) * sqrt(5*15/20 + z^2/4) = 0.1784196.
        """
        lower, upper = wilson_ci(5, 20)
        self.assertAlmostEqual(lower, 0.11186, places=4)
        self.assertAlmostEqual(upper, 0.46870, places=4)

    def test_alpha_widens_the_interval(self):
        """At alpha=0.01, z = 2.5758293, so the k=0 upper bound is
        z^2 / (100 + z^2) = 6.6348966 / 106.6348966 = 0.0622207.
        """
        _, upper = wilson_ci(0, 100, alpha=0.01)
        self.assertAlmostEqual(upper, 0.0622, places=4)

    def test_lower_bound_is_never_negative(self):
        lower, _ = wilson_ci(0, 100)
        self.assertGreaterEqual(lower, 0.0)

    def test_no_trials_gives_the_full_unit_interval(self):
        self.assertEqual(wilson_ci(0, 0), (0.0, 1.0))

    def test_more_events_than_trials_is_rejected(self):
        with self.assertRaises(ValueError):
            wilson_ci(11, 10)

    def test_negative_events_are_rejected(self):
        with self.assertRaises(ValueError):
            wilson_ci(-1, 10)


class RuleOfThreeTest(unittest.TestCase):
    """The 95% upper bound on a rate when zero events were observed."""

    def test_one_thousand_trials(self):
        self.assertEqual(rule_of_three(1000), 0.003)

    def test_two_hundred_and_fifty_three_trials(self):
        """3/253 = 0.0118577, the '95% upper bound 1.2%' a zero cell reports."""
        self.assertAlmostEqual(rule_of_three(253), 0.011858, places=6)

    def test_result_is_capped_at_one(self):
        """3/2 exceeds 1, which is not a probability."""
        self.assertEqual(rule_of_three(2), 1.0)

    def test_no_trials_gives_no_information(self):
        self.assertEqual(rule_of_three(0), 1.0)

    def test_negative_trials_are_rejected(self):
        with self.assertRaises(ValueError):
            rule_of_three(-1)


class ClusterBootstrapTest(unittest.TestCase):
    """Percentile bootstrap that resamples clusters, not individual points."""

    def test_resamples_clusters_not_points(self):
        """The reason this module exists.

        Four clusters of five identical points, two clusters all-1 and two
        all-0. Resampling four clusters with replacement makes the mean
        m/4 where m ~ Binomial(4, 1/2), so P(mean=0) = P(mean=1) = 1/16 =
        0.0625 > 0.025 and the 95% percentile interval is exactly (0, 1).

        Resampling the twenty points instead gives a mean with standard
        deviation 0.112 and an interval near (0.3, 0.7) -- far too narrow, and
        wrong in the direction that hides a real difference.
        """
        cluster_ids, values = _clustered([1.0, 1.0, 0.0, 0.0], points_per_cluster=5)
        lower, upper = cluster_bootstrap_ci(
            cluster_ids, lambda index: float(values[index].mean()), seed=7
        )
        self.assertEqual((lower, upper), (0.0, 1.0))

    def test_percentiles_follow_the_cluster_level_distribution(self):
        """Three all-1 clusters and one all-0, so m ~ Binomial(4, 3/4).

        P(m=0) = 0.0039 and P(m<=1) = 0.0508, so the 2.5th percentile falls on
        mean 0.25. P(m=4) = 0.3164 > 0.025, so the 97.5th percentile is 1.0.
        """
        cluster_ids, values = _clustered([1.0, 1.0, 1.0, 0.0], points_per_cluster=5)
        lower, upper = cluster_bootstrap_ci(
            cluster_ids, lambda index: float(values[index].mean()), seed=7
        )
        self.assertEqual((lower, upper), (0.25, 1.0))

    def test_identical_clusters_collapse_the_interval_to_the_point_estimate(self):
        """Every resample draws the same multiset, so there is nothing to vary."""
        cluster_ids, values = _clustered([0.5, 0.5, 0.5, 0.5], points_per_cluster=4)
        lower, upper = cluster_bootstrap_ci(
            cluster_ids, lambda index: float(values[index].mean()), seed=3
        )
        self.assertEqual(lower, upper)
        self.assertAlmostEqual(lower, 0.5, places=12)

    def test_same_seed_gives_the_same_interval(self):
        cluster_ids, values = _clustered([float(v) for v in range(20)], points_per_cluster=3)

        def statistic(index: NDArray[np.intp]) -> float:
            return float(values[index].mean())

        first = cluster_bootstrap_ci(cluster_ids, statistic, n_resamples=500, seed=11)
        second = cluster_bootstrap_ci(cluster_ids, statistic, n_resamples=500, seed=11)
        self.assertEqual(first, second)

    def test_different_seeds_give_different_intervals(self):
        """Guards against the seed being ignored by a hardcoded generator."""
        cluster_ids, values = _clustered([float(v) for v in range(20)], points_per_cluster=3)

        def statistic(index: NDArray[np.intp]) -> float:
            return float(values[index].mean())

        first = cluster_bootstrap_ci(cluster_ids, statistic, n_resamples=500, seed=11)
        second = cluster_bootstrap_ci(cluster_ids, statistic, n_resamples=500, seed=12)
        self.assertNotEqual(first, second)

    def test_no_clusters_is_rejected(self):
        with self.assertRaises(ValueError):
            cluster_bootstrap_ci([], lambda index: 0.0)

    def test_no_resamples_is_rejected(self):
        cluster_ids, values = _clustered([1.0, 0.0], points_per_cluster=2)
        with self.assertRaises(ValueError):
            cluster_bootstrap_ci(
                cluster_ids, lambda index: float(values[index].mean()), n_resamples=0
            )


class PairedClusterBootstrapTest(unittest.TestCase):
    """Two arms scored on the same points must share each resample's clusters."""

    def test_identical_arms_give_a_zero_width_interval_at_zero(self):
        """Pairing is the whole point: the same clusters feed both arms.

        With both arms reading the same values every resample has difference
        exactly 0. Resampling the arms independently would instead give an
        interval near (-1, 1) on this data.
        """
        cluster_ids, values = _clustered([1.0, 1.0, 0.0, 0.0], points_per_cluster=5)

        def statistic(index: NDArray[np.intp]) -> float:
            return float(values[index].mean())

        lower, upper = paired_cluster_bootstrap_diff(cluster_ids, statistic, statistic, seed=5)
        self.assertEqual((lower, upper), (0.0, 0.0))

    def test_a_constant_offset_is_preserved_with_its_sign(self):
        """arm_a - arm_b = 0.2 at every point, so every resample differs by 0.2."""
        cluster_ids, values_a = _clustered([1.0, 1.0, 1.0, 0.0], points_per_cluster=5)
        values_b = values_a - 0.2

        lower, upper = paired_cluster_bootstrap_diff(
            cluster_ids,
            lambda index: float(values_a[index].mean()),
            lambda index: float(values_b[index].mean()),
            seed=5,
        )
        self.assertAlmostEqual(lower, 0.2, places=12)
        self.assertAlmostEqual(upper, 0.2, places=12)


class DesignEffectTest(unittest.TestCase):
    """How much the naive Wilson interval understates the clustered width."""

    def test_squared_ratio_of_half_widths(self):
        """Wilson half-width at k=10, n=20 is
        z/(20 + z^2) * sqrt(10*10/20 + z^2/4) = 0.2007020.
        A bootstrap interval of (0.1, 0.9) has half-width 0.4, so the design
        effect is (0.4 / 0.2007020)^2 = 1.9930045^2 = 3.97207.
        """
        self.assertAlmostEqual(design_effect((0.1, 0.9), 10, 20, alpha=0.05), 3.97207, places=4)

    def test_matching_widths_give_a_design_effect_of_one(self):
        """The Wilson interval at k=10, n=20 is 0.5 +/- 0.2007020."""
        self.assertAlmostEqual(
            design_effect((0.2992980, 0.7007020), 10, 20, alpha=0.05), 1.0, places=5
        )

    def test_alpha_changes_the_answer_for_the_same_interval(self):
        """Why alpha is required rather than defaulted.

        The Wilson half-width at k=10, n=20 is 0.2007020 at alpha=0.05 and
        z/(20 + z^2) * sqrt(10*10/20 + z^2/4) = 0.2495523 at alpha=0.01, with
        z = 2.5758293. The same bootstrap half-width of 0.4 therefore reads as
        (0.4 / 0.2007020)^2 = 3.97207 against the first and
        (0.4 / 0.2495523)^2 = 2.56919 against the second. A caller who widens
        the bootstrap to 99% and leaves this at 95% overstates the design
        effect by 55%.
        """
        at_five_percent = design_effect((0.1, 0.9), 10, 20, alpha=0.05)
        at_one_percent = design_effect((0.1, 0.9), 10, 20, alpha=0.01)
        self.assertAlmostEqual(at_five_percent, 3.97207, places=4)
        self.assertAlmostEqual(at_one_percent, 2.56919, places=4)


class PermutationBaselineTest(unittest.TestCase):
    """Group labels are permuted across clusters, never across observations."""

    def test_assignment_stays_constant_within_each_cluster(self):
        """A cluster-level permutation keeps one group per cluster.

        Permuting the per-observation vector instead splits clusters across
        groups, which this statistic reports as 0.0 almost every time.
        """
        cluster_ids = ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
        groups = ["north"] * 8 + ["south"] * 8

        def all_clusters_uniform(assigned: NDArray[np.object_]) -> float:
            by_cluster: dict[str, set[object]] = {}
            for cluster_id, group in zip(cluster_ids, assigned, strict=True):
                by_cluster.setdefault(cluster_id, set()).add(group)
            return float(all(len(values) == 1 for values in by_cluster.values()))

        self.assertEqual(
            permutation_baseline(
                cluster_ids, groups, all_clusters_uniform, n_permutations=200, seed=2
            ),
            1.0,
        )

    def test_mean_follows_the_cluster_level_distribution(self):
        """Clusters of sizes 1, 1, 3, 3 labelled north, north, south, south.

        Permuting the two 'north' labels over the four clusters makes the
        north share of observations 2/8, 4/8 or 6/8 with probabilities
        1/6, 4/6, 1/6, so its expectation is exactly 0.5.

        Permuting the per-observation vector instead keeps exactly two north
        observations out of eight, pinning the statistic at 0.25.
        """
        cluster_ids = ["a"] + ["b"] + ["c"] * 3 + ["d"] * 3
        groups = ["north"] * 2 + ["south"] * 6

        def north_share(assigned: NDArray[np.object_]) -> float:
            return float(np.count_nonzero(assigned == "north") / len(assigned))

        baseline = permutation_baseline(
            cluster_ids, groups, north_share, n_permutations=2000, seed=4
        )
        # Monte Carlo standard error of the mean here is 0.0032; 0.02 is six of
        # those, while the point-level answer of 0.25 is seventy-eight away.
        self.assertAlmostEqual(baseline, 0.5, delta=0.02)

    def test_same_seed_gives_the_same_baseline(self):
        cluster_ids = ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2
        groups = ["north"] * 4 + ["south"] * 4

        def north_share(assigned: NDArray[np.object_]) -> float:
            return float(np.count_nonzero(assigned[:2] == "north"))

        first = permutation_baseline(cluster_ids, groups, north_share, n_permutations=100, seed=9)
        second = permutation_baseline(cluster_ids, groups, north_share, n_permutations=100, seed=9)
        self.assertEqual(first, second)

    def test_groups_varying_inside_a_cluster_are_rejected(self):
        """A region is a property of the image, so it cannot vary within one."""
        cluster_ids = ["a", "a", "b", "b"]
        groups = ["north", "south", "south", "south"]
        with self.assertRaises(ValueError):
            permutation_baseline(cluster_ids, groups, lambda assigned: 0.0)

    def test_mismatched_group_length_is_rejected(self):
        with self.assertRaises(ValueError):
            permutation_baseline(["a", "a", "b"], ["north", "south"], lambda a: 0.0)

    def test_unrecorded_region_in_groups_is_rejected(self):
        """An empty string is reserved for "unrecorded" throughout this module.

        Admitting it here would shuffle an unrecorded region in as if it were
        a real group, baselining the measured rate against a differently
        filtered set of rows.
        """
        cluster_ids = ["a", "a", "b", "b"]
        groups = ["", "", "south", "south"]
        with self.assertRaises(ValueError):
            permutation_baseline(cluster_ids, groups, lambda assigned: 0.0)


if __name__ == "__main__":
    unittest.main()
