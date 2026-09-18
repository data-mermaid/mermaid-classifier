"""Unit tests for region_eval/metrics.py.

One 16-point fixture carries every case the rates have to separate. Expected
values are counted by hand off this table and written as k/n literals, so a
rate that drifts fails against a number no code in the package produced:

    #   image  region  ground truth  prediction   out of region
    1   a      CIP     global        pacific      no
    2   a      CIP     pacific       atlantic     yes
    3   a      CIP     pacific       pacific      no
    4   a      CIP     global        no-regions   no (prediction region-unknown)
    5   b      CIP     atlantic      pacific      no (ground truth out of region)
    6   b      CIP     global        global       no
    7   b      CIP     global        pacific      no
    8   b      CIP     no-regions    global       no (ground truth region-unknown)
    9   c      TA      atlantic      pacific      yes
    10  c      TA      atlantic      atlantic     no
    11  c      TA      off-list      pacific      yes (ground truth off label space)
    12  c      TA      global        global       no
    13  d      CIP     pacific       pacific      no
    14  d      CIP     pacific       global       no
    15  d      CIP     global        atlantic     yes
    16  d      CIP     atlantic      no-regions   no (ground truth out of region)
    17  e      ""      global        atlantic     image region unrecorded
    18  e      ""      global        atlantic     image region unrecorded

Observed regions are {TA, CIP}, so 'global' (permitted in both) never
discriminates while 'atlantic', 'pacific' and 'off-list' do. The three
denominators land on 14 evaluable predictions, 10 region-discriminating
predictions and 9 region-discriminating ground truths. `oor_rate` and
`oor_rate_disc` share a numerator of 4 out-of-region predictions;
`oor_rate_disc_gt` restricts that numerator to the 3 of them whose ground
truth is itself region-discriminating.

Points 9-16 are held out, which gives the nested population its own 3 of 7.
"""

import math
import unittest
import warnings
from unittest import mock

from mermaid_classifier.region_eval.metrics import (
    ALL_POINTS,
    HELD_OUT,
    RegionMetricsOptions,
    _imprecise,
    compute_region_metrics,
    prepare_scored_points,
    required_n_for_margin,
)

TROPICAL_ATLANTIC = "1a1a1a1a-0000-4000-8000-000000000001"
CENTRAL_INDO_PACIFIC = "1a1a1a1a-0000-4000-8000-000000000002"
EASTERN_PACIFIC = "1a1a1a1a-0000-4000-8000-000000000003"

BA_GLOBAL = "2b2b2b2b-0000-4000-8000-000000000001"
BA_ATLANTIC = "2b2b2b2b-0000-4000-8000-000000000002"
BA_PACIFIC = "2b2b2b2b-0000-4000-8000-000000000003"
BA_NO_REGIONS = "2b2b2b2b-0000-4000-8000-000000000004"
BA_OFF_LIST = "2b2b2b2b-0000-4000-8000-000000000005"
BA_TWO_REGION = "2b2b2b2b-0000-4000-8000-000000000006"

GLOBAL_LABEL = f"{BA_GLOBAL}::"
ATLANTIC_LABEL = f"{BA_ATLANTIC}::"
PACIFIC_LABEL = f"{BA_PACIFIC}::"
NO_REGIONS_LABEL = f"{BA_NO_REGIONS}::"
OFF_LIST_LABEL = f"{BA_OFF_LIST}::"
TWO_REGION_LABEL = f"{BA_TWO_REGION}::"

REGION_IDS_BY_ATTRIBUTE = {
    BA_GLOBAL: frozenset({TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC}),
    BA_ATLANTIC: frozenset({TROPICAL_ATLANTIC}),
    BA_PACIFIC: frozenset({CENTRAL_INDO_PACIFIC}),
    BA_NO_REGIONS: frozenset(),
    BA_OFF_LIST: frozenset({CENTRAL_INDO_PACIFIC}),
}

MODEL_CLASSES = (GLOBAL_LABEL, ATLANTIC_LABEL, PACIFIC_LABEL, NO_REGIONS_LABEL)

REGION_NAMES = {
    TROPICAL_ATLANTIC: "Tropical Atlantic",
    CENTRAL_INDO_PACIFIC: "Central Indo-Pacific",
}
LABEL_NAMES = {
    ATLANTIC_LABEL: "Atlantic coral",
    PACIFIC_LABEL: "Pacific coral",
    OFF_LIST_LABEL: "Off-list coral",
}

# (image_id, image_region_id, ground truth label, predicted label)
FIXTURE_ROWS = (
    ("image-a", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, PACIFIC_LABEL),
    ("image-a", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, ATLANTIC_LABEL),
    ("image-a", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, PACIFIC_LABEL),
    ("image-a", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, NO_REGIONS_LABEL),
    ("image-b", CENTRAL_INDO_PACIFIC, ATLANTIC_LABEL, PACIFIC_LABEL),
    ("image-b", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, GLOBAL_LABEL),
    ("image-b", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, PACIFIC_LABEL),
    ("image-b", CENTRAL_INDO_PACIFIC, NO_REGIONS_LABEL, GLOBAL_LABEL),
    ("image-c", TROPICAL_ATLANTIC, ATLANTIC_LABEL, PACIFIC_LABEL),
    ("image-c", TROPICAL_ATLANTIC, ATLANTIC_LABEL, ATLANTIC_LABEL),
    ("image-c", TROPICAL_ATLANTIC, OFF_LIST_LABEL, PACIFIC_LABEL),
    ("image-c", TROPICAL_ATLANTIC, GLOBAL_LABEL, GLOBAL_LABEL),
    ("image-d", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, PACIFIC_LABEL),
    ("image-d", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, GLOBAL_LABEL),
    ("image-d", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, ATLANTIC_LABEL),
    ("image-d", CENTRAL_INDO_PACIFIC, ATLANTIC_LABEL, NO_REGIONS_LABEL),
    ("image-e", "", GLOBAL_LABEL, ATLANTIC_LABEL),
    ("image-e", "", GLOBAL_LABEL, ATLANTIC_LABEL),
)

FIXTURE_HELD_OUT = (False,) * 8 + (True,) * 8 + (False,) * 2

# Every point sits in one region, so nothing discriminates; no prediction is
# ever out of region. Five images of four points, which separates the
# image-level zero-cell bound (3/5) from the point-level one (3/20).
CLEAN_ROWS = tuple(
    (image_id, CENTRAL_INDO_PACIFIC, truth, prediction)
    for image_id in ("image-v", "image-w", "image-x", "image-y", "image-z")
    for truth, prediction in (
        (PACIFIC_LABEL, PACIFIC_LABEL),
        (PACIFIC_LABEL, GLOBAL_LABEL),
        (GLOBAL_LABEL, GLOBAL_LABEL),
        (GLOBAL_LABEL, PACIFIC_LABEL),
    )
)

# CLEAN_ROWS with its first prediction flipped to an out-of-region label: the
# same five images and twenty points, but one event instead of none, which is
# the boundary the zero-cell upper bound has to disappear across.
ONE_EVENT_ROWS = (("image-v", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, ATLANTIC_LABEL),) + CLEAN_ROWS[
    1:
]

# Five images predicted into the Central Indo-Pacific; two of them predict a
# label with no recorded regions on every one of their points, so neither
# contributes to the evaluable denominator and both must be absent from its
# image count.
DENOMINATOR_EXCLUDED_IMAGES_ROWS = (
    ("image-g1", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, PACIFIC_LABEL),
    ("image-g2", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, ATLANTIC_LABEL),
    ("image-g3", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, PACIFIC_LABEL),
    ("image-g4", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, NO_REGIONS_LABEL),
    ("image-g4", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, NO_REGIONS_LABEL),
    ("image-g5", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, NO_REGIONS_LABEL),
    ("image-g5", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, NO_REGIONS_LABEL),
)

# Four all-or-nothing images of five points: two predict out of region
# throughout and two never do, so resampling images spans the unit interval
# where resampling points would barely move.
CLUSTERED_ROWS = tuple(
    (image_id, CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, prediction)
    for image_id, prediction in (
        ("image-1", ATLANTIC_LABEL),
        ("image-2", ATLANTIC_LABEL),
        ("image-3", PACIFIC_LABEL),
        ("image-4", PACIFIC_LABEL),
    )
    for _ in range(5)
)

# Twenty Central Indo-Pacific images of five points. Ten predict out of region
# on four of their five points and ten never do: 40 incidents in 100 evaluable
# predictions, at a measured design effect of 2.88.
PRECISION_ROWS = tuple(
    [
        (f"image-{index:02d}", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, prediction)
        for index in range(10)
        for prediction in (
            ATLANTIC_LABEL,
            ATLANTIC_LABEL,
            ATLANTIC_LABEL,
            ATLANTIC_LABEL,
            PACIFIC_LABEL,
        )
    ]
    + [
        (f"image-{index:02d}", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, PACIFIC_LABEL)
        for index in range(10, 20)
        for _ in range(5)
    ]
)

# A region-discriminating label predicted twice each in five images with no
# out-of-region incidents, plus one image in a second region so the label
# discriminates at all: the rule-of-three bound must read five trials (the
# images), not the ten points it was predicted at.
PER_LABEL_ZERO_CELL_ROWS = tuple(
    (f"image-p{index}", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, PACIFIC_LABEL)
    for index in range(5)
    for _ in range(2)
) + (("image-ta", TROPICAL_ATLANTIC, ATLANTIC_LABEL, ATLANTIC_LABEL),)

# A label permitted in two regions, neither of which is the image's own: each
# of its four incidents is out of region towards both, so the direction rows'
# n_out_of_region sums to eight while the unduplicated event count is four.
TWO_REGION_MAP = dict(REGION_IDS_BY_ATTRIBUTE)
TWO_REGION_MAP[BA_TWO_REGION] = frozenset({TROPICAL_ATLANTIC, EASTERN_PACIFIC})
TWO_REGION_DIRECTION_ROWS = tuple(
    (f"image-two-{index}", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, TWO_REGION_LABEL)
    for index in range(4)
)

# Ground truth is a label out of region for every point, so the floor reads
# 1.0 on every draw and ratio_to_gt tracks the prediction rate exactly; two of
# these four all-or-nothing images predict out of region throughout and two
# never do, the same clustering as CLUSTERED_ROWS.
RATIO_TO_GT_CLUSTERED_ROWS = tuple(
    (image_id, CENTRAL_INDO_PACIFIC, ATLANTIC_LABEL, prediction)
    for image_id, prediction in (
        ("image-r1", ATLANTIC_LABEL),
        ("image-r2", ATLANTIC_LABEL),
        ("image-r3", PACIFIC_LABEL),
        ("image-r4", PACIFIC_LABEL),
    )
    for _ in range(5)
)

# One of four images carries the only ground-truth point out of its region, so
# a resample drawing none of it has a zero floor and an infinite ratio draw:
# (3/4)^4 = 32% of draws, far past the 2.5% upper tail. Every image predicts
# out of region once in four, so the model rate is 4/16 and the ratio 4.0.
THIN_FLOOR_ROWS = tuple(
    ("image-t1", CENTRAL_INDO_PACIFIC, truth, prediction)
    for truth, prediction in (
        (ATLANTIC_LABEL, ATLANTIC_LABEL),
        (PACIFIC_LABEL, PACIFIC_LABEL),
        (PACIFIC_LABEL, PACIFIC_LABEL),
        (PACIFIC_LABEL, PACIFIC_LABEL),
    )
) + tuple(
    (image_id, CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, prediction)
    for image_id in ("image-t2", "image-t3", "image-t4")
    for prediction in (ATLANTIC_LABEL, PACIFIC_LABEL, PACIFIC_LABEL, PACIFIC_LABEL)
)

OPTIONS = RegionMetricsOptions(n_resamples=200, seed=1)


def _prepare(rows=FIXTURE_ROWS, *, held_out=None, region_map=REGION_IDS_BY_ATTRIBUTE):
    return prepare_scored_points(
        image_ids=[row[0] for row in rows],
        image_region_ids=[row[1] for row in rows],
        gt_labels=[row[2] for row in rows],
        pred_labels=[row[3] for row in rows],
        region_ids_by_attribute=region_map,
        model_classes=MODEL_CLASSES,
        held_out=held_out,
    )


class ScoredPointsTest(unittest.TestCase):
    """Preparation filters unrecorded image regions and reports what it dropped."""

    def test_unrecorded_image_regions_are_excluded_and_counted(self):
        points = _prepare()
        self.assertEqual(points.n_points, 16)
        self.assertEqual(points.n_unrecorded_region_excluded, 2)
        self.assertNotIn("", points.image_region_ids)

    def test_observed_regions_come_from_the_scored_slice(self):
        points = _prepare()
        self.assertEqual(
            points.observed_region_ids,
            frozenset({TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC}),
        )

    def test_attribute_absent_from_the_region_map_is_region_unknown_and_traced(self):
        """A snapshot that has not heard of an attribute must leave a trace."""
        region_map = dict(REGION_IDS_BY_ATTRIBUTE)
        del region_map[BA_PACIFIC]
        points = _prepare(region_map=region_map)
        self.assertEqual(points.unmapped_attribute_ids, frozenset({BA_PACIFIC}))
        # Points 1, 3, 5, 7, 9, 11, 13 predict 'pacific'.
        self.assertEqual(int(points.pred_region_unknown.sum()), 2 + 7)

    def test_mismatched_input_lengths_are_refused(self):
        with self.assertRaises(ValueError):
            prepare_scored_points(
                image_ids=["image-a"],
                image_region_ids=[CENTRAL_INDO_PACIFIC],
                gt_labels=[GLOBAL_LABEL, GLOBAL_LABEL],
                pred_labels=[GLOBAL_LABEL],
                region_ids_by_attribute=REGION_IDS_BY_ATTRIBUTE,
                model_classes=MODEL_CLASSES,
            )


class DenominatorTest(unittest.TestCase):
    """`oor_rate` and `oor_rate_disc` share a numerator; `oor_rate_disc_gt` restricts its own."""

    def setUp(self):
        self.result = compute_region_metrics(_prepare(), options=OPTIONS)
        self.rates = self.result.overall

    def test_three_denominators_differ_and_disc_gt_restricts_its_own_numerator(self):
        self.assertEqual(self.rates.oor_rate.k, 4)
        self.assertEqual(self.rates.oor_rate_disc.k, 4)
        self.assertEqual(self.rates.oor_rate_disc_gt.k, 3)

        self.assertEqual(self.rates.oor_rate.n, 14)
        self.assertEqual(self.rates.oor_rate_disc.n, 10)
        self.assertEqual(self.rates.oor_rate_disc_gt.n, 9)

        self.assertAlmostEqual(self.rates.oor_rate.rate, 4 / 14)
        self.assertAlmostEqual(self.rates.oor_rate_disc.rate, 4 / 10)
        self.assertAlmostEqual(self.rates.oor_rate_disc_gt.rate, 3 / 9)

        denominators = {
            self.rates.oor_rate.n,
            self.rates.oor_rate_disc.n,
            self.rates.oor_rate_disc_gt.n,
        }
        self.assertEqual(len(denominators), 3)

    def test_region_unknown_predictions_leave_numerator_and_denominator(self):
        """Points 4 and 16 predict an attribute with no recorded regions."""
        self.assertEqual(self.rates.n_pred_region_unknown, 2)
        self.assertEqual(self.rates.n_points, 16)
        self.assertEqual(self.rates.oor_rate.n, 16 - 2)

    def test_region_unknown_ground_truth_leaves_the_floor_denominator(self):
        """Point 8's ground truth has no recorded regions."""
        self.assertEqual(self.rates.n_gt_region_unknown, 1)
        self.assertEqual(self.rates.gt_oor_rate.n, 16 - 1)

    def test_ground_truth_floor_and_excess(self):
        self.assertEqual(self.rates.gt_oor_rate.k, 3)
        self.assertEqual(self.rates.gt_oor_rate.n, 15)
        self.assertAlmostEqual(self.rates.gt_oor_rate.rate, 3 / 15)
        self.assertAlmostEqual(self.rates.excess.value, 4 / 14 - 3 / 15)
        self.assertAlmostEqual(self.rates.ratio_to_gt.value, (4 / 14) / (3 / 15))
        self.assertLessEqual(self.rates.excess.ci_low, self.rates.excess.value)
        self.assertGreaterEqual(self.rates.excess.ci_high, self.rates.excess.value)
        self.assertLessEqual(self.rates.ratio_to_gt.ci_low, self.rates.ratio_to_gt.value)
        self.assertGreaterEqual(self.rates.ratio_to_gt.ci_high, self.rates.ratio_to_gt.value)

    def test_accuracy_carries_the_same_clustered_interval_every_other_rate_does(self):
        """Accuracy reported bare is a rate with no width, which is what drove
        a second copy of the estimate into the reporting layer. The counts are
        the hand-derived 5 of 15, and the interval has to arrive with them.
        """
        accuracy = self.rates.accuracy
        self.assertEqual(accuracy.k, 5)
        self.assertEqual(accuracy.n, 15)
        self.assertAlmostEqual(accuracy.rate, 5 / 15)
        self.assertLessEqual(accuracy.ci_low, accuracy.rate)
        self.assertGreaterEqual(accuracy.ci_high, accuracy.rate)
        self.assertGreater(accuracy.wilson_high, accuracy.wilson_low)
        self.assertEqual(accuracy.upper_bound, None)

    def test_ground_truth_off_the_label_space_counts_for_region_but_not_accuracy(self):
        """Point 11's truth is not a model class; its prediction is still out of region."""
        self.assertEqual(self.rates.n_gt_outside_model_classes, 1)
        self.assertEqual(self.rates.n_accuracy_points, 15)
        self.assertAlmostEqual(self.rates.accuracy.rate, 5 / 15)

        confusion = self.result.confusion
        off_list = confusion[confusion["gt_label"] == OFF_LIST_LABEL]
        self.assertEqual(len(off_list), 1)
        self.assertEqual(int(off_list.iloc[0]["n"]), 1)

    def test_image_affected_rate_counts_images_not_points(self):
        """Images a, c and d carry an out-of-region point; image b does not."""
        self.assertEqual(self.rates.image_affected_rate.k, 3)
        self.assertEqual(self.rates.image_affected_rate.n, 4)
        self.assertAlmostEqual(self.rates.image_affected_rate.rate, 3 / 4)
        self.assertEqual(self.rates.n_images, 4)

    def test_excluded_count_travels_to_the_result(self):
        self.assertEqual(self.result.n_unrecorded_region_excluded, 2)


class DiscGtProportionTest(unittest.TestCase):
    """oor_rate_disc_gt is a proportion: its numerator is a subset of its denominator."""

    def test_out_of_region_predictions_at_non_discriminating_ground_truth_do_not_count(self):
        """image-p's three predictions are out of region, but their ground truth
        ('global') never discriminates, so none of them is an opportunity; image-q's
        one opportunity saw a correctly-regioned prediction."""
        rows = (
            ("image-p", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, ATLANTIC_LABEL),
            ("image-p", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, ATLANTIC_LABEL),
            ("image-p", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, ATLANTIC_LABEL),
            ("image-q", TROPICAL_ATLANTIC, ATLANTIC_LABEL, ATLANTIC_LABEL),
        )
        result = compute_region_metrics(_prepare(rows), options=OPTIONS)
        estimate = result.overall.oor_rate_disc_gt
        self.assertEqual(estimate.k, 0)
        self.assertEqual(estimate.n, 1)
        self.assertAlmostEqual(estimate.rate, 0.0)
        self.assertFalse(math.isnan(estimate.wilson_low))
        self.assertFalse(math.isnan(estimate.ci_low))
        # Every resample of a cell with no event gives the same statistic, so
        # the bootstrap measures no clustering rather than independence.
        self.assertTrue(math.isnan(estimate.design_effect))

    def test_rate_cannot_exceed_one_where_the_old_reading_gave_a_ratio_above_one(self):
        """Four points (r1-r4) predict out of region at a non-discriminating ground
        truth; under the old unrestricted numerator these would have counted toward
        oor_rate_disc_gt, pushing 6 incidents over a denominator of 3 (a ratio of 2).
        Restricting the numerator to points whose ground truth also discriminates
        leaves only image-t and image-u, bounding the rate at 2/3."""
        rows = (
            ("image-r1", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, ATLANTIC_LABEL),
            ("image-r2", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, ATLANTIC_LABEL),
            ("image-r3", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, ATLANTIC_LABEL),
            ("image-r4", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, ATLANTIC_LABEL),
            ("image-s", TROPICAL_ATLANTIC, ATLANTIC_LABEL, ATLANTIC_LABEL),
            ("image-t", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, ATLANTIC_LABEL),
            ("image-u", TROPICAL_ATLANTIC, ATLANTIC_LABEL, PACIFIC_LABEL),
        )
        result = compute_region_metrics(_prepare(rows), options=OPTIONS)
        estimate = result.overall.oor_rate_disc_gt
        self.assertEqual(estimate.k, 2)
        self.assertEqual(estimate.n, 3)
        self.assertAlmostEqual(estimate.rate, 2 / 3)
        self.assertLessEqual(estimate.rate, 1.0)


class UnrecordedRegionTest(unittest.TestCase):
    """The predicates raise on an unrecorded region; the entry point must not."""

    def test_all_regions_unrecorded_yields_an_empty_population_without_raising(self):
        rows = tuple(("image-z", "", GLOBAL_LABEL, ATLANTIC_LABEL) for _ in range(5))
        result = compute_region_metrics(_prepare(rows), options=OPTIONS)
        self.assertEqual(result.n_unrecorded_region_excluded, 5)
        self.assertEqual(result.overall.n_points, 0)
        self.assertEqual(result.overall.oor_rate.n, 0)
        self.assertTrue(math.isnan(result.overall.oor_rate.rate))

    def test_a_mixed_slice_scores_only_the_recorded_rows(self):
        result = compute_region_metrics(_prepare(), options=OPTIONS)
        self.assertEqual(result.overall.n_points, 16)
        self.assertEqual(result.n_unrecorded_region_excluded, 2)


class ZeroCellTest(unittest.TestCase):
    """A cell with no events reports an upper bound, not a bare zero."""

    def test_zero_out_of_region_reports_a_rule_of_three_upper_bound(self):
        """Twenty points, but five images: the bound counts the trials, and
        a point inside an image is not one."""
        result = compute_region_metrics(_prepare(CLEAN_ROWS), options=OPTIONS)
        estimate = result.overall.oor_rate
        self.assertEqual(estimate.k, 0)
        self.assertEqual(estimate.n, 20)
        self.assertEqual(estimate.rate, 0.0)
        self.assertEqual(estimate.upper_bound_n_images, 5)
        self.assertAlmostEqual(estimate.upper_bound, 3 / 5)

    def test_a_populated_cell_carries_no_upper_bound(self):
        result = compute_region_metrics(_prepare(), options=OPTIONS)
        self.assertIsNone(result.overall.oor_rate.upper_bound)

    def test_an_empty_denominator_yields_a_blank_estimate(self):
        """One observed region makes every label non-discriminating."""
        result = compute_region_metrics(_prepare(CLEAN_ROWS), options=OPTIONS)
        estimate = result.overall.oor_rate_disc
        self.assertEqual(estimate.n, 0)
        self.assertTrue(math.isnan(estimate.rate))

    def test_an_empty_denominator_reports_no_upper_bound(self):
        """A cell with an empty denominator carries no upper bound at all."""
        result = compute_region_metrics(_prepare(CLEAN_ROWS), options=OPTIONS)
        estimate = result.overall.oor_rate_disc
        self.assertEqual(estimate.n, 0)
        self.assertIsNone(estimate.upper_bound)
        self.assertEqual(estimate.upper_bound_n_images, 0)

    def test_the_bound_disappears_between_zero_and_one_event(self):
        """ONE_EVENT_ROWS is CLEAN_ROWS with its first prediction flipped to
        an out-of-region label: the same five images and twenty points, one
        event instead of none. The rule-of-three stand-in belongs to the
        zero cell alone; a single event must drop it rather than widen it."""
        zero_events = compute_region_metrics(_prepare(CLEAN_ROWS), options=OPTIONS).overall.oor_rate
        one_event = compute_region_metrics(
            _prepare(ONE_EVENT_ROWS), options=OPTIONS
        ).overall.oor_rate

        self.assertEqual(zero_events.k, 0)
        self.assertEqual(zero_events.upper_bound_n_images, 5)
        self.assertAlmostEqual(zero_events.upper_bound, 3 / 5)

        self.assertEqual(one_event.k, 1)
        self.assertEqual(one_event.n, zero_events.n)
        self.assertIsNone(one_event.upper_bound)

    def test_per_label_zero_cell_bound_counts_images_not_points(self):
        """Ten points predicting a region-discriminating label across five
        images with no incident: the bound must read five trials, not ten."""
        result = compute_region_metrics(_prepare(PER_LABEL_ZERO_CELL_ROWS), options=OPTIONS)
        row = result.per_label[result.per_label["label"] == PACIFIC_LABEL].iloc[0]
        self.assertEqual(int(row["n_out_of_region"]), 0)
        self.assertEqual(int(row["n_predicted"]), 10)
        self.assertEqual(int(row["upper_bound_n_images"]), 5)
        self.assertAlmostEqual(float(row["upper_bound"]), 3 / 5)


class DenominatorClusterCountTest(unittest.TestCase):
    """The rule-of-three trial count is the images the denominator mask
    keeps, not every image the points came from."""

    def test_images_excluded_by_the_denominator_mask_do_not_count(self):
        """Two of five images predict a label with no recorded regions on
        every point, so neither clears the evaluable-predictions mask; the
        trial count must be the three images that do, not all five."""
        result = compute_region_metrics(_prepare(DENOMINATOR_EXCLUDED_IMAGES_ROWS), options=OPTIONS)
        estimate = result.overall.oor_rate
        self.assertEqual(estimate.k, 1)
        self.assertEqual(estimate.n, 3)
        self.assertEqual(estimate.upper_bound_n_images, 3)


class ClusteringTest(unittest.TestCase):
    """The cluster is the image; a point-level interval would be far too narrow."""

    def setUp(self):
        self.result = compute_region_metrics(
            _prepare(CLUSTERED_ROWS), options=RegionMetricsOptions(n_resamples=2000, seed=1)
        )

    def test_cluster_bootstrap_is_wider_than_wilson_and_reports_the_design_effect(self):
        estimate = self.result.overall.oor_rate
        self.assertEqual(estimate.k, 10)
        self.assertEqual(estimate.n, 20)

        bootstrap_width = estimate.ci_high - estimate.ci_low
        wilson_width = estimate.wilson_high - estimate.wilson_low
        self.assertGreater(bootstrap_width, wilson_width)
        # Wilson half-width at 10 of 20 is 0.2007; whole-image resampling of
        # four all-or-nothing images spans the unit interval.
        self.assertGreater(estimate.design_effect, 5.0)
        self.assertLess(estimate.design_effect, 8.0)

    def test_a_per_region_row_resamples_images_too(self):
        """Resampling the region slice by point would leave the interval no
        wider than Wilson's and the design effect near 1."""
        row = self.result.per_region.set_index(["population", "region_id"]).loc[
            (ALL_POINTS, CENTRAL_INDO_PACIFIC)
        ]
        self.assertEqual(int(row["oor_rate_k"]), 10)
        self.assertEqual(int(row["oor_rate_n"]), 20)
        self.assertGreater(
            float(row["oor_rate_ci_high"]) - float(row["oor_rate_ci_low"]),
            float(row["oor_rate_wilson_high"]) - float(row["oor_rate_wilson_low"]),
        )
        self.assertGreater(float(row["oor_rate_design_effect"]), 5.0)
        self.assertLess(float(row["oor_rate_design_effect"]), 8.0)

    def test_a_per_direction_row_resamples_images_too(self):
        """The one direction these images carry, through its own estimate."""
        row = self.result.per_direction.set_index(
            ["population", "image_region_id", "excluded_region_id"]
        ).loc[(ALL_POINTS, CENTRAL_INDO_PACIFIC, TROPICAL_ATLANTIC)]
        self.assertEqual(int(row["n_out_of_region"]), 10)
        self.assertEqual(int(row["n_points"]), 20)
        self.assertGreater(
            float(row["ci_high"]) - float(row["ci_low"]),
            float(row["wilson_high"]) - float(row["wilson_low"]),
        )
        self.assertGreater(float(row["design_effect"]), 5.0)
        self.assertLess(float(row["design_effect"]), 8.0)


class RatioToGtClusteringTest(unittest.TestCase):
    """`ratio_to_gt`'s bootstrap must resample whole images, not points."""

    def test_a_point_level_resample_of_the_same_data_gives_a_visibly_narrower_interval(self):
        """The floor reads 1.0 on every draw here, so ratio_to_gt tracks the
        prediction rate exactly; splitting every point into its own image
        leaves the value unchanged but collapses the interval, the same
        contrast ClusteringTest draws for oor_rate."""
        options = RegionMetricsOptions(n_resamples=2000, seed=1)
        clustered = compute_region_metrics(_prepare(RATIO_TO_GT_CLUSTERED_ROWS), options=options)
        point_level_rows = tuple(
            (f"point-{index}", region, truth, prediction)
            for index, (_, region, truth, prediction) in enumerate(RATIO_TO_GT_CLUSTERED_ROWS)
        )
        pointwise = compute_region_metrics(_prepare(point_level_rows), options=options)

        clustered_ratio = clustered.overall.ratio_to_gt
        pointwise_ratio = pointwise.overall.ratio_to_gt
        self.assertAlmostEqual(clustered_ratio.value, 0.5)
        self.assertAlmostEqual(pointwise_ratio.value, 0.5)

        clustered_width = clustered_ratio.ci_high - clustered_ratio.ci_low
        pointwise_width = pointwise_ratio.ci_high - pointwise_ratio.ci_low
        self.assertGreater(clustered_width, 2.0 * pointwise_width)


class RatioToGtThinFloorTest(unittest.TestCase):
    """A floor a resample can miss entirely, which is the Atlantic's situation.

    The catch is silent: percentiling an array holding infinities interpolates
    inf - inf, which numpy answers with NaN after a RuntimeWarning, and a NaN
    upper bound reads as "not computed" rather than "unbounded".
    """

    def _ratio(self, rows, n_resamples: int = 200):
        options = RegionMetricsOptions(n_resamples=n_resamples, seed=1)
        return compute_region_metrics(_prepare(rows), options=options).overall.ratio_to_gt

    def test_draws_with_an_empty_floor_are_counted_rather_than_percentiled(self):
        ratio = self._ratio(THIN_FLOOR_ROWS)
        self.assertAlmostEqual(ratio.value, 4.0)
        self.assertEqual(ratio.n_draws, 200)
        # (3/4)^4 of 200 draws is about 63; anything past 5 of 200 contaminates
        # the 97.5th percentile.
        self.assertGreater(ratio.n_nonfinite_draws, 5)
        self.assertLess(ratio.n_nonfinite_draws, ratio.n_draws)

    def test_an_upper_tail_of_empty_floors_reports_an_unbounded_multiple(self):
        """A finite upper bound here would be a number the draws do not
        support: a third of them say the multiple is unbounded.
        """
        ratio = self._ratio(THIN_FLOOR_ROWS)
        self.assertEqual(ratio.ci_high, math.inf)
        self.assertTrue(math.isfinite(ratio.ci_low), ratio.ci_low)
        self.assertLessEqual(ratio.ci_low, ratio.value)

    def test_no_runtime_warning_escapes_the_ratio_bootstrap(self):
        """Asserted explicitly: the interval could be right while numpy still
        warned on the way, and a test that only reads the bounds would pass
        through the symptom this fix exists to remove.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._ratio(THIN_FLOOR_ROWS)
        self.assertEqual(
            [],
            [f"{entry.category.__name__}: {entry.message}" for entry in caught],
        )

    def test_a_floor_no_resample_can_empty_reports_no_non_finite_draws(self):
        """Every ground truth is out of region here, so the floor is 1.0 on
        every draw and the unbounded-upper-end rule must stay dormant.
        """
        ratio = self._ratio(RATIO_TO_GT_CLUSTERED_ROWS)
        self.assertEqual(ratio.n_nonfinite_draws, 0)
        self.assertEqual(ratio.n_draws, 200)
        self.assertTrue(math.isfinite(ratio.ci_high), ratio.ci_high)


class MacroF1DiscTest(unittest.TestCase):
    """Macro precision/recall/F1 over points whose ground truth discriminates.

    #   image   region  ground truth  prediction
    1   image-1 CIP     pacific       pacific
    2   image-2 TA      atlantic      pacific
    3   image-3 CIP     global        off-list   (ground truth not discriminating: dropped)
    4   image-4 TA      atlantic      atlantic

    'off-list' is predicted but never true within the scored 3-row subset, so it
    would silently drop out of an unlabeled macro average; passing `labels=` for
    every region-discriminating class keeps it in as a zero.

    Per class over the 3 scored rows (pacific, atlantic, atlantic / pacific,
    pacific, atlantic): pacific precision 1/2, recall 1/1, F1 2/3; atlantic
    precision 1/1, recall 1/2, F1 2/3; off-list precision 0, recall 0, F1 0.
    Macro precision (0.5+1+0)/3=0.5, recall (1+0.5+0)/3=0.5, F1 (2/3+2/3+0)/3=4/9.
    """

    def test_macro_f1_precision_recall_are_hand_derived_and_include_the_absent_class(self):
        rows = (
            ("image-1", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, PACIFIC_LABEL),
            ("image-2", TROPICAL_ATLANTIC, ATLANTIC_LABEL, PACIFIC_LABEL),
            ("image-3", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, OFF_LIST_LABEL),
            ("image-4", TROPICAL_ATLANTIC, ATLANTIC_LABEL, ATLANTIC_LABEL),
        )
        result = compute_region_metrics(_prepare(rows), options=OPTIONS)
        rates = result.overall
        self.assertEqual(rates.n_f1_disc_points, 3)
        self.assertAlmostEqual(rates.precision_macro_disc, 0.5)
        self.assertAlmostEqual(rates.recall_macro_disc, 0.5)
        self.assertAlmostEqual(rates.f1_macro_disc, 4 / 9)


class PerRegionTableTest(unittest.TestCase):
    def setUp(self):
        self.result = compute_region_metrics(_prepare(), options=OPTIONS)
        table = self.result.per_region
        self.rows = table[table["population"] == ALL_POINTS].set_index("region_id")

    def test_one_row_per_observed_region_with_its_own_counts(self):
        self.assertEqual(sorted(self.rows.index), sorted([TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC]))

        pacific = self.rows.loc[CENTRAL_INDO_PACIFIC]
        self.assertEqual(int(pacific["n_images"]), 3)
        self.assertEqual(int(pacific["n_points"]), 12)
        self.assertEqual(int(pacific["oor_rate_k"]), 2)
        self.assertEqual(int(pacific["oor_rate_n"]), 10)
        self.assertAlmostEqual(float(pacific["oor_rate"]), 2 / 10)
        self.assertEqual(int(pacific["oor_rate_disc_n"]), 7)
        self.assertAlmostEqual(float(pacific["oor_rate_disc"]), 2 / 7)
        self.assertEqual(int(pacific["gt_oor_rate_k"]), 2)
        self.assertEqual(int(pacific["gt_oor_rate_n"]), 11)
        self.assertAlmostEqual(float(pacific["accuracy"]), 3 / 12)

        atlantic = self.rows.loc[TROPICAL_ATLANTIC]
        self.assertEqual(int(atlantic["n_images"]), 1)
        self.assertEqual(int(atlantic["n_points"]), 4)
        self.assertEqual(int(atlantic["oor_rate_k"]), 2)
        self.assertEqual(int(atlantic["oor_rate_n"]), 4)
        self.assertAlmostEqual(float(atlantic["oor_rate"]), 2 / 4)
        self.assertEqual(int(atlantic["n_accuracy_points"]), 3)
        self.assertAlmostEqual(float(atlantic["accuracy"]), 2 / 3)

    def test_region_macro_is_the_unweighted_mean_of_the_region_rates(self):
        macro = self.result.overall.region_macro
        self.assertEqual(macro.oor_rate.n_regions, 2)
        self.assertAlmostEqual(macro.oor_rate.value, (2 / 10 + 2 / 4) / 2)

    def test_region_macro_differs_from_the_pooled_rate(self):
        """Corpus mix moves the pooled rate and leaves the macro mean alone."""
        macro = self.result.overall.region_macro.oor_rate.value
        self.assertNotAlmostEqual(macro, self.result.overall.oor_rate.rate)


class PerLabelTableTest(unittest.TestCase):
    def setUp(self):
        self.result = compute_region_metrics(
            _prepare(),
            options=OPTIONS,
            label_names={ATLANTIC_LABEL: "Atlantic coral"},
        )
        self.table = self.result.per_label

    def test_only_region_discriminating_predicted_classes_appear(self):
        self.assertEqual(set(self.table["label"]), {ATLANTIC_LABEL, PACIFIC_LABEL})

    def test_rows_are_sorted_by_incidents_and_carry_their_denominators(self):
        first, second = self.table.iloc[0], self.table.iloc[1]
        self.assertEqual(first["label"], PACIFIC_LABEL)
        self.assertEqual(int(first["n_predicted"]), 7)
        self.assertEqual(int(first["n_out_of_region"]), 2)
        self.assertAlmostEqual(float(first["rate"]), 2 / 7)
        self.assertEqual(int(first["n_ground_truth"]), 4)
        self.assertEqual(first["allowed_region_ids"], (CENTRAL_INDO_PACIFIC,))

        self.assertEqual(second["label"], ATLANTIC_LABEL)
        self.assertEqual(int(second["n_predicted"]), 3)
        self.assertEqual(int(second["n_out_of_region"]), 2)
        self.assertAlmostEqual(float(second["rate"]), 2 / 3)
        self.assertEqual(int(second["n_ground_truth"]), 4)

    def test_cumulative_share_reaches_exactly_one_on_the_last_row(self):
        shares = list(self.table["cumulative_share"])
        self.assertAlmostEqual(shares[0], 2 / 4)
        self.assertEqual(shares[-1], 1.0)

    def test_label_names_come_from_the_caller(self):
        row = self.table[self.table["label"] == ATLANTIC_LABEL].iloc[0]
        self.assertEqual(row["label_name"], "Atlantic coral")

    def test_an_unresolved_label_renders_its_id_rather_than_an_empty_cell(self):
        """An empty cell reads as "this label has no name"; the id reads as
        "unresolved", which is what actually happened. The caller here names
        only the Atlantic label, so the Pacific one has to fall back."""
        other = self.table[self.table["label"] == PACIFIC_LABEL].iloc[0]
        self.assertEqual(other["label_name"], PACIFIC_LABEL)

    def test_every_label_name_cell_is_populated(self):
        self.assertEqual([], [name for name in self.table["label_name"] if not str(name).strip()])

    def test_wilson_interval_brackets_the_rate(self):
        for _, row in self.table.iterrows():
            self.assertLessEqual(float(row["wilson_low"]), float(row["rate"]))
            self.assertGreaterEqual(float(row["wilson_high"]), float(row["rate"]))


class DirectionTest(unittest.TestCase):
    def setUp(self):
        self.result = compute_region_metrics(_prepare(), options=OPTIONS)

    def test_direction_matrix_counts_ordered_pairs(self):
        matrix = self.result.direction_matrix
        self.assertEqual(int(matrix.loc[CENTRAL_INDO_PACIFIC, TROPICAL_ATLANTIC]), 2)
        self.assertEqual(int(matrix.loc[TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC]), 2)
        self.assertEqual(int(matrix.loc[CENTRAL_INDO_PACIFIC, CENTRAL_INDO_PACIFIC]), 0)

    def test_the_two_directions_keep_their_own_denominators(self):
        """Equal incident counts, unequal exposure: the rates must not be pooled."""
        table = self.result.per_direction
        rows = table[table["population"] == ALL_POINTS].set_index(
            ["image_region_id", "excluded_region_id"]
        )
        out_of_pacific = rows.loc[(CENTRAL_INDO_PACIFIC, TROPICAL_ATLANTIC)]
        out_of_atlantic = rows.loc[(TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC)]

        self.assertEqual(int(out_of_pacific["n_out_of_region"]), 2)
        self.assertEqual(int(out_of_pacific["n_points"]), 10)
        self.assertAlmostEqual(float(out_of_pacific["rate"]), 2 / 10)

        self.assertEqual(int(out_of_atlantic["n_out_of_region"]), 2)
        self.assertEqual(int(out_of_atlantic["n_points"]), 4)
        self.assertAlmostEqual(float(out_of_atlantic["rate"]), 2 / 4)

        self.assertNotEqual(int(out_of_pacific["n_points"]), int(out_of_atlantic["n_points"]))

    def test_n_out_of_region_events_is_unduplicated_while_the_direction_sum_double_counts(self):
        """A label permitted in two regions counts once per direction: the
        four unduplicated incidents this image region carries sum to eight
        across its two direction rows."""
        result = compute_region_metrics(
            _prepare(TWO_REGION_DIRECTION_ROWS, region_map=TWO_REGION_MAP), options=OPTIONS
        )
        rows = result.per_direction[result.per_direction["population"] == ALL_POINTS].set_index(
            ["image_region_id", "excluded_region_id"]
        )
        to_atlantic = rows.loc[(CENTRAL_INDO_PACIFIC, TROPICAL_ATLANTIC)]
        to_eastern_pacific = rows.loc[(CENTRAL_INDO_PACIFIC, EASTERN_PACIFIC)]

        self.assertEqual(int(to_atlantic["n_out_of_region_events"]), 4)
        self.assertEqual(int(to_eastern_pacific["n_out_of_region_events"]), 4)
        self.assertEqual(
            int(to_atlantic["n_out_of_region"]) + int(to_eastern_pacific["n_out_of_region"]), 8
        )


def _directions(result):
    """The all-points direction rows, keyed by the ordered region pair."""
    rows = result.per_direction
    return rows[rows["population"] == ALL_POINTS].set_index(
        ["image_region_id", "excluded_region_id"]
    )


class NameRenderingTest(unittest.TestCase):
    """Names for the ids the tables key on.

    An id-only table cannot be read without a join the reader has to do by
    hand, and a name resolved at scoring time would make the table depend on a
    later state of the taxonomy than the score it annotates.
    """

    def setUp(self):
        self.result = compute_region_metrics(
            _prepare(),
            options=OPTIONS,
            label_names=LABEL_NAMES,
            region_names=REGION_NAMES,
        )

    def test_per_region_rows_carry_the_region_name_beside_the_id(self):
        rows = self.result.per_region.set_index("region_id")
        self.assertEqual(rows.loc[TROPICAL_ATLANTIC, "region_name"], "Tropical Atlantic")
        self.assertEqual(rows.loc[CENTRAL_INDO_PACIFIC, "region_name"], "Central Indo-Pacific")

    def test_per_label_rows_name_the_regions_the_label_is_allowed_in(self):
        """Ids alone leave the reader a join to do by hand, and every other
        name column in these tables was populated."""
        row = self.result.per_label.set_index("label").loc[PACIFIC_LABEL]
        self.assertEqual(row["allowed_region_ids"], (CENTRAL_INDO_PACIFIC,))
        self.assertEqual(row["allowed_region_names"], ("Central Indo-Pacific",))

    def test_per_direction_rows_name_both_ends_of_the_direction(self):
        rows = _directions(self.result)
        row = rows.loc[(CENTRAL_INDO_PACIFIC, TROPICAL_ATLANTIC)]
        self.assertEqual(row["image_region_name"], "Central Indo-Pacific")
        self.assertEqual(row["excluded_region_name"], "Tropical Atlantic")

    def test_direction_matrix_axes_render_region_names(self):
        matrix = self.result.direction_matrix
        self.assertEqual(int(matrix.loc["Central Indo-Pacific", "Tropical Atlantic"]), 2)
        self.assertEqual(int(matrix.loc["Tropical Atlantic", "Central Indo-Pacific"]), 2)

    def test_confusion_rows_name_the_region_and_both_labels(self):
        rows = self.result.confusion
        named = {
            (row["image_region_name"], row["gt_label_name"], row["pred_label_name"])
            for _, row in rows.iterrows()
        }
        self.assertIn(("Central Indo-Pacific", "Pacific coral", "Atlantic coral"), named)
        self.assertIn(("Tropical Atlantic", "Off-list coral", "Pacific coral"), named)

    def test_an_unnamed_region_renders_its_id_in_every_table(self):
        """Half a name map is the realistic failure -- a region added upstream
        after the probe was frozen. The id says "unresolved" where a blank
        would say "unnamed"."""
        result = compute_region_metrics(
            _prepare(),
            options=OPTIONS,
            label_names=LABEL_NAMES,
            region_names={TROPICAL_ATLANTIC: "Tropical Atlantic"},
        )
        per_region = result.per_region.set_index("region_id")
        self.assertEqual(per_region.loc[CENTRAL_INDO_PACIFIC, "region_name"], CENTRAL_INDO_PACIFIC)
        self.assertIn(CENTRAL_INDO_PACIFIC, result.direction_matrix.columns)
        row = _directions(result).loc[(CENTRAL_INDO_PACIFIC, TROPICAL_ATLANTIC)]
        self.assertEqual(row["image_region_name"], CENTRAL_INDO_PACIFIC)
        self.assertEqual(row["excluded_region_name"], "Tropical Atlantic")
        per_label = result.per_label.set_index("label")
        self.assertEqual(
            per_label.loc[PACIFIC_LABEL, "allowed_region_names"], (CENTRAL_INDO_PACIFIC,)
        )

    def test_names_default_to_ids_when_the_caller_supplies_none(self):
        bare = compute_region_metrics(_prepare(), options=OPTIONS)
        self.assertEqual(
            set(bare.per_region["region_name"]), {TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC}
        )
        self.assertEqual([], [name for name in bare.per_label["label_name"] if not str(name)])


class ConfusionTest(unittest.TestCase):
    def test_confusion_covers_out_of_region_predictions_only(self):
        result = compute_region_metrics(_prepare(), options=OPTIONS)
        table = result.confusion
        self.assertEqual(int(table["n"].sum()), 4)
        triples = {
            (row["image_region_id"], row["gt_label"], row["pred_label"])
            for _, row in table.iterrows()
        }
        self.assertIn((CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, ATLANTIC_LABEL), triples)
        self.assertIn((TROPICAL_ATLANTIC, OFF_LIST_LABEL, PACIFIC_LABEL), triples)
        self.assertNotIn((CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, PACIFIC_LABEL), triples)

    def test_confusion_is_capped_at_the_named_constant(self):
        with mock.patch("mermaid_classifier.region_eval.metrics.CONFUSION_ROW_LIMIT", 2):
            result = compute_region_metrics(_prepare(), options=OPTIONS)
        self.assertEqual(len(result.confusion), 2)


class HeldOutTest(unittest.TestCase):
    def test_held_out_population_carries_its_own_n_and_rate(self):
        result = compute_region_metrics(_prepare(held_out=FIXTURE_HELD_OUT), options=OPTIONS)
        self.assertIsNotNone(result.held_out)
        held_out = result.held_out
        self.assertEqual(held_out.name, HELD_OUT)
        self.assertEqual(held_out.n_points, 8)
        self.assertEqual(held_out.n_images, 2)
        self.assertEqual(held_out.oor_rate.k, 3)
        self.assertEqual(held_out.oor_rate.n, 7)
        self.assertAlmostEqual(held_out.oor_rate.rate, 3 / 7)
        self.assertEqual(result.overall.oor_rate.n, 14)

    def test_no_held_out_points_yields_no_held_out_population(self):
        result = compute_region_metrics(_prepare(), options=OPTIONS)
        self.assertIsNone(result.held_out)

    def test_tables_carry_both_populations(self):
        result = compute_region_metrics(_prepare(held_out=FIXTURE_HELD_OUT), options=OPTIONS)
        self.assertEqual(set(result.per_region["population"]), {ALL_POINTS, HELD_OUT})
        self.assertEqual(set(result.per_direction["population"]), {ALL_POINTS, HELD_OUT})

    def test_a_thin_held_out_slice_is_flagged_rather_than_read_as_a_number(self):
        result = compute_region_metrics(
            _prepare(held_out=FIXTURE_HELD_OUT),
            options=RegionMetricsOptions(n_resamples=200, seed=1, target_margin=0.01),
        )
        self.assertTrue(result.held_out.oor_rate.imprecise)

    def test_no_target_margin_leaves_the_flag_unset(self):
        result = compute_region_metrics(_prepare(held_out=FIXTURE_HELD_OUT), options=OPTIONS)
        self.assertIsNone(result.held_out.oor_rate.imprecise)


class PrecisionFlagTest(unittest.TestCase):
    """`imprecise` reads the target margin against the cell's inflated requirement."""

    def _oor_rate(self, target_margin):
        result = compute_region_metrics(
            _prepare(PRECISION_ROWS),
            options=RegionMetricsOptions(n_resamples=200, seed=1, target_margin=target_margin),
        )
        return result.overall.oor_rate

    def test_a_cell_that_meets_its_inflated_requirement_is_not_flagged(self):
        """A quarter-point margin needs 15 points unclustered and 43 at the
        design effect these images carry; 100 are available either way."""
        estimate = self._oor_rate(0.25)
        self.assertEqual(estimate.k, 40)
        self.assertEqual(estimate.n, 100)
        self.assertEqual(required_n_for_margin(0.4, 0.25, alpha=0.05), 15)
        self.assertFalse(estimate.imprecise)

    def test_the_design_effect_is_what_flags_the_cell(self):
        """At a tenth-point margin the unclustered requirement of 93 points is
        met by the 100 available, so only the design effect can flag it."""
        estimate = self._oor_rate(0.1)
        self.assertEqual(estimate.n, 100)
        self.assertEqual(required_n_for_margin(0.4, 0.1, alpha=0.05), 93)
        self.assertGreater(estimate.design_effect, 1.0)
        self.assertTrue(estimate.imprecise)

    def test_a_design_effect_below_one_is_floored_to_one_before_it_inflates_the_requirement(self):
        """Unfloored, a measured effect of 0.5 would halve the unclustered
        requirement of 97 (rate 0.5, margin 0.1) to 49; floored at 1.0 it
        stays 97, so 60 points fail the floored requirement though they
        would have passed the unfloored one."""
        options = RegionMetricsOptions(alpha=0.05, target_margin=0.1)
        self.assertEqual(required_n_for_margin(0.5, 0.1, alpha=0.05, design_effect=0.5), 49)
        self.assertEqual(required_n_for_margin(0.5, 0.1, alpha=0.05), 97)
        self.assertTrue(_imprecise(rate=0.5, n=60, n_images=60, effect=0.5, options=options))


class RequiredSampleSizeTest(unittest.TestCase):
    """The sample size needed to measure a rate to a target margin of error."""

    def test_required_n_at_the_worst_case_rate(self):
        # 3.841458820694124 * 0.25 / 0.01 = 96.0364705
        self.assertEqual(required_n_for_margin(0.5, 0.1, alpha=0.05), 97)

    def test_clustering_inflates_the_requirement(self):
        # The same figure at a design effect of 4: 384.1458820694124
        self.assertEqual(required_n_for_margin(0.5, 0.1, alpha=0.05, design_effect=4.0), 385)

    def test_an_unobserved_rate_falls_back_to_the_worst_case(self):
        self.assertEqual(required_n_for_margin(0.0, 0.1, alpha=0.05), 97)
        self.assertEqual(required_n_for_margin(math.nan, 0.1, alpha=0.05), 97)

    def test_a_non_positive_effect_is_refused(self):
        with self.assertRaises(ValueError):
            required_n_for_margin(0.5, 0.0, alpha=0.05)


if __name__ == "__main__":
    unittest.main()
