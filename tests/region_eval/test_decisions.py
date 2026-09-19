"""Unit tests for region_eval/decisions.py.

Each statistic gets a fixture small enough to count by hand, and every
expected value below is a literal derived from that table rather than from a
call into the module under test.

The permutation baseline is carried by a matched pair of fixtures, because a
ratio only means something against its opposite:

    region-blind      every image holds one Atlantic and one Pacific
                      prediction, so exactly half are out of region whatever
                      region the image is in -> baseline == observed -> 1.0
    region-aware      Atlantic images hold only Atlantic predictions and
                      Pacific images only Pacific ones -> observed 0, while
                      shuffling the regions puts half of them out -> 0.0

Both fixtures span two regions, which is what makes the Atlantic and Pacific
labels region-discriminating: a label permitted everywhere the data goes
cannot be out of region and is not in any denominator here.
"""

import math
import unittest

import numpy as np

from mermaid_classifier.region_eval.decisions import (
    confidence_stratification,
    masking_counterfactual,
    region_blind_baseline,
    within_branch_share,
)
from mermaid_classifier.region_eval.metrics import prepare_scored_points

TROPICAL_ATLANTIC = "1a1a1a1a-0000-4000-8000-000000000001"
CENTRAL_INDO_PACIFIC = "1a1a1a1a-0000-4000-8000-000000000002"
EASTERN_PACIFIC = "1a1a1a1a-0000-4000-8000-000000000003"

BA_GLOBAL = "2b2b2b2b-0000-4000-8000-000000000001"
BA_ATLANTIC = "2b2b2b2b-0000-4000-8000-000000000002"
BA_PACIFIC = "2b2b2b2b-0000-4000-8000-000000000003"
BA_NO_REGIONS = "2b2b2b2b-0000-4000-8000-000000000004"
BA_OFF_LIST = "2b2b2b2b-0000-4000-8000-000000000005"

GLOBAL_LABEL = f"{BA_GLOBAL}::"
ATLANTIC_LABEL = f"{BA_ATLANTIC}::"
PACIFIC_LABEL = f"{BA_PACIFIC}::"
NO_REGIONS_LABEL = f"{BA_NO_REGIONS}::"
OFF_LIST_LABEL = f"{BA_OFF_LIST}::"

REGION_IDS_BY_ATTRIBUTE = {
    BA_GLOBAL: frozenset({TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC}),
    BA_ATLANTIC: frozenset({TROPICAL_ATLANTIC}),
    BA_PACIFIC: frozenset({CENTRAL_INDO_PACIFIC}),
    BA_NO_REGIONS: frozenset(),
    BA_OFF_LIST: frozenset({TROPICAL_ATLANTIC}),
}

# One Atlantic and one Pacific prediction in every image, so the out-of-region
# count is 4 of 8 under the true regions and under every permutation of them.
BLIND_ROWS = (
    ("image-a", TROPICAL_ATLANTIC, ATLANTIC_LABEL),
    ("image-a", TROPICAL_ATLANTIC, PACIFIC_LABEL),
    ("image-b", TROPICAL_ATLANTIC, ATLANTIC_LABEL),
    ("image-b", TROPICAL_ATLANTIC, PACIFIC_LABEL),
    ("image-c", CENTRAL_INDO_PACIFIC, ATLANTIC_LABEL),
    ("image-c", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL),
    ("image-d", CENTRAL_INDO_PACIFIC, ATLANTIC_LABEL),
    ("image-d", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL),
)

# Every prediction matches its image's region, so none is out of region.
AWARE_ROWS = (
    ("image-a", TROPICAL_ATLANTIC, ATLANTIC_LABEL),
    ("image-a", TROPICAL_ATLANTIC, ATLANTIC_LABEL),
    ("image-b", TROPICAL_ATLANTIC, ATLANTIC_LABEL),
    ("image-b", TROPICAL_ATLANTIC, ATLANTIC_LABEL),
    ("image-c", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL),
    ("image-c", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL),
    ("image-d", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL),
    ("image-d", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL),
)


def _baseline(rows, **kwargs):
    """The baseline over `rows`, which carry no ground truth to prepare with.

    The statistic reads predictions alone, so the predictions stand in as the
    truth column `prepare_scored_points` requires.
    """
    predictions = [row[2] for row in rows]
    points = prepare_scored_points(
        image_ids=[row[0] for row in rows],
        image_region_ids=[row[1] for row in rows],
        gt_labels=predictions,
        pred_labels=predictions,
        region_ids_by_attribute=REGION_IDS_BY_ATTRIBUTE,
        model_classes=(),
    )
    return region_blind_baseline(points, n_permutations=400, seed=0, **kwargs)


class RegionBlindBaselineTest(unittest.TestCase):
    def test_region_blind_predictions_score_the_ratio_at_one(self):
        """Predictions carrying no region information must not read as a
        region-aware model. Half of the 8 discriminating predictions are out
        of region under the true assignment and under every permutation, so
        observed and baseline are both 4/8 and the ratio is exactly 1."""
        result = _baseline(BLIND_ROWS)

        self.assertEqual(result.n_out_of_region, 4)
        self.assertEqual(result.n_discriminating, 8)
        self.assertEqual(result.observed_rate, 0.5)
        self.assertEqual(result.baseline_rate, 0.5)
        self.assertEqual(result.ratio, 1.0)

    def test_region_blind_baseline_has_no_spread_when_every_permutation_agrees(self):
        """The dispersion must come from the permutation distribution, not
        from a constant. Every permutation of this fixture yields 4/8, so the
        spread is zero and the interval collapses on the centre."""
        result = _baseline(BLIND_ROWS)

        self.assertEqual(result.baseline_sd, 0.0)
        self.assertEqual((result.baseline_ci_low, result.baseline_ci_high), (0.5, 0.5))

    def test_region_aware_predictions_score_the_ratio_far_below_one(self):
        """A model whose predictions track region must separate from the
        blind case. Observed is 0 of 8; shuffling the four image regions
        leaves each image mismatched with probability 1/2, so the baseline
        averages 4/8 and the ratio is 0."""
        result = _baseline(AWARE_ROWS)

        self.assertEqual(result.n_out_of_region, 0)
        self.assertEqual(result.observed_rate, 0.0)
        self.assertAlmostEqual(result.baseline_rate, 0.5, delta=0.05)
        self.assertEqual(result.ratio, 0.0)

    def test_region_aware_baseline_spread_spans_the_three_reachable_rates(self):
        """The permutation distribution here is 0, 1/2 or 1 with weights
        1/6, 4/6, 1/6, whose standard deviation is sqrt(1/12) = 0.2887 and
        whose 2.5th and 97.5th percentiles are the extremes themselves."""
        result = _baseline(AWARE_ROWS)

        self.assertAlmostEqual(result.baseline_sd, 0.2887, delta=0.03)
        self.assertEqual((result.baseline_ci_low, result.baseline_ci_high), (0.0, 1.0))

    def test_labels_with_no_recorded_regions_leave_the_denominator(self):
        """An attribute nobody has recorded a region for is unrecorded, not
        permitted nowhere. Counting the two extra rows would read 4/10 = 0.4
        instead of the 4/8 the answerable predictions support."""
        rows = (
            *BLIND_ROWS,
            ("image-a", TROPICAL_ATLANTIC, NO_REGIONS_LABEL),
            ("image-c", CENTRAL_INDO_PACIFIC, NO_REGIONS_LABEL),
        )

        result = _baseline(rows)

        self.assertEqual(result.n_points, 10)
        self.assertEqual(result.n_region_unknown, 2)
        self.assertEqual(result.n_discriminating, 8)
        self.assertEqual(result.observed_rate, 0.5)
        self.assertEqual(result.baseline_rate, 0.5)
        self.assertEqual(result.ratio, 1.0)

    def test_ratio_is_undefined_when_no_permutation_can_place_a_label_out_of_region(self):
        """A single-region corpus of globally permitted labels has nothing to
        shuffle: the baseline is 0 and dividing by it would report an
        infinite or zero ratio as if it were a finding."""
        rows = (
            ("image-a", TROPICAL_ATLANTIC, GLOBAL_LABEL),
            ("image-b", TROPICAL_ATLANTIC, GLOBAL_LABEL),
        )

        result = _baseline(rows)

        self.assertEqual(result.n_discriminating, 0)
        self.assertTrue(math.isnan(result.ratio))


# Probability columns are ordered (global, Atlantic, Pacific). Read against the
# region map above, the masked argmax of each row is hand-derivable:
#
#  #   image  region  ground truth  probabilities        unmasked  masked
#  M1  a      CIP     Pacific       0.20 0.50 0.30       Atlantic  Pacific   fixed
#  M2  a      CIP     global        0.45 0.40 0.15       global    global
#  M3  b      TA      Atlantic      0.30 0.45 0.25       Atlantic  Atlantic
#  M4  b      TA      Pacific       0.10 0.30 0.60       Pacific   Atlantic  broken
#  M5  b      TA      off-list      0.60 0.20 0.20       global    global
#
# M5's truth is outside the model's label space, so it is never correct and
# leaves accuracy; it stays in the changed-share denominator, which needs no
# ground truth. One fix and one break cancel to a delta of zero, which is the
# reason the two directions are counted apart.
MASKING_CLASSES = (GLOBAL_LABEL, ATLANTIC_LABEL, PACIFIC_LABEL)

MASKING_ROWS = (
    ("image-a", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, (0.20, 0.50, 0.30)),
    ("image-a", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, (0.45, 0.40, 0.15)),
    ("image-b", TROPICAL_ATLANTIC, ATLANTIC_LABEL, (0.30, 0.45, 0.25)),
    ("image-b", TROPICAL_ATLANTIC, PACIFIC_LABEL, (0.10, 0.30, 0.60)),
    ("image-b", TROPICAL_ATLANTIC, OFF_LIST_LABEL, (0.60, 0.20, 0.20)),
)

# Every top prediction already sits in its image's region.
IN_REGION_ROWS = (
    ("image-a", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, (0.20, 0.30, 0.50)),
    ("image-b", TROPICAL_ATLANTIC, ATLANTIC_LABEL, (0.30, 0.50, 0.20)),
)


def _masking(rows, *, model_classes=MASKING_CLASSES, **kwargs):
    """The counterfactual over `rows`, whose argmax is the model's prediction.

    The probability matrix stays indexed over every row, unrecorded regions
    included, which is the alignment `source_positions` carries.
    """
    probabilities = np.array([row[3] for row in rows], dtype=np.float64)
    points = prepare_scored_points(
        image_ids=[row[0] for row in rows],
        image_region_ids=[row[1] for row in rows],
        gt_labels=[row[2] for row in rows],
        pred_labels=[model_classes[int(index)] for index in probabilities.argmax(axis=1)],
        region_ids_by_attribute=REGION_IDS_BY_ATTRIBUTE,
        model_classes=model_classes,
    )
    return masking_counterfactual(
        points,
        probabilities=probabilities,
        model_classes=model_classes,
        region_ids_by_attribute=REGION_IDS_BY_ATTRIBUTE,
        **kwargs,
    )


class MaskingCounterfactualTest(unittest.TestCase):
    def test_masking_changes_only_the_out_of_region_top_predictions(self):
        """Zeroing a class that was never the argmax must leave the prediction
        alone. M2 and M3 each lose a class to the mask and keep their top-1;
        only M1 and M4, whose top-1 is itself out of region, move."""
        result = _masking(MASKING_ROWS)

        self.assertEqual(result.n_points, 5)
        self.assertEqual(result.n_changed, 2)
        self.assertEqual(result.changed_share, 0.4)

    def test_masking_counts_what_it_fixes_apart_from_what_it_breaks(self):
        """The accuracy delta is zero here and the mitigation is not neutral:
        one wrong prediction becomes right and one right one becomes wrong.
        A delta alone would price this as costless."""
        result = _masking(MASKING_ROWS)

        self.assertEqual(result.n_fixed, 1)
        self.assertEqual(result.n_broken, 1)
        self.assertEqual(result.accuracy_unmasked, 0.75)
        self.assertEqual(result.accuracy_masked, 0.75)
        self.assertEqual(result.accuracy_delta, 0.0)

    def test_margin_over_out_of_region_events_prices_the_confidence_masking_fights(self):
        """M1 gives 0.50 - 0.30 and M4 gives 0.60 - 0.30, so the median margin
        is 0.25. Reading the margin over every point, rather than over the
        out-of-region events, would drown it in zeros."""
        result = _masking(MASKING_ROWS)

        self.assertEqual(result.margin_n, 2)
        self.assertAlmostEqual(result.margin_median, 0.25, places=10)
        self.assertAlmostEqual(result.margin_p90, 0.29, places=10)
        self.assertAlmostEqual(result.margin_max, 0.30, places=10)

    def test_accuracy_delta_interval_resamples_images_not_points(self):
        """Image a supplies both fixes and image b both breaks, so a resample
        of the two images yields +0.5, 0.0 or -0.5 and the percentile interval
        is exactly (-0.5, 0.5). Resampling points would split the two images
        and return a narrower interval."""
        result = _masking(MASKING_ROWS, n_resamples=2000, seed=0)

        self.assertEqual(result.accuracy_delta_ci_low, -0.5)
        self.assertEqual(result.accuracy_delta_ci_high, 0.5)

    def test_a_lone_fix_is_counted_as_a_fix(self):
        """One argmax moves and it moves the right way. The shared fixture
        holds one fix against one break, so the two counts could be swapped
        there without any assertion noticing; here the swap reads (0, 1)."""
        rows = (
            ("image-a", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, (0.20, 0.50, 0.30)),
            ("image-a", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, (0.45, 0.40, 0.15)),
        )

        result = _masking(rows)

        self.assertEqual(result.n_changed, 1)
        self.assertEqual(result.n_fixed, 1)
        self.assertEqual(result.n_broken, 0)
        self.assertEqual(result.accuracy_unmasked, 0.5)
        self.assertEqual(result.accuracy_masked, 1.0)
        self.assertEqual(result.accuracy_delta, 0.5)

    def test_masking_is_inert_where_every_top_prediction_is_already_in_region(self):
        """Nothing to fix and nothing to break. A mask that renormalised its
        way into a different argmax here would be corrupting predictions it
        was never meant to touch."""
        result = _masking(IN_REGION_ROWS)

        self.assertEqual(result.n_changed, 0)
        self.assertEqual(result.n_fixed, 0)
        self.assertEqual(result.n_broken, 0)
        self.assertEqual(result.accuracy_delta, 0.0)
        self.assertEqual(result.margin_n, 0)
        self.assertTrue(math.isnan(result.margin_median))

    def test_unrecorded_image_regions_are_dropped_and_counted_without_raising(self):
        """A "" region reaching `is_out_of_region` raises, and the metrics
        orchestrator swallows what a group raises, so the whole counterfactual
        would vanish rather than report four points and one exclusion. The
        unrecorded row leads the list, so `source_positions` falling back to
        the identity mapping would pair M1's image and region with this row's
        global-leaning probabilities instead of M1's own out-of-region ones,
        changing which predictions masking counts as fixed."""
        rows = (("image-e", "", PACIFIC_LABEL, (0.90, 0.05, 0.05)), *MASKING_ROWS)

        result = _masking(rows)

        self.assertEqual(result.n_unrecorded_region_excluded, 1)
        self.assertEqual(result.n_points, 5)
        self.assertEqual(result.n_changed, 2)
        self.assertEqual(result.n_fixed, 1)
        self.assertEqual(result.n_broken, 1)


# Ancestry arrives as data, root first and ending in the attribute itself, the
# shape `build_ba_paths` produces. Two Acropora species share a genus; the
# macroalga shares nothing above the root.
BA_ACROPORA_TA = "3c3c3c3c-0000-4000-8000-000000000001"
BA_ACROPORA_CIP = "3c3c3c3c-0000-4000-8000-000000000002"
BA_PORITES_CIP = "3c3c3c3c-0000-4000-8000-000000000003"
BA_ALGAE_CIP = "3c3c3c3c-0000-4000-8000-000000000004"
BA_UNPLACED_CIP = "3c3c3c3c-0000-4000-8000-000000000005"

HARD_CORAL = "4d4d4d4d-0000-4000-8000-000000000001"
ACROPORA = "4d4d4d4d-0000-4000-8000-000000000002"
PORITES = "4d4d4d4d-0000-4000-8000-000000000003"
MACROALGAE = "4d4d4d4d-0000-4000-8000-000000000004"

ACROPORA_TA_LABEL = f"{BA_ACROPORA_TA}::"
ACROPORA_CIP_LABEL = f"{BA_ACROPORA_CIP}::"
PORITES_CIP_LABEL = f"{BA_PORITES_CIP}::"
ALGAE_CIP_LABEL = f"{BA_ALGAE_CIP}::"
UNPLACED_CIP_LABEL = f"{BA_UNPLACED_CIP}::"

BRANCH_REGION_IDS = {
    BA_ACROPORA_TA: frozenset({TROPICAL_ATLANTIC}),
    BA_ACROPORA_CIP: frozenset({CENTRAL_INDO_PACIFIC}),
    BA_PORITES_CIP: frozenset({CENTRAL_INDO_PACIFIC}),
    BA_ALGAE_CIP: frozenset({CENTRAL_INDO_PACIFIC}),
    BA_UNPLACED_CIP: frozenset({CENTRAL_INDO_PACIFIC}),
    BA_NO_REGIONS: frozenset(),
}

ANCESTRY = {
    BA_ACROPORA_TA: (HARD_CORAL, ACROPORA, BA_ACROPORA_TA),
    BA_ACROPORA_CIP: (HARD_CORAL, ACROPORA, BA_ACROPORA_CIP),
    BA_PORITES_CIP: (HARD_CORAL, PORITES, BA_PORITES_CIP),
    BA_ALGAE_CIP: (MACROALGAE, BA_ALGAE_CIP),
    BA_NO_REGIONS: (MACROALGAE, BA_NO_REGIONS),
}

# Every image is Tropical Atlantic, so each Indo-Pacific prediction is an
# out-of-region event and the only question left is how far the confusion is.
#
#  #   image  ground truth   prediction        event  shares an ancestor
#  B1  a      Acropora TA    Acropora CIP      yes    yes (genus Acropora)
#  B2  a      Acropora TA    macroalga CIP     yes    no  (diverges at the root)
#  B3  a      Acropora TA    Acropora TA       no
BRANCH_ROWS = (
    (TROPICAL_ATLANTIC, ACROPORA_TA_LABEL, ACROPORA_CIP_LABEL),
    (TROPICAL_ATLANTIC, ACROPORA_TA_LABEL, ALGAE_CIP_LABEL),
    (TROPICAL_ATLANTIC, ACROPORA_TA_LABEL, ACROPORA_TA_LABEL),
)


def _branch(rows, **kwargs):
    """The share over `rows`, every one of which sits in the same image."""
    points = prepare_scored_points(
        image_ids=["image-a"] * len(rows),
        image_region_ids=[row[0] for row in rows],
        gt_labels=[row[1] for row in rows],
        pred_labels=[row[2] for row in rows],
        region_ids_by_attribute=BRANCH_REGION_IDS,
        model_classes=(),
    )
    return within_branch_share(points, ancestry_by_attribute=ANCESTRY, **kwargs)


class WithinBranchShareTest(unittest.TestCase):
    def test_share_separates_a_same_genus_confusion_from_a_cross_branch_one(self):
        """One of the two events keeps the genus and swaps the ocean, the
        other is an ordinary misclassification that masking would not touch.
        Counting them together would report either 0 or 1 and hide which
        mitigation applies."""
        result = _branch(BRANCH_ROWS)

        self.assertEqual(result.n_out_of_region, 2)
        self.assertEqual(result.n_within_branch, 1)
        self.assertEqual(result.share, 0.5)

    def test_in_region_predictions_are_not_events(self):
        """B3 is a correct in-region prediction. Scoring it would put a third
        point in the denominator and read the share as 1/3."""
        result = _branch(BRANCH_ROWS)

        self.assertEqual(result.n_points, 3)
        self.assertEqual(result.n_evaluable, 2)

    def test_a_shared_root_counts_as_the_same_branch(self):
        """Acropora against Porites diverges below hard coral but not at it.
        The question is whether the model has the broad group right, so an
        implementation demanding a deeper common ancestor would read 0."""
        rows = ((TROPICAL_ATLANTIC, ACROPORA_TA_LABEL, PORITES_CIP_LABEL),)

        result = _branch(rows)

        self.assertEqual(result.n_within_branch, 1)
        self.assertEqual(result.share, 1.0)

    def test_attributes_missing_from_the_ancestry_leave_both_halves_of_the_share(self):
        """An attribute the ancestry snapshot omits cannot be placed in or out
        of a branch. Defaulting it either way moves the share; here it would
        read 1/3 or 2/3 instead of the 1/2 the placeable events support."""
        rows = (*BRANCH_ROWS, (TROPICAL_ATLANTIC, ACROPORA_TA_LABEL, UNPLACED_CIP_LABEL))

        result = _branch(rows)

        self.assertEqual(result.n_out_of_region, 3)
        self.assertEqual(result.n_ancestry_unknown, 1)
        self.assertEqual(result.n_evaluable, 2)
        self.assertEqual(result.share, 0.5)

    def test_labels_with_no_recorded_regions_never_become_events(self):
        """An attribute with no recorded regions is unrecorded, not permitted
        nowhere, so it is not out of region anywhere and cannot be counted as
        a cross-branch confusion."""
        rows = (*BRANCH_ROWS, (TROPICAL_ATLANTIC, ACROPORA_TA_LABEL, NO_REGIONS_LABEL))

        result = _branch(rows)

        self.assertEqual(result.n_region_unknown, 1)
        self.assertEqual(result.n_out_of_region, 2)
        self.assertEqual(result.share, 0.5)

    def test_share_is_undefined_where_nothing_went_out_of_region(self):
        """An empty denominator has no share. Reporting 0.0 would read as
        "every event is an ordinary misclassification" on a model that made no
        out-of-region prediction at all."""
        rows = ((TROPICAL_ATLANTIC, ACROPORA_TA_LABEL, ACROPORA_TA_LABEL),)

        result = _branch(rows)

        self.assertEqual(result.n_evaluable, 0)
        self.assertTrue(math.isnan(result.share))


# Both regions appear, which is what makes the Atlantic and Pacific labels
# region-discriminating; the global label is permitted everywhere the data goes
# and can never be out of region, so it carries no information about whether
# confidence separates the two.
#
#  #   region  prediction  confidence  status
#  C1  TA      Pacific     0.90        out of region
#  C2  TA      Pacific     0.30        out of region
#  C3  CIP     Pacific     0.80        in region
#  C4  CIP     Pacific     0.20        in region
#  C5  CIP     global      0.95        not region-discriminating
#  C6  CIP     no-regions  0.55        region unrecorded
#  C7  ""      Pacific     0.99        image region unrecorded
#
# Of the four ordered (out-of-region, in-region) pairs, three put the higher
# confidence on the out-of-region member, so the AUROC is 3/4.
CONFIDENCE_ROWS = (
    (TROPICAL_ATLANTIC, PACIFIC_LABEL, 0.90),
    (TROPICAL_ATLANTIC, PACIFIC_LABEL, 0.30),
    (CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, 0.80),
    (CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, 0.20),
    (CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, 0.95),
    (CENTRAL_INDO_PACIFIC, NO_REGIONS_LABEL, 0.55),
    ("", PACIFIC_LABEL, 0.99),
)

QUARTILE_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _stratify(rows, **kwargs):
    """The stratification over `rows`, which carry no ground truth.

    The statistic reads predictions alone, so the predictions stand in as the
    truth column `prepare_scored_points` requires, and the confidences stay
    indexed over every row.
    """
    predictions = [row[1] for row in rows]
    points = prepare_scored_points(
        image_ids=["image-a"] * len(rows),
        image_region_ids=[row[0] for row in rows],
        gt_labels=predictions,
        pred_labels=predictions,
        region_ids_by_attribute=REGION_IDS_BY_ATTRIBUTE,
        model_classes=(),
    )
    return confidence_stratification(points, pred_confidences=[row[2] for row in rows], **kwargs)


def _auroc_of(positives, negatives):
    """AUROC over Pacific predictions, out of region in TA and in region in CIP."""
    rows = [(TROPICAL_ATLANTIC, PACIFIC_LABEL, score) for score in positives]
    rows += [(CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, score) for score in negatives]
    return _stratify(tuple(rows)).auroc


class ConfidenceStratificationTest(unittest.TestCase):
    def test_auroc_reads_how_far_confidence_ranks_out_of_region_events_apart(self):
        """Three of the four cross pairs rank the out-of-region prediction
        above the in-region one. A value this far from 0.5 is what would keep
        a confidence threshold on the table as a mitigation."""
        self.assertEqual(_stratify(CONFIDENCE_ROWS, bin_edges=QUARTILE_EDGES).auroc, 0.75)

    def test_bins_count_and_rate_the_discriminating_predictions_per_quartile(self):
        """Counted by hand off the table: 0.20 alone below 0.25; 0.30 alone
        and out of region in the second quartile; nothing in the third; 0.90
        and 0.80 in the fourth, one of the two out of region."""
        result = _stratify(CONFIDENCE_ROWS, bin_edges=QUARTILE_EDGES)

        counts = [(bin_.n, bin_.n_out_of_region) for bin_ in result.bins]
        self.assertEqual(counts, [(1, 0), (1, 1), (0, 0), (2, 1)])
        self.assertEqual([bin_.rate for bin_ in result.bins[:2]], [0.0, 1.0])
        self.assertEqual(result.bins[3].rate, 0.5)
        self.assertTrue(math.isnan(result.bins[2].rate))

    def test_predictions_permitted_everywhere_leave_the_bins(self):
        """C5 is the most confident prediction in the fixture and could never
        have been out of region. Binning it would put three points in the top
        quartile and drag that bin's rate from 0.5 to 1/3."""
        result = _stratify(CONFIDENCE_ROWS, bin_edges=QUARTILE_EDGES)

        self.assertEqual(result.n_points, 6)
        self.assertEqual(result.n_discriminating, 4)
        self.assertEqual(result.n_region_unknown, 1)
        self.assertEqual(result.bins[3].n, 2)

    def test_auroc_is_one_half_where_confidence_carries_no_signal(self):
        """Interleaved scores: each out-of-region event beats one in-region
        prediction and loses to the other. This is the reading that kills a
        confidence threshold as a mitigation."""
        self.assertEqual(_auroc_of(positives=(0.2, 0.8), negatives=(0.4, 0.6)), 0.5)

    def test_auroc_is_one_where_confidence_separates_perfectly(self):
        """Every out-of-region event outranks every in-region prediction."""
        self.assertEqual(_auroc_of(positives=(0.8, 0.9), negatives=(0.1, 0.2)), 1.0)

    def test_tied_confidences_score_one_half_rather_than_a_rank_artefact(self):
        """Ranks broken by input order would read 0 or 1 here depending on
        which class the sort happened to put first."""
        self.assertEqual(_auroc_of(positives=(0.5, 0.5), negatives=(0.5, 0.5)), 0.5)

    def test_auroc_is_undefined_with_no_out_of_region_events(self):
        """There is nothing for confidence to separate. Reporting 0.5 would
        read as "confidence carries no signal" on a model that made no
        out-of-region prediction at all."""
        rows = (
            (TROPICAL_ATLANTIC, ATLANTIC_LABEL, 0.80),
            (CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, 0.30),
        )

        result = _stratify(rows)

        self.assertEqual(result.n_out_of_region, 0)
        self.assertEqual(result.n_discriminating, 2)
        self.assertTrue(math.isnan(result.auroc))

    def test_auroc_is_undefined_with_no_in_region_predictions(self):
        """The other empty class. Both predictions are out of region, so there
        is no negative to rank them against."""
        rows = (
            (TROPICAL_ATLANTIC, PACIFIC_LABEL, 0.80),
            (CENTRAL_INDO_PACIFIC, ATLANTIC_LABEL, 0.30),
        )

        result = _stratify(rows)

        self.assertEqual(result.n_out_of_region, 2)
        self.assertTrue(math.isnan(result.auroc))

    def test_unrecorded_image_regions_are_dropped_and_counted_without_raising(self):
        """C7 would raise inside `is_out_of_region` and take the whole
        stratification with it. Its 0.99 is also the highest score present, so
        admitting it would move the AUROC as well as the top bin. C7 leads the
        list here, so `source_positions` falling back to the identity mapping
        would shift every remaining confidence onto the wrong point instead of
        merely dropping a trailing one."""
        rows = (CONFIDENCE_ROWS[-1], *CONFIDENCE_ROWS[:-1])

        result = _stratify(rows, bin_edges=QUARTILE_EDGES)

        self.assertEqual(result.n_unrecorded_region_excluded, 1)
        self.assertEqual(result.n_points, 6)
        self.assertEqual(result.auroc, 0.75)

    def test_confidence_of_nan_is_rejected_as_outside_the_bins(self):
        """NaN falls neither below the lowest edge nor above the highest, so a
        range test built from those two comparisons alone would wave it
        through. It would then count in `n_discriminating` while landing in no
        bin, so the per-bin counts would stop summing to the total."""
        rows = ((TROPICAL_ATLANTIC, PACIFIC_LABEL, math.nan),)

        with self.assertRaises(ValueError):
            _stratify(rows)


class SourceRowAlignmentTest(unittest.TestCase):
    """Both statistics take a column indexed over the rows the points were
    prepared from, and read it through `source_positions`. A column of any
    other length still indexes, so it would pair confidences and probabilities
    with the wrong points and move the result rather than raise."""

    def test_a_probability_matrix_shorter_than_its_source_rows_is_refused(self):
        probabilities = np.array([row[3] for row in MASKING_ROWS], dtype=np.float64)
        points = prepare_scored_points(
            image_ids=[row[0] for row in MASKING_ROWS],
            image_region_ids=[row[1] for row in MASKING_ROWS],
            gt_labels=[row[2] for row in MASKING_ROWS],
            pred_labels=[MASKING_CLASSES[int(i)] for i in probabilities.argmax(axis=1)],
            region_ids_by_attribute=REGION_IDS_BY_ATTRIBUTE,
            model_classes=MASKING_CLASSES,
        )

        with self.assertRaisesRegex(ValueError, r"probabilities must hold one row per source"):
            masking_counterfactual(
                points,
                probabilities=probabilities[:-1],
                model_classes=MASKING_CLASSES,
                region_ids_by_attribute=REGION_IDS_BY_ATTRIBUTE,
            )

    def test_a_confidence_column_shorter_than_its_source_rows_is_refused(self):
        predictions = [row[1] for row in CONFIDENCE_ROWS]
        points = prepare_scored_points(
            image_ids=["image-a"] * len(CONFIDENCE_ROWS),
            image_region_ids=[row[0] for row in CONFIDENCE_ROWS],
            gt_labels=predictions,
            pred_labels=predictions,
            region_ids_by_attribute=REGION_IDS_BY_ATTRIBUTE,
            model_classes=(),
        )

        with self.assertRaisesRegex(ValueError, r"pred_confidences must hold one row per source"):
            confidence_stratification(
                points,
                pred_confidences=[row[2] for row in CONFIDENCE_ROWS[:-1]],
                bin_edges=QUARTILE_EDGES,
            )
