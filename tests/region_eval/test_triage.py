"""Unit tests for region_eval/triage.py.

An eight-point fixture carries one point for each bucket plus the cases that
must not become events:

    #  image  region  ground truth  prediction   bucket
    1  a      CIP     pacific       atlantic     list_suspect
    2  a      CIP     global        atlantic     list_suspect
    3  a      CIP     pacific       pacific      not an event
    4  a      CIP     global        no-regions   unknown_list
    5  b      TA      atlantic      pacific      model_error (2 annotations < 5)
    6  b      TA      atlantic      atlantic     not an event
    7  b      TA      global        no-regions   unknown_list
    8  b      TA      global        rare-pacific model_error (pair unannotated)
    9  c      ""      pacific       atlantic     image region unrecorded

The 'atlantic' attribute carries 7 confirmed annotations in the Central
Indo-Pacific, which is what makes points 1 and 2 a suspect region list rather
than two model mistakes.
"""

import unittest

from mermaid_classifier.region_eval.metrics import prepare_scored_points
from mermaid_classifier.region_eval.triage import (
    DEFAULT_LIST_SUSPECT_THRESHOLD,
    TriageBucket,
    triage_events,
)
from region_eval.fixtures import (
    ATLANTIC_LABEL,
    BA_ATLANTIC,
    BA_NO_REGIONS,
    BA_PACIFIC,
    BASE_REGION_IDS_BY_ATTRIBUTE,
    CENTRAL_INDO_PACIFIC,
    GLOBAL_LABEL,
    NO_REGIONS_LABEL,
    PACIFIC_LABEL,
    TROPICAL_ATLANTIC,
)

BA_RARE_PACIFIC = "2b2b2b2b-0000-4000-8000-000000000005"

RARE_PACIFIC_LABEL = f"{BA_RARE_PACIFIC}::"

REGION_IDS_BY_ATTRIBUTE = {
    **BASE_REGION_IDS_BY_ATTRIBUTE,
    BA_RARE_PACIFIC: frozenset({CENTRAL_INDO_PACIFIC}),
}

MODEL_CLASSES = (
    GLOBAL_LABEL,
    ATLANTIC_LABEL,
    PACIFIC_LABEL,
    NO_REGIONS_LABEL,
    RARE_PACIFIC_LABEL,
)

FIXTURE_ROWS = (
    ("image-a", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, ATLANTIC_LABEL),
    ("image-a", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, ATLANTIC_LABEL),
    ("image-a", CENTRAL_INDO_PACIFIC, PACIFIC_LABEL, PACIFIC_LABEL),
    ("image-a", CENTRAL_INDO_PACIFIC, GLOBAL_LABEL, NO_REGIONS_LABEL),
    ("image-b", TROPICAL_ATLANTIC, ATLANTIC_LABEL, PACIFIC_LABEL),
    ("image-b", TROPICAL_ATLANTIC, ATLANTIC_LABEL, ATLANTIC_LABEL),
    ("image-b", TROPICAL_ATLANTIC, GLOBAL_LABEL, NO_REGIONS_LABEL),
    ("image-b", TROPICAL_ATLANTIC, GLOBAL_LABEL, RARE_PACIFIC_LABEL),
    ("image-c", "", PACIFIC_LABEL, ATLANTIC_LABEL),
)

GROUND_TRUTH_COUNTS = {
    (BA_ATLANTIC, CENTRAL_INDO_PACIFIC): 7,
    (BA_PACIFIC, TROPICAL_ATLANTIC): 2,
}

ATTRIBUTE_NAMES = {BA_ATLANTIC: "Agaricia tenuifolia", BA_PACIFIC: "Goniopora"}
REGION_NAMES = {
    TROPICAL_ATLANTIC: "Tropical Atlantic",
    CENTRAL_INDO_PACIFIC: "Central Indo-Pacific",
}


def _points():
    return prepare_scored_points(
        image_ids=[row[0] for row in FIXTURE_ROWS],
        image_region_ids=[row[1] for row in FIXTURE_ROWS],
        gt_labels=[row[2] for row in FIXTURE_ROWS],
        pred_labels=[row[3] for row in FIXTURE_ROWS],
        region_ids_by_attribute=REGION_IDS_BY_ATTRIBUTE,
        model_classes=MODEL_CLASSES,
    )


def _triage(threshold=DEFAULT_LIST_SUSPECT_THRESHOLD):
    return triage_events(_points(), ground_truth_counts=GROUND_TRUTH_COUNTS, threshold=threshold)


class BucketAssignmentTest(unittest.TestCase):
    def test_every_flagged_point_gets_exactly_one_bucket(self):
        events = _triage().events
        self.assertEqual(len(events), 6)
        self.assertEqual(len(events.drop_duplicates(["image_id", "point_position"])), 6)
        self.assertEqual(set(events["bucket"]), set(TriageBucket))

    def test_repeatedly_annotated_pairs_are_a_suspect_region_list(self):
        events = _triage().events
        suspects = events[events["bucket"] == TriageBucket.LIST_SUSPECT]
        self.assertEqual(len(suspects), 2)
        self.assertEqual(set(suspects["pred_label"]), {ATLANTIC_LABEL})
        self.assertEqual(set(suspects["ground_truth_count"]), {7})

    def test_predictions_without_recorded_regions_are_their_own_bucket(self):
        events = _triage().events
        unknown = events[events["bucket"] == TriageBucket.UNKNOWN_LIST]
        self.assertEqual(len(unknown), 2)
        self.assertEqual(set(unknown["pred_label"]), {NO_REGIONS_LABEL})

    def test_everything_else_is_model_error(self):
        events = _triage().events
        errors = events[events["bucket"] == TriageBucket.MODEL_ERROR]
        self.assertEqual(len(errors), 2)
        self.assertEqual(set(errors["pred_label"]), {PACIFIC_LABEL, RARE_PACIFIC_LABEL})

    def test_a_missing_region_list_outranks_a_heavily_annotated_pair(self):
        """With no list at all there is nothing for the annotations to contradict."""
        counts = dict(GROUND_TRUTH_COUNTS)
        counts[(BA_NO_REGIONS, CENTRAL_INDO_PACIFIC)] = 50
        counts[(BA_NO_REGIONS, TROPICAL_ATLANTIC)] = 50
        events = triage_events(_points(), ground_truth_counts=counts).events
        unknown = events[events["pred_label"] == NO_REGIONS_LABEL]
        self.assertEqual(len(unknown), 2)
        self.assertEqual(set(unknown["bucket"]), {TriageBucket.UNKNOWN_LIST})

    def test_an_unannotated_pair_counts_as_zero_rather_than_going_missing(self):
        events = _triage().events
        rare = events[events["pred_label"] == RARE_PACIFIC_LABEL].iloc[0]
        self.assertEqual(int(rare["ground_truth_count"]), 0)
        self.assertEqual(rare["bucket"], TriageBucket.MODEL_ERROR)


class ThresholdTest(unittest.TestCase):
    """Seven annotations sit either side of the threshold at T = 7 and T = 8."""

    def test_threshold_at_the_observed_count_keeps_the_pair_suspect(self):
        events = _triage(threshold=7).events
        suspects = events[events["bucket"] == TriageBucket.LIST_SUSPECT]
        self.assertEqual(len(suspects), 2)

    def test_threshold_one_above_flips_the_pair_to_model_error(self):
        events = _triage(threshold=8).events
        self.assertEqual(len(events[events["bucket"] == TriageBucket.LIST_SUSPECT]), 0)
        self.assertEqual(len(events[events["bucket"] == TriageBucket.MODEL_ERROR]), 4)

    def test_a_non_positive_threshold_is_refused(self):
        with self.assertRaises(ValueError):
            _triage(threshold=0)


class RegionListSuspectsTest(unittest.TestCase):
    """The suspects table is a taxonomy bug report, not an exclusion list."""

    def test_one_row_per_suspect_attribute_and_region(self):
        table = _triage().region_list_suspects
        self.assertEqual(len(table), 1)
        row = table.iloc[0]
        self.assertEqual(row["attribute_id"], BA_ATLANTIC)
        self.assertEqual(row["region_id"], CENTRAL_INDO_PACIFIC)
        self.assertEqual(int(row["n_ground_truth"]), 7)
        self.assertEqual(int(row["n_predicted"]), 2)

    def test_sorted_by_ground_truth_count(self):
        counts = dict(GROUND_TRUTH_COUNTS)
        counts[(BA_PACIFIC, TROPICAL_ATLANTIC)] = 9
        result = triage_events(_points(), ground_truth_counts=counts)
        table = result.region_list_suspects
        self.assertEqual(list(table["n_ground_truth"]), [9, 7])
        self.assertEqual(list(table["attribute_id"]), [BA_PACIFIC, BA_ATLANTIC])

    def test_suspects_name_the_attribute_and_the_region(self):
        """This table is a bug report handed to the data team. Two UUIDs name
        neither the coral to check nor the ocean to add it to.
        """
        result = triage_events(
            _points(),
            ground_truth_counts=GROUND_TRUTH_COUNTS,
            attribute_names=ATTRIBUTE_NAMES,
            region_names=REGION_NAMES,
        )
        row = result.region_list_suspects.iloc[0]
        self.assertEqual(row["attribute_id"], BA_ATLANTIC)
        self.assertEqual(row["attribute_name"], "Agaricia tenuifolia")
        self.assertEqual(row["region_id"], CENTRAL_INDO_PACIFIC)
        self.assertEqual(row["region_name"], "Central Indo-Pacific")

    def test_an_unresolved_name_renders_its_id_rather_than_an_empty_cell(self):
        """A probe frozen before the names were is the case this covers: the
        id says "unresolved" where a blank would say "unnamed".
        """
        row = _triage().region_list_suspects.iloc[0]
        self.assertEqual(row["attribute_name"], BA_ATLANTIC)
        self.assertEqual(row["region_name"], CENTRAL_INDO_PACIFIC)


if __name__ == "__main__":
    unittest.main()
