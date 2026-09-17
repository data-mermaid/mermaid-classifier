"""Tests for mermaid_classifier.pyspacer.metrics.region.

One 6-point fixture carries every case the group has to separate. Expected
values are counted by hand off this table and written as k/n literals:

    #  image  region  ground truth  prediction  out of region
    0  img-a  TA      A1            A1          no
    1  img-a  TA      A1            A2          yes (A2 is Pacific only)
    2  img-b  CIP     A2            A1          yes (A1 is Atlantic only)
    3  img-b  CIP     A1            B1          no (B1 spans both regions);
                                                ground truth A1 is out of region
    4  img-c  ''      A1            A2          image region unrecorded
    5  img-c  ''      A1            A2          image region unrecorded

Observed regions are {TA, CIP}, so B1 never discriminates while A1 and A2 do.
Four points survive the unrecorded-region filter, over two images.
"""

import unittest
from collections import OrderedDict

from spacer.data_classes import DataLocation

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import MetricGroupResult
from mermaid_classifier.pyspacer.metrics.region import compute_region
from mermaid_classifier.pyspacer.settings import settings

from .metrics_test_helpers import (
    MockBALibrary,
    MockGFLibrary,
    format_metric,
    make_val_results,
)
from .test_train import override_settings

TROPICAL_ATLANTIC = "region-ta"
CENTRAL_INDO_PACIFIC = "region-cip"
UNRECORDED = ("", "")

CLASSES = ["A1::", "A2::", "B1::"]

# (image key, (region id, region name), n_points)
FIXTURE_IMAGES = (
    ("img-a", (TROPICAL_ATLANTIC, "Tropical Atlantic"), 2),
    ("img-b", (CENTRAL_INDO_PACIFIC, "Central Indo-Pacific"), 2),
    ("img-c", UNRECORDED, 2),
)
FIXTURE_GT = [0, 0, 1, 0, 0, 0]
FIXTURE_EST = [0, 1, 0, 2, 1, 1]

# Five Central Indo-Pacific images of two points. The first two predict an
# Atlantic-only label on both their points and the last three never do, so a
# resample that swaps whole images moves the rate in steps of a fifth.
CLUSTERED_IMAGES = tuple(
    (f"img-{index}", (CENTRAL_INDO_PACIFIC, "Central Indo-Pacific"), 2) for index in range(5)
)
CLUSTERED_GT = [1] * 10
CLUSTERED_EST = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def _make_loc(key: str) -> DataLocation:
    return DataLocation(storage_type="filesystem", key=key)


class _MockValLabels:
    """Mock val ImageLabels: maps DataLocation -> list of (row, col, label)."""

    def __init__(self, image_specs):
        """image_specs: list of (DataLocation, n_points)."""
        self._data = OrderedDict()
        for loc, n in image_specs:
            self._data[loc] = [(0, 0, "dummy")] * n

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]


class _MockDataset:
    """Mock TrainingDataset exposing labels.val and feature_loc_to_region."""

    def __init__(self, image_specs, region_map):
        val_labels = _MockValLabels(image_specs)
        self.labels = type("obj", (object,), {"val": val_labels})()
        self.feature_loc_to_region = region_map


def _make_ctx(images=FIXTURE_IMAGES, gt=FIXTURE_GT, est=FIXTURE_EST, classes=None):
    classes = CLASSES if classes is None else classes
    image_specs = []
    region_map = {}
    for key, region, n_points in images:
        loc = _make_loc(key)
        image_specs.append((loc, n_points))
        region_map[loc] = region
    return MetricsContext(
        val_results=make_val_results(gt, est, classes),
        ba_library=MockBALibrary(),
        gf_library=MockGFLibrary(),
        format_func=format_metric,
        dataset=_MockDataset(image_specs, region_map),
    )


def _scalars(result: MetricGroupResult) -> dict[str, float]:
    return {scalar.name: scalar.value for scalar in result.scalars}


class ComputeRegionTest(unittest.TestCase):
    """The rates the group reads off a validation split."""

    def setUp(self):
        self.result = compute_region(_make_ctx())
        self.scalars = _scalars(self.result)

    def test_out_of_region_rate_is_incidents_over_evaluable_predictions(self):
        """Two of the four scored predictions land outside their image's
        region, and every predicted attribute has a recorded region."""
        self.assertAlmostEqual(self.scalars["region_val/oor_rate"], 2 / 4)
        self.assertEqual(self.scalars["region_val/oor_rate_k"], 2)
        self.assertEqual(self.scalars["region_val/oor_rate_n"], 4)

    def test_discriminating_denominator_drops_the_globally_permitted_label(self):
        """B1 is permitted in both observed regions, so it can never be out of
        region and dilutes the rate it sits in the denominator of."""
        self.assertAlmostEqual(self.scalars["region_val/oor_rate_disc"], 2 / 3)
        self.assertEqual(self.scalars["region_val/oor_rate_disc_n"], 3)

    def test_ground_truth_floor_travels_beside_the_model_rate(self):
        """Point 3's ground truth is itself out of its image's region: the
        floor no model can go below."""
        self.assertAlmostEqual(self.scalars["region_val/gt_oor_rate"], 1 / 4)
        self.assertEqual(self.scalars["region_val/gt_oor_rate_k"], 1)
        self.assertAlmostEqual(self.scalars["region_val/excess"], 2 / 4 - 1 / 4)
        self.assertAlmostEqual(self.scalars["region_val/ratio_to_gt"], 2.0)

    def test_scored_population_counts_exclude_the_unrecorded_images(self):
        self.assertEqual(self.scalars["region_val/n_points"], 4)
        self.assertEqual(self.scalars["region_val/n_images"], 2)
        self.assertEqual(self.scalars["region_val/n_regions"], 2)

    def test_unrecorded_region_points_are_counted_rather_than_silently_dropped(self):
        """The two CoralNet points leave every denominator, and the exclusion
        is a recorded number rather than a log line nobody reads."""
        self.assertEqual(self.scalars["region_val/n_points_unrecorded_region"], 2)

    def test_every_rate_carries_its_interval_bounds(self):
        """A rate logged without bounds cannot be read or diffed across runs."""
        for name in (
            "oor_rate",
            "oor_rate_disc",
            "oor_rate_disc_gt",
            "gt_oor_rate",
            "image_affected_rate",
            "accuracy",
            "excess",
        ):
            with self.subTest(metric=name):
                self.assertIn(f"region_val/{name}_lo95", self.scalars)
                self.assertIn(f"region_val/{name}_hi95", self.scalars)

    def test_no_scalar_is_non_finite(self):
        """MLflow logging skips NaN, so a NaN cell would vanish without trace."""
        import math

        for name, value in self.scalars.items():
            with self.subTest(metric=name):
                self.assertTrue(math.isfinite(value), msg=f"{name} = {value}")

    def test_every_name_carries_the_validation_prefix(self):
        """The validation split and the frozen probe measure different
        denominators over different populations; neither borrows the other's
        names."""
        for name in self.scalars:
            with self.subTest(metric=name):
                self.assertTrue(name.startswith("region_val/"), msg=name)
        for frame in self.result.dataframes:
            with self.subTest(artifact=frame.artifact_path):
                self.assertTrue(frame.artifact_path.startswith("region_val/"))

    def test_tables_that_explain_the_rate_are_emitted(self):
        paths = {frame.artifact_path for frame in self.result.dataframes}
        self.assertEqual(
            paths,
            {
                "region_val/per_region",
                "region_val/per_label",
                "region_val/per_direction",
                "region_val/confusion",
            },
        )

    def test_per_region_table_splits_the_two_regions(self):
        frame = next(
            f.df for f in self.result.dataframes if f.artifact_path == "region_val/per_region"
        )
        rows = frame.set_index("region_id")
        self.assertEqual(int(rows.loc[TROPICAL_ATLANTIC, "oor_rate_k"]), 1)
        self.assertEqual(int(rows.loc[TROPICAL_ATLANTIC, "oor_rate_n"]), 2)
        self.assertEqual(int(rows.loc[CENTRAL_INDO_PACIFIC, "oor_rate_k"]), 1)

    def test_region_names_reach_the_tables(self):
        frame = next(
            f.df for f in self.result.dataframes if f.artifact_path == "region_val/per_region"
        )
        rows = frame.set_index("region_id")
        self.assertEqual(rows.loc[TROPICAL_ATLANTIC, "region_name"], "Tropical Atlantic")


class GracefulNoOpTest(unittest.TestCase):
    """Configurations where the group has nothing to measure."""

    def test_no_validation_image_has_a_region_returns_empty(self):
        """An `include_mermaid: false` run: every point is CoralNet, so the
        group is a no-op rather than a swallowed exception."""
        images = (("img-a", UNRECORDED, 2), ("img-b", UNRECORDED, 2))
        result = compute_region(_make_ctx(images=images, gt=[0, 0, 1, 1], est=[0, 1, 1, 0]))
        self.assertEqual(result, MetricGroupResult())

    def test_no_dataset_returns_empty(self):
        ctx = MetricsContext(
            val_results=make_val_results([0, 1], [0, 1], CLASSES),
            ba_library=MockBALibrary(),
            gf_library=MockGFLibrary(),
            format_func=format_metric,
            dataset=None,
        )
        self.assertEqual(compute_region(ctx), MetricGroupResult())

    def test_empty_region_map_returns_empty(self):
        """A dataset built by a path that never populated the region map."""
        ctx = _make_ctx()
        ctx.dataset.feature_loc_to_region = {}
        self.assertEqual(compute_region(ctx), MetricGroupResult())


class EmptyDenominatorTest(unittest.TestCase):
    """A rate whose denominator holds no point at all.

    Every prediction here is an attribute with no recorded regions, so it can
    be neither in nor out of one and the out-of-region denominator is empty.
    """

    def setUp(self):
        images = (
            ("img-a", (TROPICAL_ATLANTIC, "Tropical Atlantic"), 2),
            ("img-b", (CENTRAL_INDO_PACIFIC, "Central Indo-Pacific"), 2),
        )
        self.scalars = _scalars(
            compute_region(
                _make_ctx(
                    images=images,
                    gt=[1, 1, 1, 1],
                    est=[0, 0, 0, 0],
                    classes=["B2::", "B1::"],
                )
            )
        )

    def test_empty_cell_reads_as_zero_beside_a_zero_count(self):
        """MLflow logging skips NaN, so the empty cell would vanish from the
        run entirely; a zero beside an n of zero says what happened."""
        self.assertEqual(self.scalars["region_val/oor_rate"], 0.0)
        self.assertEqual(self.scalars["region_val/oor_rate_n"], 0)

    def test_empty_cell_publishes_no_interval_it_cannot_measure(self):
        self.assertNotIn("region_val/oor_rate_lo95", self.scalars)
        self.assertNotIn("region_val/oor_rate_hi95", self.scalars)

    def test_a_floor_of_zero_publishes_no_multiple_of_itself(self):
        """Dividing by an empty floor is unbounded, not large."""
        self.assertNotIn("region_val/ratio_to_gt", self.scalars)

    def test_no_scalar_is_non_finite(self):
        import math

        for name, value in self.scalars.items():
            with self.subTest(metric=name):
                self.assertTrue(math.isfinite(value), msg=f"{name} = {value}")


class IndexAlignmentTest(unittest.TestCase):
    def test_index_count_mismatch_raises(self):
        """The region array is expanded assuming each image's points are
        contiguous in val_results; a divergence must fail loudly rather than
        attribute every point to the wrong image."""
        ctx = _make_ctx(gt=FIXTURE_GT + [0], est=FIXTURE_EST + [0])
        with self.assertRaisesRegex(ValueError, "val_results"):
            compute_region(ctx)


class ResampleSettingTest(unittest.TestCase):
    """The resample count is what a large validation split pays for its
    intervals, so it has to be reachable from configuration."""

    def _bounds(self, n_resamples: int) -> tuple[float, float]:
        with override_settings(region_val_n_resamples=n_resamples):
            scalars = _scalars(
                compute_region(
                    _make_ctx(images=CLUSTERED_IMAGES, gt=CLUSTERED_GT, est=CLUSTERED_EST)
                )
            )
        return scalars["region_val/oor_rate_lo95"], scalars["region_val/oor_rate_hi95"]

    def test_resample_count_moves_the_interval(self):
        self.assertNotEqual(self._bounds(2), self._bounds(200))

    def test_default_is_cheap_enough_for_a_full_validation_split(self):
        """The whole split can run to millions of points, and each rate in
        each table draws this many resamples."""
        self.assertLessEqual(settings.region_val_n_resamples, 500)


if __name__ == "__main__":
    unittest.main()
