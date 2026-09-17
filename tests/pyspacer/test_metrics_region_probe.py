"""Tests for mermaid_classifier.pyspacer.metrics.region_probe.

A four-point probe is written to a temp dir in the layout
`build_region_probe.py` emits, and scored through a predictor whose
probabilities are the feature rows themselves, so every prediction in the
table below is hand-chosen. Nothing here reaches S3 or the network.

    #  image  region  ground truth  prediction  out of region
    0  ta1    TA      ba_atl        ba_atl      no
    1  ta1    TA      ba_atl        ba_pac      yes (ba_pac is Pacific only)
    2  cip1   CIP     ba_pac        ba_atl      yes (ba_atl is Atlantic only)
    3  cip1   CIP     ba_atl        ba_glob     no (ba_glob spans both);
                                                ground truth ba_atl is out of
                                                its image's region

Observed regions are {TA, CIP}, so ba_glob never discriminates: four
evaluable predictions, three region-discriminating ones, two incidents, and a
ground-truth floor of one in four.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import MetricGroupResult
from mermaid_classifier.pyspacer.metrics.region_probe import compute_region_probe
from mermaid_classifier.region_eval.features import FeatureCache, write_feature_cache
from mermaid_classifier.region_eval.probe_set import PROBE_COLUMNS

from .metrics_test_helpers import (
    MockBALibrary,
    MockGFLibrary,
    format_metric,
    make_val_results,
)
from .test_train import override_settings

TROPICAL_ATLANTIC = "region-ta"
CENTRAL_INDO_PACIFIC = "region-cip"

BA_ATLANTIC = "ba_atl"
BA_PACIFIC = "ba_pac"
BA_GLOBAL = "ba_glob"

ATLANTIC_LABEL = f"{BA_ATLANTIC}::"
PACIFIC_LABEL = f"{BA_PACIFIC}::"
GLOBAL_LABEL = f"{BA_GLOBAL}::"

# The predictor's column order; a feature row is the probability vector.
CLASSES = (ATLANTIC_LABEL, PACIFIC_LABEL, GLOBAL_LABEL)

FROZEN_REGIONS = {
    BA_ATLANTIC: [TROPICAL_ATLANTIC],
    BA_PACIFIC: [CENTRAL_INDO_PACIFIC],
    BA_GLOBAL: [TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC],
}
REGION_DISPLAY_NAMES = {
    TROPICAL_ATLANTIC: "Tropical Atlantic",
    CENTRAL_INDO_PACIFIC: "Central Indo-Pacific",
}

# (image_id, region_id, ground-truth attribute, predicted attribute)
PROBE_POINTS = (
    ("ta1", TROPICAL_ATLANTIC, BA_ATLANTIC, BA_ATLANTIC),
    ("ta1", TROPICAL_ATLANTIC, BA_ATLANTIC, BA_PACIFIC),
    ("cip1", CENTRAL_INDO_PACIFIC, BA_PACIFIC, BA_ATLANTIC),
    ("cip1", CENTRAL_INDO_PACIFIC, BA_ATLANTIC, BA_GLOBAL),
)


class _ArgmaxPredictor:
    """The Predictor interface `runner.py` hands the metrics coordinator.

    `classes_` in column order and `predict_proba` returning one row per
    feature vector. The probabilities here are the feature rows themselves,
    which is what lets the fixture table choose every prediction by hand.
    """

    def __init__(self, classes=CLASSES):
        self.classes_ = list(classes)

    def predict_proba(self, features):
        return np.asarray(features, dtype=np.float64)


def _features(points=PROBE_POINTS) -> np.ndarray:
    """One-hot rows selecting each point's intended predicted class."""
    rows = np.zeros((len(points), len(CLASSES)), dtype=np.float32)
    for index, (_image, _region, _truth, prediction) in enumerate(points):
        rows[index, CLASSES.index(f"{prediction}::")] = 1.0
    return rows


def _write_probe(probe_dir: Path, points=PROBE_POINTS) -> None:
    """Write the probe dir in the layout build_region_probe.py emits."""
    probe_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    counters: dict[str, int] = {}
    for image_id, region_id, attribute_id, _prediction in points:
        index = counters.get(image_id, 0)
        counters[image_id] = index + 1
        rows.append(
            {
                "image_id": image_id,
                "point_id": f"{image_id}-p{index}",
                "row": 10 * index,
                "col": 20 * index,
                "benthic_attribute_id": attribute_id,
                "benthic_attribute_name": attribute_id.upper(),
                "growth_form_id": "",
                "growth_form_name": "",
                "gt_label": f"{attribute_id}::",
                "region_id": region_id,
                "region_name": REGION_DISPLAY_NAMES[region_id],
                "site_id": "",
                "held_out": False,
            }
        )
    frame = pd.DataFrame(rows, columns=pd.Index(PROBE_COLUMNS))
    frame.to_parquet(probe_dir / "probe_points.parquet", index=False)
    (probe_dir / "ba_regions.json").write_text(json.dumps(FROZEN_REGIONS, sort_keys=True))
    (probe_dir / "names.json").write_text(
        json.dumps(
            {
                "benthic_attributes": {BA_ATLANTIC: "Atlantic coral"},
                "growth_forms": {},
                "regions": REGION_DISPLAY_NAMES,
            }
        )
    )
    write_feature_cache(
        FeatureCache(
            features=_features(points),
            image_ids=tuple(frame["image_id"]),
            point_ids=tuple(frame["point_id"]),
            rows=np.asarray(frame["row"], dtype=np.int64),
            cols=np.asarray(frame["col"], dtype=np.int64),
            gt_labels=tuple(frame["gt_label"]),
            region_ids=tuple(frame["region_id"]),
            held_out=np.asarray(frame["held_out"], dtype=bool),
            n_points_requested=len(frame),
            n_points_missing_row_col=0,
            n_points_missing_image=0,
            missing_image_ids=(),
        ),
        probe_dir / "probe_features.npz",
    )


def _make_ctx(clf=None) -> MetricsContext:
    """A context carrying only what the probe group reads: the predictor."""
    return MetricsContext(
        val_results=make_val_results([0, 1], [0, 1], ["A1::", "A2::"]),
        ba_library=MockBALibrary(),
        gf_library=MockGFLibrary(),
        format_func=format_metric,
        clf=_ArgmaxPredictor() if clf is None else clf,
    )


def _scalars(result: MetricGroupResult) -> dict[str, float]:
    return {scalar.name: scalar.value for scalar in result.scalars}


class ComputeRegionProbeTest(unittest.TestCase):
    """The rates the group reads off the frozen probe."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        probe_dir = Path(self._tmpdir.name) / "probe"
        _write_probe(probe_dir)
        with override_settings(region_probe_dir=str(probe_dir)):
            self.result = compute_region_probe(_make_ctx())
        self.scalars = _scalars(self.result)

    def test_out_of_region_rate_is_incidents_over_evaluable_predictions(self):
        self.assertAlmostEqual(self.scalars["region_probe/oor_rate"], 2 / 4)
        self.assertEqual(self.scalars["region_probe/oor_rate_k"], 2)
        self.assertEqual(self.scalars["region_probe/oor_rate_n"], 4)

    def test_discriminating_denominator_drops_the_globally_permitted_label(self):
        self.assertAlmostEqual(self.scalars["region_probe/oor_rate_disc"], 2 / 3)
        self.assertEqual(self.scalars["region_probe/oor_rate_disc_n"], 3)

    def test_ground_truth_floor_travels_beside_the_model_rate(self):
        self.assertAlmostEqual(self.scalars["region_probe/gt_oor_rate"], 1 / 4)
        self.assertAlmostEqual(self.scalars["region_probe/ratio_to_gt"], 2.0)

    def test_a_probe_that_was_scored_says_so_in_a_metric(self):
        """The coordinator logs a 0 before the group runs, so a probe dir it
        could not read leaves a 0 behind rather than no metric at all."""
        self.assertEqual(self.scalars["region_probe/scored"], 1.0)

    def test_population_counts_cover_the_whole_probe(self):
        self.assertEqual(self.scalars["region_probe/n_points"], 4)
        self.assertEqual(self.scalars["region_probe/n_images"], 2)
        self.assertEqual(self.scalars["region_probe/n_points_unrecorded_region"], 0)

    def test_every_rate_carries_its_interval_bounds(self):
        for name in ("oor_rate", "oor_rate_disc", "gt_oor_rate", "accuracy"):
            with self.subTest(metric=name):
                self.assertIn(f"region_probe/{name}_lo95", self.scalars)
                self.assertIn(f"region_probe/{name}_hi95", self.scalars)

    def test_no_scalar_is_non_finite(self):
        for name, value in self.scalars.items():
            with self.subTest(metric=name):
                self.assertTrue(math.isfinite(value), msg=f"{name} = {value}")

    def test_every_name_carries_the_probe_prefix(self):
        """The probe is a fixed image set and the validation split is composed
        by the training config; neither borrows the other's names."""
        for name in self.scalars:
            with self.subTest(metric=name):
                self.assertTrue(name.startswith("region_probe/"), msg=name)
        for frame in self.result.dataframes:
            with self.subTest(artifact=frame.artifact_path):
                self.assertTrue(frame.artifact_path.startswith("region_probe/"))

    def test_tables_that_explain_the_rate_are_emitted(self):
        paths = {frame.artifact_path for frame in self.result.dataframes}
        self.assertEqual(
            paths,
            {
                "region_probe/per_region",
                "region_probe/per_label",
                "region_probe/per_direction",
                "region_probe/confusion",
            },
        )

    def test_frozen_names_reach_the_tables(self):
        frame = next(
            f.df for f in self.result.dataframes if f.artifact_path == "region_probe/per_region"
        )
        rows = frame.set_index("region_id")
        self.assertEqual(rows.loc[TROPICAL_ATLANTIC, "region_name"], "Tropical Atlantic")


class ScoresTheContextModelTest(unittest.TestCase):
    """The probe is scored through the predictor the run exported, not
    through a model the group loads for itself."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.probe_dir = Path(self._tmpdir.name) / "probe"
        _write_probe(self.probe_dir)

    def _rate(self, clf) -> float:
        with override_settings(region_probe_dir=str(self.probe_dir)):
            return _scalars(compute_region_probe(_make_ctx(clf=clf)))["region_probe/oor_rate"]

    def test_a_model_that_never_leaves_the_region_reads_zero(self):
        """Every point predicted as the globally-permitted class, which can
        never be out of region."""

        class _AlwaysGlobal(_ArgmaxPredictor):
            def predict_proba(self, features):
                probabilities = np.zeros((len(features), len(CLASSES)))
                probabilities[:, CLASSES.index(GLOBAL_LABEL)] = 1.0
                return probabilities

        self.assertEqual(self._rate(_AlwaysGlobal()), 0.0)

    def test_a_model_that_always_leaves_the_region_reads_one(self):
        """Every Atlantic image predicted Pacific and every Pacific image
        Atlantic."""

        class _AlwaysWrongOcean(_ArgmaxPredictor):
            def predict_proba(self, features):
                probabilities = np.zeros((len(features), len(CLASSES)))
                for index, (_image, region, _truth, _pred) in enumerate(PROBE_POINTS):
                    wrong = PACIFIC_LABEL if region == TROPICAL_ATLANTIC else ATLANTIC_LABEL
                    probabilities[index, CLASSES.index(wrong)] = 1.0
                return probabilities

        self.assertEqual(self._rate(_AlwaysWrongOcean()), 1.0)


class FrozenRegionMapTest(unittest.TestCase):
    def test_the_probe_map_decides_rather_than_the_live_library(self):
        """A region added to a coral upstream must not move a probe score;
        the map frozen beside the points is what the predicates read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            probe_dir = Path(tmpdir) / "probe"
            _write_probe(probe_dir)
            ctx = _make_ctx()
            # The live library would permit every attribute everywhere, which
            # would read as no incidents at all.
            ctx.ba_library.region_ids_by_id = {
                attribute: frozenset({TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC})
                for attribute in FROZEN_REGIONS
            }
            with override_settings(region_probe_dir=str(probe_dir)):
                scalars = _scalars(compute_region_probe(ctx))
        self.assertAlmostEqual(scalars["region_probe/oor_rate"], 2 / 4)


class GracefulNoOpTest(unittest.TestCase):
    def test_no_configured_probe_is_a_no_op(self):
        """Most runs have no probe to score against, and that is not a
        failure."""
        with override_settings(region_probe_dir=None):
            self.assertEqual(compute_region_probe(_make_ctx()), MetricGroupResult())

    def test_no_clf_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            probe_dir = Path(tmpdir) / "probe"
            _write_probe(probe_dir)
            with override_settings(region_probe_dir=str(probe_dir)):
                self.assertEqual(compute_region_probe(_no_clf_ctx()), MetricGroupResult())

    def test_a_configured_probe_that_is_not_there_fails_loudly(self):
        """Silently scoring nothing would read as a model with no incidents."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            override_settings(region_probe_dir=str(Path(tmpdir) / "absent")),
            self.assertRaises(FileNotFoundError),
        ):
            compute_region_probe(_make_ctx())


def _no_clf_ctx() -> MetricsContext:
    return MetricsContext(
        val_results=make_val_results([0, 1], [0, 1], ["A1::", "A2::"]),
        ba_library=MockBALibrary(),
        gf_library=MockGFLibrary(),
        format_func=format_metric,
        clf=None,
    )


if __name__ == "__main__":
    unittest.main()
