"""Unit tests for region_eval/score.py.

Every test scores a real exported artifact -- a TorchScript head built from
the shared calibrated-model fixture and loaded through the production
`load_predictor` -- against a 25-point probe written to a temp dir in the
layout `build_region_probe.py` emits. Nothing here reaches S3, the MERMAID
API or the network: the feature cache is a local npz and the live region map
arrives as an injected callable.

The probe carries the fixture model's own labels, `ba0::gf0`..`ba4::gf4`,
plus `ba9::gf9`, whose attribute and growth form the model has never seen:

    image  region  points  ground-truth attributes
    ta1    TA      4       ba1, ba0, ba1, ba9
    ta2    TA      4       ba0, ba1, ba3, ba1
    cip1   CIP     5       ba2, ba2, ba4, ba0, ba2
    cip2   CIP     4       ba4, ba2, ba3, ba0
    cip3   CIP     4       ba2, ba4, ba2, ba0
    cip4   CIP     4       ba4, ba2, ba1, ba0

Frozen region map: ba0 -> {TA, CIP}, ba1 -> {TA}, ba2 -> {CIP},
ba3 -> {EP}, ba4 -> {CIP}, ba9 -> {TA}. Observed regions are {TA, CIP}, so
ba0 alone never discriminates and ba3 is untestable -- no probe image comes
from the one region it is recorded in.

Counts derived by hand off that table, and asserted as literals:

    25 points over 6 images; 17 held out (the four CIP images)
    24 points whose ground truth is inside the model's label space
    19 points whose ground-truth attribute region-discriminates
     3 points whose ground truth is out of its image's region
       (ba3 on ta2, ba3 on cip2, ba1 on cip4)
    25 points whose predicted attribute has recorded regions -- every model
       class does, so the out-of-region denominator cannot depend on which
       class the model happens to pick
"""

import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from pyspacer._calibrated_model_fixture import make_calibrated_model

from mermaid_classifier.pyspacer.inference import export_artifact
from mermaid_classifier.region_eval.decisions import RegionBlindBaseline
from mermaid_classifier.region_eval.features import (
    DEFAULT_FEATURE_SUFFIX,
    FEATURE_DIM,
    FeatureCache,
    write_feature_cache,
)
from mermaid_classifier.region_eval.metrics import RateEstimate, RegionMetricsOptions
from mermaid_classifier.region_eval.probe_set import (
    PROBE_COLUMNS,
    NameSnapshot,
    ancestry_snapshot_hash,
    ground_truth_counts_hash,
    name_snapshot_hash,
    probe_content_hash,
)
from mermaid_classifier.region_eval.report import (
    _ratio_to_baseline_interval,
    build_manifest,
    decisions_table,
    summary_table,
    write_report,
)
from mermaid_classifier.region_eval.score import (
    ModelScore,
    check_feature_coverage,
    load_probe,
    score_model,
)

# Allow importing scripts/evaluate_region_probe.py (mirrors test_release_artifact).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import evaluate_region_probe  # noqa: E402

TROPICAL_ATLANTIC = "1a1a1a1a-0000-4000-8000-000000000001"
CENTRAL_INDO_PACIFIC = "1a1a1a1a-0000-4000-8000-000000000002"
EASTERN_PACIFIC = "1a1a1a1a-0000-4000-8000-000000000003"

FROZEN_REGIONS = {
    "ba0": [TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC],
    "ba1": [TROPICAL_ATLANTIC],
    "ba2": [CENTRAL_INDO_PACIFIC],
    "ba3": [EASTERN_PACIFIC],
    "ba4": [CENTRAL_INDO_PACIFIC],
    "ba9": [TROPICAL_ATLANTIC],
}

# (image_id, region_id, ground-truth attribute id) in probe order.
PROBE_POINTS = [
    ("ta1", TROPICAL_ATLANTIC, "ba1"),
    ("ta1", TROPICAL_ATLANTIC, "ba0"),
    ("ta1", TROPICAL_ATLANTIC, "ba1"),
    ("ta1", TROPICAL_ATLANTIC, "ba9"),
    ("ta2", TROPICAL_ATLANTIC, "ba0"),
    ("ta2", TROPICAL_ATLANTIC, "ba1"),
    ("ta2", TROPICAL_ATLANTIC, "ba3"),
    ("ta2", TROPICAL_ATLANTIC, "ba1"),
    ("cip1", CENTRAL_INDO_PACIFIC, "ba2"),
    ("cip1", CENTRAL_INDO_PACIFIC, "ba2"),
    ("cip1", CENTRAL_INDO_PACIFIC, "ba4"),
    ("cip1", CENTRAL_INDO_PACIFIC, "ba0"),
    ("cip1", CENTRAL_INDO_PACIFIC, "ba2"),
    ("cip2", CENTRAL_INDO_PACIFIC, "ba4"),
    ("cip2", CENTRAL_INDO_PACIFIC, "ba2"),
    ("cip2", CENTRAL_INDO_PACIFIC, "ba3"),
    ("cip2", CENTRAL_INDO_PACIFIC, "ba0"),
    ("cip3", CENTRAL_INDO_PACIFIC, "ba2"),
    ("cip3", CENTRAL_INDO_PACIFIC, "ba4"),
    ("cip3", CENTRAL_INDO_PACIFIC, "ba2"),
    ("cip3", CENTRAL_INDO_PACIFIC, "ba0"),
    ("cip4", CENTRAL_INDO_PACIFIC, "ba4"),
    ("cip4", CENTRAL_INDO_PACIFIC, "ba2"),
    ("cip4", CENTRAL_INDO_PACIFIC, "ba1"),
    ("cip4", CENTRAL_INDO_PACIFIC, "ba0"),
]

# The frozen display names. ba4 and gf4 are deliberately absent, as is the
# Eastern Pacific, so every table has one id that no name resolves.
BA_NAMES = {
    "ba0": "Porites astreoides",
    "ba1": "Acropora cervicornis",
    "ba2": "Goniopora",
    "ba3": "Agaricia tenuifolia",
    "ba9": "Orbicella faveolata",
}
GF_NAMES = {
    "gf0": "Encrusting",
    "gf1": "Branching",
    "gf2": "Massive",
    "gf3": "Foliose",
    "gf9": "Columnar",
}
REGION_DISPLAY_NAMES = {
    TROPICAL_ATLANTIC: "Tropical Atlantic",
    CENTRAL_INDO_PACIFIC: "Central Indo-Pacific",
}
NAMES = NameSnapshot(
    benthic_attributes=BA_NAMES, growth_forms=GF_NAMES, regions=REGION_DISPLAY_NAMES
)

CORAL_ROOT = "root-coral"
# Every attribute under one root, so every out-of-region prediction shares a
# branch with its ground truth and the share is exactly 1.
ONE_BRANCH_ANCESTRY = {
    attribute: [CORAL_ROOT, attribute] for attribute in ("ba0", "ba1", "ba2", "ba3", "ba4", "ba9")
}
# Each attribute its own root, so none shares a branch and the share is 0.
SEPARATE_BRANCH_ANCESTRY = {
    attribute: [f"root-{attribute}", attribute]
    for attribute in ("ba0", "ba1", "ba2", "ba3", "ba4", "ba9")
}

# Permutations enough for a stable baseline on six images, few enough to keep
# the suite quick.
N_PERMUTATIONS = 200

# Corpus-wide confirmed annotations for every (attribute, region) pair a
# prediction on this probe can be flagged on: an out-of-region prediction of
# ba1 lands on a Central Indo-Pacific image, of ba2/ba4 on an Atlantic one,
# and ba3 is recorded only in the Eastern Pacific. The probe's own ground
# truth carries at most four of any of them, short of a threshold of 5.
CORPUS_COUNTS = {
    "ba1": {CENTRAL_INDO_PACIFIC: 605},
    "ba2": {TROPICAL_ATLANTIC: 605},
    "ba3": {TROPICAL_ATLANTIC: 605, CENTRAL_INDO_PACIFIC: 605},
    "ba4": {TROPICAL_ATLANTIC: 605},
}

N_POINTS = 25
N_IMAGES = 6
N_HELD_OUT = 17
N_IN_MODEL_CLASSES = 24
N_GT_OUT_OF_REGION = 3
N_GT_DISCRIMINATING = 19

# Small enough to keep six-image resampling quick, large enough that the
# percentile interval is not degenerate.
N_RESAMPLES = 60


def _probe_rows(points=PROBE_POINTS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    counters: dict[str, int] = {}
    for image_id, region_id, attribute_id in points:
        index = counters.get(image_id, 0)
        counters[image_id] = index + 1
        growth_form_id = attribute_id.replace("ba", "gf")
        rows.append(
            {
                "image_id": image_id,
                "point_id": f"{image_id}-p{index}",
                "row": 10 * index,
                "col": 20 * index,
                "benthic_attribute_id": attribute_id,
                "benthic_attribute_name": attribute_id.upper(),
                "growth_form_id": growth_form_id,
                "growth_form_name": growth_form_id.upper(),
                "gt_label": f"{attribute_id}::{growth_form_id}",
                "region_id": region_id,
                "region_name": region_id[:8],
                "site_id": "",
                "held_out": image_id.startswith("cip"),
            }
        )
    return pd.DataFrame(rows, columns=pd.Index(PROBE_COLUMNS))


def _feature_file_bytes(points: list[tuple[int, int]], *, dim: int = FEATURE_DIM) -> bytes:
    """A minimal real `.featurevector` npz for the given (row, col) points."""
    buffer = io.BytesIO()
    np.savez(
        buffer,
        meta=np.array([1, len(points), dim], dtype=np.int64),
        rows=np.array([row for row, _ in points], dtype=np.uint16),
        cols=np.array([col for _, col in points], dtype=np.uint16),
        feat=np.zeros((len(points), dim), dtype=np.float64),
    )
    return buffer.getvalue()


def _flat_ground_truth_counts(counts: dict[str, dict[str, int]]) -> dict[tuple[str, str], int]:
    """Nested (attribute -> region -> count) as the flat mapping the hash helper takes.

    Mirrors `score.read_ground_truth_counts`, so a hash computed here over the
    same `counts` a probe writes to `ba_region_counts.json` matches the hash
    `load_probe` recomputes after reading that file back.
    """
    return {
        (attribute_id, region_id): count
        for attribute_id, by_region in counts.items()
        for region_id, count in by_region.items()
    }


def _write_probe(
    probe_dir: Path,
    features: np.ndarray,
    points=PROBE_POINTS,
    counts=None,
    names: NameSnapshot | None = NAMES,
    ancestry=ONE_BRANCH_ANCESTRY,
) -> pd.DataFrame:
    """Write the probe dir in the layout build_region_probe.py emits.

    `counts`, `names` and `ancestry` write the frozen snapshots; passing None
    for one is a probe built before that snapshot was frozen beside the points.
    """
    probe_dir.mkdir(parents=True, exist_ok=True)
    rows = _probe_rows(points)
    rows.to_parquet(probe_dir / "probe_points.parquet", index=False)
    (probe_dir / "ba_regions.json").write_text(json.dumps(FROZEN_REGIONS, sort_keys=True))
    if counts is not None:
        (probe_dir / "ba_region_counts.json").write_text(json.dumps(counts, sort_keys=True))
    if names is not None:
        (probe_dir / "names.json").write_text(
            json.dumps(
                {
                    "benthic_attributes": dict(names.benthic_attributes),
                    "growth_forms": dict(names.growth_forms),
                    "regions": dict(names.regions),
                },
                sort_keys=True,
            )
        )
    if ancestry is not None:
        (probe_dir / "ba_ancestry.json").write_text(json.dumps(ancestry, sort_keys=True))
    (probe_dir / "manifest.json").write_text(
        json.dumps(
            {
                "probe_version": "1",
                "content_hash": probe_content_hash(rows),
                "n_points": len(rows),
                "n_images": rows["image_id"].nunique(),
                "names_hash": None if names is None else name_snapshot_hash(names),
                "ancestry_hash": (None if ancestry is None else ancestry_snapshot_hash(ancestry)),
                "ground_truth_counts_hash": (
                    None
                    if counts is None
                    else ground_truth_counts_hash(_flat_ground_truth_counts(counts))
                ),
            }
        )
    )
    write_feature_cache(
        FeatureCache(
            features=features.astype(np.float32),
            image_ids=tuple(rows["image_id"]),
            point_ids=tuple(rows["point_id"]),
            rows=np.asarray(rows["row"], dtype=np.int64),
            cols=np.asarray(rows["col"], dtype=np.int64),
            gt_labels=tuple(rows["gt_label"]),
            region_ids=tuple(rows["region_id"]),
            held_out=np.asarray(rows["held_out"], dtype=bool),
            n_points_requested=len(rows),
            n_points_missing_row_col=0,
            n_points_missing_image=0,
            missing_image_ids=(),
            n_points_missing_download_failed=0,
            download_failed_image_ids=(),
        ),
        rows,
        probe_dir / "probe_features.npz",
    )
    return rows


def _export_model(model_dir: Path, seed: int = 0) -> tuple[Path, Path]:
    """A real TorchScript artifact, exported the way a release is."""
    model, batch = make_calibrated_model(seed=seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_pt, _manifest, _diff = export_artifact(model, model_dir, batch)
    return Path(model_pt), model_dir / "model.json"


def _live_map(**overrides: list[str]):
    """A live-map loader returning the frozen map with the given edits."""
    live = {key: frozenset(value) for key, value in FROZEN_REGIONS.items()}
    live.update({key: frozenset(value) for key, value in overrides.items()})

    def load() -> dict[str, frozenset[str]]:
        return live

    return load


def _unreachable_live_map():
    def load() -> dict[str, frozenset[str]]:
        raise ConnectionError("api.datamermaid.org is unreachable")

    return load


def _read_csv(path: Path) -> pd.DataFrame:
    """The written CSV as text, so an empty cell stays distinguishable."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


class ScoreReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _model, batch = make_calibrated_model()
        self.features = np.asarray(batch)[:N_POINTS]
        self.rows = _write_probe(self.root / "probe", self.features)
        self.probe = load_probe(self.root / "probe")

    def _counted_probe(self) -> Path:
        """A probe dir carrying the frozen corpus-wide annotation counts."""
        probe_dir = self.root / "probe_counted"
        _write_probe(probe_dir, self.features, counts=CORPUS_COUNTS)
        return probe_dir

    def _probe(self, suffix: str, **overrides):
        """A loaded probe dir written with the given snapshots."""
        probe_dir = self.root / f"probe_{suffix}"
        _write_probe(probe_dir, self.features, **overrides)
        return load_probe(probe_dir)

    def _score(self, name: str = "v1", *, seed: int = 0, live_map=None, probe=None):
        model_pt, model_json = _export_model(self.root / f"model_{name}", seed=seed)
        return score_model(
            name,
            model_pt_path=model_pt,
            model_json_path=model_json,
            probe=self.probe if probe is None else probe,
            options=RegionMetricsOptions(n_resamples=N_RESAMPLES),
            live_region_map_loader=_live_map() if live_map is None else live_map,
            n_permutations=N_PERMUTATIONS,
        )

    def test_scores_a_real_artifact_and_writes_the_report_files(self):
        """A missing output file, or a summary row without its denominator
        spelled out, is what this catches: the denominator column is the only
        thing stopping a rate being quoted without what it was computed over.
        """
        score = self._score()
        out_dir = self.root / "out"
        write_report(score, out_dir)

        for name in (
            "summary.csv",
            "per_region.csv",
            "per_label.csv",
            "per_direction.csv",
            "direction_matrix.csv",
            "confusion.csv",
            "region_list_suspects.csv",
            "decisions.csv",
            "manifest.json",
        ):
            self.assertTrue((out_dir / name).exists(), f"{name} was not written")

        summary = _read_csv(out_dir / "summary.csv")
        self.assertEqual(sorted(set(summary["population"])), ["all", "held_out"])
        for metric in (
            "oor_rate",
            "oor_rate_disc",
            "oor_rate_disc_gt",
            "gt_oor_rate",
            "image_affected_rate",
            "accuracy",
            "excess",
            "ratio_to_gt",
        ):
            self.assertIn(metric, set(summary["metric"]), f"{metric} is absent from summary.csv")

        self.assertEqual(
            [],
            summary.loc[summary["denominator"].str.strip() == "", "metric"].tolist(),
            "every summary row must carry the denominator it was read from",
        )

        overall = summary[summary["population"] == "all"].set_index("metric")
        self.assertEqual(overall.loc["oor_rate", "n"], str(N_POINTS))
        self.assertEqual(overall.loc["gt_oor_rate", "k"], str(N_GT_OUT_OF_REGION))
        self.assertEqual(overall.loc["gt_oor_rate", "n"], str(N_POINTS))

        held = summary[summary["population"] == "held_out"].set_index("metric")
        self.assertEqual(held.loc["gt_oor_rate", "n"], str(N_HELD_OUT))

    def test_every_summary_row_carries_interval_bounds(self):
        """A rate quoted without its width is the failure this file exists to
        prevent, so no summary row may leave ci_low/ci_high blank or NaN --
        including accuracy, which the metrics layer reports bare.
        """
        score = self._score()
        out_dir = self.root / "out"
        write_report(score, out_dir)

        summary = _read_csv(out_dir / "summary.csv")
        self.assertGreater(len(summary), 0)
        for _, row in summary.iterrows():
            for column in ("ci_low", "ci_high"):
                cell = row[column].strip()
                label = f"{row['population']}/{row['metric']}.{column}"
                self.assertNotEqual(cell, "", f"{label} is blank")
                self.assertFalse(math.isnan(float(cell)), f"{label} is nan")

    def test_accuracy_reaches_the_summary_from_the_metrics_layer(self):
        """Two implementations of one cluster-bootstrap estimate drift apart.
        The row in summary.csv is asserted to be the metrics layer's own k, n
        and bounds, so a second copy here cannot quietly diverge from them.
        """
        score = self._score()
        out_dir = self.root / "out"
        write_report(score, out_dir)

        summary = _read_csv(out_dir / "summary.csv").set_index(["population", "metric"])
        self.assertIsNotNone(score.metrics.held_out)
        for population, rates in (
            ("all", score.metrics.overall),
            ("held_out", score.metrics.held_out),
        ):
            estimate = rates.accuracy
            row = summary.loc[(population, "accuracy")]
            self.assertEqual(int(row["k"]), estimate.k, population)
            self.assertEqual(int(row["n"]), estimate.n, population)
            self.assertAlmostEqual(float(row["ci_low"]), estimate.ci_low, msg=population)
            self.assertAlmostEqual(float(row["ci_high"]), estimate.ci_high, msg=population)
        self.assertEqual(int(summary.loc[("all", "accuracy"), "n"]), N_IN_MODEL_CLASSES)

    def test_frozen_corpus_counts_bucket_events_the_probes_own_counts_call_mistakes(self):
        """Stypopodium is annotated 605 times in the Western Indo-Pacific, which
        marks a broadly distributed genus rather than 605 mistakes. A probe
        carrying a handful of those does not clear a threshold of 5, so counts
        derived from it push the same events into model_error and read the
        model-error headline high.
        """
        counted = load_probe(self._counted_probe())
        corpus = self._score(probe=counted)
        probe_only = self._score()

        events = len(probe_only.triage.events)
        self.assertGreater(events, 0, "the fixture model must produce out-of-region events")
        self.assertEqual(len(corpus.triage.events), events)

        before = probe_only.triage.bucket_counts
        after = corpus.triage.bucket_counts
        self.assertEqual(before["unknown_list"], 0)
        self.assertEqual(before["list_suspect"], 0)
        self.assertEqual(before["model_error"], events)
        self.assertEqual(after["list_suspect"], events)
        self.assertEqual(after["model_error"], 0)

        suspects = corpus.triage.region_list_suspects
        self.assertGreater(len(suspects), 0)
        self.assertEqual(set(suspects["n_ground_truth"]), {605})

    def test_region_list_drift_reports_a_hash_comparison(self):
        """A hash comparison, not a rescore: same map unmoved, changed map
        moved, and an unreachable loader degrades with its reason rather than
        raising or leaving `moved` looking like a measurement.
        """
        unmoved = self._score(live_map=_live_map())
        self.assertEqual(unmoved.drift["status"], "computed")
        self.assertIsNone(unmoved.drift["reason"])
        self.assertFalse(unmoved.drift["moved"])
        self.assertEqual(unmoved.drift["live_snapshot_hash"], unmoved.drift["frozen_snapshot_hash"])

        moved = self._score(
            name="v2", live_map=_live_map(ba1=[TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC])
        )
        self.assertEqual(moved.drift["status"], "computed")
        self.assertTrue(moved.drift["moved"])
        self.assertNotEqual(moved.drift["live_snapshot_hash"], moved.drift["frozen_snapshot_hash"])

        unreachable = self._score(name="v3", live_map=_unreachable_live_map())
        self.assertEqual(unreachable.drift["status"], "not_computed")
        self.assertIn("ConnectionError", unreachable.drift["reason"])
        self.assertIsNone(unreachable.drift["moved"])
        self.assertIsNone(unreachable.drift["live_snapshot_hash"])
        self.assertEqual(
            unreachable.drift["frozen_snapshot_hash"], unmoved.drift["frozen_snapshot_hash"]
        )


class ProbeIntegrityTest(unittest.TestCase):
    """What binds the parquet and the feature cache together.

    A probe dir rebuilt in place after the selection moved holds two
    selections at once: a fresh parquet beside a cache from the previous one.
    Nothing about the cache's shape says it is stale -- only the hash it
    carries of the rows it was built from does.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _model, batch = make_calibrated_model()
        self.features = np.asarray(batch)[:N_POINTS]
        self.probe_dir = self.root / "probe"
        self.rows = _write_probe(self.probe_dir, self.features)

    def _overwrite_cache(
        self, rows: pd.DataFrame, features: np.ndarray, *, hash_rows: pd.DataFrame | None = None
    ) -> None:
        """Replace the probe's cache with one built for `rows`.

        `hash_rows` is the selection the hash is computed over, and defaults
        to `rows`; a cache legitimately short by a missing image's points is
        still hashed against the full selection it was asked to fetch.
        """
        write_feature_cache(
            FeatureCache(
                features=features.astype(np.float32),
                image_ids=tuple(rows["image_id"]),
                point_ids=tuple(rows["point_id"]),
                rows=np.asarray(rows["row"], dtype=np.int64),
                cols=np.asarray(rows["col"], dtype=np.int64),
                gt_labels=tuple(rows["gt_label"]),
                region_ids=tuple(rows["region_id"]),
                held_out=np.asarray(rows["held_out"], dtype=bool),
                n_points_requested=len(rows),
                n_points_missing_row_col=0,
                n_points_missing_image=0,
                missing_image_ids=(),
                n_points_missing_download_failed=0,
                download_failed_image_ids=(),
            ),
            rows if hash_rows is None else hash_rows,
            self.probe_dir / "probe_features.npz",
        )

    def test_a_cache_frozen_against_other_values_of_these_points_is_refused(self):
        """`--skip-features` rewrites the parquet and leaves the npz where it
        is, so a corrected label, a redrawn region map or a different
        held-out partition reaches the score at its previous value. The
        selection never moved, so only the hash of the changed column gives
        it away.
        """
        for column, stale_value in (
            ("gt_label", "ba9::gf9"),
            ("region_id", CENTRAL_INDO_PACIFIC),
            ("held_out", True),
            ("row", 99),
            ("col", 99),
        ):
            with self.subTest(column=column):
                stale = self.rows.copy()
                stale.loc[0, column] = stale_value
                stale.to_parquet(self.probe_dir / "probe_points.parquet", index=False)

                with self.assertRaisesRegex(ValueError, "content_hash"):
                    load_probe(self.probe_dir)

    def test_a_cache_short_by_one_images_points_still_loads(self):
        """An image whose feature file never existed legitimately shrinks the
        cache below the row count its hash was computed over. A check that
        refused that would refuse every real probe.
        """
        kept = self.rows[self.rows["image_id"] != "ta1"].reset_index(drop=True)
        self._overwrite_cache(
            kept, self.features[len(self.rows) - len(kept) :], hash_rows=self.rows
        )

        probe = load_probe(self.probe_dir)

        self.assertEqual(probe.features.n_points, len(kept))
        self.assertNotIn("ta1", probe.features.image_ids)

    def test_a_permuted_cache_scores_identically_to_the_canonical_one(self):
        """The cluster bootstrap draws resample indices against the order
        clusters first appear in, so a cache that shuffles which image comes
        first draws a different resample from the same seed even though the
        row set and content_hash are unchanged. Sorting the cache to
        (image_id, point_id) order on load is what makes the two the same
        probe.
        """
        canonical_probe = load_probe(self.probe_dir)

        reversed_positions = list(range(len(self.rows)))[::-1]
        permuted_rows = self.rows.iloc[reversed_positions].reset_index(drop=True)
        permuted_features = self.features[reversed_positions]
        self._overwrite_cache(permuted_rows, permuted_features, hash_rows=self.rows)
        permuted_probe = load_probe(self.probe_dir)

        self.assertEqual(permuted_probe.content_hash, canonical_probe.content_hash)
        self.assertEqual(
            sorted(
                zip(
                    permuted_probe.features.image_ids,
                    permuted_probe.features.point_ids,
                    strict=True,
                )
            ),
            sorted(
                zip(
                    canonical_probe.features.image_ids,
                    canonical_probe.features.point_ids,
                    strict=True,
                )
            ),
        )

        model_pt, model_json = _export_model(self.root / "model_permutation")

        def _score(probe, name):
            return score_model(
                name,
                model_pt_path=model_pt,
                model_json_path=model_json,
                probe=probe,
                options=RegionMetricsOptions(n_resamples=N_RESAMPLES),
                live_region_map_loader=_live_map(),
                n_permutations=N_PERMUTATIONS,
            )

        canonical_score = _score(canonical_probe, "canonical")
        permuted_score = _score(permuted_probe, "permuted")

        pd.testing.assert_frame_equal(
            summary_table(canonical_score).drop(columns=["model"]),
            summary_table(permuted_score).drop(columns=["model"]),
        )
        pd.testing.assert_frame_equal(
            decisions_table(canonical_score).drop(columns=["model"]),
            decisions_table(permuted_score).drop(columns=["model"]),
        )

    def test_load_probe_build_branch_hashes_the_full_row_set_not_the_survivors(self):
        """`load_probe`'s build-from-scratch branch is the one production
        call site of `write_feature_cache`; a test that never passes
        `download_dir` never reaches it. A cache legitimately short by a
        missing image's points must still be written against the parquet's
        full row set -- hashing only the surviving rows would carry a
        different hash than the one this same call re-reads the parquet at,
        so an incomplete-but-legitimate cache would refuse to load.
        """
        points = [
            ("kept1", TROPICAL_ATLANTIC, "ba1"),
            ("kept1", TROPICAL_ATLANTIC, "ba0"),
            ("absent", TROPICAL_ATLANTIC, "ba0"),
        ]
        rows = _probe_rows(points)
        probe_dir = self.root / "probe_from_scratch"
        probe_dir.mkdir()
        rows.to_parquet(probe_dir / "probe_points.parquet", index=False)
        (probe_dir / "ba_regions.json").write_text(json.dumps(FROZEN_REGIONS, sort_keys=True))

        download_dir = self.root / "downloads"
        download_dir.mkdir()
        kept_rows = rows[rows["image_id"] == "kept1"]
        (download_dir / f"kept1{DEFAULT_FEATURE_SUFFIX}").write_bytes(
            _feature_file_bytes(list(zip(kept_rows["row"], kept_rows["col"], strict=True)))
        )

        with mock.patch(
            "mermaid_classifier.region_eval.features.download_features_parallel",
            return_value=set(),
        ):
            probe = load_probe(probe_dir, download_dir=download_dir)

        self.assertEqual(probe.features.n_points, 2)
        self.assertNotIn("absent", probe.features.image_ids)

        archive = np.load(probe_dir / "probe_features.npz", allow_pickle=False)
        self.assertEqual(str(archive["content_hash"]), probe_content_hash(rows))

    def test_load_probe_build_branch_reports_a_failed_download_apart_from_a_missing_file(self):
        """A throttled or credential-expired transfer must be visible on the
        probe this build produces, not folded into the same silence a
        genuinely absent feature file gets. `throttled` carries two points on
        one image, so a count that reports images rather than points would
        under-report this by one.
        """
        points = [
            ("kept1", TROPICAL_ATLANTIC, "ba1"),
            ("absent", TROPICAL_ATLANTIC, "ba0"),
            ("throttled", TROPICAL_ATLANTIC, "ba0"),
            ("throttled", TROPICAL_ATLANTIC, "ba1"),
        ]
        rows = _probe_rows(points)
        probe_dir = self.root / "probe_download_failure"
        probe_dir.mkdir()
        rows.to_parquet(probe_dir / "probe_points.parquet", index=False)
        (probe_dir / "ba_regions.json").write_text(json.dumps(FROZEN_REGIONS, sort_keys=True))

        download_dir = self.root / "downloads_failure"
        download_dir.mkdir()
        kept_rows = rows[rows["image_id"] == "kept1"]
        (download_dir / f"kept1{DEFAULT_FEATURE_SUFFIX}").write_bytes(
            _feature_file_bytes(list(zip(kept_rows["row"], kept_rows["col"], strict=True)))
        )

        with mock.patch(
            "mermaid_classifier.region_eval.features.download_features_parallel",
            return_value={("coral-reef-training", "mermaid/throttled_featurevector")},
        ):
            probe = load_probe(probe_dir, download_dir=download_dir)

        self.assertEqual(probe.download_failed_image_ids, ("throttled",))
        self.assertEqual(probe.features.n_points_missing_download_failed, 2)

        # This test's feature files are real, FEATURE_DIM-wide npz archives
        # `build_feature_cache` parses for real, so the model must accept
        # that width too -- unlike `_export_model`'s 8-feature default,
        # sized for the other tests' hand-built `FeatureCache`s.
        model, batch = make_calibrated_model(n_features=FEATURE_DIM, seed=0)
        model_dir = self.root / "model_download_failure"
        model_dir.mkdir()
        model_pt, _manifest, _diff = export_artifact(model, model_dir, batch)
        model_json = model_dir / "model.json"
        score = score_model(
            "v1",
            model_pt_path=Path(model_pt),
            model_json_path=model_json,
            probe=probe,
            options=RegionMetricsOptions(n_resamples=N_RESAMPLES),
            live_region_map_loader=_live_map(),
            n_permutations=N_PERMUTATIONS,
        )
        manifest = build_manifest(score)
        self.assertEqual(manifest["probe"]["n_points_download_failed"], 2)
        self.assertEqual(manifest["probe"]["download_failed_image_ids"], ["throttled"])

    def test_a_cache_read_as_is_carries_no_download_failure_record(self):
        """`ProbeIntegrityTest.setUp` writes a complete cache directly, never
        through a build in this process, so no download was ever attempted
        here to report on -- the cache itself says so, since `_write_probe`
        persists a zero count and an empty image list into it.
        """
        probe = load_probe(self.probe_dir)
        self.assertEqual(probe.download_failed_image_ids, ())
        self.assertEqual(probe.features.n_points_missing_download_failed, 0)

    def test_a_cache_predating_the_download_failure_record_reads_back_as_unknown(self):
        """An npz written before this record existed cannot say whether a
        download failed; it reads back as unknown rather than as the zero
        the manifest would otherwise assert as fact.
        """
        path = self.probe_dir / "probe_features.npz"
        archive = dict(np.load(path, allow_pickle=False))
        del archive["n_points_missing_download_failed"]
        del archive["download_failed_image_ids"]
        np.savez_compressed(path, **archive)

        probe = load_probe(self.probe_dir)
        self.assertIsNone(probe.download_failed_image_ids)
        self.assertIsNone(probe.features.n_points_missing_download_failed)

        model_pt, model_json = _export_model(self.root / "model_predating_download_record")
        score = score_model(
            "v1",
            model_pt_path=model_pt,
            model_json_path=model_json,
            probe=probe,
            options=RegionMetricsOptions(n_resamples=N_RESAMPLES),
            live_region_map_loader=_live_map(),
            n_permutations=N_PERMUTATIONS,
        )
        manifest = build_manifest(score)
        self.assertIsNone(manifest["probe"]["n_points_download_failed"])
        self.assertIsNone(manifest["probe"]["download_failed_image_ids"])


class FeatureCoverageTest(unittest.TestCase):
    """How much of the requested probe actually made it into the cache.

    A cache short by a few images' points still loads -- that tolerance is
    `ProbeIntegrityTest`'s, and stays -- but nothing before this told a reader
    how short, or let a caller refuse a run that fell short by more than it
    could tolerate. These exercise the coverage `write_feature_cache` and
    `read_feature_cache` round-trip, and what it surfaces.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _model, batch = make_calibrated_model()
        self.features = np.asarray(batch)[:N_POINTS]
        self.probe_dir = self.root / "probe"
        self.rows = _write_probe(self.probe_dir, self.features)

    def _shrink_cache(self, n_drop: int) -> pd.DataFrame:
        """Overwrite the probe's cache with `n_drop` fewer points, hashed
        against the full row set the way a legitimately short cache is.
        """
        kept = self.rows.iloc[: len(self.rows) - n_drop].reset_index(drop=True)
        write_feature_cache(
            FeatureCache(
                features=self.features[: len(kept)].astype(np.float32),
                image_ids=tuple(kept["image_id"]),
                point_ids=tuple(kept["point_id"]),
                rows=np.asarray(kept["row"], dtype=np.int64),
                cols=np.asarray(kept["col"], dtype=np.int64),
                gt_labels=tuple(kept["gt_label"]),
                region_ids=tuple(kept["region_id"]),
                held_out=np.asarray(kept["held_out"], dtype=bool),
                n_points_requested=len(self.rows),
                n_points_missing_row_col=0,
                n_points_missing_image=n_drop,
                missing_image_ids=(),
                n_points_missing_download_failed=0,
                download_failed_image_ids=(),
            ),
            self.rows,
            self.probe_dir / "probe_features.npz",
        )
        return kept

    def _strip_n_points_requested(self) -> None:
        """Simulate a cache npz written before this field existed."""
        path = self.probe_dir / "probe_features.npz"
        archive = dict(np.load(path, allow_pickle=False))
        del archive["n_points_requested"]
        np.savez_compressed(path, **archive)

    def _score(self, probe) -> ModelScore:
        model_pt, model_json = _export_model(self.root / "model")
        return score_model(
            "v1",
            model_pt_path=model_pt,
            model_json_path=model_json,
            probe=probe,
            options=RegionMetricsOptions(n_resamples=N_RESAMPLES),
            live_region_map_loader=_live_map(),
            n_permutations=N_PERMUTATIONS,
        )

    def test_load_probe_reads_back_n_points_requested(self):
        probe = load_probe(self.probe_dir)
        self.assertEqual(probe.features.n_points_requested, len(self.rows))

    def test_a_cache_written_before_the_field_existed_reads_back_as_unknown(self):
        self._strip_n_points_requested()

        probe = load_probe(self.probe_dir)

        self.assertIsNone(probe.features.n_points_requested)

    def test_coverage_below_one_logs_a_warning_naming_both_counts(self):
        self._shrink_cache(n_drop=5)

        with self.assertLogs("mermaid_classifier.region_eval.score", level="WARNING") as logs:
            load_probe(self.probe_dir)

        message = "\n".join(logs.output)
        self.assertIn(str(len(self.rows) - 5), message)
        self.assertIn(str(len(self.rows)), message)

    def test_complete_coverage_does_not_warn(self):
        with self.assertNoLogs("mermaid_classifier.region_eval.score", level="WARNING"):
            load_probe(self.probe_dir)

    def test_manifest_reports_a_fractional_coverage_for_a_short_cache(self):
        self._shrink_cache(n_drop=5)
        manifest = build_manifest(self._score(load_probe(self.probe_dir)))

        n_kept = len(self.rows) - 5
        self.assertEqual(manifest["probe"]["n_points_requested"], len(self.rows))
        self.assertEqual(manifest["probe"]["n_points_cached"], n_kept)
        self.assertAlmostEqual(manifest["probe"]["coverage"], n_kept / len(self.rows))

    def test_manifest_leaves_coverage_null_when_the_request_count_is_unknown(self):
        self._strip_n_points_requested()
        manifest = build_manifest(self._score(load_probe(self.probe_dir)))

        self.assertIsNone(manifest["probe"]["n_points_requested"])
        self.assertIsNone(manifest["probe"]["coverage"])


class CheckFeatureCoverageTest(unittest.TestCase):
    """`check_feature_coverage`, the function `--min-coverage` refuses through."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _model, batch = make_calibrated_model()
        self.features = np.asarray(batch)[:N_POINTS]
        self.probe_dir = self.root / "probe"
        self.rows = _write_probe(self.probe_dir, self.features)

    def test_refuses_a_cache_short_of_the_floor(self):
        probe = load_probe(self.probe_dir)

        with self.assertRaisesRegex(ValueError, "25.*25|coverage"):
            # A full cache scored against an impossible floor above 1.0
            # exercises the same refusal path a real shortfall would.
            check_feature_coverage(probe.features, 1.5)

    def test_the_refusal_names_the_remedy(self):
        """A local probe dir's short cache is written to disk and refused
        identically on every later run; the operator needs to know that
        deleting it is what unblocks a retried download.
        """
        probe = load_probe(self.probe_dir)

        with self.assertRaisesRegex(ValueError, r"delete.*probe_features\.npz"):
            check_feature_coverage(probe.features, 1.5)

    def test_accepts_a_cache_at_or_above_the_floor(self):
        probe = load_probe(self.probe_dir)
        check_feature_coverage(probe.features, 1.0)  # must not raise

    def test_an_unknown_coverage_is_not_refused(self):
        """A cache whose request count cannot be read is not proof it fell
        short; the floor cannot be checked against a number that is not
        there, so it passes rather than fabricating a refusal.
        """
        path = self.probe_dir / "probe_features.npz"
        archive = dict(np.load(path, allow_pickle=False))
        del archive["n_points_requested"]
        np.savez_compressed(path, **archive)
        probe = load_probe(self.probe_dir)

        check_feature_coverage(probe.features, 1.0)  # must not raise

    def test_an_unknown_coverage_logs_that_the_floor_could_not_be_applied(self):
        """A silent accept here leaves the operator who set --min-coverage
        believing it was honored; a WARNING is the only signal that it was
        not applied at all.
        """
        path = self.probe_dir / "probe_features.npz"
        archive = dict(np.load(path, allow_pickle=False))
        del archive["n_points_requested"]
        np.savez_compressed(path, **archive)
        probe = load_probe(self.probe_dir)

        with self.assertLogs("mermaid_classifier.region_eval.score", level="WARNING") as logs:
            check_feature_coverage(probe.features, 0.99)

        self.assertIn("99.0", "\n".join(logs.output))


class MinCoverageCliTest(unittest.TestCase):
    """`scripts/evaluate_region_probe.py --min-coverage`, opt-in end to end."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _model, batch = make_calibrated_model()
        self.features = np.asarray(batch)[:N_POINTS]
        self.probe_dir = self.root / "probe"
        self.rows = _write_probe(self.probe_dir, self.features)
        self.model_dir = self.root / "model"
        _export_model(self.model_dir)

    def _shrink_cache(self, n_drop: int) -> None:
        kept = self.rows.iloc[: len(self.rows) - n_drop].reset_index(drop=True)
        write_feature_cache(
            FeatureCache(
                features=self.features[: len(kept)].astype(np.float32),
                image_ids=tuple(kept["image_id"]),
                point_ids=tuple(kept["point_id"]),
                rows=np.asarray(kept["row"], dtype=np.int64),
                cols=np.asarray(kept["col"], dtype=np.int64),
                gt_labels=tuple(kept["gt_label"]),
                region_ids=tuple(kept["region_id"]),
                held_out=np.asarray(kept["held_out"], dtype=bool),
                n_points_requested=len(self.rows),
                n_points_missing_row_col=0,
                n_points_missing_image=n_drop,
                missing_image_ids=(),
                n_points_missing_download_failed=0,
                download_failed_image_ids=(),
            ),
            self.rows,
            self.probe_dir / "probe_features.npz",
        )

    def test_a_run_below_the_requested_floor_is_refused_naming_both_counts(self):
        self._shrink_cache(n_drop=5)

        with self.assertRaisesRegex(ValueError, str(len(self.rows) - 5)):
            evaluate_region_probe.main(
                [
                    "--model",
                    f"v1={self.model_dir}",
                    "--probe-dir",
                    str(self.probe_dir),
                    "--out-dir",
                    str(self.root / "out"),
                    "--min-coverage",
                    "0.9",
                    "--n-resamples",
                    str(N_RESAMPLES),
                ]
            )

    def test_the_same_short_run_is_not_refused_without_the_flag(self):
        self._shrink_cache(n_drop=5)

        exit_code = evaluate_region_probe.main(
            [
                "--model",
                f"v1={self.model_dir}",
                "--probe-dir",
                str(self.probe_dir),
                "--out-dir",
                str(self.root / "out"),
                "--n-resamples",
                str(N_RESAMPLES),
            ]
        )
        self.assertEqual(exit_code, 0)


class _StubS3Client:
    """A local double for the `download_file` calls `load_probe` makes.

    A present key writes its bytes to `Filename`; an absent one raises the
    same `ClientError` shape a missing S3 object does, so `load_probe`'s
    404-is-optional handling runs against the real exception type rather than
    an assumption about it. `errors` raises a caller-supplied `ClientError`
    for a given key regardless of whether it is also present in `objects`,
    for exercising a code that is not 404 -- a 403 or a throttle -- on a file
    the 404-only tolerance must not swallow.
    """

    def __init__(self, objects: dict[str, bytes], *, errors: dict[str, ClientError] | None = None):
        self.objects = objects
        self.errors = errors or {}
        self.download_file_calls: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.download_file_calls.append((bucket, key, filename))
        if key in self.errors:
            raise self.errors[key]
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
        Path(filename).write_bytes(self.objects[key])


def _s3_objects(
    probe_dir: Path, prefix: str, *, exclude: frozenset[str] = frozenset()
) -> dict[str, bytes]:
    """Every file under `probe_dir`, keyed as `load_probe` would look it up."""
    return {
        f"{prefix}{path.name}": path.read_bytes()
        for path in probe_dir.iterdir()
        if path.is_file() and path.name not in exclude
    }


class S3ProbeLoadingTest(unittest.TestCase):
    """`load_probe` given an s3://bucket/prefix/ URI instead of a local dir.

    The stub never touches the network; it serves the same bytes a local
    probe dir holds, so the s3 path is checked against the local path it must
    reproduce rather than against a guess at what S3 would return.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _model, batch = make_calibrated_model()
        self.features = np.asarray(batch)[:N_POINTS]
        self.probe_dir = self.root / "probe"
        _write_probe(self.probe_dir, self.features, counts=CORPUS_COUNTS)

    def test_reads_the_same_probe_as_the_local_equivalent(self):
        local_probe = load_probe(self.probe_dir)

        client = _StubS3Client(_s3_objects(self.probe_dir, "region_probe/v1/"))
        with mock.patch("mermaid_classifier.region_eval.score.boto3.client", return_value=client):
            s3_probe = load_probe("s3://bucket/region_probe/v1/")

        self.assertEqual(s3_probe.content_hash, local_probe.content_hash)
        self.assertEqual(s3_probe.region_snapshot_hash, local_probe.region_snapshot_hash)
        self.assertEqual(s3_probe.features.n_points, local_probe.features.n_points)
        self.assertEqual(s3_probe.manifest, local_probe.manifest)
        pd.testing.assert_frame_equal(s3_probe.rows, local_probe.rows)

    def test_a_probe_missing_its_optional_snapshots_still_loads_from_s3(self):
        """A probe published before names/ancestry/counts were frozen has no
        object at those keys. A 404 on any of the three is exactly what a
        missing local file already tolerates, so the s3 path must too.
        """
        bare_dir = self.root / "probe_bare"
        _write_probe(bare_dir, self.features, counts=None, names=None, ancestry=None)
        local_probe = load_probe(bare_dir)

        objects = _s3_objects(
            bare_dir,
            "region_probe/v2/",
            exclude=frozenset({"ba_region_counts.json", "names.json", "ba_ancestry.json"}),
        )
        client = _StubS3Client(objects)
        with mock.patch("mermaid_classifier.region_eval.score.boto3.client", return_value=client):
            s3_probe = load_probe("s3://bucket/region_probe/v2/")

        self.assertEqual(s3_probe.content_hash, local_probe.content_hash)
        self.assertIsNone(s3_probe.ground_truth_counts)
        self.assertFalse(s3_probe.names_present)
        self.assertIsNone(s3_probe.ancestry_by_attribute)

    def test_a_non_404_error_on_an_optional_file_propagates(self):
        """A 403 or a throttle on an optional file is not "this probe
        predates that snapshot" -- only a 404 means absence. `load_probe`
        must propagate anything else rather than loading a probe that has
        silently degraded on an error it never diagnosed.
        """
        objects = _s3_objects(self.probe_dir, "region_probe/v4/")
        forbidden = ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "GetObject")
        client = _StubS3Client(objects, errors={"region_probe/v4/ba_region_counts.json": forbidden})
        with (
            mock.patch("mermaid_classifier.region_eval.score.boto3.client", return_value=client),
            self.assertRaises(ClientError),
        ):
            load_probe("s3://bucket/region_probe/v4/")

    def test_the_content_hash_check_fires_on_a_mismatched_pair_from_s3(self):
        """A parquet from one probe published beside another's feature cache
        is the two-selections-at-once failure `load_probe` refuses locally;
        the s3 path must refuse it identically rather than trusting whatever
        bytes happen to sit under the prefix.
        """
        other_points = [
            (f"o{image_id}", region_id, attribute_id)
            for image_id, region_id, attribute_id in PROBE_POINTS
        ]
        mismatched_dir = self.root / "mismatched"
        _write_probe(mismatched_dir, self.features, points=other_points)

        objects = _s3_objects(self.probe_dir, "region_probe/v3/")
        objects["region_probe/v3/probe_features.npz"] = (
            mismatched_dir / "probe_features.npz"
        ).read_bytes()

        client = _StubS3Client(objects)
        with (
            mock.patch("mermaid_classifier.region_eval.score.boto3.client", return_value=client),
            self.assertRaisesRegex(ValueError, "content_hash"),
        ):
            load_probe("s3://bucket/region_probe/v3/")


class NameResolutionTest(unittest.TestCase):
    """The names frozen with the probe, rendered into the artifacts.

    A table of UUIDs is one a scientist cannot act on without joining it by
    hand, which is what this covers end to end: the names come off the frozen
    snapshot, and an id the snapshot does not name renders as the id rather
    than as a blank a reader would take for "unnamed".
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _model, batch = make_calibrated_model()
        self.features = np.asarray(batch)[:N_POINTS]
        _write_probe(self.root / "probe", self.features, counts=CORPUS_COUNTS)
        self.probe = load_probe(self.root / "probe")
        model_pt, model_json = _export_model(self.root / "model")
        self.score = score_model(
            "v1",
            model_pt_path=model_pt,
            model_json_path=model_json,
            probe=self.probe,
            options=RegionMetricsOptions(n_resamples=N_RESAMPLES),
            live_region_map_loader=_live_map(),
            n_permutations=N_PERMUTATIONS,
        )
        self.out_dir = self.root / "out"
        write_report(self.score, self.out_dir)

    def test_per_label_renders_the_frozen_name_for_every_row(self):
        table = _read_csv(self.out_dir / "per_label.csv").set_index("label")
        self.assertEqual(table.loc["ba1::gf1", "label_name"], "Acropora cervicornis::Branching")
        self.assertEqual(
            [], [name for name in table["label_name"] if not name.strip()], "a name cell is blank"
        )

    def test_an_unnamed_label_renders_its_id_rather_than_an_empty_cell(self):
        """ba4 and gf4 are absent from the frozen names, which is the state a
        probe frozen before an attribute was added leaves behind."""
        table = _read_csv(self.out_dir / "per_label.csv").set_index("label")
        self.assertEqual(table.loc["ba4::gf4", "label_name"], "ba4::gf4")

    def test_a_label_without_a_growth_form_renders_the_attribute_name_alone(self):
        self.assertEqual(
            self.score.probe.label_display_name("ba1::"),
            "Acropora cervicornis",
        )

    def test_region_list_suspects_name_the_attribute_and_the_region(self):
        suspects = _read_csv(self.out_dir / "region_list_suspects.csv").set_index("attribute_id")
        self.assertGreater(len(suspects), 0)
        self.assertEqual(suspects.loc["ba1", "attribute_name"], "Acropora cervicornis")
        self.assertEqual(
            set(suspects["region_name"]) - {"Tropical Atlantic", "Central Indo-Pacific"},
            set(),
            "every suspect row must name the region it was predicted in",
        )

    def test_the_direction_tables_render_region_names(self):
        directions = _read_csv(self.out_dir / "per_direction.csv")
        self.assertEqual(
            set(directions["image_region_name"]),
            {"Tropical Atlantic", "Central Indo-Pacific"},
        )
        self.assertIn("Tropical Atlantic", set(directions["excluded_region_name"]))

        matrix = _read_csv(self.out_dir / "direction_matrix.csv")
        self.assertEqual(
            set(matrix["image_region_name"]), {"Tropical Atlantic", "Central Indo-Pacific"}
        )
        self.assertIn("Central Indo-Pacific", matrix.columns)
        self.assertIn(
            EASTERN_PACIFIC,
            matrix.columns,
            "an unnamed region keeps its id, which reads as unresolved",
        )

    def test_manifest_records_the_frozen_name_and_ancestry_hashes(self):
        """Two scores render the same names only if they read the same
        snapshot, which the hash is what makes checkable."""
        manifest = json.loads((self.out_dir / "manifest.json").read_text())
        self.assertEqual(manifest["probe"]["names_hash"], name_snapshot_hash(NAMES))
        self.assertEqual(
            manifest["probe"]["ancestry_hash"], ancestry_snapshot_hash(ONE_BRANCH_ANCESTRY)
        )

    def test_manifest_records_the_frozen_ground_truth_counts_hash(self):
        """A partial publish dropping only `ba_region_counts.json` must be
        detectable the same way a dropped names or ancestry snapshot already
        is: from the recorded hash disagreeing with the computed one, rather
        than silently falling back and shifting the triage split with only a
        log line to show for it.
        """
        manifest = json.loads((self.out_dir / "manifest.json").read_text())
        expected = ground_truth_counts_hash(_flat_ground_truth_counts(CORPUS_COUNTS))
        self.assertEqual(manifest["probe"]["ground_truth_counts_hash"], expected)
        self.assertEqual(manifest["probe"]["recorded_ground_truth_counts_hash"], expected)

    def test_a_probe_without_frozen_counts_leaves_the_hash_null(self):
        """A probe built before `ba_region_counts.json` was frozen must leave
        both hashes null rather than one computed against an empty
        substitute -- a null reads as absent, where a hash of nothing would
        read as a real, comparable snapshot.
        """
        probe_dir = self.root / "probe_no_counts"
        _write_probe(probe_dir, self.features, counts=None)
        probe = load_probe(probe_dir)
        model_pt, model_json = _export_model(self.root / "model_no_counts")
        score = score_model(
            "no_counts",
            model_pt_path=model_pt,
            model_json_path=model_json,
            probe=probe,
            options=RegionMetricsOptions(n_resamples=N_RESAMPLES),
            live_region_map_loader=_live_map(),
            n_permutations=N_PERMUTATIONS,
        )
        out_dir = self.root / "out_no_counts"
        write_report(score, out_dir)

        manifest = json.loads((out_dir / "manifest.json").read_text())
        self.assertIsNone(manifest["probe"]["ground_truth_counts_hash"])
        self.assertIsNone(manifest["probe"]["recorded_ground_truth_counts_hash"])

    def _bare_probe(self):
        probe_dir = self.root / "probe_bare"
        _write_probe(probe_dir, self.features, counts=CORPUS_COUNTS, names=None, ancestry=None)
        return load_probe(probe_dir)


class DecisionStatisticsTest(unittest.TestCase):
    """The statistics that choose between the mitigations.

    A rate says how bad the problem is; these say what to do about it, and a
    report that omits them leaves the reader to pick between a hard constraint,
    reweighting and dropping labels with no evidence.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _model, batch = make_calibrated_model()
        self.features = np.asarray(batch)[:N_POINTS]
        _write_probe(self.root / "probe", self.features, counts=CORPUS_COUNTS)
        self.probe = load_probe(self.root / "probe")
        self.score = self._score(self.probe)
        self.out_dir = self.root / "out"
        write_report(self.score, self.out_dir)

    def _score(self, probe, name: str = "v1"):
        model_pt, model_json = _export_model(self.root / f"model_{name}")
        return score_model(
            name,
            model_pt_path=model_pt,
            model_json_path=model_json,
            probe=probe,
            options=RegionMetricsOptions(n_resamples=N_RESAMPLES),
            live_region_map_loader=_live_map(),
            n_permutations=N_PERMUTATIONS,
        )

    def _decisions(self):
        return _read_csv(self.out_dir / "decisions.csv").set_index(["statistic", "quantity"])

    def test_the_permutation_ratio_reaches_the_summary_with_an_interval(self):
        """The ratio is the number that picks between a hard constraint and a
        reweighting, so it belongs in the file a reader opens first -- with the
        spread of the null beside it, since a ratio of 0.8 means one thing
        against a baseline spread of 0.01 and nothing against a spread of 0.3.
        """
        summary = _read_csv(self.out_dir / "summary.csv")
        overall = summary[summary["population"] == "all"].set_index("metric")
        self.assertIn("region_blind_ratio", overall.index)
        self.assertIn("region_blind_rate", overall.index)

        ratio = overall.loc["region_blind_ratio"]
        self.assertAlmostEqual(float(ratio["estimate"]), self.score.decisions.region_blind.ratio)
        self.assertLess(float(ratio["ci_low"]), float(ratio["ci_high"]))
        self.assertIn("permutation", ratio["method"])

        baseline = overall.loc["region_blind_rate"]
        self.assertAlmostEqual(
            float(baseline["estimate"]), self.score.decisions.region_blind.baseline_rate
        )
        self.assertEqual(int(baseline["n"]), self.score.decisions.region_blind.n_discriminating)

    def test_the_ratio_interval_carries_both_of_its_terms(self):
        """Inverting the permutation interval alone reports how tightly the
        null is pinned as the ratio's precision. On the real run that is 2.5x
        too narrow, so the caveat has to be in the bounds themselves: each end
        divides one end of the measured rate's cluster bootstrap interval by
        the opposite end of the permutation interval.
        """
        baseline = self.score.decisions.region_blind
        measured = self.score.metrics.overall.oor_rate_disc
        self.assertAlmostEqual(
            measured.rate,
            baseline.observed_rate,
            msg="the ratio's numerator and oor_rate_disc must share a denominator",
        )
        self.assertLess(measured.ci_low, measured.ci_high)

        summary = _read_csv(self.out_dir / "summary.csv")
        ratio = summary[summary["metric"] == "region_blind_ratio"].iloc[0]
        self.assertLess(float(ratio["ci_low"]), float(ratio["estimate"]))
        self.assertGreater(float(ratio["ci_high"]), float(ratio["estimate"]))

        self.assertLess(
            float(ratio["ci_low"]),
            baseline.observed_rate / baseline.baseline_ci_high,
            "the lower bound must sit below the null-only inversion",
        )
        self.assertGreater(
            float(ratio["ci_high"]),
            baseline.observed_rate / baseline.baseline_ci_low,
            "the upper bound must sit above the null-only inversion",
        )

        row = self._decisions().loc[("region_blind", "ratio")]
        self.assertAlmostEqual(float(row["ci_low"]), float(ratio["ci_low"]))
        self.assertAlmostEqual(float(row["ci_high"]), float(ratio["ci_high"]))

    def test_a_region_blind_model_scores_near_the_permutation_baseline(self):
        """The fixture model reads features that carry nothing about region, so
        shuffling regions between images must not move its out-of-region rate.
        A scorer that reported the raw rate (0.70 here) as the ratio, or never
        computed the baseline at all, lands outside this band.
        """
        baseline = self.score.decisions.region_blind
        self.assertGreater(baseline.ratio, 0.8)
        self.assertLess(baseline.ratio, 1.4)
        self.assertGreaterEqual(baseline.observed_rate, baseline.baseline_ci_low)
        self.assertLessEqual(baseline.observed_rate, baseline.baseline_ci_high)
        self.assertEqual(baseline.n_permutations, N_PERMUTATIONS)

    def test_masking_reports_what_it_fixes_apart_from_what_it_breaks(self):
        """A delta of zero is one fix against one break as readily as it is no
        change at all, and a scientist reads those two very differently. On
        this probe masking fixes three predictions and breaks one.
        """
        masking = self.score.decisions.masking
        self.assertEqual(masking.n_fixed, 3)
        self.assertEqual(masking.n_broken, 1)

        rows = self._decisions()
        self.assertEqual(int(rows.loc[("masking", "n_fixed"), "k"]), 3)
        self.assertEqual(int(rows.loc[("masking", "n_broken"), "k"]), 1)
        self.assertEqual(float(rows.loc[("masking", "n_fixed"), "value"]), 3.0)

        delta = rows.loc[("masking", "accuracy_delta")]
        self.assertAlmostEqual(float(delta["value"]), masking.accuracy_delta)
        self.assertFalse(math.isnan(float(delta["ci_low"])))
        self.assertFalse(math.isnan(float(delta["ci_high"])))

    def test_confidence_stratification_reaches_the_report_with_its_bins(self):
        """An AUROC near 0.5 rules a confidence threshold out before anyone
        tunes a cutoff, so it has to be in the report rather than derivable
        from it."""
        rows = self._decisions()
        self.assertAlmostEqual(
            float(rows.loc[("confidence", "auroc"), "value"]),
            self.score.decisions.confidence.auroc,
        )
        bins = [key for key in rows.index if key[0] == "confidence" and key[1].startswith("rate_")]
        self.assertEqual(len(bins), len(self.score.decisions.confidence.bins))

    def test_separate_branches_leave_almost_no_within_branch_share(self):
        """Giving each attribute its own root inverts the statistic: only a
        prediction of the very attribute in the truth still shares a branch.
        One of the sixteen events is that case -- ba1 predicted where the truth
        is ba1, on a Pacific image ba1 does not cover. A test that only ever
        saw one ancestry could not tell the comparison from a constant.
        """
        probe = self.root / "probe_split"
        _write_probe(probe, self.features, counts=CORPUS_COUNTS, ancestry=SEPARATE_BRANCH_ANCESTRY)
        score = self._score(load_probe(probe), name="split")
        share = score.decisions.within_branch
        self.assertEqual(share.n_within_branch, 1)
        self.assertEqual(share.n_evaluable, 16)
        self.assertAlmostEqual(share.share, 1 / 16)

    def test_within_branch_degrades_to_not_computed_without_an_ancestry_snapshot(self):
        """A probe built before the ancestry was frozen still scores. The
        statistic that cannot be computed says so; it does not fail the run and
        it does not read as a share of zero.
        """
        probe = self.root / "probe_no_ancestry"
        _write_probe(probe, self.features, counts=CORPUS_COUNTS, ancestry=None)
        score = self._score(load_probe(probe), name="no_ancestry")
        out_dir = self.root / "out_no_ancestry"
        write_report(score, out_dir)

        self.assertIsNone(score.decisions.within_branch)
        self.assertEqual(score.decisions.within_branch_status, "not_computed")

        rows = _read_csv(out_dir / "decisions.csv")
        within = rows[rows["statistic"] == "within_branch"]
        self.assertEqual(set(within["status"]), {"not_computed"})

        summary = _read_csv(out_dir / "summary.csv")
        self.assertEqual(
            summary[summary["population"] == "all"].set_index("metric").loc["oor_rate", "n"],
            str(N_POINTS),
            "the rest of the score is complete without the ancestry",
        )

    def test_decisions_csv_carries_every_statistic_for_the_model(self):
        rows = _read_csv(self.out_dir / "decisions.csv")
        self.assertEqual(set(rows["model"]), {"v1"})
        self.assertEqual(
            set(rows["statistic"]),
            {"region_blind", "masking", "within_branch", "confidence"},
        )
        self.assertEqual(
            [], [method for method in rows["method"] if not method.strip()], "a method is blank"
        )


class RatioIntervalTest(unittest.TestCase):
    """The degenerate ends of the region-blind ratio's interval.

    A permutation distribution with no spread and a bootstrap with no spread
    are both reachable on a small or homogeneous probe and neither is
    reachable from the fixture, so the bounds are read off the helper with the
    two estimates handed to it directly.
    """

    def _baseline(self, *, observed: float, low: float, high: float) -> RegionBlindBaseline:
        return RegionBlindBaseline(
            n_points=100,
            n_images=10,
            n_unrecorded_region_excluded=0,
            n_region_unknown=0,
            n_discriminating=100,
            n_out_of_region=int(observed * 100),
            observed_rate=observed,
            baseline_rate=(low + high) / 2.0,
            baseline_sd=(high - low) / 4.0,
            baseline_ci_low=low,
            baseline_ci_high=high,
            ratio=observed / ((low + high) / 2.0),
            n_permutations=200,
        )

    def _measured(self, *, low: float, high: float) -> RateEstimate:
        return RateEstimate(
            k=20,
            n=100,
            rate=(low + high) / 2.0,
            ci_low=low,
            ci_high=high,
            wilson_low=low,
            wilson_high=high,
            design_effect=1.0,
            upper_bound=None,
            upper_bound_n_images=10,
            imprecise=None,
        )

    def test_both_uncertainties_widen_the_bounds(self):
        low, high = _ratio_to_baseline_interval(
            self._baseline(observed=0.20, low=0.25, high=0.35),
            self._measured(low=0.16, high=0.24),
        )
        self.assertAlmostEqual(low, 0.16 / 0.35)
        self.assertAlmostEqual(high, 0.24 / 0.25)

    def test_agreeing_finite_quotients_report_a_precise_value(self):
        """Dividing a point by a point still lands both ends on the same
        finite quotient, which is a real, zero-width answer -- the ratio
        this null and this measurement agree on -- not an unresolved bound.
        """
        low, high = _ratio_to_baseline_interval(
            self._baseline(observed=0.20, low=0.25, high=0.25),
            self._measured(low=0.20, high=0.20),
        )
        self.assertAlmostEqual(low, 0.8)
        self.assertAlmostEqual(high, 0.8)

    def test_a_null_pinned_to_zero_leaves_an_unbounded_end(self):
        low, high = _ratio_to_baseline_interval(
            self._baseline(observed=0.20, low=0.0, high=0.35),
            self._measured(low=0.16, high=0.24),
        )
        self.assertAlmostEqual(low, 0.16 / 0.35)
        self.assertTrue(math.isnan(high))

    def test_zero_out_of_region_predictions_report_a_precise_zero(self):
        """A model with no out-of-region predictions at all holds the
        measured rate at 0.0 across the whole bootstrap, so it divides to
        0.0 against any non-degenerate null -- the best result this metric
        can report, not an unresolved bound.
        """
        low, high = _ratio_to_baseline_interval(
            self._baseline(observed=0.0, low=0.03, high=0.08),
            self._measured(low=0.0, high=0.0),
        )
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 0.0)


class ModelSpecTest(unittest.TestCase):
    def test_splits_on_the_first_equals_so_a_path_may_carry_its_own(self):
        """Splitting on the last "=" would score the wrong object whenever a
        model path carries one, silently and with no error to read.
        """
        self.assertEqual(
            evaluate_region_probe.parse_model_spec("v1=s3://mermaid-config/classifier/v1"),
            ("v1", "s3://mermaid-config/classifier/v1"),
        )
        self.assertEqual(
            evaluate_region_probe.parse_model_spec("v2=/models/run=7/artifact"),
            ("v2", "/models/run=7/artifact"),
        )

    def test_rejects_a_spec_missing_its_name_or_its_path(self):
        for bad in ("v1", "=s3://bucket/key", "v1=", "", "   "):
            with self.assertRaises(ValueError, msg=bad):
                evaluate_region_probe.parse_model_spec(bad)
