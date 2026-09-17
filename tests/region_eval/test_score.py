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

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyspacer._calibrated_model_fixture import make_calibrated_model

from mermaid_classifier.pyspacer.inference import export_artifact
from mermaid_classifier.region_eval.decisions import RegionBlindBaseline
from mermaid_classifier.region_eval.features import FeatureCache, write_feature_cache
from mermaid_classifier.region_eval.metrics import RateEstimate, RegionMetricsOptions
from mermaid_classifier.region_eval.probe_set import (
    PROBE_COLUMNS,
    NameSnapshot,
    ancestry_snapshot_hash,
    name_snapshot_hash,
    probe_content_hash,
)
from mermaid_classifier.region_eval.score import (
    _ratio_to_baseline_interval,
    load_probe,
    paired_comparison,
    score_model,
    write_report,
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

# Ids a report renders: the six attributes and six growth forms the model
# classes and the probe ground truth span between them, and the three regions
# the directions run between.
N_RENDERED_ATTRIBUTES = 6
N_RENDERED_GROWTH_FORMS = 6
N_RENDERED_REGIONS = 3

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
N_CORPUS_COUNT_PAIRS = 5

# Distinct (ground-truth attribute, region) pairs the probe itself carries:
# {ba0, ba1, ba3, ba9} in the Atlantic and {ba0..ba4} in the Pacific.
N_PROBE_COUNT_PAIRS = 9

N_POINTS = 25
N_IMAGES = 6
N_HELD_OUT = 17
N_HELD_OUT_IMAGES = 4
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
        ),
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


def _bucket_counts(score) -> dict[str, int]:
    return {str(row["bucket"]): int(row["n"]) for _, row in score.triage.bucket_counts.iterrows()}


def _limitation(out_dir: Path, entry_id: str) -> dict:
    payload = yaml.safe_load((out_dir / "limitations.yaml").read_text())
    return next(entry for entry in payload["limitations"] if entry["id"] == entry_id)["magnitude"]


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
            "limitations.yaml",
            "manifest.json",
            "summary.md",
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

    def test_ground_truth_outside_the_label_space_counts_for_region_not_accuracy(self):
        """Dropping the point whose truth the model was never trained on would
        shrink the out-of-region denominator to 24 and flatter a narrow label
        space; counting it in accuracy would score the model on a class it
        cannot emit. Both denominators are asserted, so either move fails.
        """
        score = self._score()
        out_dir = self.root / "out"
        write_report(score, out_dir)

        self.assertIn("ba9::gf9", score.points.gt_labels)
        self.assertNotIn("ba9::gf9", score.classes)
        self.assertEqual(score.metrics.overall.n_gt_outside_model_classes, 1)

        overall = _read_csv(out_dir / "summary.csv")
        overall = overall[overall["population"] == "all"].set_index("metric")
        self.assertEqual(overall.loc["oor_rate", "n"], str(N_POINTS))
        self.assertEqual(overall.loc["accuracy", "n"], str(N_IN_MODEL_CLASSES))
        self.assertEqual(
            overall.loc["oor_rate_disc_gt", "n"],
            str(N_GT_DISCRIMINATING),
            "the off-label-space truth region-discriminates and belongs in this denominator",
        )

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

    def test_limitations_carry_a_measured_magnitude_for_every_caveat(self):
        """A caveat reduced to prose is one a reader can wave away, so each
        entry is asserted to carry numbers -- and the numbers are the ones
        counted by hand off the probe table.
        """
        score = self._score()
        out_dir = self.root / "out"
        write_report(score, out_dir)

        payload = yaml.safe_load((out_dir / "limitations.yaml").read_text())
        entries = {entry["id"]: entry for entry in payload["limitations"]}
        self.assertEqual(
            set(entries),
            {
                "per_direction_sample_size",
                "realized_design_effect",
                "clusters_are_images_not_sites",
                "model_class_region_coverage",
                "ground_truth_out_of_region_floor",
                "region_polygons_disjoint",
                "triage_ground_truth_counts",
                "name_resolution",
                "within_branch_ancestry",
                "held_out_training_exclusion",
                "masking_upper_bound",
            },
        )
        for name, entry in entries.items():
            self.assertTrue(entry["statement"].strip(), f"{name} has no statement")
            self.assertIsInstance(entry["magnitude"], dict, f"{name} magnitude is not a mapping")
            self.assertTrue(entry["magnitude"], f"{name} carries no measured magnitude")

        floor = entries["ground_truth_out_of_region_floor"]["magnitude"]
        self.assertEqual(floor["k"], N_GT_OUT_OF_REGION)
        self.assertEqual(floor["n"], N_POINTS)

        clusters = entries["clusters_are_images_not_sites"]["magnitude"]
        self.assertEqual(clusters["n_images"], N_IMAGES)
        self.assertEqual(clusters["n_points"], N_POINTS)
        self.assertEqual(clusters["n_points_with_site_id"], 0)

        coverage = entries["model_class_region_coverage"]["magnitude"]
        self.assertEqual(coverage["n_model_classes"], 5)
        self.assertEqual(coverage["n_region_discriminating"], 4)
        self.assertEqual(coverage["n_never_discriminating"], 1)
        self.assertEqual(coverage["n_untestable_no_probe_region"], 1)
        self.assertEqual(coverage["untestable_classes"], ["ba3::gf3"])

        polygons = entries["region_polygons_disjoint"]["magnitude"]
        self.assertEqual(polygons["n_regions"], 12)
        self.assertEqual(polygons["n_pairs_tested"], 66)
        self.assertEqual(polygons["n_pairs_intersecting"], 0)

        directions = entries["per_direction_sample_size"]["magnitude"]
        self.assertGreater(directions["n_directions"], 0)
        self.assertEqual(len(directions["directions"]), directions["n_directions"])

        effects = entries["realized_design_effect"]["magnitude"]
        self.assertEqual(effects["n_measured"] + effects["n_degenerate"], 4)

        held_out = entries["held_out_training_exclusion"]["magnitude"]
        self.assertEqual(held_out["n_held_out_points"], N_HELD_OUT)
        self.assertEqual(held_out["n_held_out_images"], N_HELD_OUT_IMAGES)
        self.assertAlmostEqual(held_out["share_points_held_out"], N_HELD_OUT / N_POINTS)
        self.assertIn("not verifiable", held_out["training_exclusion"])

    def test_the_held_out_block_says_the_exclusion_cannot_be_verified(self):
        """`held_out` means only that the image is outside a census region.
        A model whose training run never consumed the matching exclusion list
        produces a held-out block that is a training-set measurement, and the
        block is what the mitigation argument quotes, so the qualifier travels
        with the rates rather than living only in limitations.yaml.
        """
        score = self._score()
        out_dir = self.root / "out"
        write_report(score, out_dir)
        text = (out_dir / "summary.md").read_text()

        held_out = text.index("### Held-out points only")
        qualifier = text.index("not verifiable from here", held_out)
        self.assertLess(qualifier, text.index("## Other denominators"))
        self.assertIn("training-set measurement", text[held_out:])

        manifest = json.loads((out_dir / "manifest.json").read_text())
        self.assertIn("census region", manifest["held_out"]["definition"])
        self.assertIn("not verifiable", manifest["held_out"]["training_exclusion"])

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

        before = _bucket_counts(probe_only)
        after = _bucket_counts(corpus)
        self.assertEqual(before["unknown_list"], 0)
        self.assertEqual(before["list_suspect"], 0)
        self.assertEqual(before["model_error"], events)
        self.assertEqual(after["list_suspect"], events)
        self.assertEqual(after["model_error"], 0)

        suspects = corpus.triage.region_list_suspects
        self.assertGreater(len(suspects), 0)
        self.assertEqual(set(suspects["n_ground_truth"]), {605})

    def test_limitations_name_the_frozen_counts_as_the_triage_source(self):
        """A suspect table read off corpus counts is authoritative; one read off
        the probe is a lower bound. A reader must be able to tell which.
        """
        score = self._score(probe=load_probe(self._counted_probe()))
        write_report(score, self.root / "out")

        magnitude = _limitation(self.root / "out", "triage_ground_truth_counts")
        self.assertEqual(magnitude["source"], "frozen_corpus")
        self.assertFalse(magnitude["is_lower_bound"])
        self.assertEqual(magnitude["n_pairs"], N_CORPUS_COUNT_PAIRS)
        self.assertEqual(magnitude["threshold"], 5)

    def test_a_probe_without_frozen_counts_falls_back_and_says_so(self):
        """A probe built before the counts were frozen still scores; what it
        must not do is present a lower-bound suspect table as the corpus one.
        """
        score = self._score()
        write_report(score, self.root / "out")

        magnitude = _limitation(self.root / "out", "triage_ground_truth_counts")
        self.assertEqual(magnitude["source"], "probe_lower_bound")
        self.assertTrue(magnitude["is_lower_bound"])
        self.assertEqual(magnitude["n_pairs"], N_PROBE_COUNT_PAIRS)

    def test_the_multiple_of_the_floor_reports_how_many_draws_it_was_read_over(self):
        """The interval is read over the draws whose ground-truth floor was not
        empty, so it is conditional; without the two counts a reader cannot see
        on how much, and an infinite bound looks like a defect.
        """
        score = self._score()
        out_dir = self.root / "out"
        write_report(score, out_dir)

        magnitude = _limitation(out_dir, "ground_truth_out_of_region_floor")
        self.assertEqual(magnitude["n_ratio_draws"], N_RESAMPLES)
        self.assertGreaterEqual(magnitude["n_ratio_draws_empty_floor"], 0)
        self.assertLessEqual(magnitude["n_ratio_draws_empty_floor"], N_RESAMPLES)

        summary = _read_csv(out_dir / "summary.csv").set_index(["population", "metric"])
        self.assertIn("floor", summary.loc[("all", "ratio_to_gt"), "method"])

    def test_markdown_leads_with_the_headline_rate_beside_the_floor(self):
        """The all-points headline has to come first: a reader who stops after
        the first number must have read the rate a model is judged on, beside
        the floor it is judged against, not the held-out subset. Above the
        drift diagnostic no rate may appear without its interval.
        """
        score = self._score()
        out_dir = self.root / "out"
        write_report(score, out_dir)
        text = (out_dir / "summary.md").read_text()

        headline = text.index("## Headline")
        out_of_region = text.index("- Out-of-region predictions:")
        floor = text.index("- Ground-truth floor:")
        held_out = text.index("Held-out points only")
        self.assertLess(headline, out_of_region)
        self.assertLess(out_of_region, floor)
        self.assertLess(floor, held_out)

        results = text[: text.index("## Region list drift")]
        for line in results.splitlines():
            if line.startswith("- ") and "%" in line:
                self.assertIn("[", line, f"rate rendered without its interval: {line}")

    def test_live_region_map_difference_produces_a_non_zero_drift_diagnostic(self):
        """A coral gaining a region upstream moves the score; without this
        diagnostic that movement is indistinguishable from a model change.
        Giving ba1 the Pacific takes the ground-truth floor from 3 to 2.
        """
        score = self._score(
            live_map=_live_map(ba1=[TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC]),
        )
        drift = score.drift

        self.assertEqual(drift["status"], "computed")
        self.assertEqual(drift["n_attributes_region_changed"], 1)
        self.assertEqual(drift["n_probe_attributes_changed"], 1)
        self.assertEqual(drift["probe_attributes_changed"], ["ba1"])
        self.assertNotEqual(drift["live_snapshot_hash"], drift["frozen_snapshot_hash"])

        rates = {entry["metric"]: entry for entry in drift["rates"]}
        self.assertEqual(rates["gt_oor_rate"]["frozen_k"], N_GT_OUT_OF_REGION)
        self.assertEqual(rates["gt_oor_rate"]["live_k"], N_GT_OUT_OF_REGION - 1)
        self.assertAlmostEqual(rates["gt_oor_rate"]["delta"], -1 / N_POINTS)

    def test_unreachable_live_region_map_degrades_without_failing_the_run(self):
        """The frozen-map results are complete whether or not the API answers,
        so an unreachable library costs the diagnostic, not the run.
        """
        score = self._score(live_map=_unreachable_live_map())
        out_dir = self.root / "out"
        write_report(score, out_dir)

        self.assertEqual(score.drift["status"], "not_computed")
        self.assertIn("ConnectionError", score.drift["reason"])

        summary = _read_csv(out_dir / "summary.csv")
        self.assertEqual(
            summary[summary["population"] == "all"].set_index("metric").loc["oor_rate", "n"],
            str(N_POINTS),
        )
        manifest = json.loads((out_dir / "manifest.json").read_text())
        self.assertEqual(manifest["region_list_drift"]["status"], "not_computed")
        self.assertIn("Not computed", (out_dir / "summary.md").read_text())

    def test_two_models_are_compared_pairwise_on_identical_points(self):
        """An unpaired two-proportion test here would discard the variation the
        two models share and most of the power with it. The comparison is
        asserted to be paired, and to be read over one denominator both models
        share -- 24 points for accuracy, the label space they have in common.
        """
        first = self._score("v1", seed=0)
        second = self._score("v2", seed=1)

        self.assertEqual(first.probe.points_fingerprint, second.probe.points_fingerprint)
        self.assertEqual(first.points.image_ids, second.points.image_ids)
        self.assertEqual(first.points.gt_labels, second.points.gt_labels)
        self.assertNotEqual(first.points.pred_labels, second.points.pred_labels)

        comparison = paired_comparison([first, second])
        self.assertEqual(set(comparison["model_a"]), {"v1"})
        self.assertEqual(set(comparison["model_b"]), {"v2"})
        self.assertEqual(
            sorted(comparison["metric"]),
            ["accuracy", "oor_rate", "oor_rate_disc", "oor_rate_disc_gt"],
        )
        self.assertEqual(set(comparison["points_fingerprint"]), {first.probe.points_fingerprint})

        for _, row in comparison.iterrows():
            self.assertIn("paired", row["method"], row["metric"])
            self.assertIn("McNemar", row["method"], row["metric"])
            self.assertFalse(math.isnan(row["ci_low"]), row["metric"])
            self.assertFalse(math.isnan(row["ci_high"]), row["metric"])
            self.assertAlmostEqual(row["difference"], row["rate_a"] - row["rate_b"])
            self.assertLessEqual(
                row["n_discordant_a_only"] + row["n_discordant_b_only"], row["n_paired"]
            )
            self.assertGreaterEqual(row["mcnemar_p"], 0.0)
            self.assertLessEqual(row["mcnemar_p"], 1.0)

        accuracy = comparison.set_index("metric").loc["accuracy"]
        self.assertEqual(int(accuracy["n_paired"]), N_IN_MODEL_CLASSES)

    def test_paired_comparison_refuses_models_scored_on_different_points(self):
        """Pairing two scores taken on different point sets would report a
        difference the shared-draw interval cannot support, so the mismatch
        has to stop the comparison rather than quietly widen it.

        The second probe carries the same 25 points in reverse, so the arrays
        still line up by shape: only the fingerprint check can catch it, and a
        broadcast error cannot stand in for the guard.
        """
        other_dir = self.root / "probe_reversed"
        _write_probe(other_dir, self.features, points=PROBE_POINTS[::-1])
        model_pt, model_json = _export_model(self.root / "model_reversed")
        reversed_score = score_model(
            "reversed",
            model_pt_path=model_pt,
            model_json_path=model_json,
            probe=load_probe(other_dir),
            options=RegionMetricsOptions(n_resamples=N_RESAMPLES),
            live_region_map_loader=_live_map(),
        )

        self.assertNotEqual(self.probe.points_fingerprint, reversed_score.probe.points_fingerprint)
        with self.assertRaisesRegex(ValueError, "different probe points"):
            paired_comparison([self._score(), reversed_score])


class ProbeIntegrityTest(unittest.TestCase):
    """What binds the parquet, the manifest and the feature cache together.

    A probe dir rebuilt in place after the selection moved holds two
    selections at once: a fresh parquet beside a cache -- or a shard -- from
    the previous one. Every integrity counter still reads clean, because the
    npz's own metadata is written from the new rows, so the only thing that
    can catch it is a check that the cache's points are the parquet's.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _model, batch = make_calibrated_model()
        self.features = np.asarray(batch)[:N_POINTS]
        self.probe_dir = self.root / "probe"
        self.rows = _write_probe(self.probe_dir, self.features)

    def _overwrite_cache(self, rows: pd.DataFrame, features: np.ndarray) -> None:
        """Replace the probe's cache with one built for `rows`."""
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
            ),
            self.probe_dir / "probe_features.npz",
        )

    def test_a_cache_built_for_another_selection_is_refused(self):
        """Twenty-five vectors for twenty-five rows, cached and requested
        counts equal, nothing missing -- and every vector belongs to another
        image. Scoring it would price each point against another point's
        features while reporting this point's ground truth and region.
        """
        other = _probe_rows(
            [
                (f"o{image_id}", region_id, attribute_id)
                for image_id, region_id, attribute_id in PROBE_POINTS
            ]
        )
        self._overwrite_cache(other, self.features)

        with self.assertRaisesRegex(ValueError, "does not hold the points"):
            load_probe(self.probe_dir)

    def test_a_cache_reordered_against_the_parquet_is_refused(self):
        """The reversed cache carries the same 25 (image, point) pairs and the
        same 25 vectors, so every count balances and only the order gives it
        away.
        """
        self._overwrite_cache(self.rows.iloc[::-1].reset_index(drop=True), self.features[::-1])

        with self.assertRaisesRegex(ValueError, "does not hold the points"):
            load_probe(self.probe_dir)

    def test_a_cache_short_by_one_images_points_still_loads(self):
        """An image whose feature file never existed legitimately shrinks the
        cache. A check that refused that would refuse every real probe.
        """
        kept = self.rows[self.rows["image_id"] != "ta1"].reset_index(drop=True)
        self._overwrite_cache(kept, self.features[len(self.rows) - len(kept) :])

        probe = load_probe(self.probe_dir)

        self.assertEqual(probe.features.n_points, len(kept))
        self.assertNotIn("ta1", probe.features.image_ids)

    def test_a_manifest_recording_other_points_is_refused(self):
        """The manifest's content hash and the parquet disagreeing means the
        points moved after the probe was frozen, which makes every other file
        in the directory a description of something else.
        """
        manifest_path = self.probe_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["content_hash"] = probe_content_hash(_probe_rows(PROBE_POINTS[:20]))
        manifest_path.write_text(json.dumps(manifest))

        with self.assertRaisesRegex(ValueError, "content_hash"):
            load_probe(self.probe_dir)


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

    def test_limitations_count_the_ids_no_name_resolved(self):
        """A blank cell hides the gap; the count makes it a measured caveat.
        The frozen names omit ba4, gf4 and the Eastern Pacific.
        """
        magnitude = _limitation(self.out_dir, "name_resolution")
        self.assertEqual(magnitude["source"], "frozen_probe")
        self.assertEqual(magnitude["n_benthic_attributes"], N_RENDERED_ATTRIBUTES)
        self.assertEqual(magnitude["n_benthic_attributes_unresolved"], 1)
        self.assertEqual(magnitude["n_growth_forms"], N_RENDERED_GROWTH_FORMS)
        self.assertEqual(magnitude["n_growth_forms_unresolved"], 1)
        self.assertEqual(magnitude["n_regions"], N_RENDERED_REGIONS)
        self.assertEqual(magnitude["n_regions_unresolved"], 1)

    def test_manifest_records_the_frozen_name_and_ancestry_hashes(self):
        """Two scores render the same names only if they read the same
        snapshot, which the hash is what makes checkable."""
        manifest = json.loads((self.out_dir / "manifest.json").read_text())
        self.assertEqual(manifest["probe"]["names_hash"], name_snapshot_hash(NAMES))
        self.assertEqual(
            manifest["probe"]["ancestry_hash"], ancestry_snapshot_hash(ONE_BRANCH_ANCESTRY)
        )

    def test_a_probe_frozen_without_names_renders_ids_and_counts_them_all(self):
        """A probe built before the names were frozen still scores; what it
        must not do is emit blank name cells that read as "unnamed".
        """
        bare = self._bare_probe()
        model_pt, model_json = _export_model(self.root / "model_bare")
        score = score_model(
            "bare",
            model_pt_path=model_pt,
            model_json_path=model_json,
            probe=bare,
            options=RegionMetricsOptions(n_resamples=N_RESAMPLES),
            live_region_map_loader=_live_map(),
            n_permutations=N_PERMUTATIONS,
        )
        out_dir = self.root / "out_bare"
        write_report(score, out_dir)

        table = _read_csv(out_dir / "per_label.csv")
        self.assertEqual(list(table["label"]), list(table["label_name"]))

        magnitude = _limitation(out_dir, "name_resolution")
        self.assertEqual(magnitude["source"], "none")
        self.assertEqual(
            magnitude["n_benthic_attributes_unresolved"], magnitude["n_benthic_attributes"]
        )
        self.assertEqual(magnitude["n_growth_forms_unresolved"], magnitude["n_growth_forms"])
        self.assertEqual(magnitude["n_regions_unresolved"], magnitude["n_regions"])
        self.assertEqual(magnitude["n_benthic_attributes"], N_RENDERED_ATTRIBUTES)

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
        too narrow, and the ratio reaches summary.md without its method
        string, so the caveat has to be in the bounds themselves: each end
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
        self.assertAlmostEqual(float(ratio["ci_low"]), measured.ci_low / baseline.baseline_ci_high)
        self.assertAlmostEqual(float(ratio["ci_high"]), measured.ci_high / baseline.baseline_ci_low)

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

    def test_the_markdown_renders_the_widened_ratio_interval(self):
        """summary.md carries no method column, so the bounds a reader takes
        for the ratio's confidence interval are the only place the second
        source of uncertainty can appear.
        """
        baseline = self.score.decisions.region_blind
        measured = self.score.metrics.overall.oor_rate_disc
        text = (self.out_dir / "summary.md").read_text()

        low = measured.ci_low / baseline.baseline_ci_high
        high = measured.ci_high / baseline.baseline_ci_low
        self.assertIn(f"[{low:.4f}, {high:.4f}]", text)
        self.assertNotIn(
            f"[{baseline.observed_rate / baseline.baseline_ci_high:.4f},"
            f" {baseline.observed_rate / baseline.baseline_ci_low:.4f}]",
            text,
            "the null-only inversion must not reach the human summary",
        )

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

    def test_within_branch_share_reads_the_frozen_ancestry(self):
        """Every fixture attribute hangs off one root, so every out-of-region
        prediction is the right branch in the wrong ocean and the share is 1.
        """
        share = self.score.decisions.within_branch
        self.assertIsNotNone(share)
        self.assertEqual(share.share, 1.0)
        self.assertEqual(share.n_within_branch, share.n_out_of_region)
        self.assertGreater(share.n_out_of_region, 0)

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

        magnitude = _limitation(out_dir, "within_branch_ancestry")
        self.assertEqual(magnitude["status"], "not_computed")
        self.assertTrue(str(magnitude["reason"]).strip())

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

    def test_markdown_leads_the_interpretation_with_the_ratio(self):
        """The ratio is what picks the mitigation, so a reader who stops after
        the first line of the section must have read it -- not the masking
        delta, which is only worth pricing once the ratio says masking is the
        answer.
        """
        text = (self.out_dir / "summary.md").read_text()
        section = text.index("## What to do about it")
        ratio = text.index("region-blind rate", section)
        masking = text.index("Masking", section)
        self.assertLess(section, ratio)
        self.assertLess(ratio, masking)
        self.assertIn("fixes", text[section:])
        self.assertIn("breaks", text[section:])


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

    def test_a_null_and_a_measurement_with_no_spread_leave_no_bounds(self):
        """Dividing a point by a point is a zero-width interval on the number
        the mitigation argument rests on, which reads as certainty nothing
        measured.
        """
        low, high = _ratio_to_baseline_interval(
            self._baseline(observed=0.20, low=0.25, high=0.25),
            self._measured(low=0.20, high=0.20),
        )
        self.assertTrue(math.isnan(low))
        self.assertTrue(math.isnan(high))

    def test_a_null_pinned_to_zero_leaves_an_unbounded_end(self):
        low, high = _ratio_to_baseline_interval(
            self._baseline(observed=0.20, low=0.0, high=0.35),
            self._measured(low=0.16, high=0.24),
        )
        self.assertAlmostEqual(low, 0.16 / 0.35)
        self.assertTrue(math.isnan(high))


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
