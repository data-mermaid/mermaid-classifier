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
from mermaid_classifier.region_eval.features import FeatureCache, write_feature_cache
from mermaid_classifier.region_eval.metrics import RegionMetricsOptions
from mermaid_classifier.region_eval.probe_set import PROBE_COLUMNS, probe_content_hash
from mermaid_classifier.region_eval.score import (
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
    probe_dir: Path, features: np.ndarray, points=PROBE_POINTS, counts=None
) -> pd.DataFrame:
    """Write the probe dir in the layout build_region_probe.py emits.

    `counts` writes the frozen corpus-wide annotation counts; omitting it is a
    probe built before they were frozen.
    """
    probe_dir.mkdir(parents=True, exist_ok=True)
    rows = _probe_rows(points)
    rows.to_parquet(probe_dir / "probe_points.parquet", index=False)
    (probe_dir / "ba_regions.json").write_text(json.dumps(FROZEN_REGIONS, sort_keys=True))
    if counts is not None:
        (probe_dir / "ba_region_counts.json").write_text(json.dumps(counts, sort_keys=True))
    (probe_dir / "manifest.json").write_text(
        json.dumps(
            {
                "probe_version": "1",
                "content_hash": probe_content_hash(rows),
                "n_points": len(rows),
                "n_images": rows["image_id"].nunique(),
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

    def _score(self, name: str = "v1", *, seed: int = 0, live_map=None, probe=None):
        model_pt, model_json = _export_model(self.root / f"model_{name}", seed=seed)
        return score_model(
            name,
            model_pt_path=model_pt,
            model_json_path=model_json,
            probe=self.probe if probe is None else probe,
            options=RegionMetricsOptions(n_resamples=N_RESAMPLES),
            live_region_map_loader=_live_map() if live_map is None else live_map,
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
