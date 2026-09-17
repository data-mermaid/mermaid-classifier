"""Scoring a published model artifact against the frozen region probe.

Loads the portable artifact through `load_predictor` -- the production
serve-time loader -- so the thing measured is the thing shipped. Prediction is
the argmax of `predict_proba` mapped back through `predictor.classes_`, which
is what the inference lane does with the same probabilities.

A probe point whose ground truth falls outside the model's label space is
kept. The out-of-region statistics need only the image's region and the
prediction, and dropping such points would shrink the denominator by exactly
the labels a model was never trained on -- flattering a narrow label space.
The metrics layer excludes them from accuracy and F1 on its own, given the
model's class list.

Every model is scored twice: once against the region map frozen with the
probe, and once against the map the MERMAID API serves now. MERMAID curates
those region lists, so without the second pass a score that moved because a
coral gained a region is indistinguishable from a score that moved because
the model changed. Fetching the live map is the one network call this module
makes, and an unreachable API degrades the diagnostic to "not computed"
rather than failing a run that is otherwise complete.

Triage reads the corpus-wide annotation counts frozen with the probe, for the
same reason the region map is frozen. A probe built before they were frozen
falls back to the probe's own ground truth, which counts each pair only as
often as the probe samples it and so reads the model-error headline high;
`limitations.yaml` records which of the two a score was taken on.

Every table renders the display names frozen with the probe beside the ids it
keys on, so the artifact that says which labels cause this can be read without
a join. An id the snapshot does not name renders as the id, which reads as
unresolved rather than as unnamed, and `limitations.yaml` counts those too.

`decisions` runs alongside the rates, because a rate says how bad the problem
is and says nothing about which mitigation answers it. The region-blind
permutation baseline is the load-bearing one -- the ratio to it separates a
model that reads nothing about region from one leaking only a tail -- and the
masking counterfactual prices the constraint that ratio would argue for. The
within-branch share needs the taxonomic ancestry frozen with the probe, and a
probe carrying none leaves that one statistic uncomputed rather than failing a
run whose others are complete.

Outputs are written to a local directory. Nothing is uploaded: publishing a
score is a deliberate step of its own.
"""

import dataclasses
import hashlib
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from mermaid_classifier.common.benthic_attributes import (
    BAGF_SEP,
    get_benthic_attribute_library,
    split_ba_gf,
)
from mermaid_classifier.common.region_rules import (
    is_region_discriminating,
    mcnemar_paired,
    paired_cluster_bootstrap_diff,
)
from mermaid_classifier.pyspacer.inference import load_predictor
from mermaid_classifier.region_eval.decisions import (
    DEFAULT_N_PERMUTATIONS,
    ConfidenceStratification,
    MaskingCounterfactual,
    RegionBlindBaseline,
    WithinBranchShare,
    confidence_stratification,
    masking_counterfactual,
    region_blind_baseline,
    within_branch_share,
)
from mermaid_classifier.region_eval.features import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_WORKERS,
    FeatureLoader,
    build_feature_cache,
    write_feature_cache,
)
from mermaid_classifier.region_eval.metrics import (
    ALL_POINTS,
    HELD_OUT,
    DiffEstimate,
    MacroEstimate,
    PopulationRates,
    RateEstimate,
    RatioEstimate,
    RegionMetricsOptions,
    RegionMismatchMetrics,
    ScoredPoints,
    compute_region_metrics,
    prepare_scored_points,
)
from mermaid_classifier.region_eval.probe_set import (
    PROBE_ANCESTRY_FILE,
    PROBE_COLUMNS,
    PROBE_COUNTS_FILE,
    PROBE_FEATURES_FILE,
    PROBE_MANIFEST_FILE,
    PROBE_NAMES_FILE,
    PROBE_POINTS_FILE,
    PROBE_REGIONS_FILE,
    NameSnapshot,
    ancestry_snapshot_hash,
    name_snapshot_hash,
    probe_content_hash,
    region_snapshot_hash,
)
from mermaid_classifier.region_eval.triage import (
    DEFAULT_LIST_SUSPECT_THRESHOLD,
    TriageResult,
    triage_events,
)

logger = logging.getLogger(__name__)

SUMMARY_FILE = "summary.csv"
LIMITATIONS_FILE = "limitations.yaml"
MANIFEST_FILE = "manifest.json"
MARKDOWN_FILE = "summary.md"
DECISIONS_FILE = "decisions.csv"
COMPARISON_FILE = "paired_comparison.csv"

DECISIONS_COLUMNS = (
    "model",
    "statistic",
    "status",
    "quantity",
    "value",
    "ci_low",
    "ci_high",
    "k",
    "n",
    "method",
)

SUMMARY_COLUMNS = (
    "model",
    "population",
    "metric",
    "kind",
    "denominator",
    "estimate",
    "k",
    "n",
    "ci_low",
    "ci_high",
    "wilson_low",
    "wilson_high",
    "design_effect",
    "imprecise",
    "method",
)

COMPARISON_COLUMNS = (
    "metric",
    "model_a",
    "model_b",
    "rate_a",
    "rate_b",
    "difference",
    "ci_low",
    "ci_high",
    "n_paired",
    "n_discordant_a_only",
    "n_discordant_b_only",
    "mcnemar_p",
    "points_fingerprint",
    "method",
)

# The denominator each rate is read over, in words. A rate quoted without it
# is the misreading this column exists to prevent: the same incidents divided
# by evaluable predictions, by region-discriminating predictions, or by
# region-discriminating ground truths are three different numbers.
RATE_DENOMINATORS = {
    "oor_rate": ("predictions whose benthic attribute has at least one recorded region"),
    "oor_rate_disc": (
        "predictions whose benthic attribute is region-discriminating"
        " across the regions this probe contains"
    ),
    "oor_rate_disc_gt": (
        "points whose ground-truth benthic attribute is region-discriminating"
        " across the regions this probe contains"
    ),
    "gt_oor_rate": ("points whose ground-truth benthic attribute has at least one recorded region"),
    "image_affected_rate": "images in this population, one trial each",
    "accuracy": "points whose ground-truth label is inside the model's label space",
}
REGION_BLIND_DENOMINATOR = (
    "predictions whose benthic attribute is region-discriminating across the"
    " regions this probe contains -- the denominator permuting regions between"
    " images leaves fixed, so it cancels out of the ratio"
)

MACRO_DENOMINATOR = (
    "regions in this population, each contributing its own rate unweighted,"
    " so the mean does not move with the corpus mix"
)
EXCESS_DENOMINATOR = (
    "the out-of-region denominator against the ground-truth denominator,"
    " on one shared draw of images"
)

# All 12 MEOW region polygons were measured mutually disjoint: none of the 66
# pairs intersects, and their areas sum exactly to their union. The upstream
# first-intersecting-polygon assignment is therefore deterministic except
# exactly on a shared boundary.
MEOW_N_REGIONS = 12
MEOW_N_PAIRS_TESTED = 66
MEOW_N_PAIRS_INTERSECTING = 0

# `held_out` marks a point whose image region is not a census region of the
# probe. The portable artifact records its label space and the libraries it
# was built with, and nothing about which points its training run excluded, so
# the two cannot be reconciled from a score.
HELD_OUT_DEFINITION = "the point's image region is not a census region of the probe"
HELD_OUT_TRAINING_EXCLUSION = (
    "not verifiable from here: the model artifact carries no training"
    " exclusion list, so nothing reconciles what the run excluded against the"
    " probe's census regions"
)

# Diagnostics that a missing input can leave uncomputable carry one of these
# rather than a number a reader would take for a measurement.
STATUS_COMPUTED = "computed"
STATUS_NOT_COMPUTED = "not_computed"

DRIFT_COMPUTED = STATUS_COMPUTED
DRIFT_NOT_COMPUTED = STATUS_NOT_COMPUTED

# Where the display names came from. Only a probe carrying the frozen
# snapshot can name anything; without it every id renders as itself.
NAMES_SOURCE_PROBE = "frozen_probe"
NAMES_SOURCE_NONE = "none"

# Where the (attribute, region) annotation counts triage reads came from. Only
# the corpus-wide ones carry the count the list-suspect threshold is set for;
# the probe's own ground truth counts each pair at most as often as the probe
# samples it.
COUNTS_SOURCE_CORPUS = "frozen_corpus"
COUNTS_SOURCE_PROBE = "probe_lower_bound"
COUNTS_SOURCE_CALLER = "caller_supplied"

CORE_RATE_METRICS = ("oor_rate", "oor_rate_disc", "oor_rate_disc_gt", "gt_oor_rate")

# Rates a v1-vs-v2 comparison is read on. The ground-truth floor is not one
# of them: it is a property of the probe, identical for every model.
PAIRED_METRICS = ("oor_rate", "oor_rate_disc", "oor_rate_disc_gt", "accuracy")

RegionMap = Mapping[str, frozenset[str]]
LiveRegionMapLoader = Callable[[], RegionMap]


@dataclasses.dataclass(frozen=True)
class ProbeFeatures:
    """The probe's cached feature matrix and the metadata it lines up with.

    `features[i]` belongs to the point every other array describes at index
    `i`, which is the alignment `features.build_feature_cache` establishes by
    (row, col) rather than by position.
    """

    features: NDArray[np.float32]
    image_ids: tuple[str, ...]
    point_ids: tuple[str, ...]
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    gt_labels: tuple[str, ...]
    region_ids: tuple[str, ...]
    held_out: NDArray[np.bool_]

    @property
    def n_points(self) -> int:
        return len(self.image_ids)


@dataclasses.dataclass(frozen=True)
class LoadedProbe:
    """Everything a scoring run reads off disk, plus the hashes that pin it.

    `ground_truth_counts`, `names` and `ancestry_by_attribute` are absent for a
    probe built before each was frozen beside the points. Each is a fallback
    for the caller to resolve -- ids for names, an uncomputed statistic for
    ancestry -- rather than a reason to refuse the probe.
    """

    rows: pd.DataFrame
    region_ids_by_attribute: dict[str, frozenset[str]]
    ground_truth_counts: dict[tuple[str, str], int] | None
    names: NameSnapshot
    names_present: bool
    ancestry_by_attribute: dict[str, tuple[str, ...]] | None
    manifest: dict[str, Any]
    features: ProbeFeatures
    content_hash: str
    region_snapshot_hash: str
    points_fingerprint: str

    def label_display_name(self, label: str) -> str:
        """A BA-GF label as text, falling back to the id it cannot name.

        A label with no growth form reads as the attribute name alone, which
        is the convention the MERMAID library renders BA-only combos in.
        """
        attribute_id, growth_form_id = split_ba_gf(label)
        attribute = self.names.benthic_attributes.get(attribute_id) or attribute_id
        if not growth_form_id:
            return attribute
        growth_form = self.names.growth_forms.get(growth_form_id) or growth_form_id
        return f"{attribute}{BAGF_SEP}{growth_form}"

    def label_display_names(self, labels: Sequence[str]) -> dict[str, str]:
        """Display names for a set of labels, keyed by the label itself."""
        return {label: self.label_display_name(label) for label in set(labels)}


@dataclasses.dataclass(frozen=True)
class DecisionStatistics:
    """The four statistics that choose between the mitigations.

    `within_branch` is None where the probe carries no ancestry snapshot to
    read, and `within_branch_status` says so: a share of zero would read as
    "none of these is the right taxon", which is a measurement rather than the
    absence of one.
    """

    region_blind: RegionBlindBaseline
    masking: MaskingCounterfactual
    confidence: ConfidenceStratification
    within_branch: WithinBranchShare | None
    within_branch_status: str
    within_branch_reason: str | None
    n_permutations: int


@dataclasses.dataclass(frozen=True)
class ModelScore:
    """One model's measured region-mismatch behaviour on one probe."""

    name: str
    model_pt_path: str
    model_json_path: str
    classes: tuple[str, ...]
    probe: LoadedProbe
    options: RegionMetricsOptions
    points: ScoredPoints
    metrics: RegionMismatchMetrics
    triage: TriageResult
    decisions: DecisionStatistics
    ground_truth_counts_source: str
    n_ground_truth_pairs: int
    drift: dict[str, Any]


def default_live_region_map() -> RegionMap:
    """The benthic-attribute region map the MERMAID API serves right now."""
    return get_benthic_attribute_library().region_ids_by_id


def read_region_snapshot(path: Path) -> dict[str, frozenset[str]]:
    """The frozen `ba_regions.json` as the mapping the predicates expect."""
    payload = json.loads(path.read_text())
    return {str(key): frozenset(str(value) for value in values) for key, values in payload.items()}


def read_ground_truth_counts(path: Path) -> dict[tuple[str, str], int]:
    """The frozen `ba_region_counts.json` keyed the way triage looks counts up."""
    payload = json.loads(path.read_text())
    return {
        (str(attribute_id), str(region_id)): int(count)
        for attribute_id, by_region in payload.items()
        for region_id, count in by_region.items()
    }


def read_name_snapshot(path: Path) -> NameSnapshot:
    """The frozen `names.json` as the three lookups the reports render from."""
    payload = json.loads(path.read_text())
    return NameSnapshot(
        benthic_attributes=_name_section(payload, "benthic_attributes"),
        growth_forms=_name_section(payload, "growth_forms"),
        regions=_name_section(payload, "regions"),
    )


def read_ancestry_snapshot(path: Path) -> dict[str, tuple[str, ...]]:
    """The frozen `ba_ancestry.json` as root-to-leaf paths per attribute."""
    payload = json.loads(path.read_text())
    return {
        str(attribute_id): tuple(str(ancestor) for ancestor in path_)
        for attribute_id, path_ in payload.items()
    }


def read_feature_cache(path: Path) -> ProbeFeatures:
    """Read back the npz `features.write_feature_cache` wrote."""
    archive = np.load(path, allow_pickle=False)
    return ProbeFeatures(
        features=np.asarray(archive["features"], dtype=np.float32),
        image_ids=tuple(str(value) for value in archive["image_id"]),
        point_ids=tuple(str(value) for value in archive["point_id"]),
        rows=np.asarray(archive["row"], dtype=np.int64),
        cols=np.asarray(archive["col"], dtype=np.int64),
        gt_labels=tuple(str(value) for value in archive["gt_label"]),
        region_ids=tuple(str(value) for value in archive["region_id"]),
        held_out=np.asarray(archive["held_out"], dtype=bool),
    )


def points_fingerprint(features: ProbeFeatures) -> str:
    """A hash of the scored points themselves, in the order they are scored.

    Two scores carrying the same fingerprint were taken on identical points,
    which is what makes a comparison between them paired rather than a
    comparison of two differently-composed samples.
    """
    digest = hashlib.sha256()
    for image_id, point_id in zip(features.image_ids, features.point_ids, strict=True):
        digest.update(f"{image_id}\x1f{point_id}\x1e".encode())
    return digest.hexdigest()


def load_probe(
    probe_dir: Path,
    *,
    feature_loader: FeatureLoader | None = None,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> LoadedProbe:
    """Load the probe points, the frozen region map and the feature cache.

    The cache is built through `feature_loader` and written back when the
    probe dir has none; without a loader a missing cache raises, since
    silently scoring zero points would read as a model with no incidents.
    A build checkpoints into `shards/` beside the cache, so an interrupted
    one restarts where it stopped.

    Two things have to agree before a score means anything: the manifest's
    recorded content hash against the points on disk, and the cache against
    those same points -- their identities, their order, and the ground truth,
    region, held-out flag and (row, col) frozen beside each vector. Either
    disagreement is a probe dir holding two selections at once, which scores
    each point against another point's vector or another point's truth, so
    both raise.
    """
    rows = pd.read_parquet(probe_dir / PROBE_POINTS_FILE)
    missing = [column for column in PROBE_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"{probe_dir} probe points are missing: {', '.join(missing)}")

    region_ids_by_attribute = read_region_snapshot(probe_dir / PROBE_REGIONS_FILE)

    counts_path = probe_dir / PROBE_COUNTS_FILE
    ground_truth_counts = read_ground_truth_counts(counts_path) if counts_path.exists() else None

    names_path = probe_dir / PROBE_NAMES_FILE
    names = (
        read_name_snapshot(names_path)
        if names_path.exists()
        else NameSnapshot(benthic_attributes={}, growth_forms={}, regions={})
    )

    ancestry_path = probe_dir / PROBE_ANCESTRY_FILE
    ancestry = read_ancestry_snapshot(ancestry_path) if ancestry_path.exists() else None

    manifest_path = probe_dir / PROBE_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    content_hash = probe_content_hash(rows)
    _check_recorded_content_hash(manifest, content_hash, manifest_path)

    features_path = probe_dir / PROBE_FEATURES_FILE
    if features_path.exists():
        features = read_feature_cache(features_path)
    elif feature_loader is None:
        raise FileNotFoundError(
            f"{features_path} does not exist and no feature loader was given;"
            " build the cache with scripts/build_region_probe.py first"
        )
    else:
        cache = build_feature_cache(
            rows,
            feature_loader,
            workers=workers,
            batch_size=batch_size,
            shard_dir=probe_dir / "shards",
        )
        write_feature_cache(cache, features_path)
        features = read_feature_cache(features_path)
    _check_features_match_rows(features, rows, features_path)

    return LoadedProbe(
        rows=rows,
        region_ids_by_attribute=region_ids_by_attribute,
        ground_truth_counts=ground_truth_counts,
        names=names,
        names_present=names_path.exists(),
        ancestry_by_attribute=ancestry,
        manifest=manifest,
        features=features,
        content_hash=content_hash,
        region_snapshot_hash=region_snapshot_hash(region_ids_by_attribute),
        points_fingerprint=points_fingerprint(features),
    )


def _check_recorded_content_hash(
    manifest: Mapping[str, Any], content_hash: str, manifest_path: Path
) -> None:
    """Refuse points that have moved since the manifest recorded them.

    A probe frozen before the hash was recorded carries none, and passes.
    """
    recorded = manifest.get("content_hash")
    if recorded is not None and str(recorded) != content_hash:
        raise ValueError(
            f"{manifest_path} records content_hash {recorded} but"
            f" {PROBE_POINTS_FILE} hashes to {content_hash}: the points have"
            " changed since the probe was frozen, so nothing else in this"
            " directory describes them"
        )


def _check_features_match_rows(features: ProbeFeatures, rows: pd.DataFrame, path: Path) -> None:
    """Refuse a feature cache built for points other than these.

    The cache is allowed to be shorter than the parquet -- a point with no
    matching (row, col) and every point of an image with no feature file are
    both dropped -- but what it carries has to be these points in their order.
    A cache left behind by an earlier selection fails here rather than pairing
    each point with another point's vector.
    """
    cached = list(zip(features.image_ids, features.point_ids, strict=True))
    present = set(cached)
    probe_points = list(
        zip(
            (str(value) for value in rows["image_id"]),
            (str(value) for value in rows["point_id"]),
            strict=True,
        )
    )
    expected = [pair for pair in probe_points if pair in present]
    if expected != cached:
        wanted = set(probe_points)
        foreign = sum(1 for pair in cached if pair not in wanted)
        raise ValueError(
            f"{path} does not hold the points in {PROBE_POINTS_FILE}: it"
            f" carries {len(cached)} point(s), {foreign} of which the probe"
            f" does not contain, against {len(expected)} that line up in"
            " order. Rebuild the cache."
        )
    kept = [index for index, pair in enumerate(probe_points) if pair in present]
    _check_frozen_columns(features, rows.iloc[kept], path)


def _check_frozen_columns(features: ProbeFeatures, rows: pd.DataFrame, path: Path) -> None:
    """Refuse a cache whose frozen cells are no longer the parquet's.

    `write_feature_cache` freezes each point's ground truth, region, held-out
    flag and (row, col) beside its vector, and scoring reads them from there.
    A probe rebuilt with --skip-features keeps that npz, so a corrected label,
    a redrawn region map or a different held-out partition would otherwise be
    scored at the value it carried when the vectors were downloaded.
    """
    frozen: dict[str, tuple[tuple[object, ...], tuple[object, ...]]] = {
        "gt_label": (features.gt_labels, tuple(str(value) for value in rows["gt_label"])),
        "region_id": (features.region_ids, tuple(str(value) for value in rows["region_id"])),
        "held_out": (
            tuple(bool(value) for value in features.held_out),
            tuple(bool(value) for value in rows["held_out"]),
        ),
        "row": (
            tuple(int(value) for value in features.rows),
            tuple(int(value) for value in rows["row"]),
        ),
        "col": (
            tuple(int(value) for value in features.cols),
            tuple(int(value) for value in rows["col"]),
        ),
    }
    disagreeing = [
        f"{column} ({count} point(s))"
        for column, (cached, recorded) in frozen.items()
        if (count := sum(1 for a, b in zip(cached, recorded, strict=True) if a != b))
    ]
    if disagreeing:
        raise ValueError(
            f"{path} disagrees with {PROBE_POINTS_FILE} on"
            f" {', '.join(disagreeing)}: the cache carries what these points"
            " held when it was built, not what the probe now records."
            " Rebuild the cache."
        )


def predict_with_probabilities(
    predictor: Any, features: NDArray[np.float32]
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Each feature vector's argmax label and the whole probability matrix.

    One forward pass answers both: the label the inference lane would emit,
    and the matrix the masking counterfactual re-argmaxes with the
    out-of-region classes zeroed.
    """
    classes = list(predictor.classes_)
    probabilities = np.asarray(predictor.predict_proba(features), dtype=np.float64)
    labels = tuple(classes[int(index)] for index in np.argmax(probabilities, axis=1))
    return labels, probabilities


def score_model(
    name: str,
    *,
    model_pt_path: Path,
    model_json_path: Path,
    probe: LoadedProbe,
    options: RegionMetricsOptions | None = None,
    live_region_map_loader: LiveRegionMapLoader = default_live_region_map,
    ground_truth_counts: Mapping[tuple[str, str], int] | None = None,
    triage_threshold: int = DEFAULT_LIST_SUSPECT_THRESHOLD,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
) -> ModelScore:
    """Score one portable artifact on the probe and assemble every output.

    `ground_truth_counts` maps (benthic attribute id, region id) to confirmed
    human annotations of that pair corpus-wide, and overrides whatever the
    probe carries. Left out, the counts frozen with the probe are used, and a
    probe without them falls back to its own ground truth --
    `ground_truth_counts_source` says which, because only the first two carry
    the count the list-suspect threshold is set for.

    `n_permutations` sizes the region-blind null. It dominates the cost of the
    decision statistics, since each permutation re-scores every prediction.
    """
    resolved = RegionMetricsOptions() if options is None else options
    predictor = load_predictor(model_pt_path, model_json_path)
    classes = tuple(str(label) for label in predictor.classes_)
    predictions, probabilities = predict_with_probabilities(predictor, probe.features.features)

    points = prepare_scored_points(
        image_ids=probe.features.image_ids,
        image_region_ids=probe.features.region_ids,
        gt_labels=probe.features.gt_labels,
        pred_labels=predictions,
        region_ids_by_attribute=probe.region_ids_by_attribute,
        model_classes=classes,
        held_out=probe.features.held_out.tolist(),
    )
    metrics = compute_region_metrics(
        points,
        options=resolved,
        label_names=probe.label_display_names([*classes, *probe.features.gt_labels, *predictions]),
        region_names=probe.names.regions,
    )
    counts, counts_source = _resolve_ground_truth_counts(points, probe, ground_truth_counts)
    triage = triage_events(
        points,
        ground_truth_counts=counts,
        threshold=triage_threshold,
        attribute_names=probe.names.benthic_attributes,
        region_names=probe.names.regions,
    )

    return ModelScore(
        name=name,
        model_pt_path=str(model_pt_path),
        model_json_path=str(model_json_path),
        classes=classes,
        probe=probe,
        options=resolved,
        points=points,
        metrics=metrics,
        triage=triage,
        decisions=compute_decisions(
            probe=probe,
            predictions=predictions,
            probabilities=probabilities,
            classes=classes,
            options=resolved,
            n_permutations=n_permutations,
        ),
        ground_truth_counts_source=counts_source,
        n_ground_truth_pairs=len(counts),
        drift=region_list_drift(
            probe=probe,
            predictions=predictions,
            classes=classes,
            live_region_map_loader=live_region_map_loader,
        ),
    )


def compute_decisions(
    *,
    probe: LoadedProbe,
    predictions: Sequence[str],
    probabilities: NDArray[np.float64],
    classes: Sequence[str],
    options: RegionMetricsOptions,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
) -> DecisionStatistics:
    """The four statistics that separate the mitigations from one another.

    All four read the probe's frozen region map, so a curation change upstream
    cannot move them. The within-branch share also needs the frozen ancestry,
    and a probe without it leaves that one statistic uncomputed rather than
    failing a run whose other three are complete.
    """
    features = probe.features
    confidences = probabilities.max(axis=1).tolist()

    ancestry = probe.ancestry_by_attribute
    within_branch = (
        None
        if ancestry is None
        else within_branch_share(
            image_region_ids=features.region_ids,
            gt_labels=features.gt_labels,
            pred_labels=predictions,
            region_ids_by_attribute=probe.region_ids_by_attribute,
            ancestry_by_attribute=ancestry,
        )
    )

    return DecisionStatistics(
        region_blind=region_blind_baseline(
            image_ids=features.image_ids,
            image_region_ids=features.region_ids,
            pred_labels=predictions,
            region_ids_by_attribute=probe.region_ids_by_attribute,
            n_permutations=n_permutations,
            alpha=options.alpha,
            seed=options.seed,
        ),
        masking=masking_counterfactual(
            image_ids=features.image_ids,
            image_region_ids=features.region_ids,
            gt_labels=features.gt_labels,
            probabilities=probabilities,
            model_classes=classes,
            region_ids_by_attribute=probe.region_ids_by_attribute,
            n_resamples=options.n_resamples,
            alpha=options.alpha,
            seed=options.seed,
        ),
        confidence=confidence_stratification(
            image_region_ids=features.region_ids,
            pred_labels=predictions,
            pred_confidences=confidences,
            region_ids_by_attribute=probe.region_ids_by_attribute,
        ),
        within_branch=within_branch,
        within_branch_status=(STATUS_NOT_COMPUTED if within_branch is None else STATUS_COMPUTED),
        within_branch_reason=(
            None
            if within_branch is not None
            else f"probe carries no {PROBE_ANCESTRY_FILE}; taxonomic ancestry is unavailable"
        ),
        n_permutations=n_permutations,
    )


def region_list_drift(
    *,
    probe: LoadedProbe,
    predictions: Sequence[str],
    classes: Sequence[str],
    live_region_map_loader: LiveRegionMapLoader,
) -> dict[str, Any]:
    """How far the live region map has moved from the frozen one, and what that costs.

    Rates are recomputed under both maps as bare k/n. The diagnostic answers
    "would this score read differently against today's region lists", and a
    point estimate answers it; the intervals in `summary.csv` all belong to
    the frozen map the probe is pinned to.

    A live map that cannot be fetched leaves the diagnostic not computed,
    carrying the reason, rather than failing a run whose frozen-map results
    are complete.
    """
    frozen = probe.region_ids_by_attribute
    try:
        live = dict(live_region_map_loader())
    except Exception as error:  # noqa: BLE001 - any fetch failure degrades the diagnostic
        logger.warning("live region map unavailable; drift not computed: %s", error)
        return {
            "status": DRIFT_NOT_COMPUTED,
            "reason": f"{type(error).__name__}: {error}",
            "frozen_snapshot_hash": probe.region_snapshot_hash,
            "live_snapshot_hash": None,
            "n_attributes_frozen": len(frozen),
        }

    added = sorted(set(live) - set(frozen))
    removed = sorted(set(frozen) - set(live))
    changed = sorted(
        attribute_id
        for attribute_id in set(frozen) & set(live)
        if frozenset(frozen[attribute_id]) != frozenset(live[attribute_id])
    )
    touched = {split_ba_gf(label)[0] for label in probe.features.gt_labels}
    touched |= {split_ba_gf(label)[0] for label in predictions}

    frozen_counts = _core_counts(
        _prepare(probe, predictions, classes, region_map=frozen),
    )
    live_counts = _core_counts(
        _prepare(probe, predictions, classes, region_map=live),
    )

    return {
        "status": DRIFT_COMPUTED,
        "reason": None,
        "frozen_snapshot_hash": probe.region_snapshot_hash,
        "live_snapshot_hash": region_snapshot_hash(
            {key: frozenset(value) for key, value in live.items()}
        ),
        "n_attributes_frozen": len(frozen),
        "n_attributes_live": len(live),
        "n_attributes_added": len(added),
        "n_attributes_removed": len(removed),
        "n_attributes_region_changed": len(changed),
        "n_probe_attributes_changed": len(set(changed) & touched),
        "probe_attributes_changed": sorted(set(changed) & touched),
        "rates": [
            {
                "metric": metric,
                "frozen_k": frozen_counts[metric][0],
                "frozen_n": frozen_counts[metric][1],
                "frozen_rate": _ratio(*frozen_counts[metric]),
                "live_k": live_counts[metric][0],
                "live_n": live_counts[metric][1],
                "live_rate": _ratio(*live_counts[metric]),
                "delta": _ratio(*live_counts[metric]) - _ratio(*frozen_counts[metric]),
            }
            for metric in CORE_RATE_METRICS
        ],
    }


def paired_comparison(
    scores: Sequence[ModelScore], *, options: RegionMetricsOptions | None = None
) -> pd.DataFrame:
    """Compare every pair of models on the points they share.

    Models scored on one probe are scored on identical points, so the
    difference between them is a paired quantity: one draw of images feeds
    both rates and the variation they share cancels instead of inflating the
    interval. An unpaired two-proportion test would throw that shared
    variation away along with most of the power.

    Each pair is read over the intersection of the two denominators, so
    `rate_a` and `rate_b` can differ from either model's own reported rate --
    that is the price of making them comparable.
    """
    if len(scores) < 2:
        return pd.DataFrame(columns=pd.Index(COMPARISON_COLUMNS))

    fingerprints = {score.probe.points_fingerprint for score in scores}
    if len(fingerprints) > 1:
        raise ValueError(
            "models were scored on different probe points, so no comparison"
            f" between them is paired; fingerprints: {sorted(fingerprints)}"
        )

    resolved = scores[0].options if options is None else options
    rows = [
        _comparison_row(metric, first, second, resolved)
        for position, first in enumerate(scores)
        for second in scores[position + 1 :]
        for metric in PAIRED_METRICS
    ]
    return pd.DataFrame(rows, columns=pd.Index(COMPARISON_COLUMNS))


def summary_table(score: ModelScore) -> pd.DataFrame:
    """One row per metric, each carrying the denominator it was read over.

    The region-blind baseline and the ratio to it sit here beside the rates
    rather than in `decisions.csv` alone: the ratio is what picks between a
    hard constraint and a reweighting, and a reader opens this file first.
    """
    rows: list[dict[str, object]] = []
    for population, rates in _populations(score):
        rows.extend(_population_summary_rows(score, population, rates))
    rows.extend(_decision_summary_rows(score))
    table = pd.DataFrame(rows, columns=pd.Index(SUMMARY_COLUMNS))
    # Counts stay integral: a difference row carries no k, and a float column
    # would render every denominator as "25.0".
    return table.astype({"k": "Int64", "n": "Int64"})


def decisions_table(score: ModelScore) -> pd.DataFrame:
    """Every decision statistic in long form, one quantity per row.

    A quantity with no interval carries NaN bounds rather than a bare number
    dressed as an estimate; `method` says how each was read.
    """
    rows = [
        *_region_blind_rows(score),
        *_masking_rows(score),
        *_within_branch_rows(score),
        *_confidence_rows(score),
    ]
    # Counts stay integral: a quantity carrying no k renders as a blank rather
    # than as "3.0".
    return pd.DataFrame(rows, columns=pd.Index(DECISIONS_COLUMNS)).astype(
        {"k": "Int64", "n": "Int64"}
    )


def build_limitations(score: ModelScore) -> dict[str, Any]:
    """Every caveat on this score, each with the magnitude it was measured at.

    Machine-readable and uniform: a renderer that drops one drops a named
    entry rather than a sentence, and a caveat whose magnitude is small is
    visibly small rather than absent.
    """
    metrics = score.metrics
    points = score.points
    floor = metrics.overall.gt_oor_rate
    ratio = metrics.overall.ratio_to_gt
    directions = metrics.per_direction
    overall_directions = (
        directions[directions["population"] == ALL_POINTS] if len(directions) else directions
    )
    n_direction_points = (
        [int(value) for value in overall_directions["n_points"]] if len(overall_directions) else []
    )
    site_ids = [str(value) for value in score.probe.rows["site_id"]]
    n_with_site = sum(1 for value in site_ids if value)

    return {
        "model": score.name,
        "limitations": [
            {
                "id": "per_direction_sample_size",
                "statement": (
                    "Opposite directions differ in available points by orders of"
                    " magnitude, so a direction's rate is readable only against its"
                    " own denominator and directions are never pooled."
                ),
                "magnitude": {
                    "n_directions": len(n_direction_points),
                    "min_n_points": min(n_direction_points, default=0),
                    "max_n_points": max(n_direction_points, default=0),
                    "directions": [
                        {
                            "image_region_id": str(row["image_region_id"]),
                            "excluded_region_id": str(row["excluded_region_id"]),
                            "n_points": int(row["n_points"]),
                            "n_out_of_region": int(row["n_out_of_region"]),
                        }
                        for _, row in overall_directions.iterrows()
                    ],
                },
            },
            {
                "id": "realized_design_effect",
                "statement": (
                    "Points inside one image are correlated, so each interval is"
                    " wider than the independent-sample interval for the same k of"
                    " n. The design effect is that widening, squared."
                ),
                "magnitude": _design_effects(metrics),
            },
            {
                "id": "clusters_are_images_not_sites",
                "statement": (
                    "Resampling draws whole images. Images from one dive site are"
                    " correlated with each other too, so every interval here is a"
                    " lower bound on the true width."
                ),
                "magnitude": {
                    "n_images": int(points.n_images),
                    "n_points": int(points.n_points),
                    "mean_points_per_image": (
                        float(points.n_points / points.n_images) if points.n_images else 0.0
                    ),
                    "n_points_with_site_id": int(n_with_site),
                    "share_points_with_site_id": (
                        float(n_with_site / len(site_ids)) if site_ids else 0.0
                    ),
                },
            },
            {
                "id": "model_class_region_coverage",
                "statement": (
                    "Only a region-discriminating class can ever be out of region,"
                    " and a class whose recorded regions this probe contains none of"
                    " is untestable: every prediction of it counts as out of region"
                    " and none can be confirmed in its own region."
                ),
                "magnitude": _class_coverage(score),
            },
            {
                "id": "ground_truth_out_of_region_floor",
                "statement": (
                    "Confirmed human annotations are themselves out of region at"
                    " this rate -- label noise plus region-polygon error. No model"
                    " can read below it, so a model rate is read against it."
                ),
                "magnitude": {
                    "k": int(floor.k),
                    "n": int(floor.n),
                    "rate": _clean(floor.rate),
                    "ci_low": _clean(floor.ci_low),
                    "ci_high": _clean(floor.ci_high),
                    "n_images": int(points.n_images),
                    "n_ratio_draws": int(ratio.n_draws),
                    "n_ratio_draws_empty_floor": int(ratio.n_nonfinite_draws),
                },
            },
            {
                "id": "triage_ground_truth_counts",
                "statement": (
                    "An out-of-region event reads as a suspect region list rather"
                    " than a model error once experts have annotated that"
                    " (attribute, region) pair at least `threshold` times"
                    " corpus-wide. Counted on the probe alone the same pair is"
                    " counted only as often as the probe samples it, which moves"
                    " events out of list_suspect and into model_error."
                ),
                "magnitude": {
                    "source": score.ground_truth_counts_source,
                    "is_lower_bound": score.ground_truth_counts_source == COUNTS_SOURCE_PROBE,
                    "threshold": int(score.triage.threshold),
                    "n_pairs": int(score.n_ground_truth_pairs),
                    "bucket_counts": _bucket_counts(score.triage),
                },
            },
            {
                "id": "name_resolution",
                "statement": (
                    "Ids render as the display names frozen with the probe. An"
                    " id the snapshot does not name renders as the id itself,"
                    " which reads as unresolved rather than as unnamed."
                ),
                "magnitude": _name_resolution(score),
            },
            {
                "id": "within_branch_ancestry",
                "statement": (
                    "The share of out-of-region predictions that are the right"
                    " branch in the wrong ocean is read against the taxonomic"
                    " ancestry frozen with the probe. A probe carrying none"
                    " leaves the statistic uncomputed, since a share of zero"
                    " would read as a measurement rather than the absence of"
                    " one."
                ),
                "magnitude": _within_branch_magnitude(score),
            },
            {
                "id": "held_out_training_exclusion",
                "statement": (
                    "`held_out` says only that the point's image region is not"
                    " a census region of the probe, which is what the probe"
                    " builder can see. Whether the scored model's training run"
                    " consumed the matching exclusion list is not readable"
                    " from the artifact, so for a model trained without it the"
                    " held-out block is a training-set measurement."
                ),
                "magnitude": _held_out_magnitude(score),
            },
            {
                "id": "masking_upper_bound",
                "statement": (
                    "The masking counterfactual assumes every image's recorded"
                    " region is correct, so it is an upper bound on what a hard"
                    " constraint would buy: a wrong region turns the same"
                    " machinery into a source of error this number cannot see."
                ),
                "magnitude": _masking_magnitude(score),
            },
            {
                "id": "region_polygons_disjoint",
                "statement": (
                    "All MEOW region polygons were measured mutually disjoint, so"
                    " the upstream first-intersecting-polygon assignment is"
                    " deterministic except exactly on a shared boundary."
                ),
                "magnitude": {
                    "n_regions": MEOW_N_REGIONS,
                    "n_pairs_tested": MEOW_N_PAIRS_TESTED,
                    "n_pairs_intersecting": MEOW_N_PAIRS_INTERSECTING,
                },
            },
        ],
    }


def build_manifest(score: ModelScore) -> dict[str, Any]:
    """What the score was taken on, and what would make it non-comparable."""
    probe = score.probe
    return {
        "probe": {
            "probe_version": probe.manifest.get("probe_version"),
            "recorded_content_hash": probe.manifest.get("content_hash"),
            "computed_content_hash": probe.content_hash,
            "region_snapshot_hash": probe.region_snapshot_hash,
            "names_hash": name_snapshot_hash(probe.names) if probe.names_present else None,
            "recorded_names_hash": probe.manifest.get("names_hash"),
            "ancestry_hash": (
                None
                if probe.ancestry_by_attribute is None
                else ancestry_snapshot_hash(probe.ancestry_by_attribute)
            ),
            "recorded_ancestry_hash": probe.manifest.get("ancestry_hash"),
            "points_fingerprint": probe.points_fingerprint,
            "n_points_cached": int(probe.features.n_points),
            "n_rows": int(len(probe.rows)),
        },
        "model": {
            "name": score.name,
            "model_pt": score.model_pt_path,
            "model_json": score.model_json_path,
            "n_classes": len(score.classes),
        },
        "options": {
            "alpha": score.options.alpha,
            "n_resamples": score.options.n_resamples,
            "seed": score.options.seed,
            "target_margin": score.options.target_margin,
        },
        "held_out": {
            "definition": HELD_OUT_DEFINITION,
            "training_exclusion": HELD_OUT_TRAINING_EXCLUSION,
        },
        "scored": {
            "n_points": int(score.points.n_points),
            "n_images": int(score.points.n_images),
            "n_unrecorded_region_excluded": int(score.points.n_unrecorded_region_excluded),
            "unmapped_attribute_ids": list(score.metrics.unmapped_attribute_ids),
            "observed_region_ids": list(score.metrics.observed_region_ids),
        },
        "triage": {
            "threshold": score.triage.threshold,
            "ground_truth_counts_source": score.ground_truth_counts_source,
            "n_ground_truth_pairs": score.n_ground_truth_pairs,
            "bucket_counts": _bucket_counts(score.triage),
        },
        "decisions": {
            "n_permutations": score.decisions.n_permutations,
            "within_branch_status": score.decisions.within_branch_status,
            "within_branch_reason": score.decisions.within_branch_reason,
        },
        "region_list_drift": score.drift,
    }


def render_markdown(score: ModelScore) -> str:
    """The human summary, leading with the headline rate beside its floor."""
    overall = score.metrics.overall
    lines = [
        f"# Region mismatch — {score.name}",
        "",
        f"{score.points.n_points} points over {score.points.n_images} images"
        f" from {len(score.metrics.observed_region_ids)} region(s).",
        "",
        "## Headline",
        "",
        f"- Out-of-region predictions: {_rate_text(overall.oor_rate)}",
        f"- Ground-truth floor: {_rate_text(overall.gt_oor_rate)}",
        f"- Excess over the floor: {_diff_text(overall.excess)}",
        f"- As a multiple of the floor: {_ratio_text(overall.ratio_to_gt)}",
        "",
        f"Denominator: {RATE_DENOMINATORS['oor_rate']}.",
        "",
        "### Held-out points only",
        "",
        *_held_out_lines(score),
        "## Other denominators for the same incidents",
        "",
        f"- Region-discriminating predictions: {_rate_text(overall.oor_rate_disc)}",
        f"- Region-discriminating ground truths: {_rate_text(overall.oor_rate_disc_gt)}",
        f"- Images carrying at least one incident: {_rate_text(overall.image_affected_rate)}",
        "",
        "## Model quality on this probe",
        "",
        f"- Accuracy: {_rate_text(overall.accuracy)}",
        f"- Over region-discriminating ground truths, point estimates carrying"
        f" no interval: macro precision {_number(overall.precision_macro_disc)},"
        f" macro recall {_number(overall.recall_macro_disc)},"
        f" macro F1 {_number(overall.f1_macro_disc)},"
        f" over {overall.n_f1_disc_points} points",
        "",
        "## What to do about it",
        "",
        *_decision_lines(score),
        "## Region list drift",
        "",
        _drift_text(score.drift),
        "",
        "## Limitations",
        "",
        f"Every caveat, with the magnitude it was measured at, is in {LIMITATIONS_FILE}.",
    ]
    return "\n".join(lines) + "\n"


def _held_out_lines(score: ModelScore) -> list[str]:
    """The held-out rates, below the all-points headline they qualify.

    The qualifier travels with the rates: `held_out` is a property of the
    probe, and the assumption that this model's training run excluded the same
    points is one the score cannot check.
    """
    held = score.metrics.held_out
    if held is None:
        return ["No probe point is held out.", ""]
    return [
        f"- Out-of-region predictions: {_rate_text(held.oor_rate)}",
        f"- Ground-truth floor: {_rate_text(held.gt_oor_rate)}",
        "",
        f"Held out means {HELD_OUT_DEFINITION}. The model artifact carries no"
        " training exclusion list, so whether this run excluded those points"
        " is not verifiable from here: trained without that exclusion, these"
        f" two rates are a training-set measurement. {LIMITATIONS_FILE} counts"
        " what they are read over.",
        "",
    ]


def write_report(score: ModelScore, out_dir: Path) -> dict[str, Path]:
    """Write every artifact for one model into `out_dir`. Nothing is uploaded."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = score.metrics

    written: dict[str, Path] = {}
    for name, frame in (
        (SUMMARY_FILE, summary_table(score)),
        ("per_region.csv", metrics.per_region),
        ("per_label.csv", metrics.per_label),
        ("per_direction.csv", metrics.per_direction),
        ("confusion.csv", metrics.confusion),
        ("region_list_suspects.csv", score.triage.region_list_suspects),
        (DECISIONS_FILE, decisions_table(score)),
    ):
        path = out_dir / name
        frame.to_csv(path, index=False, na_rep="nan")
        written[name] = path

    # Both axes carry region names; per_direction.csv is the long form of the
    # same counts and carries each pair's ids beside them.
    matrix_path = out_dir / "direction_matrix.csv"
    metrics.direction_matrix.to_csv(matrix_path, index=True, index_label="image_region_name")
    written["direction_matrix.csv"] = matrix_path

    limitations_path = out_dir / LIMITATIONS_FILE
    limitations_path.write_text(yaml.safe_dump(build_limitations(score), sort_keys=False))
    written[LIMITATIONS_FILE] = limitations_path

    manifest_path = out_dir / MANIFEST_FILE
    manifest_path.write_text(json.dumps(build_manifest(score), indent=2))
    written[MANIFEST_FILE] = manifest_path

    markdown_path = out_dir / MARKDOWN_FILE
    markdown_path.write_text(render_markdown(score))
    written[MARKDOWN_FILE] = markdown_path

    return written


def _decision_summary_rows(score: ModelScore) -> list[dict[str, object]]:
    """The region-blind baseline and the ratio to it, as summary rows.

    The ratio's interval carries both of its terms -- the cluster bootstrap of
    the measured rate and the percentile spread of the null -- so its width is
    the two combined rather than the null's alone.
    """
    baseline = score.decisions.region_blind
    permutation = (
        f"out-of-region rate under {baseline.n_permutations} permutations of"
        f" image regions between images, seed={score.options.seed}"
    )
    low, high = _ratio_to_baseline_interval(baseline, score.metrics.overall.oor_rate_disc)
    return [
        {
            "model": score.name,
            "population": ALL_POINTS,
            "metric": "region_blind_rate",
            "kind": "baseline",
            "denominator": REGION_BLIND_DENOMINATOR,
            "estimate": baseline.baseline_rate,
            "k": None,
            "n": baseline.n_discriminating,
            "ci_low": baseline.baseline_ci_low,
            "ci_high": baseline.baseline_ci_high,
            "wilson_low": None,
            "wilson_high": None,
            "design_effect": None,
            "imprecise": None,
            "method": (
                f"mean {permutation}; the interval is the"
                f" {1.0 - score.options.alpha:.0%} percentile spread of the"
                f" permutation distribution itself, not a sampling error"
            ),
        },
        {
            "model": score.name,
            "population": ALL_POINTS,
            "metric": "region_blind_ratio",
            "kind": "ratio",
            "denominator": REGION_BLIND_DENOMINATOR,
            "estimate": baseline.ratio,
            "k": baseline.n_out_of_region,
            "n": baseline.n_discriminating,
            "ci_low": low,
            "ci_high": high,
            "wilson_low": None,
            "wilson_high": None,
            "design_effect": None,
            "imprecise": None,
            "method": (
                f"measured rate over the mean {permutation}; each end of the"
                f" interval divides one end of the measured rate's cluster"
                f" bootstrap interval by the opposite end of the permutation"
                f" interval, so it carries both terms' uncertainty"
            ),
        },
    ]


def _ratio_to_baseline_interval(
    baseline: RegionBlindBaseline, measured: RateEstimate
) -> tuple[float, float]:
    """The ratio's interval, carrying the uncertainty of both its terms.

    Each end divides one end of the measured rate's cluster bootstrap interval
    by the opposite end of the permutation interval, so the bounds widen with
    the sampling error of the numerator as well as the dispersion of the null.
    Inverting the permutation interval alone reports the null's spread as the
    ratio's precision, and on a real run the numerator is the larger of the
    two.

    A baseline end of zero leaves that bound unbounded, and a bound neither
    term resolved collapses the interval to a point; both read as NaN rather
    than as a number.
    """
    low = _quotient(measured.ci_low, baseline.baseline_ci_high)
    high = _quotient(measured.ci_high, baseline.baseline_ci_low)
    if low == high:
        return math.nan, math.nan
    return low, high


def _quotient(numerator: float, denominator: float) -> float:
    """numerator/denominator, NaN wherever the division says nothing."""
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return math.nan
    return numerator / denominator


def _decision_row(
    score: ModelScore,
    statistic: str,
    quantity: str,
    *,
    value: float,
    method: str,
    status: str = STATUS_COMPUTED,
    ci_low: float = math.nan,
    ci_high: float = math.nan,
    k: int | None = None,
    n: int | None = None,
) -> dict[str, object]:
    return {
        "model": score.name,
        "statistic": statistic,
        "status": status,
        "quantity": quantity,
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "k": k,
        "n": n,
        "method": method,
    }


def _region_blind_rows(score: ModelScore) -> list[dict[str, object]]:
    """The measured rate, the region-blind null, and the ratio between them."""
    baseline = score.decisions.region_blind
    measured = score.metrics.overall.oor_rate_disc
    status = STATUS_COMPUTED if baseline.n_discriminating else STATUS_NOT_COMPUTED
    permutation = (
        f"{baseline.n_permutations} permutations of image regions between images,"
        f" seed={score.options.seed}"
    )
    low, high = _ratio_to_baseline_interval(baseline, measured)
    return [
        _decision_row(
            score,
            "region_blind",
            "observed_rate",
            status=status,
            value=baseline.observed_rate,
            ci_low=measured.ci_low,
            ci_high=measured.ci_high,
            k=baseline.n_out_of_region,
            n=baseline.n_discriminating,
            method=(
                "the measured out-of-region rate over region-discriminating"
                " predictions; interval is the cluster bootstrap over images"
                " reported for oor_rate_disc"
            ),
        ),
        _decision_row(
            score,
            "region_blind",
            "baseline_rate",
            status=status,
            value=baseline.baseline_rate,
            ci_low=baseline.baseline_ci_low,
            ci_high=baseline.baseline_ci_high,
            n=baseline.n_discriminating,
            method=f"mean rate under {permutation}; interval is the permutation percentile spread",
        ),
        _decision_row(
            score,
            "region_blind",
            "baseline_sd",
            status=status,
            value=baseline.baseline_sd,
            n=baseline.n_permutations,
            method=f"standard deviation of the {permutation}",
        ),
        _decision_row(
            score,
            "region_blind",
            "ratio",
            status=status,
            value=baseline.ratio,
            ci_low=low,
            ci_high=high,
            k=baseline.n_out_of_region,
            n=baseline.n_discriminating,
            method=(
                "measured rate over the region-blind baseline; near 1 the model"
                " reads nothing about region from the image, well below 1 only a"
                " tail leaks. The interval divides each end of the measured"
                " rate's cluster bootstrap interval by the opposite end of the"
                " permutation interval, carrying both terms' uncertainty"
            ),
        ),
    ]


def _masking_rows(score: ModelScore) -> list[dict[str, object]]:
    """What a hard region constraint at inference would buy, and cost."""
    masking = score.decisions.masking
    bootstrap = (
        f"cluster bootstrap over images, percentile,"
        f" {score.options.n_resamples} resamples, alpha={score.options.alpha}"
    )
    counted = "counted over the points whose ground truth is inside the model's label space"
    return [
        _decision_row(
            score,
            "masking",
            "accuracy_unmasked",
            value=masking.accuracy_unmasked,
            n=masking.n_accuracy_points,
            method="argmax of the model's own probabilities",
        ),
        _decision_row(
            score,
            "masking",
            "accuracy_masked",
            value=masking.accuracy_masked,
            n=masking.n_accuracy_points,
            method="argmax after zeroing every class the image's region does not permit",
        ),
        _decision_row(
            score,
            "masking",
            "accuracy_delta",
            value=masking.accuracy_delta,
            ci_low=masking.accuracy_delta_ci_low,
            ci_high=masking.accuracy_delta_ci_high,
            n=masking.n_accuracy_points,
            method=f"masked minus unmasked accuracy; {bootstrap}",
        ),
        _decision_row(
            score,
            "masking",
            "n_fixed",
            value=float(masking.n_fixed),
            k=masking.n_fixed,
            n=masking.n_accuracy_points,
            method=f"predictions masking turns right, {counted}",
        ),
        _decision_row(
            score,
            "masking",
            "n_broken",
            value=float(masking.n_broken),
            k=masking.n_broken,
            n=masking.n_accuracy_points,
            method=f"predictions masking turns wrong, {counted}",
        ),
        _decision_row(
            score,
            "masking",
            "changed_share",
            value=masking.changed_share,
            k=masking.n_changed,
            n=masking.n_points,
            method="predictions masking moves at all, over every scored point",
        ),
        _decision_row(
            score,
            "masking",
            "n_no_permitted_class",
            value=float(masking.n_no_permitted_class),
            n=masking.n_points,
            method="points whose region permits no class at all, left as the model made them",
        ),
        *(
            _decision_row(
                score,
                "masking",
                f"margin_{name}",
                value=value,
                n=masking.margin_n,
                method=(
                    "p_top minus the best in-region probability over the"
                    " predictions masking overturns; near zero masking is nearly"
                    " free"
                ),
            )
            for name, value in (
                ("median", masking.margin_median),
                ("p90", masking.margin_p90),
                ("max", masking.margin_max),
            )
        ),
    ]


def _within_branch_rows(score: ModelScore) -> list[dict[str, object]]:
    """The right-taxon-wrong-ocean share, or the reason there is none."""
    share = score.decisions.within_branch
    if share is None:
        return [
            _decision_row(
                score,
                "within_branch",
                "share",
                status=STATUS_NOT_COMPUTED,
                value=math.nan,
                method=str(score.decisions.within_branch_reason),
            )
        ]
    return [
        _decision_row(
            score,
            "within_branch",
            "share",
            value=share.share,
            k=share.n_within_branch,
            n=share.n_evaluable,
            method=(
                "out-of-region predictions sharing a root-to-leaf branch with"
                " the truth, over those the frozen ancestry places"
            ),
        ),
        _decision_row(
            score,
            "within_branch",
            "n_out_of_region",
            value=float(share.n_out_of_region),
            n=share.n_points,
            method="out-of-region predictions the share is read over",
        ),
        _decision_row(
            score,
            "within_branch",
            "n_ancestry_unknown",
            value=float(share.n_ancestry_unknown),
            n=share.n_out_of_region,
            method="events whose predicted or true attribute the frozen ancestry does not place",
        ),
    ]


def _confidence_rows(score: ModelScore) -> list[dict[str, object]]:
    """Whether a confidence threshold could suppress these predictions at all."""
    confidence = score.decisions.confidence
    return [
        _decision_row(
            score,
            "confidence",
            "auroc",
            value=confidence.auroc,
            k=confidence.n_out_of_region,
            n=confidence.n_discriminating,
            method=(
                "probability a drawn out-of-region prediction outranks a drawn"
                " in-region one; 0.5 is no separation and rules a threshold out"
            ),
        ),
        *(
            _decision_row(
                score,
                "confidence",
                f"rate_{bin_.lower:.2f}_{bin_.upper:.2f}",
                value=bin_.rate,
                k=bin_.n_out_of_region,
                n=bin_.n,
                method="out-of-region share of the predictions in this confidence band",
            )
            for bin_ in confidence.bins
        ),
    ]


def _decision_lines(score: ModelScore) -> list[str]:
    """The interpretation, led by the ratio that picks the mitigation."""
    decisions = score.decisions
    baseline = decisions.region_blind
    masking = decisions.masking
    low, high = _ratio_to_baseline_interval(baseline, score.metrics.overall.oor_rate_disc)
    return [
        f"- Region-blind baseline: {_percent(baseline.baseline_rate)}"
        f" [{_percent(baseline.baseline_ci_low)}, {_percent(baseline.baseline_ci_high)}]"
        f" of region-discriminating predictions fall out of region once region is"
        f" shuffled between images, against {_percent(baseline.observed_rate)}"
        f" [{_percent(score.metrics.overall.oor_rate_disc.ci_low)},"
        f" {_percent(score.metrics.overall.oor_rate_disc.ci_high)}] measured"
        f" — {_number(baseline.ratio)}x [{_number(low)}, {_number(high)}]"
        f" the region-blind rate.",
        "- Near 1 the model reads nothing about region from the image, which is"
        " the case a hard constraint answers; well below 1 region is already"
        " implicit in the features and only a tail leaks, which is the case for"
        " reweighting, regional training data, or dropping the labels"
        " responsible.",
        f"- Masking out-of-region classes at inference moves accuracy by"
        f" {_percent(masking.accuracy_delta)} points"
        f" [{_percent(masking.accuracy_delta_ci_low)},"
        f" {_percent(masking.accuracy_delta_ci_high)}]: it fixes"
        f" {masking.n_fixed} prediction(s) and breaks {masking.n_broken},"
        f" changing {masking.n_changed} of {masking.n_points}.",
        f"- Confidence separates out-of-region predictions at AUROC"
        f" {_number(decisions.confidence.auroc)} over"
        f" {decisions.confidence.n_discriminating} prediction(s).",
        _within_branch_line(score),
        "",
    ]


def _within_branch_line(score: ModelScore) -> str:
    share = score.decisions.within_branch
    if share is None:
        return f"- Right taxon, wrong ocean: not computed ({score.decisions.within_branch_reason})."
    return (
        f"- Right taxon, wrong ocean: {share.n_within_branch} of"
        f" {share.n_evaluable} out-of-region prediction(s) share a branch with"
        f" the truth, which is what masking cleans up rather than reshuffles."
    )


def _name_section(payload: Mapping[str, Any], section: str) -> dict[str, str]:
    """One section of the frozen name snapshot, as text keyed by id."""
    entries = payload.get(section) or {}
    return {str(key): str(value) for key, value in entries.items()}


def _name_resolution(score: ModelScore) -> dict[str, Any]:
    """How much of what the report renders the frozen names could resolve."""
    names = score.probe.names
    labels = {*score.classes, *score.points.gt_labels, *score.points.pred_labels}
    split = [split_ba_gf(label) for label in labels]
    attributes = {attribute_id for attribute_id, _growth_form in split}
    growth_forms = {growth_form for _attribute, growth_form in split if growth_form}
    regions = set(score.metrics.direction_region_ids)

    return {
        "source": NAMES_SOURCE_PROBE if score.probe.names_present else NAMES_SOURCE_NONE,
        "n_benthic_attributes": len(attributes),
        "n_benthic_attributes_unresolved": _n_unresolved(attributes, names.benthic_attributes),
        "n_growth_forms": len(growth_forms),
        "n_growth_forms_unresolved": _n_unresolved(growth_forms, names.growth_forms),
        "n_regions": len(regions),
        "n_regions_unresolved": _n_unresolved(regions, names.regions),
    }


def _n_unresolved(identifiers: set[str], names: Mapping[str, str]) -> int:
    """Ids the name map does not carry a name for."""
    return sum(1 for identifier in identifiers if not names.get(identifier))


def _held_out_magnitude(score: ModelScore) -> dict[str, Any]:
    """How much of the probe is held out, and what pins the exclusion to it."""
    points = score.points
    held = points.held_out
    n_held_out = int(np.count_nonzero(held))
    images = {image_id for image_id, flag in zip(points.image_ids, held, strict=True) if bool(flag)}
    return {
        "definition": HELD_OUT_DEFINITION,
        "n_held_out_points": n_held_out,
        "n_held_out_images": len(images),
        "share_points_held_out": (float(n_held_out / points.n_points) if points.n_points else 0.0),
        "training_exclusion": HELD_OUT_TRAINING_EXCLUSION,
    }


def _within_branch_magnitude(score: ModelScore) -> dict[str, Any]:
    share = score.decisions.within_branch
    return {
        "status": score.decisions.within_branch_status,
        "reason": score.decisions.within_branch_reason,
        "n_out_of_region": None if share is None else int(share.n_out_of_region),
        "n_evaluable": None if share is None else int(share.n_evaluable),
        "n_ancestry_unknown": None if share is None else int(share.n_ancestry_unknown),
        "share": None if share is None else _clean(share.share),
    }


def _masking_magnitude(score: ModelScore) -> dict[str, Any]:
    masking = score.decisions.masking
    return {
        "n_fixed": int(masking.n_fixed),
        "n_broken": int(masking.n_broken),
        "n_changed": int(masking.n_changed),
        "n_no_permitted_class": int(masking.n_no_permitted_class),
        "n_region_unknown_classes": int(masking.n_region_unknown_classes),
        "n_accuracy_points": int(masking.n_accuracy_points),
        "accuracy_delta": _clean(masking.accuracy_delta),
        "accuracy_delta_ci_low": _clean(masking.accuracy_delta_ci_low),
        "accuracy_delta_ci_high": _clean(masking.accuracy_delta_ci_high),
    }


def _paired_masks(metric: str, points: ScoredPoints) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """The (numerator, denominator) masks one metric is read from.

    Each numerator is a subset of its denominator, which is what lets the
    bootstrap statistic count it without masking again.
    """
    if metric == "oor_rate":
        return points.pred_out_of_region, ~points.pred_region_unknown
    if metric == "oor_rate_disc":
        return points.pred_out_of_region, points.pred_discriminating
    if metric == "oor_rate_disc_gt":
        return (
            points.pred_out_of_region & points.gt_discriminating,
            points.gt_discriminating,
        )
    if metric == "accuracy":
        return points.correct & points.gt_in_model_classes, points.gt_in_model_classes
    raise ValueError(f"no paired masks for metric {metric!r}")


def _comparison_row(
    metric: str,
    first: ModelScore,
    second: ModelScore,
    options: RegionMetricsOptions,
) -> dict[str, object]:
    numerator_a, denominator_a = _paired_masks(metric, first.points)
    numerator_b, denominator_b = _paired_masks(metric, second.points)
    shared = denominator_a & denominator_b

    index = np.flatnonzero(shared)
    image_ids = tuple(first.points.image_ids[int(position)] for position in index)
    n_paired = len(index)

    events_a = numerator_a[index]
    events_b = numerator_b[index]
    rate_a = float(np.count_nonzero(events_a) / n_paired) if n_paired else math.nan
    rate_b = float(np.count_nonzero(events_b) / n_paired) if n_paired else math.nan

    a_only = int(np.count_nonzero(events_a & ~events_b))
    b_only = int(np.count_nonzero(events_b & ~events_a))

    if n_paired:
        ones = np.ones(n_paired, dtype=bool)
        ci_low, ci_high = paired_cluster_bootstrap_diff(
            image_ids,
            _rate_statistic(events_a, ones, rate_a),
            _rate_statistic(events_b, ones, rate_b),
            n_resamples=options.n_resamples,
            alpha=options.alpha,
            seed=options.seed,
        )
    else:
        ci_low = ci_high = math.nan

    return {
        "metric": metric,
        "model_a": first.name,
        "model_b": second.name,
        "rate_a": rate_a,
        "rate_b": rate_b,
        "difference": rate_a - rate_b,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_paired": n_paired,
        "n_discordant_a_only": a_only,
        "n_discordant_b_only": b_only,
        "mcnemar_p": mcnemar_paired(a_only, b_only),
        "points_fingerprint": first.probe.points_fingerprint,
        "method": (
            f"paired cluster bootstrap over images,"
            f" {options.n_resamples} resamples, alpha={options.alpha};"
            f" McNemar exact binomial on the {n_paired} shared points"
        ),
    }


def _populations(score: ModelScore) -> list[tuple[str, PopulationRates]]:
    populations = [(ALL_POINTS, score.metrics.overall)]
    if score.metrics.held_out is not None:
        populations.append((HELD_OUT, score.metrics.held_out))
    return populations


def _population_summary_rows(
    score: ModelScore,
    population: str,
    rates: PopulationRates,
) -> list[dict[str, object]]:
    bootstrap = (
        f"cluster bootstrap over images, percentile,"
        f" {score.options.n_resamples} resamples, alpha={score.options.alpha}"
    )
    paired = f"paired {bootstrap}"

    rows = [
        _rate_row(score.name, population, metric, estimate, bootstrap)
        for metric, estimate in (
            ("oor_rate", rates.oor_rate),
            ("oor_rate_disc", rates.oor_rate_disc),
            ("oor_rate_disc_gt", rates.oor_rate_disc_gt),
            ("gt_oor_rate", rates.gt_oor_rate),
            ("image_affected_rate", rates.image_affected_rate),
            ("accuracy", rates.accuracy),
        )
    ]
    rows.append(_diff_row(score.name, population, rates.excess, paired))
    rows.append(_ratio_row(score.name, population, rates.ratio_to_gt, _ratio_method(rates, paired)))
    rows.extend(
        _macro_row(score.name, population, f"macro_{metric}", estimate, bootstrap)
        for metric, estimate in (
            ("oor_rate", rates.region_macro.oor_rate),
            ("oor_rate_disc", rates.region_macro.oor_rate_disc),
            ("oor_rate_disc_gt", rates.region_macro.oor_rate_disc_gt),
            ("gt_oor_rate", rates.region_macro.gt_oor_rate),
        )
    )
    return rows


def _rate_row(
    model: str, population: str, metric: str, estimate: RateEstimate, method: str
) -> dict[str, object]:
    return {
        "model": model,
        "population": population,
        "metric": metric,
        "kind": "rate",
        "denominator": RATE_DENOMINATORS[metric],
        "estimate": estimate.rate,
        "k": estimate.k,
        "n": estimate.n,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "wilson_low": estimate.wilson_low,
        "wilson_high": estimate.wilson_high,
        "design_effect": estimate.design_effect,
        "imprecise": estimate.imprecise,
        "method": method,
    }


def _diff_row(
    model: str, population: str, estimate: DiffEstimate, method: str
) -> dict[str, object]:
    return {
        "model": model,
        "population": population,
        "metric": "excess",
        "kind": "difference",
        "denominator": EXCESS_DENOMINATOR,
        "estimate": estimate.value,
        "k": None,
        "n": None,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "wilson_low": None,
        "wilson_high": None,
        "design_effect": None,
        "imprecise": None,
        "method": method,
    }


def _ratio_method(rates: PopulationRates, paired: str) -> str:
    """How the multiple's interval was read, and what it is conditional on.

    A draw whose floor holds no event has no multiple, so the interval covers
    the draws that do and the reader is told how many did not.
    """
    ratio = rates.ratio_to_gt
    return (
        f"{paired}; interval over the"
        f" {ratio.n_draws - ratio.n_nonfinite_draws} of {ratio.n_draws} draws"
        f" whose ground-truth floor held an event, the rest unbounded"
    )


def _ratio_row(
    model: str, population: str, estimate: RatioEstimate, method: str
) -> dict[str, object]:
    return {
        "model": model,
        "population": population,
        "metric": "ratio_to_gt",
        "kind": "ratio",
        "denominator": EXCESS_DENOMINATOR,
        "estimate": estimate.value,
        "k": None,
        "n": None,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "wilson_low": None,
        "wilson_high": None,
        "design_effect": None,
        "imprecise": None,
        "method": method,
    }


def _macro_row(
    model: str, population: str, metric: str, estimate: MacroEstimate, method: str
) -> dict[str, object]:
    return {
        "model": model,
        "population": population,
        "metric": metric,
        "kind": "macro",
        "denominator": MACRO_DENOMINATOR,
        "estimate": estimate.value,
        "k": None,
        "n": estimate.n_regions,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "wilson_low": None,
        "wilson_high": None,
        "design_effect": None,
        "imprecise": None,
        "method": f"{method}, unweighted mean of per-region rates",
    }


def _rate_statistic(
    numerator: NDArray[np.bool_], denominator: NDArray[np.bool_], fallback: float
) -> Callable[[NDArray[np.intp]], float]:
    """numerator/denominator over drawn positions, with `fallback` on an empty draw."""

    def statistic(index: NDArray[np.intp]) -> float:
        drawn = int(np.count_nonzero(denominator[index]))
        if drawn == 0:
            return fallback
        return float(np.count_nonzero(numerator[index]) / drawn)

    return statistic


def _prepare(
    probe: LoadedProbe,
    predictions: Sequence[str],
    classes: Sequence[str],
    *,
    region_map: RegionMap,
) -> ScoredPoints:
    return prepare_scored_points(
        image_ids=probe.features.image_ids,
        image_region_ids=probe.features.region_ids,
        gt_labels=probe.features.gt_labels,
        pred_labels=predictions,
        region_ids_by_attribute=region_map,
        model_classes=classes,
        held_out=probe.features.held_out.tolist(),
    )


def _core_counts(points: ScoredPoints) -> dict[str, tuple[int, int]]:
    """Each core rate as bare (k, n), with no interval."""
    return {
        "oor_rate": (
            int(np.count_nonzero(points.pred_out_of_region)),
            int(np.count_nonzero(~points.pred_region_unknown)),
        ),
        "oor_rate_disc": (
            int(np.count_nonzero(points.pred_out_of_region)),
            int(np.count_nonzero(points.pred_discriminating)),
        ),
        "oor_rate_disc_gt": (
            int(np.count_nonzero(points.pred_out_of_region & points.gt_discriminating)),
            int(np.count_nonzero(points.gt_discriminating)),
        ),
        "gt_oor_rate": (
            int(np.count_nonzero(points.gt_out_of_region)),
            int(np.count_nonzero(~points.gt_region_unknown)),
        ),
    }


def _bucket_counts(triage: TriageResult) -> dict[str, int]:
    """Events per triage bucket, including the buckets that caught none."""
    return {str(row["bucket"]): int(row["n"]) for _, row in triage.bucket_counts.iterrows()}


def _resolve_ground_truth_counts(
    points: ScoredPoints,
    probe: LoadedProbe,
    supplied: Mapping[tuple[str, str], int] | None,
) -> tuple[Mapping[tuple[str, str], int], str]:
    """The counts triage reads, and the name of where they came from.

    The counts frozen with the probe are corpus-wide, which is the population
    the list-suspect threshold is set against. The probe's own ground truth
    counts a pair only as often as the probe samples it, so falling back to it
    buckets as model errors the events a corpus count would call a suspect
    region list -- the direction that overstates the model-error headline.
    """
    if supplied is not None:
        return supplied, COUNTS_SOURCE_CALLER
    if probe.ground_truth_counts is not None:
        return probe.ground_truth_counts, COUNTS_SOURCE_CORPUS
    logger.warning(
        "probe carries no %s; triage counts fall back to the probe's own ground"
        " truth, a lower bound on the corpus count the threshold reads",
        PROBE_COUNTS_FILE,
    )
    return _probe_ground_truth_counts(points), COUNTS_SOURCE_PROBE


def _probe_ground_truth_counts(points: ScoredPoints) -> dict[tuple[str, str], int]:
    """(attribute, region) annotation counts read off the probe's own ground truth."""
    counts: dict[tuple[str, str], int] = {}
    for attribute_id, region_id in zip(
        points.gt_attribute_ids, points.image_region_ids, strict=True
    ):
        key = (attribute_id, region_id)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _class_coverage(score: ModelScore) -> dict[str, Any]:
    """How much of the model's label space this probe can say anything about."""
    observed = frozenset(score.metrics.observed_region_ids)
    region_map = score.probe.region_ids_by_attribute

    discriminating: list[str] = []
    never: list[str] = []
    unknown: list[str] = []
    untestable: list[str] = []
    for label in score.classes:
        allowed = region_map.get(split_ba_gf(label)[0], frozenset())
        if not allowed:
            unknown.append(label)
            continue
        if is_region_discriminating(allowed, observed):
            discriminating.append(label)
        else:
            never.append(label)
        if not allowed & observed:
            untestable.append(label)

    return {
        "n_model_classes": len(score.classes),
        "n_region_discriminating": len(discriminating),
        "n_never_discriminating": len(never),
        "n_region_unknown": len(unknown),
        "n_untestable_no_probe_region": len(untestable),
        "untestable_classes": untestable,
    }


def _design_effects(metrics: RegionMismatchMetrics) -> dict[str, Any]:
    overall = metrics.overall
    realized = {
        metric: _clean(estimate.design_effect)
        for metric, estimate in (
            ("oor_rate", overall.oor_rate),
            ("oor_rate_disc", overall.oor_rate_disc),
            ("oor_rate_disc_gt", overall.oor_rate_disc_gt),
            ("gt_oor_rate", overall.gt_oor_rate),
        )
    }
    measured = [value for value in realized.values() if value is not None]
    return {
        **realized,
        "max": max(measured) if measured else None,
        "n_measured": len(measured),
        "n_degenerate": len(realized) - len(measured),
    }


def _ratio(k: int, n: int) -> float:
    return k / n if n else math.nan


def _clean(value: float) -> float | None:
    """A float YAML and JSON can carry, with NaN as an explicit absence."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _number(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.4f}"


def _percent(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value * 100:.3f}%"


def _rate_text(estimate: RateEstimate) -> str:
    """A rate, its interval and its counts. Never a rate on its own."""
    interval = f"[{_percent(estimate.ci_low)}, {_percent(estimate.ci_high)}]"
    body = f"{_percent(estimate.rate)} {interval} ({estimate.k} of {estimate.n})"
    if estimate.k == 0 and estimate.upper_bound is not None:
        return (
            f"{body}; none seen, 95% upper bound {_percent(estimate.upper_bound)}"
            f" over {estimate.upper_bound_n_images} images"
        )
    return body


def _diff_text(estimate: DiffEstimate) -> str:
    return (
        f"{_percent(estimate.value)} points"
        f" [{_percent(estimate.ci_low)}, {_percent(estimate.ci_high)}]"
    )


def _ratio_text(estimate: RatioEstimate) -> str:
    return f"{_number(estimate.value)}x [{_number(estimate.ci_low)}, {_number(estimate.ci_high)}]"


def _drift_text(drift: Mapping[str, Any]) -> str:
    if drift.get("status") != DRIFT_COMPUTED:
        return f"Not computed: {drift.get('reason')}"
    lines = [
        f"Against the live library: {drift['n_attributes_region_changed']} attribute(s)"
        f" changed region ({drift['n_probe_attributes_changed']} of them on this probe),"
        f" {drift['n_attributes_added']} added, {drift['n_attributes_removed']} removed."
        f" A removed attribute has no live regions, so the rates below read n/a"
        f" rather than zero.",
        "",
    ]
    lines += [
        f"- {entry['metric']}: frozen {_percent(entry['frozen_rate'])},"
        f" live {_percent(entry['live_rate'])},"
        f" delta {_percent(entry['delta'])}"
        for entry in drift["rates"]
    ]
    return "\n".join(lines)
