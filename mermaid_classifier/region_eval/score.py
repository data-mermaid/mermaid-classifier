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

The score itself is read once, against the region map frozen with the probe.
`region_list_drift` separately compares that frozen map's hash against a hash
of the map the MERMAID API serves now, so a coral gaining a region between the
probe's freeze and this run shows up as a hash mismatch rather than being
folded silently into the rates. Fetching the live map is the one network call
this module makes, and an unreachable API degrades the diagnostic to "not
computed" rather than failing a run that is otherwise complete.

Triage reads the corpus-wide annotation counts frozen with the probe, for the
same reason the region map is frozen. A probe built before they were frozen
falls back to the probe's own ground truth, which counts each pair only as
often as the probe samples it and so reads the model-error headline high;
`manifest.json`'s `ground_truth_counts_source` records which of the two a
score was taken on.

Every table renders the display names frozen with the probe beside the ids it
keys on, so the artifact that says which labels cause this can be read without
a join. An id the snapshot does not name renders as the id, which reads as
unresolved rather than as unnamed.

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
import json
import logging
import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from numpy.typing import NDArray

from mermaid_classifier.common.benthic_attributes import (
    BAGF_SEP,
    get_benthic_attribute_library,
    split_ba_gf,
)
from mermaid_classifier.common.s3_utils import is_s3_uri, parse_s3_uri
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
    DEFAULT_FEATURE_BUCKET,
    DEFAULT_FEATURE_PREFIX,
    DEFAULT_WORKERS,
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
    PROBE_HELD_OUT_IMAGES_FILE,
    PROBE_MANIFEST_FILE,
    PROBE_NAMES_FILE,
    PROBE_POINTS_FILE,
    PROBE_REGIONS_FILE,
    NameSnapshot,
    ancestry_snapshot_hash,
    ground_truth_counts_hash,
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

DEFAULT_REGION = "us-east-1"

SUMMARY_FILE = "summary.csv"
MANIFEST_FILE = "manifest.json"
DECISIONS_FILE = "decisions.csv"

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

# Where the (attribute, region) annotation counts triage reads came from. Only
# the corpus-wide ones carry the count the list-suspect threshold is set for;
# the probe's own ground truth counts each pair at most as often as the probe
# samples it.
COUNTS_SOURCE_CORPUS = "frozen_corpus"
COUNTS_SOURCE_PROBE = "probe_lower_bound"
COUNTS_SOURCE_CALLER = "caller_supplied"

RegionMap = Mapping[str, frozenset[str]]
LiveRegionMapLoader = Callable[[], RegionMap]


@dataclasses.dataclass(frozen=True)
class ProbeFeatures:
    """The probe's cached feature matrix and the metadata it lines up with.

    `features[i]` belongs to the point every other array describes at index
    `i`, which is the alignment `features.build_feature_cache` establishes by
    (row, col) rather than by position. `content_hash` is the hash of the
    probe rows the cache was built from, frozen in the npz at write time; an
    archive written before that field existed reads back with an empty one.
    `n_points_requested` is `None` for the same reason on an archive written
    before it was persisted -- coverage against it reads as unknown rather
    than assumed complete. `n_points_missing_download_failed` and
    `download_failed_image_ids` are `None` on the same grounds for an archive
    written before either was persisted: a download failure is then unknown
    rather than known absent.
    """

    features: NDArray[np.float32]
    image_ids: tuple[str, ...]
    point_ids: tuple[str, ...]
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    gt_labels: tuple[str, ...]
    region_ids: tuple[str, ...]
    held_out: NDArray[np.bool_]
    content_hash: str
    n_points_requested: int | None
    n_points_missing_download_failed: int | None
    download_failed_image_ids: tuple[str, ...] | None

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

    `download_failed_image_ids` names the images whose feature file failed to
    download, read from the feature cache itself -- the one this call just
    built, or one already on disk from an earlier run. It is `None` for a
    cache written before that record was persisted, since a download failure
    is then unknown rather than known absent, and empty when the cache does
    carry the record and nothing failed.
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
    download_failed_image_ids: tuple[str, ...] | None

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
    """Read back the npz `features.write_feature_cache` wrote, in canonical order.

    Every array comes back permuted to the (image_id, point_id) order
    `probe_content_hash` sorts by before hashing -- the order `_probe_rows`
    already writes a correctly-built cache in, so the sort is a no-op there.
    The cluster bootstrap draws resample indices against cluster
    first-appearance order, so two caches holding the same rows in different
    orders would otherwise draw different resamples from the same seed
    despite sharing a `content_hash`.

    A cache written before `content_hash` existed reads back with an empty
    one, which cannot equal a real probe's hash and so still fails the check
    in `load_probe` rather than passing silently. A cache written before
    `n_points_requested`, `n_points_missing_download_failed` or
    `download_failed_image_ids` existed reads that field back as `None`
    instead, since there is no comparable substitute a missing value could
    take.
    """
    archive = np.load(path, allow_pickle=False)
    image_ids = tuple(str(value) for value in archive["image_id"])
    point_ids = tuple(str(value) for value in archive["point_id"])
    order = sorted(
        range(len(image_ids)), key=lambda position: (image_ids[position], point_ids[position])
    )
    index = np.asarray(order, dtype=np.int64)

    gt_labels = tuple(str(value) for value in archive["gt_label"])
    region_ids = tuple(str(value) for value in archive["region_id"])
    return ProbeFeatures(
        features=np.asarray(archive["features"], dtype=np.float32)[index],
        image_ids=tuple(image_ids[position] for position in order),
        point_ids=tuple(point_ids[position] for position in order),
        rows=np.asarray(archive["row"], dtype=np.int64)[index],
        cols=np.asarray(archive["col"], dtype=np.int64)[index],
        gt_labels=tuple(gt_labels[position] for position in order),
        region_ids=tuple(region_ids[position] for position in order),
        held_out=np.asarray(archive["held_out"], dtype=bool)[index],
        content_hash=str(archive["content_hash"]) if "content_hash" in archive.files else "",
        n_points_requested=(
            int(archive["n_points_requested"]) if "n_points_requested" in archive.files else None
        ),
        n_points_missing_download_failed=(
            int(archive["n_points_missing_download_failed"])
            if "n_points_missing_download_failed" in archive.files
            else None
        ),
        download_failed_image_ids=(
            tuple(str(value) for value in archive["download_failed_image_ids"])
            if "download_failed_image_ids" in archive.files
            else None
        ),
    )


# Every file scripts/build_region_probe.py may write under --out-dir, in the
# order an s3:// probe dir is downloaded. probe_points.parquet and
# ba_regions.json are load_probe's hard requirements; every other name here is
# optional, exactly as the local path already treats a missing one.
_PROBE_FILE_NAMES = (
    PROBE_POINTS_FILE,
    PROBE_HELD_OUT_IMAGES_FILE,
    PROBE_REGIONS_FILE,
    PROBE_COUNTS_FILE,
    PROBE_NAMES_FILE,
    PROBE_ANCESTRY_FILE,
    PROBE_MANIFEST_FILE,
    PROBE_FEATURES_FILE,
)
_REQUIRED_PROBE_FILE_NAMES = frozenset({PROBE_POINTS_FILE, PROBE_REGIONS_FILE})

_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


def _download_probe_dir(uri: str, destination: Path, *, region_name: str) -> None:
    """Download every probe file published at `uri` into `destination`.

    A 404 on an optional file is skipped, the same tolerance the local path
    gives a missing one via `.exists()`; a 404 on `probe_points.parquet` or
    `ba_regions.json` propagates, since both are read unconditionally below.
    """
    bucket, prefix = parse_s3_uri(uri)
    prefix = prefix.rstrip("/") + "/"
    client = boto3.client("s3", region_name=region_name)
    for name in _PROBE_FILE_NAMES:
        try:
            client.download_file(bucket, f"{prefix}{name}", str(destination / name))
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if name in _REQUIRED_PROBE_FILE_NAMES or code not in _NOT_FOUND_CODES:
                raise


def load_probe(
    probe_dir: Path | str,
    *,
    download_dir: Path | None = None,
    bucket: str = DEFAULT_FEATURE_BUCKET,
    prefix: str = DEFAULT_FEATURE_PREFIX,
    workers: int = DEFAULT_WORKERS,
    region_name: str = DEFAULT_REGION,
) -> LoadedProbe:
    """Load the probe points, the frozen region map and the feature cache.

    `probe_dir` is a local directory, or the s3://bucket/prefix/ one was
    published to (`scripts/build_region_probe.py --publish`). An s3 one is
    downloaded into a scratch directory that does not outlive this call, then
    read exactly as a local directory would be -- the content-hash check
    below runs unchanged regardless of where the pair came from.

    The cache is built into `download_dir` and written back when the probe
    dir has none; without a directory a missing cache raises, since silently
    scoring zero points would read as a model with no incidents. Downloads run
    in parallel into `download_dir` through `download_features_parallel`.
    `download_dir` is a scratch directory that does not outlive the run. For
    an s3:// probe this means the rebuilt cache is written into that same
    scratch directory and discarded when the call returns, rather than beside
    `probe_points.parquet` the way a local probe dir keeps it -- a probe
    published without `probe_features.npz` is rebuilt from S3 on every run
    that scores it. Publish with features included when the same probe will
    be scored repeatedly.

    The cache carries the hash of the exact rows it was built from. A parquet
    that hashes to something else means the directory holds two selections at
    once -- the points moved since the cache was built, the cache was built
    for a different selection, or a `--skip-features` rebuild changed a value
    the cache still carries the old version of -- so the cache is refused
    rather than scored against a point it does not describe.
    """
    if is_s3_uri(probe_dir):
        with tempfile.TemporaryDirectory() as scratch:
            local_dir = Path(scratch)
            _download_probe_dir(probe_dir, local_dir, region_name=region_name)
            return _load_local_probe(
                local_dir,
                source=str(probe_dir),
                download_dir=download_dir,
                bucket=bucket,
                prefix=prefix,
                workers=workers,
            )
    return _load_local_probe(
        Path(probe_dir),
        source=str(probe_dir),
        download_dir=download_dir,
        bucket=bucket,
        prefix=prefix,
        workers=workers,
    )


def _load_local_probe(
    probe_dir: Path,
    *,
    source: str,
    download_dir: Path | None,
    bucket: str,
    prefix: str,
    workers: int,
) -> LoadedProbe:
    """`load_probe`'s local-directory path, read from `probe_dir` as is.

    `source` is what the caller passed to `load_probe` -- `probe_dir` itself
    for a local directory, or the original s3:// URI when `probe_dir` is a
    scratch download that will not outlive this call. Error messages name
    `source`, since a caller of the s3 path never saw the scratch path and it
    no longer exists by the time an error reaches them.
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

    features_path = probe_dir / PROBE_FEATURES_FILE
    if features_path.exists():
        features = read_feature_cache(features_path)
    elif download_dir is None:
        remedy = (
            "republish the probe with features included"
            if is_s3_uri(source)
            else "build the cache with scripts/build_region_probe.py first"
        )
        raise FileNotFoundError(
            f"{source} has no {PROBE_FEATURES_FILE} and no download_dir was given; {remedy}"
        )
    else:
        cache = build_feature_cache(
            rows, download_dir, bucket=bucket, prefix=prefix, workers=workers
        )
        write_feature_cache(cache, rows, features_path)
        features = read_feature_cache(features_path)
    _check_feature_cache_content_hash(features, content_hash, features_path)
    _warn_if_coverage_incomplete(features)

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
        download_failed_image_ids=features.download_failed_image_ids,
    )


def _feature_coverage(features: ProbeFeatures) -> float | None:
    """`n_points_cached` over `n_points_requested`, or `None` when the
    request count is absent or zero and so gives no fraction to compute.
    """
    requested = features.n_points_requested
    if not requested:
        return None
    return features.n_points / requested


def _warn_if_coverage_incomplete(features: ProbeFeatures) -> None:
    """Log how much of the requested probe arrived, whenever it is short.

    Silence otherwise: a cache the tolerance in `load_probe` already accepts
    should not read as an error on every run that scores it, only as a number
    a reader can act on when it drops below what they expect.
    """
    coverage = _feature_coverage(features)
    if coverage is not None and coverage < 1.0:
        logger.warning(
            "feature cache covers %d/%d requested point(s) (%.1f%% coverage)",
            features.n_points,
            features.n_points_requested,
            coverage * 100,
        )


def check_feature_coverage(features: ProbeFeatures, min_coverage: float) -> None:
    """Refuse a cache covering less of the requested probe than `min_coverage`.

    A cache whose request count is unknown -- built before that field was
    persisted -- cannot be checked against a floor and is accepted rather
    than refused on a number this call cannot see; a WARNING says so, since an
    operator who set a floor and gets no refusal has no other way to learn it
    was never applied.
    """
    coverage = _feature_coverage(features)
    if coverage is None:
        logger.warning(
            "--min-coverage %.1f%% cannot be checked: this cache's request count"
            " is unknown (built before n_points_requested was persisted); the"
            " floor was not applied",
            min_coverage * 100,
        )
        return
    if coverage < min_coverage:
        raise ValueError(
            f"feature cache covers {features.n_points}/{features.n_points_requested}"
            f" requested point(s) ({coverage:.1%}), below the required {min_coverage:.1%};"
            f" for a local probe dir, delete {PROBE_FEATURES_FILE} there and rerun to retry"
            f" the download (a probe read from s3:// already rebuilds into a scratch"
            f" directory on every run)"
        )


def _check_feature_cache_content_hash(
    features: ProbeFeatures, content_hash: str, path: Path
) -> None:
    """Refuse a cache that was not built from these exact probe rows.

    The npz's own hash disagreeing with the parquet just read collapses three
    causes into one symptom: the points moved since the cache was built, the
    cache belongs to a different selection, or a `--skip-features` rebuild
    changed a value the cache still carries the old version of.
    """
    if features.content_hash != content_hash:
        raise ValueError(
            f"{path} content_hash {features.content_hash or '(none recorded)'!r} disagrees with"
            f" {PROBE_POINTS_FILE}'s {content_hash!r}: the cache was not built for these rows."
            " Rebuild it with scripts/build_region_probe.py."
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
        drift=region_list_drift(probe=probe, live_region_map_loader=live_region_map_loader),
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
    *, probe: LoadedProbe, live_region_map_loader: LiveRegionMapLoader
) -> dict[str, Any]:
    """Whether the region map has moved since the probe's map was frozen.

    A hash comparison rather than a rescore: the probe already carries the
    frozen map's hash, and hashing the live map the same way answers "has this
    moved" without a second pass over every point or a set of rates a reader
    could mistake for the score itself.

    A live map that cannot be fetched leaves `moved` unanswerable, carrying
    the reason, rather than failing a run whose frozen-map results are
    complete.
    """
    frozen_hash = probe.region_snapshot_hash
    try:
        live = dict(live_region_map_loader())
    except Exception as error:  # any fetch failure degrades the diagnostic
        logger.warning("live region map unavailable; drift not computed: %s", error)
        return {
            "status": STATUS_NOT_COMPUTED,
            "reason": f"{type(error).__name__}: {error}",
            "frozen_snapshot_hash": frozen_hash,
            "live_snapshot_hash": None,
            "moved": None,
        }

    live_hash = region_snapshot_hash({key: frozenset(value) for key, value in live.items()})
    return {
        "status": STATUS_COMPUTED,
        "reason": None,
        "frozen_snapshot_hash": frozen_hash,
        "live_snapshot_hash": live_hash,
        "moved": live_hash != frozen_hash,
    }


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
            "ground_truth_counts_hash": (
                None
                if probe.ground_truth_counts is None
                else ground_truth_counts_hash(probe.ground_truth_counts)
            ),
            "recorded_ground_truth_counts_hash": probe.manifest.get("ground_truth_counts_hash"),
            "n_points_requested": probe.features.n_points_requested,
            "n_points_cached": int(probe.features.n_points),
            "coverage": _feature_coverage(probe.features),
            "n_points_download_failed": probe.features.n_points_missing_download_failed,
            "download_failed_image_ids": (
                None
                if probe.download_failed_image_ids is None
                else list(probe.download_failed_image_ids)
            ),
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

    manifest_path = out_dir / MANIFEST_FILE
    manifest_path.write_text(json.dumps(build_manifest(score), indent=2))
    written[MANIFEST_FILE] = manifest_path

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

    A baseline end of zero leaves that bound unbounded, and `_quotient`
    reports it as NaN rather than as a number; the two ends resolve
    independently, so a measured rate of zero against a non-degenerate null
    reports a precise (0.0, 0.0) rather than an unresolved bound.
    """
    low = _quotient(measured.ci_low, baseline.baseline_ci_high)
    high = _quotient(measured.ci_high, baseline.baseline_ci_low)
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


def _name_section(payload: Mapping[str, Any], section: str) -> dict[str, str]:
    """One section of the frozen name snapshot, as text keyed by id."""
    entries = payload.get(section) or {}
    return {str(key): str(value) for key, value in entries.items()}


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
