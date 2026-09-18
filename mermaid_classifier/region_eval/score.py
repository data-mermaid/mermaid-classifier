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
    ProbeFeatures,
    build_feature_cache,
    feature_coverage,
    read_feature_cache,
    write_feature_cache,
)
from mermaid_classifier.region_eval.metrics import (
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
    PROBE_FILE_NAMES,
    PROBE_MANIFEST_FILE,
    PROBE_NAMES_FILE,
    PROBE_POINTS_FILE,
    PROBE_REGIONS_FILE,
    REQUIRED_PROBE_FILE_NAMES,
    NameSnapshot,
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


def _name_section(payload: Mapping[str, Any], section: str) -> dict[str, str]:
    """One section of the frozen name snapshot, as text keyed by id."""
    entries = payload.get(section) or {}
    return {str(key): str(value) for key, value in entries.items()}


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
    for name in PROBE_FILE_NAMES:
        try:
            client.download_file(bucket, f"{prefix}{name}", str(destination / name))
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if name in REQUIRED_PROBE_FILE_NAMES or code not in _NOT_FOUND_CODES:
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


def _warn_if_coverage_incomplete(features: ProbeFeatures) -> None:
    """Log how much of the requested probe arrived, whenever it is short.

    Silence otherwise: a cache the tolerance in `load_probe` already accepts
    should not read as an error on every run that scores it, only as a number
    a reader can act on when it drops below what they expect.
    """
    coverage = feature_coverage(features)
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
    coverage = feature_coverage(features)
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
