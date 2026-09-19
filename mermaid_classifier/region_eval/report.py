"""Rendering one model's region-mismatch score into report artifacts.

`summary_table` and `decisions_table` are the two DataFrames `write_report`
writes to CSV, alongside the tables `metrics` already computed and the
`region_list_suspects` table `triage` already computed; `build_manifest` is
the JSON companion that pins what the score was taken on -- the probe's
hashes, the model paths, and the region-list-drift diagnostic. Nothing here
reaches S3 or MLflow: `write_report` writes only to a local directory, and
publishing a score is a deliberate step of its own.

Every summary and decision quantity renders through the row-builder helpers
below it, so a rate, a difference, a ratio and a macro mean share one row
shape (`model`, `population`/`statistic`, `value`, `ci_low`, `ci_high`, ...)
regardless of which metric or which of the four decision statistics produced
it. `_populations` and `_population_summary_rows` walk `overall` and, where
present, `held_out`; the region-blind ratio's interval combines the measured
rate's own cluster bootstrap with the permutation baseline's percentile
spread, so its width carries both terms' uncertainty rather than the null's
alone.
"""

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from mermaid_classifier.region_eval.decisions import RegionBlindBaseline
from mermaid_classifier.region_eval.features import feature_coverage
from mermaid_classifier.region_eval.metrics import (
    ALL_POINTS,
    HELD_OUT,
    DiffEstimate,
    MacroEstimate,
    PopulationRates,
    RateEstimate,
    RatioEstimate,
)
from mermaid_classifier.region_eval.probe_set import (
    ancestry_snapshot_hash,
    ground_truth_counts_hash,
    name_snapshot_hash,
)
from mermaid_classifier.region_eval.score import (
    STATUS_COMPUTED,
    STATUS_NOT_COMPUTED,
    ModelScore,
)

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
            "coverage": feature_coverage(probe.features),
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
            "bucket_counts": dict(score.triage.bucket_counts),
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
        _summary_row(
            score.name,
            ALL_POINTS,
            "region_blind_rate",
            "baseline",
            REGION_BLIND_DENOMINATOR,
            estimate=baseline.baseline_rate,
            n=baseline.n_discriminating,
            ci_low=baseline.baseline_ci_low,
            ci_high=baseline.baseline_ci_high,
            method=(
                f"mean {permutation}; the interval is the"
                f" {1.0 - score.options.alpha:.0%} percentile spread of the"
                f" permutation distribution itself, not a sampling error"
            ),
        ),
        _summary_row(
            score.name,
            ALL_POINTS,
            "region_blind_ratio",
            "ratio",
            REGION_BLIND_DENOMINATOR,
            estimate=baseline.ratio,
            k=baseline.n_out_of_region,
            n=baseline.n_discriminating,
            ci_low=low,
            ci_high=high,
            method=(
                f"measured rate over the mean {permutation}; each end of the"
                f" interval divides one end of the measured rate's cluster"
                f" bootstrap interval by the opposite end of the permutation"
                f" interval, so it carries both terms' uncertainty"
            ),
        ),
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


def _summary_row(
    model: str,
    population: str,
    metric: str,
    kind: str,
    denominator: str,
    *,
    estimate: float,
    method: str,
    k: int | None = None,
    n: int | None = None,
    ci_low: float = math.nan,
    ci_high: float = math.nan,
    wilson_low: float | None = None,
    wilson_high: float | None = None,
    design_effect: float | None = None,
    imprecise: bool | None = None,
) -> dict[str, object]:
    """One `SUMMARY_COLUMNS` row; a field a caller omits reports as absent."""
    return {
        "model": model,
        "population": population,
        "metric": metric,
        "kind": kind,
        "denominator": denominator,
        "estimate": estimate,
        "k": k,
        "n": n,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "design_effect": design_effect,
        "imprecise": imprecise,
        "method": method,
    }


def _rate_row(
    model: str, population: str, metric: str, estimate: RateEstimate, method: str
) -> dict[str, object]:
    return _summary_row(
        model,
        population,
        metric,
        "rate",
        RATE_DENOMINATORS[metric],
        estimate=estimate.rate,
        k=estimate.k,
        n=estimate.n,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        wilson_low=estimate.wilson_low,
        wilson_high=estimate.wilson_high,
        design_effect=estimate.design_effect,
        imprecise=estimate.imprecise,
        method=method,
    )


def _diff_row(
    model: str, population: str, estimate: DiffEstimate, method: str
) -> dict[str, object]:
    return _summary_row(
        model,
        population,
        "excess",
        "difference",
        EXCESS_DENOMINATOR,
        estimate=estimate.value,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        method=method,
    )


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
    return _summary_row(
        model,
        population,
        "ratio_to_gt",
        "ratio",
        EXCESS_DENOMINATOR,
        estimate=estimate.value,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        method=method,
    )


def _macro_row(
    model: str, population: str, metric: str, estimate: MacroEstimate, method: str
) -> dict[str, object]:
    return _summary_row(
        model,
        population,
        metric,
        "macro",
        MACRO_DENOMINATOR,
        estimate=estimate.value,
        n=estimate.n_regions,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        method=f"{method}, unweighted mean of per-region rates",
    )
