"""Paired comparison of two training runs on one source-grouped holdout.

Each run logs ``annotations_holdout.csv``. Two runs that held out the same
groups score the same points, so their difference has a paired bootstrap:
one draw of cluster weights (``site:project_id``) evaluates both runs, and the
interval is taken over the per-resample difference B - A. Points the two runs
score differently (different ground truth, or a point only one run holds)
make the comparison invalid, so alignment rejects them by default.

Metrics follow ``cv_aggregate``: they use only points whose ground-truth class
both models know, and macro metrics average over the classes present in the
ground truth. Cover bias is reported two ways per class: pooled,
``(predicted - true) / points`` over the whole set, and per image, the mean
over images of (predicted share - true share) as in
``pyspacer/metrics/cover.py``. Overall metrics (accuracy and the macro
averages) carry ``class_key`` ``OVERALL_KEY`` (``"__overall__"``).
"""

from __future__ import annotations

import collections
import dataclasses
import warnings
from collections.abc import Collection, Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from mermaid_classifier.training.cv_aggregate import (
    CI_PERCENTILES,
    DEFAULT_REPS,
    POINT_KEY,
    SCALAR_METRICS,
    ClusterCounts,
    FloatArray,
    _gt_column,  # pyright: ignore[reportPrivateUsage]
    _known_mask,  # pyright: ignore[reportPrivateUsage]
    _metric_arrays,  # pyright: ignore[reportPrivateUsage]
    _scopes,  # pyright: ignore[reportPrivateUsage]
    bootstrap_weights,
    cluster_counts,
)

OVERALL_KEY = "__overall__"
CLASS_METRICS = ("recall", "precision", "f1", "cover_bias_pooled", "cover_bias_image")
MATCH_COLUMNS = ("gt_bagf", "site", "project_id", "image_id")
SUMMARY_COLUMNS = (
    "scope",
    "level",
    "class_key",
    "metric",
    "est_a",
    "est_b",
    "diff",
    "diff_ci_low",
    "diff_ci_high",
    "a_ci_low",
    "a_ci_high",
    "b_ci_low",
    "b_ci_high",
    "support",
)


@dataclasses.dataclass(frozen=True)
class AlignmentReport:
    """Point counts from aligning two holdout tables."""

    n_a: int
    n_b: int
    only_a: int
    only_b: int
    relabelled: int
    unknown: int  # shared, agreeing points whose class either model lacks
    aligned: int

    def to_text(self) -> str:
        return "\n".join(f"{f.name}: {getattr(self, f.name)}" for f in dataclasses.fields(self))


def _prepare(df: pd.DataFrame, run: str) -> pd.DataFrame:
    out = pd.DataFrame({k: df[k].astype(str) for k in POINT_KEY})
    duplicated = int(out.duplicated().sum())
    if duplicated:
        raise ValueError(f"Run {run} has {duplicated} duplicated point key(s) {POINT_KEY}.")
    out["gt_bagf"] = _gt_column(df)
    for column in ("site", "project_id", "image_id"):
        out[column] = df[column].astype(str)
    out["pred"] = df["pred_bagf"].astype(str)
    out["known"] = _known_mask(df)
    return out


def align_holdout(
    a: pd.DataFrame, b: pd.DataFrame, *, allow_mismatch: bool = False
) -> tuple[pd.DataFrame, AlignmentReport]:
    """
    Join two runs' holdout tables point by point.

    Points match on (bucket, feature_vector, row, col) and must agree on
    ground truth, site, project and image. Raises ValueError on a point only
    one run holds or a relabelled point, unless `allow_mismatch`, which keeps
    the agreeing intersection. Returns the points both models know, with
    columns POINT_KEY, site, project_id, image_id, gt_bagf, pred_a, pred_b.
    """
    pa, pb = _prepare(a, "A"), _prepare(b, "B")
    merged = pa.merge(pb, on=POINT_KEY, how="outer", suffixes=("_a", "_b"), indicator=True)
    side = merged["_merge"].astype(str)
    only_a = int((side == "left_only").sum())
    only_b = int((side == "right_only").sum())
    both = cast(pd.DataFrame, merged[side == "both"])
    differs = np.zeros(len(both), dtype=bool)
    for column in MATCH_COLUMNS:
        differs |= (both[f"{column}_a"] != both[f"{column}_b"]).to_numpy()
    relabelled = int(differs.sum())

    if not allow_mismatch and (only_a or only_b or relabelled):
        parts = []
        if only_a:
            parts.append(f"{only_a} point(s) only in A")
        if only_b:
            parts.append(f"{only_b} point(s) only in B")
        if relabelled:
            parts.append(f"{relabelled} point(s) relabelled (ground truth, site, project or image)")
        raise ValueError("The runs' holdout points differ: " + "; ".join(parts) + ".")

    agree = cast(pd.DataFrame, both[~differs])
    known = (agree["known_a"] & agree["known_b"]).to_numpy()
    kept = cast(pd.DataFrame, agree[known])
    aligned = pd.DataFrame(
        {
            **{k: kept[k] for k in POINT_KEY},
            "site": kept["site_a"],
            "project_id": kept["project_id_a"],
            "image_id": kept["image_id_a"],
            "gt_bagf": kept["gt_bagf_a"],
            "pred_a": kept["pred_a"],
            "pred_b": kept["pred_b"],
        }
    ).reset_index(drop=True)
    report = AlignmentReport(
        n_a=len(pa),
        n_b=len(pb),
        only_a=only_a,
        only_b=only_b,
        relabelled=relabelled,
        unknown=int((~known).sum()),
        aligned=len(aligned),
    )
    return aligned, report


@dataclasses.dataclass(frozen=True)
class _RunStats:
    counts: ClusterCounts
    image_bias: FloatArray  # (S, K) per cluster, sum over images of pred share - true share
    n_images: FloatArray  # (S,)


def _run_stats(
    gt: Sequence[str],
    pred: Sequence[str],
    clusters: Sequence[str],
    images: Sequence[str],
    classes: list[str],
    weights: FloatArray,
) -> _RunStats:
    counts = cluster_counts(gt, pred, clusters, classes=classes, point_weights=weights.tolist())
    cls_idx = {c: i for i, c in enumerate(classes)}
    cl_idx = {c: i for i, c in enumerate(counts.cluster_ids)}
    image_codes, image_keys = pd.factorize(pd.Series(images))
    n_img = len(image_keys)
    g = np.array([cls_idx[c] for c in gt], dtype=int)
    p = np.array([cls_idx[c] for c in pred], dtype=int)
    true_share = np.zeros((n_img, len(classes)))
    pred_share = np.zeros((n_img, len(classes)))
    np.add.at(true_share, (image_codes, g), weights)
    np.add.at(pred_share, (image_codes, p), weights)
    image_weight = np.bincount(image_codes, weights=weights, minlength=n_img)
    diff = (pred_share - true_share) / image_weight[:, None]
    # Image keys embed the cluster, so each image lies in exactly one cluster.
    image_cluster = np.zeros(n_img, dtype=int)
    image_cluster[image_codes] = [cl_idx[c] for c in clusters]
    image_bias = np.zeros((len(counts.cluster_ids), len(classes)))
    np.add.at(image_bias, image_cluster, diff)
    n_images = np.bincount(image_cluster, minlength=len(counts.cluster_ids)).astype(float)
    return _RunStats(counts, image_bias, n_images)


def _evaluate(stats: _RunStats, w: FloatArray) -> dict[str, FloatArray]:
    """Metrics for each row of cluster weights `w` (R, S): arrays (R, K) or (R,)."""
    c = stats.counts
    tp, gt, pred, n = w @ c.tp, w @ c.gt, w @ c.pred, w @ c.n
    metrics = _metric_arrays(tp, gt, pred, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics["cover_bias_pooled"] = (pred - gt) / n[:, None]
        metrics["cover_bias_image"] = (w @ stats.image_bias) / (w @ stats.n_images)[:, None]
    return metrics


def _interval(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    with warnings.catch_warnings():
        # A class absent from every resample's ground truth has an all-NaN column.
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanpercentile(values, CI_PERCENTILES, axis=0)
    return np.asarray(low, dtype=float), np.asarray(high, dtype=float)


def _scope_rows(
    scope: str, level: str, sub: pd.DataFrame, reps: int, seed: int
) -> list[dict[str, object]]:
    gt = sub["gt_bagf"].tolist()
    clusters = (sub["site"] + ":" + sub["project_id"]).tolist()
    images = (
        sub["site"] + ":" + sub["project_id"] + "|" + sub["bucket"] + "/" + sub["feature_vector"]
    ).tolist()
    classes = sorted(set(gt) | set(sub["pred_a"]) | set(sub["pred_b"]))
    weights = sub["_weight"].to_numpy(dtype=float)
    stats_a = _run_stats(gt, sub["pred_a"].tolist(), clusters, images, classes, weights)
    stats_b = _run_stats(gt, sub["pred_b"].tolist(), clusters, images, classes, weights)

    n_clusters = len(stats_a.counts.cluster_ids)
    w = bootstrap_weights(n_clusters, reps, seed)
    full = np.ones((1, n_clusters))
    est_a, est_b = _evaluate(stats_a, full), _evaluate(stats_b, full)
    boot_a, boot_b = _evaluate(stats_a, w), _evaluate(stats_b, w)
    support = sub["gt_bagf"].value_counts().reindex(classes, fill_value=0)

    def row(class_key: str, metric: str, index: int | None, n: int) -> dict[str, object]:
        def col(values: FloatArray) -> FloatArray:
            return values if index is None else values[:, index]

        a, b = col(est_a[metric])[0], col(est_b[metric])[0]
        diff_lo, diff_hi = _interval(col(boot_b[metric]) - col(boot_a[metric]))
        a_lo, a_hi = _interval(col(boot_a[metric]))
        b_lo, b_hi = _interval(col(boot_b[metric]))
        return dict(
            zip(
                SUMMARY_COLUMNS,
                (scope, level, class_key, metric, float(a), float(b), float(b - a))
                + tuple(float(v) for v in (diff_lo, diff_hi, a_lo, a_hi, b_lo, b_hi))
                + (n,),
                strict=True,
            )
        )

    rows = [row(OVERALL_KEY, metric, None, len(sub)) for metric in SCALAR_METRICS]
    for i, class_key in enumerate(classes):
        rows += [row(class_key, metric, i, int(support[class_key])) for metric in CLASS_METRICS]
    return rows


def paired_summary(
    aligned: pd.DataFrame,
    reps: int = DEFAULT_REPS,
    seed: int = 0,
    *,
    class_map: Mapping[str, str] | None = None,
    point_weights: Sequence[float] | FloatArray | None = None,
) -> pd.DataFrame:
    """
    Long table comparing run B against run A per scope (all, coralnet, mermaid).

    `class_map` relabels ground truth and both predictions first and marks the
    rows level "top_level" (else "class"). `point_weights`, one per aligned
    row, weight every count. Support is the unweighted ground-truth point count
    of a class, or all points for an `OVERALL_KEY` row.
    """
    df = aligned.copy()
    if point_weights is None:
        df["_weight"] = 1.0
    else:
        weights = np.asarray(point_weights, dtype=float)
        if len(weights) != len(df):
            raise ValueError(f"{len(weights)} point weights for {len(df)} aligned points.")
        df["_weight"] = weights
    level = "class"
    if class_map is not None:
        level = "top_level"
        labels = set(df["gt_bagf"]) | set(df["pred_a"]) | set(df["pred_b"])
        missing = sorted(labels - set(class_map))
        if missing:
            raise ValueError(f"class_map has no entry for {missing}.")
        for column in ("gt_bagf", "pred_a", "pred_b"):
            df[column] = [class_map[v] for v in df[column]]

    rows: list[dict[str, object]] = []
    for scope, sub in _scopes(df):
        if not sub.empty:
            rows += _scope_rows(scope, level, sub, reps, seed)
    return pd.DataFrame(rows, columns=pd.Index(SUMMARY_COLUMNS))


def top_level_map(
    classes: Sequence[str],
    rollup_df: pd.DataFrame,
    top_level_ids: Collection[str] | None = None,
) -> dict[str, str]:
    """
    Map "ba::gf" class keys to their top-level key via a rollup table.

    `rollup_df` has from_ba_id, from_gf_id, to_ba_id, to_gf_id. A rollup target
    maps to itself, as does "<id>::" for each id in `top_level_ids`, which
    covers a top-level class with no rollup rows. Raises ValueError listing
    every class that does not map.
    """
    r = rollup_df.fillna("").astype(str)
    src = (r["from_ba_id"] + "::" + r["from_gf_id"]).tolist()
    dst = (r["to_ba_id"] + "::" + r["to_gf_id"]).tolist()
    mapping = {d: d for d in dst}
    mapping.update({f"{i}::": f"{i}::" for i in top_level_ids or ()})
    explicit: dict[str, str] = {}
    for s, d in zip(src, dst, strict=True):
        if explicit.setdefault(s, d) != d:
            raise ValueError(f"Rollup maps {s} to both {explicit[s]} and {d}.")
    mapping.update(explicit)
    unmapped = sorted(set(classes) - set(mapping))
    if unmapped:
        raise ValueError(f"{len(unmapped)} class(es) have no top-level rollup: {unmapped}.")
    return {c: mapping[c] for c in classes}


def real_mix_weights(per_class_counts_df: pd.DataFrame, gt: Sequence[str]) -> FloatArray:
    """
    Per-point weight available / (points in `gt` of that class) for its class.

    This undoes a per-class holdout cap, restoring the holdout's real class
    mix. The denominator counts the compared points rather than the logged
    `realized`, so each class still totals `available` when alignment keeps
    only an intersection. Per-class recall is unchanged by it, since every
    point of a class gets the same weight; precision, accuracy and cover bias
    change.
    """
    c = per_class_counts_df
    keys = (
        c["benthic_attribute_id"].fillna("").astype(str)
        + "::"
        + c["growth_form_id"].fillna("").astype(str)
    )
    available = dict(zip(keys.tolist(), c["available"].astype(float).tolist(), strict=True))
    missing = sorted(set(gt) - set(available))
    if missing:
        raise ValueError(f"Per-class counts lack class(es) {missing}.")
    present = collections.Counter(gt)
    return np.array([available[g] / present[g] for g in gt], dtype=float)


def check_group_disjointness(
    project_stats_df: pd.DataFrame,
    val_groups: set[tuple[str, str]],
    test_groups: set[tuple[str, str]],
) -> list[str]:
    """
    Problems with a run's group split, from its project_stats_train_data table.

    Each (site, project_id) group must have points in at most one of
    train+ref, val and holdout; groups with val points must be in
    `val_groups`, and groups with holdout points in `test_groups`.
    """
    problems: list[str] = []
    for r in project_stats_df.to_dict("records"):
        key = (str(r["site"]), str(r["project_id"]))
        name = f"{key[0]}:{key[1]}"
        sets = {
            "train+ref": float(r["train"]) + float(r["ref"]),
            "val": float(r["val"]),
            "holdout": float(r["holdout"]),
        }
        used = [s for s, n in sets.items() if n > 0]
        if len(used) > 1:
            problems.append(f"Group {name} has points in more than one set: {', '.join(used)}.")
        if sets["val"] > 0 and key not in val_groups:
            problems.append(f"Group {name} has val points but is not a val group.")
        if sets["holdout"] > 0 and key not in test_groups:
            problems.append(f"Group {name} has holdout points but is not a test group.")
    return problems


def _cell(r: Mapping[str, object]) -> str:
    def f(key: str) -> float:
        return float(cast(float, r[key]))

    return (
        f"{f('est_a'):.3f} → {f('est_b'):.3f}"
        f" ({f('diff'):+.3f} [{f('diff_ci_low'):+.3f}, {f('diff_ci_high'):+.3f}])"
    )


def with_class_names(summary_df: pd.DataFrame, names: Mapping[str, str]) -> pd.DataFrame:
    """`summary_df` with a `class_name` column: the name of `class_key`, else the key."""
    keys = summary_df["class_key"].astype(str).tolist()
    return summary_df.assign(class_name=[names.get(k, k) for k in keys])


def to_markdown_paired(summary_df: pd.DataFrame) -> str:
    """
    One section per scope and level: overall metrics, then per-class metrics.

    Each cell reads "A → B (B - A [95% CI of B - A])". Classes are labelled
    by the `class_name` column when present, else by `class_key`.
    """
    lines: list[str] = []
    groups = summary_df[["scope", "level"]].drop_duplicates()
    for scope, level in groups.itertuples(index=False):
        sub = cast(
            pd.DataFrame,
            summary_df[(summary_df["scope"] == scope) & (summary_df["level"] == level)],
        )
        lines += [
            f"## {scope} ({level})",
            "",
            "| metric | A → B (B - A [95% CI]) |",
            "| --- | --- |",
        ]
        overall = cast(pd.DataFrame, sub[sub["class_key"] == OVERALL_KEY])
        for r in overall.to_dict("records"):
            lines.append(f"| {r['metric']} | {_cell(r)} |")
        lines += [
            "",
            "| class | support | " + " | ".join(CLASS_METRICS) + " |",
            "| --- | --- | " + " | ".join("---" for _ in CLASS_METRICS) + " |",
        ]
        per_class = cast(pd.DataFrame, sub[sub["class_key"] != OVERALL_KEY])
        for class_key, rows in per_class.groupby("class_key", sort=False):
            by_metric = {str(r["metric"]): r for r in rows.to_dict("records")}
            support = int(cast(int, rows["support"].iloc[0]))
            label = rows["class_name"].iloc[0] if "class_name" in rows.columns else class_key
            cells = " | ".join(_cell(by_metric[m]) for m in CLASS_METRICS)
            lines.append(f"| {label} | {support} | {cells} |")
        lines.append("")
    return "\n".join(lines)
