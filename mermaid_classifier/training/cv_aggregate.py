"""Aggregate held-out predictions from the K cross-validation fold runs.

Each fold run logs ``annotations_holdout.csv`` (its holdout rows plus
``pred_bagf``, ``pred_score``, ``in_model_classes``). The folds hold disjoint
groups, so concatenating them covers every non-test point once. Metrics use
only points whose ground-truth class the model knows. Confidence intervals
come from a cluster bootstrap that resamples whole groups (``site:project_id``)
with replacement, because points from one source are not independent.

Macro metrics average over the classes present in the ground truth of the
full set, a fixed class list across resamples. A class never predicted has
precision 0. A prediction outside the ground-truth class list counts against
recall and accuracy.
"""

from __future__ import annotations

import collections
import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

POINT_KEY = ["bucket", "feature_vector", "row", "col"]
SITES = ("coralnet", "mermaid")
PER_CLASS_METRICS = ("recall", "precision", "f1")
SCALAR_METRICS = ("accuracy", "macro_recall", "macro_precision", "macro_f1")
CI_PERCENTILES = (2.5, 97.5)
SUMMARY_COLUMNS = (
    "scope",
    "level",
    "class_key",
    "metric",
    "estimate",
    "ci_low",
    "ci_high",
    "support",
)
DEFAULT_REPS = 2000

FloatArray = npt.NDArray[np.float64]


def concat_folds(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-run holdout tables, tagging each row with its `run_id`."""
    parts = [df.assign(run_id=run_id) for run_id, df in frames.items()]
    combined = pd.concat(parts, ignore_index=True)
    points_per_run = combined.groupby(POINT_KEY)["run_id"].nunique()
    shared = cast(pd.Series, points_per_run[points_per_run > 1])
    if len(shared) > 0:
        raise ValueError(
            f"{len(shared)} point(s) appear in more than one run, so the folds are not"
            f" disjoint. First: {shared.index[0]}."
        )
    return combined


def _fold_number(run_id: str, value: str | None) -> int:
    if value is None or not value.isascii() or not value.isdigit():
        raise ValueError(
            f"run {run_id} has holdout_fold={value!r}; a CV fold run needs an integer"
            f" holdout_fold param (the 'test' run is not a fold)."
        )
    return int(value)


def combine_cv_runs(
    runs: Mapping[str, tuple[str | None, bytes, pd.DataFrame]], folds: int
) -> pd.DataFrame:
    """
    Validate that `runs` are exactly the K folds of one grouped selection, then concatenate.

    `runs` maps run id to (its `holdout_fold` param, its `holdout_groups.csv`
    bytes, its holdout table). Raises ValueError unless the fold params are
    exactly 0..folds-1 once each, every run used identical group bytes, and no
    group (`site:project_id`) is held out by two runs.
    """
    fold_of = {run_id: _fold_number(run_id, fold) for run_id, (fold, _, _) in runs.items()}
    counts = collections.Counter(fold_of.values())
    duplicated = sorted(f for f, n in counts.items() if n > 1)
    if duplicated:
        raise ValueError(f"holdout_fold {duplicated} appears in more than one run.")
    if set(counts) != set(range(folds)):
        raise ValueError(
            f"Runs cover folds {sorted(counts)}, expected exactly folds 0..{folds - 1}."
        )
    if len({groups for _, groups, _ in runs.values()}) > 1:
        raise ValueError("The runs' holdout_groups.csv artifacts differ.")
    owner: dict[str, str] = {}
    for run_id, (_, _, df) in runs.items():
        for cluster in (df["site"] + ":" + df["project_id"].astype(str)).unique():
            if cluster in owner:
                raise ValueError(
                    f"Group {cluster} is held out by both run {owner[cluster]} and {run_id}."
                )
            owner[cluster] = run_id
    return concat_folds({run_id: df for run_id, (_, _, df) in runs.items()})


def _gt_column(df: pd.DataFrame) -> pd.Series:
    if "gt_bagf" in df.columns:
        return cast(pd.Series, df["gt_bagf"].astype(str))
    ba = cast(pd.Series, df["benthic_attribute_id"].astype(str))
    return ba + "::" + cast(pd.Series, df["growth_form_id"].astype(str))


def _known_mask(df: pd.DataFrame) -> pd.Series:
    """`in_model_classes` as booleans; a missing prediction counts as unknown."""
    return df["in_model_classes"].astype(str).str.lower() == "true"


@dataclasses.dataclass(frozen=True)
class ClusterCounts:
    """Per-cluster sufficient statistics: rows are clusters, columns are classes."""

    classes: list[str]
    tp: FloatArray  # (S, K) correct predictions of class k
    gt: FloatArray  # (S, K) points whose ground truth is class k
    pred: FloatArray  # (S, K) predictions of class k
    n: FloatArray  # (S,) points


def cluster_counts(
    gt: Sequence[str], pred: Sequence[str], clusters: Sequence[str]
) -> ClusterCounts:
    gt_arr = np.asarray(gt, dtype=str)
    pred_arr = np.asarray(pred, dtype=str)
    cluster_arr = np.asarray(clusters, dtype=str)
    classes = sorted(set(gt_arr.tolist()) | set(pred_arr.tolist()))
    cluster_ids = sorted(set(cluster_arr.tolist()))
    cls_idx = {c: i for i, c in enumerate(classes)}
    cl_idx = {c: i for i, c in enumerate(cluster_ids)}
    s = np.array([cl_idx[c] for c in cluster_arr], dtype=int)
    g = np.array([cls_idx[c] for c in gt_arr], dtype=int)
    p = np.array([cls_idx[c] for c in pred_arr], dtype=int)
    shape = (len(cluster_ids), len(classes))
    tp = np.zeros(shape)
    gt_counts = np.zeros(shape)
    pred_counts = np.zeros(shape)
    hit = p == g
    np.add.at(tp, (s[hit], g[hit]), 1)
    np.add.at(gt_counts, (s, g), 1)
    np.add.at(pred_counts, (s, p), 1)
    n = np.bincount(s, minlength=len(cluster_ids)).astype(float)
    return ClusterCounts(classes, tp, gt_counts, pred_counts, n)


def _metric_arrays(
    tp: FloatArray, gt: FloatArray, pred: FloatArray, n: FloatArray
) -> dict[str, FloatArray]:
    """Metrics over a leading resample axis: inputs are (R, K) and (R,)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(gt > 0, tp / gt, np.nan)
        precision = np.where(pred > 0, tp / pred, 0.0)
        denom = precision + np.nan_to_num(recall)
        f1 = np.where(denom > 0, 2 * precision * np.nan_to_num(recall) / denom, 0.0)
        f1 = np.where(gt > 0, f1, np.nan)
        precision_for_macro = np.where(gt > 0, precision, np.nan)
        accuracy = tp.sum(axis=-1) / n
    with warnings.catch_warnings():
        # A resample with no class in the ground truth has no macro average.
        warnings.simplefilter("ignore", RuntimeWarning)
        macros = {
            "macro_recall": np.nanmean(recall, axis=-1),
            "macro_precision": np.nanmean(precision_for_macro, axis=-1),
            "macro_f1": np.nanmean(f1, axis=-1),
        }
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        **macros,
    }


def point_estimates(c: ClusterCounts) -> dict[str, FloatArray | float]:
    """Metrics on the full set: per-class arrays (K,) and scalar floats."""
    m = _metric_arrays(c.tp.sum(0)[None], c.gt.sum(0)[None], c.pred.sum(0)[None], c.n.sum()[None])
    out: dict[str, FloatArray | float] = {k: m[k][0] for k in PER_CLASS_METRICS}
    out.update({k: float(m[k][0]) for k in SCALAR_METRICS})
    return out


def cluster_bootstrap_intervals(
    c: ClusterCounts, reps: int = DEFAULT_REPS, seed: int = 0
) -> tuple[dict[str, FloatArray | float], dict[str, FloatArray | float]]:
    """95% percentile (low, high) intervals from resampling whole clusters."""
    n_clusters = len(c.n)
    rng = np.random.default_rng(seed)
    # (R, S): how many times each cluster appears in each resample.
    w = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=reps).astype(float)
    m = _metric_arrays(w @ c.tp, w @ c.gt, w @ c.pred, w @ c.n)
    lo: dict[str, FloatArray | float] = {}
    hi: dict[str, FloatArray | float] = {}
    with warnings.catch_warnings():
        # A class absent from every resample's ground truth has an all-NaN column.
        warnings.simplefilter("ignore", RuntimeWarning)
        for name in PER_CLASS_METRICS:
            lo[name], hi[name] = np.nanpercentile(m[name], CI_PERCENTILES, axis=0)
        for name in SCALAR_METRICS:
            low, high = np.nanpercentile(m[name], CI_PERCENTILES)
            lo[name], hi[name] = float(low), float(high)
    return lo, hi


def _scopes(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    scopes: list[tuple[str, pd.DataFrame]] = [("all", df)]
    for site in SITES:
        if (df["site"] == site).any():
            scopes.append((site, cast(pd.DataFrame, df[df["site"] == site])))
    return scopes


def unknown_class_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per scope: holdout points, those whose class the model lacks, and groups."""
    rows = []
    for scope, sub in _scopes(df):
        known = _known_mask(sub)
        rows.append(
            {
                "scope": scope,
                "n_points": len(sub),
                "n_unknown_class": int((~known).sum()),
                "n_clusters": int((sub["site"] + ":" + sub["project_id"].astype(str)).nunique()),
            }
        )
    return pd.DataFrame(rows)


def _metric_row(
    scope: str,
    level: str,
    class_key: str,
    metric: str,
    values: tuple[dict[str, FloatArray | float], ...],
    name: str,
    index: int | None,
    support: int,
) -> dict[str, object]:
    def pick(source: dict[str, FloatArray | float]) -> float:
        value = source[name]
        return float(value) if index is None else float(np.asarray(value)[index])

    est, lo, hi = values
    return {
        "scope": scope,
        "level": level,
        "class_key": class_key,
        "metric": metric,
        "estimate": pick(est),
        "ci_low": pick(lo),
        "ci_high": pick(hi),
        "support": support,
    }


def summarize(df: pd.DataFrame, reps: int = DEFAULT_REPS, seed: int = 0) -> pd.DataFrame:
    """
    Long table of metrics with bootstrap CIs per scope (all, coralnet, mermaid).

    Columns: scope, level (overall, macro, class), class_key, metric,
    estimate, ci_low, ci_high, support (points, or ground-truth points
    for a class).
    """
    rows: list[dict[str, object]] = []
    for scope, sub in _scopes(df):
        known = cast(pd.DataFrame, sub[_known_mask(sub)])
        if known.empty:
            continue
        clusters = known["site"] + ":" + known["project_id"].astype(str)
        counts = cluster_counts(
            _gt_column(known).tolist(),
            known["pred_bagf"].astype(str).tolist(),
            clusters.tolist(),
        )
        est = point_estimates(counts)
        lo, hi = cluster_bootstrap_intervals(counts, reps=reps, seed=seed)
        values = (est, lo, hi)

        rows.append(
            _metric_row(scope, "overall", "", "accuracy", values, "accuracy", None, len(known))
        )
        for metric in PER_CLASS_METRICS:
            rows.append(
                _metric_row(scope, "macro", "", metric, values, f"macro_{metric}", None, len(known))
            )
        gt_totals = counts.gt.sum(0)
        for i, class_key in enumerate(counts.classes):
            if gt_totals[i] == 0:
                continue
            for metric in PER_CLASS_METRICS:
                rows.append(
                    _metric_row(
                        scope, "class", class_key, metric, values, metric, i, int(gt_totals[i])
                    )
                )
    return pd.DataFrame(rows, columns=pd.Index(SUMMARY_COLUMNS))


def _fmt(row: pd.Series) -> str:
    return f"{row['estimate']:.3f} ({row['ci_low']:.3f}-{row['ci_high']:.3f})"


def to_markdown(summary: pd.DataFrame, unknowns: pd.DataFrame | None = None) -> str:
    """One section per scope: overall and macro metrics, then a per-class table."""
    lines: list[str] = []
    for scope in summary["scope"].drop_duplicates().tolist():
        sub = cast(pd.DataFrame, summary[summary["scope"] == scope])
        lines += [f"## {scope}", ""]
        if unknowns is not None:
            info = cast(pd.DataFrame, unknowns[unknowns["scope"] == scope])
            if not info.empty:
                r = info.iloc[0]
                lines += [
                    f"{r['n_points']} points, {r['n_unknown_class']} with a class the model"
                    f" lacks (excluded), {r['n_clusters']} clusters.",
                    "",
                ]
        lines += ["| metric | estimate (95% CI) |", "| --- | --- |"]
        for _, r in cast(pd.DataFrame, sub[sub["level"] != "class"]).iterrows():
            name = r["metric"] if r["level"] == "overall" else f"macro {r['metric']}"
            lines.append(f"| {name} | {_fmt(r)} |")
        lines += [
            "",
            "| class | support | recall | precision | f1 |",
            "| --- | --- | --- | --- | --- |",
        ]
        per_class = cast(pd.DataFrame, sub[sub["level"] == "class"])
        for class_key in per_class["class_key"].drop_duplicates().tolist():
            rows = cast(pd.DataFrame, per_class[per_class["class_key"] == class_key]).set_index(
                "metric"
            )
            cells = " | ".join(_fmt(cast(pd.Series, rows.loc[m])) for m in PER_CLASS_METRICS)
            lines.append(f"| {class_key} | {int(rows['support'].iloc[0])} | {cells} |")
        lines.append("")
    return "\n".join(lines)
