"""Telemetry extraction from MLflow runs for the autoresearch loop.

After each training run the harness needs more than `balanced_accuracy`
to reason about what to try next. This module pulls every scalar
metric, every small/textual artifact, and a compact summary of each
per-epoch curve from the most recent MLflow run, and renders it as a
single markdown blob suitable for inlining in the next Claude prompt.

Single entry point: ``extract_run_telemetry``.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────

# Headline scalars surfaced as their own ``results.tsv`` columns and
# rendered at the top of the markdown summary.
HEADLINE_KEYS = (
    "balanced_accuracy",
    "mcc",
    "ece",
    "top_5_accuracy",
    "cross_branch_error_rate",
    "within_branch_error_rate",
    "precision_macro",
    "recall_macro",
)

# Per-epoch metrics we want a min/max/last summary for.
EPOCH_KEYS = (
    "epoch/ref_accuracy",
    "epoch/val_accuracy",
    "epoch/val_loss",
    "epoch/training_loss",
)

# Artifacts logged via mlflow.log_text after DuckDB CSV conversion all
# end up at ``<filestem>.csv``. Other artifacts use explicit suffixes.
# Each entry is (artifact_path, renderer_key).
RENDERED_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("metrics_overall.yaml", "metrics_overall"),
    ("metrics_per_label.csv", "metrics_per_label"),
    ("confusion_matrix/percents.csv", "confusion_top_pairs"),
    ("calibration/per_bin_details.csv", "calibration_bins"),
    ("calibration/per_category_ece.csv", "calibration_per_category"),
    ("taxonomic/error_attribution.csv", "error_attribution"),
    ("taxonomic/top_level_confusions.csv", "top_level_confusions"),
    ("taxonomic/gf_precision_recall_f1.csv", "gf_prf"),
    ("per_source/metrics.csv", "per_source"),
    ("cover/per_class_cover_metrics.csv", "cover"),
    ("probability/per_category_log_loss.csv", "log_loss_per_category"),
    ("ranking/per_category_topk.csv", "ranking_per_category"),
    ("ranking/hierarchical_topk.csv", "ranking_hierarchical"),
    ("profiled_sections.csv", "profiling"),
    ("weighting/per_class_weights.csv", "weighting"),
    ("subsample/per_class_counts.csv", "subsample"),
)

# Skip outright: too large or non-textual.
# valresult.json (full predictions), confusion_matrix/frequencies.csv
# (redundant with percents), all *.png plots.

# Scalar metrics we never want in the prompt (system telemetry from
# the MLflow system metrics monitor).
_SYSTEM_METRIC_PREFIXES = ("system/",)


# ── Dataclass ──────────────────────────────────────────────────────


@dataclass
class RunTelemetry:
    run_id: str
    headline: dict[str, float] = field(default_factory=dict)
    all_scalars: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    epoch_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    artifact_sections: dict[str, str] = field(default_factory=dict)
    full_markdown: str = ""


# ── Public entry point ─────────────────────────────────────────────


def extract_run_telemetry(
    experiment_name: str = "autoresearch",
    run_id: str | None = None,
) -> RunTelemetry | None:
    """Extract a markdown-rendered telemetry summary for an MLflow run.

    By default targets the most recent FINISHED run in the named
    experiment. Pass ``run_id`` to target a specific run. Returns
    ``None`` if no matching run is found.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    if run_id is None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(
                f"MLflow experiment '{experiment_name}' not found")
            return None
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attribute.start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            logger.warning(
                f"No FINISHED runs in experiment '{experiment_name}'")
            return None
        run_row = runs.iloc[0]
        run_id = run_row["run_id"]
        all_scalars = _flatten_metrics(run_row)
        params = _flatten_params(run_row)
    else:
        run = client.get_run(run_id)
        all_scalars = {
            k: float(v) for k, v in run.data.metrics.items()
            if not _is_system_metric(k)
        }
        params = dict(run.data.params)

    headline = {k: all_scalars[k] for k in HEADLINE_KEYS if k in all_scalars}

    epoch_summary = _summarize_epoch_curves(client, run_id)

    artifact_sections: dict[str, str] = {}
    for artifact_path, renderer_key in RENDERED_ARTIFACTS:
        try:
            block = _render_artifact(run_id, artifact_path, renderer_key)
        except Exception as e:
            block = f"_(unavailable: {type(e).__name__}: {e})_"
        if block:
            artifact_sections[artifact_path] = block

    full_markdown = _render_full_markdown(
        run_id=run_id,
        headline=headline,
        all_scalars=all_scalars,
        params=params,
        epoch_summary=epoch_summary,
        artifact_sections=artifact_sections,
    )

    return RunTelemetry(
        run_id=run_id,
        headline=headline,
        all_scalars=all_scalars,
        params=params,
        epoch_summary=epoch_summary,
        artifact_sections=artifact_sections,
        full_markdown=full_markdown,
    )


# ── Scalar extraction ──────────────────────────────────────────────


def _is_system_metric(key: str) -> bool:
    return any(key.startswith(p) for p in _SYSTEM_METRIC_PREFIXES)


def _flatten_metrics(run_row: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in run_row.index:
        if not col.startswith("metrics."):
            continue
        key = col[len("metrics."):]
        if _is_system_metric(key):
            continue
        value = run_row[col]
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if f != f:  # NaN check.
            continue
        out[key] = f
    return out


def _flatten_params(run_row: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for col in run_row.index:
        if not col.startswith("params."):
            continue
        key = col[len("params."):]
        value = run_row[col]
        if value is None or (isinstance(value, float) and value != value):
            continue
        out[key] = str(value)
    return out


# ── Epoch curve summary ────────────────────────────────────────────


def _summarize_epoch_curves(
    client: Any, run_id: str,
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in EPOCH_KEYS:
        try:
            history = client.get_metric_history(run_id, key)
        except Exception:
            continue
        if not history:
            continue
        values = [(m.step, m.value) for m in history]
        values.sort()
        steps = [s for s, _ in values]
        vals = [v for _, v in values]
        argmin = steps[vals.index(min(vals))]
        argmax = steps[vals.index(max(vals))]
        summary[key] = {
            "first": vals[0],
            "last": vals[-1],
            "min": min(vals),
            "max": max(vals),
            "argmin_step": argmin,
            "argmax_step": argmax,
            "n_steps": len(vals),
        }
    return summary


# ── Artifact rendering ─────────────────────────────────────────────


def _render_artifact(
    run_id: str, artifact_path: str, renderer_key: str,
) -> str:
    """Download an artifact and render it as a compact markdown block.

    Each renderer returns either a non-empty markdown string or "" if
    the artifact's contents weren't useful.
    """
    import mlflow
    import pandas as pd

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path)

    if artifact_path.endswith(".yaml") or artifact_path.endswith(".json"):
        # Read as text, truncate aggressively if huge.
        with open(local_path) as f:
            text = f.read()
        if len(text) > 4000:
            text = text[:4000] + "\n…(truncated)…"
        return f"```\n{text}\n```"

    df = pd.read_csv(local_path)

    renderer = _RENDERERS.get(renderer_key, _render_default_csv)
    return renderer(df)


def _render_default_csv(df: Any) -> str:
    """Fallback: render the whole table if small, else summary stats."""
    if len(df) <= 30:
        return _df_to_md(df)
    return f"_(too large: {len(df)} rows; showing first 20)_\n\n" + _df_to_md(df.head(20))


def _render_metrics_overall(df: Any) -> str:
    return _df_to_md(df)


def _render_metrics_per_label(df: Any) -> str:
    """Worst 10 by recall + best 10 by recall + summary stats."""
    if "recall" not in df.columns:
        return _render_default_csv(df)
    keep_cols = [c for c in (
        "label", "bagf_id", "precision", "recall", "f1", "n_samples",
    ) if c in df.columns]
    df_keep = df[keep_cols].copy()
    df_sorted = df_keep.sort_values("recall")
    worst = df_sorted.head(10)
    best = df_sorted.tail(10).iloc[::-1]
    parts = ["**10 worst classes by recall:**", _df_to_md(worst), ""]
    parts += ["**10 best classes by recall:**", _df_to_md(best), ""]

    def _stats(col: str) -> str:
        if col not in df.columns:
            return ""
        s = df[col].astype(float)
        return (
            f"- {col}: median={s.median():.4f}, "
            f"p10={s.quantile(0.1):.4f}, p90={s.quantile(0.9):.4f}"
        )

    parts.append("**Summary stats across all classes:**")
    for col in ("precision", "recall", "f1", "n_samples"):
        line = _stats(col)
        if line:
            parts.append(line)
    return "\n".join(parts)


def _render_confusion_top_pairs(df: Any) -> str:
    """Top-15 most-confused (true → predicted) pairs from percent matrix.

    The percent matrix's first column is the true-class label and the
    remaining columns are predicted-class headers. Off-diagonal entries
    are misclassification percentages already on a 0–100 scale.
    """
    if df.empty or len(df.columns) < 2:
        return ""
    label_col = df.columns[0]
    classes = list(df[label_col].astype(str))
    class_set = set(classes)
    pred_cols = [c for c in df.columns[1:] if str(c) in class_set]
    pairs: list[tuple[str, str, float]] = []
    for i, true_class in enumerate(classes):
        for pred_class in pred_cols:
            if str(pred_class) == true_class:
                continue
            try:
                v = float(df.iloc[i][pred_class])
            except (TypeError, ValueError):
                continue
            if v != v or v <= 0:
                continue
            pairs.append((true_class, str(pred_class), v))
    pairs.sort(key=lambda t: t[2], reverse=True)
    top = pairs[:15]
    if not top:
        return ""
    lines = [
        "**Top-15 confused pairs (true → predicted, % of true class):**",
        "| true | predicted | pct |",
        "|------|-----------|-----|",
    ]
    for t, p, v in top:
        lines.append(f"| {t} | {p} | {v:.1f}% |")
    return "\n".join(lines)


def _render_per_source(df: Any) -> str:
    if "balanced_accuracy" not in df.columns:
        return _render_default_csv(df)
    df_sorted = df.sort_values("balanced_accuracy")
    parts = [
        "**5 worst sources by balanced_accuracy:**",
        _df_to_md(df_sorted.head(5)),
        "",
        "**5 best sources by balanced_accuracy:**",
        _df_to_md(df_sorted.tail(5).iloc[::-1]),
    ]
    return "\n".join(parts)


def _render_cover(df: Any) -> str:
    if "bias" not in df.columns:
        return _render_default_csv(df)
    df_sorted = df.sort_values("bias")
    parts = [
        "**5 most-negatively-biased classes (under-predicted cover):**",
        _df_to_md(df_sorted.head(5)),
        "",
        "**5 most-positively-biased classes (over-predicted cover):**",
        _df_to_md(df_sorted.tail(5).iloc[::-1]),
    ]
    if "r_squared" in df.columns:
        s = df["r_squared"].astype(float)
        parts += [
            "",
            f"**r_squared distribution:** median={s.median():.3f}, "
            f"p10={s.quantile(0.1):.3f}, p90={s.quantile(0.9):.3f}",
        ]
    return "\n".join(parts)


def _render_error_attribution(df: Any) -> str:
    if "pct_of_errors" in df.columns:
        df_sorted = df.sort_values("pct_of_errors", ascending=False)
        return "**Top-10 LCA error nodes:**\n\n" + _df_to_md(df_sorted.head(10))
    return _render_default_csv(df)


def _render_profiling(df: Any) -> str:
    if "seconds" not in df.columns:
        return _render_default_csv(df)
    df_sorted = df.sort_values("seconds", ascending=False)
    keep = [c for c in ("name", "seconds", "hms") if c in df.columns]
    return "**Top-5 sections by wall time:**\n\n" + _df_to_md(df_sorted[keep].head(5))


def _render_weighting_summary(df: Any) -> str:
    weight_col = next(
        (c for c in df.columns if "weight" in c.lower()), None)
    if weight_col is None:
        return _render_default_csv(df)
    s = df[weight_col].astype(float)
    return (
        f"**Per-class weight distribution ({weight_col}, n={len(s)}):**\n"
        f"- min={s.min():.4f}, p25={s.quantile(0.25):.4f}, "
        f"median={s.median():.4f}, p75={s.quantile(0.75):.4f}, "
        f"max={s.max():.4f}\n"
        f"- sum={s.sum():.2f}, max/min ratio="
        f"{(s.max() / s.min()) if s.min() > 0 else float('inf'):.1f}"
    )


def _render_subsample_summary(df: Any) -> str:
    count_col = next(
        (c for c in df.columns if "count" in c.lower() or c == "n"), None)
    if count_col is None:
        return _render_default_csv(df)
    s = df[count_col].astype(float)
    return (
        f"**Per-class realized counts ({count_col}, n_classes={len(s)}):**\n"
        f"- min={int(s.min())}, p25={int(s.quantile(0.25))}, "
        f"median={int(s.median())}, p75={int(s.quantile(0.75))}, "
        f"max={int(s.max())}\n"
        f"- total={int(s.sum())}"
    )


_RENDERERS: dict[str, Any] = {
    "metrics_overall": _render_metrics_overall,
    "metrics_per_label": _render_metrics_per_label,
    "confusion_top_pairs": _render_confusion_top_pairs,
    "calibration_bins": _render_default_csv,
    "calibration_per_category": _render_default_csv,
    "error_attribution": _render_error_attribution,
    "top_level_confusions": _render_default_csv,
    "gf_prf": _render_default_csv,
    "per_source": _render_per_source,
    "cover": _render_cover,
    "log_loss_per_category": _render_default_csv,
    "ranking_per_category": _render_default_csv,
    "ranking_hierarchical": _render_default_csv,
    "profiling": _render_profiling,
    "weighting": _render_weighting_summary,
    "subsample": _render_subsample_summary,
}


# ── Markdown helpers ───────────────────────────────────────────────


def _df_to_md(df: Any) -> str:
    """Render a small DataFrame as a markdown table without pulling in tabulate."""
    if len(df) == 0:
        return "_(empty)_"
    cols = list(df.columns)
    buf = io.StringIO()
    buf.write("| " + " | ".join(str(c) for c in cols) + " |\n")
    buf.write("|" + "|".join(["---"] * len(cols)) + "|\n")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        buf.write("| " + " | ".join(cells) + " |\n")
    return buf.getvalue().rstrip()


def _render_full_markdown(
    *,
    run_id: str,
    headline: dict[str, float],
    all_scalars: dict[str, float],
    params: dict[str, str],
    epoch_summary: dict[str, dict[str, float]],
    artifact_sections: dict[str, str],
) -> str:
    parts: list[str] = []
    parts.append(f"_run_id: `{run_id}`_")
    parts.append("")

    parts.append("## Headline metrics")
    if headline:
        parts.append(_kv_table(headline))
    else:
        parts.append("_(no headline metrics available)_")
    parts.append("")

    parts.append("## All scalar metrics")
    non_headline = {
        k: v for k, v in sorted(all_scalars.items()) if k not in headline}
    if non_headline:
        parts.append(_kv_table(non_headline))
    else:
        parts.append("_(none)_")
    parts.append("")

    parts.append("## Run params")
    if params:
        parts.append(_kv_table(params, value_fmt="{}"))
    else:
        parts.append("_(none)_")
    parts.append("")

    parts.append("## Per-epoch curves")
    if epoch_summary:
        lines = ["| metric | first | last | min | min_step | max | max_step | n_steps |",
                 "|---|---|---|---|---|---|---|---|"]
        for k, s in epoch_summary.items():
            lines.append(
                f"| {k} | {s['first']:.4g} | {s['last']:.4g} | "
                f"{s['min']:.4g} | {int(s['argmin_step'])} | "
                f"{s['max']:.4g} | {int(s['argmax_step'])} | "
                f"{int(s['n_steps'])} |")
        parts.append("\n".join(lines))
    else:
        parts.append("_(no per-epoch metrics logged)_")
    parts.append("")

    for artifact_path, _ in RENDERED_ARTIFACTS:
        block = artifact_sections.get(artifact_path)
        if not block:
            continue
        parts.append(f"## {artifact_path}")
        parts.append(block)
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _kv_table(d: dict, value_fmt: str = "{:.6g}") -> str:
    if not d:
        return "_(empty)_"
    lines = ["| key | value |", "|---|---|"]
    for k, v in d.items():
        if isinstance(v, (int, float)) and value_fmt != "{}":
            lines.append(f"| {k} | {value_fmt.format(v)} |")
        else:
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines)
