"""Calibration metrics.

Metrics:
- Adaptive ECE (equal-mass binning)
- Reliability diagram
- Per-category ECE

Uses val_results.scores (max probabilities) — no clf needed.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import ArrayLike

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import (
    DataFrameResult,
    FigureResult,
    MetricGroupResult,
    ScalarMetric,
)
from mermaid_classifier.pyspacer.metrics._taxonomy_helpers import (
    build_ba_to_top,
    group_by_top_level,
)


def _adaptive_ece(
    confidences: ArrayLike,
    gt_indices: ArrayLike,
    est_indices: ArrayLike,
    n_bins: int = 20,
) -> tuple[float, list[dict[str, float | int]]]:
    """Compute ECE using adaptive (equal-mass) binning.

    Returns (ece, bin_data) where bin_data is a list of dicts with
    per-bin statistics.
    """
    confidences = np.asarray(confidences, dtype=float)
    gt_indices = np.asarray(gt_indices)
    est_indices = np.asarray(est_indices)
    accuracies = (est_indices == gt_indices).astype(float)

    order = np.argsort(confidences)
    confidences = confidences[order]
    accuracies = accuracies[order]

    n = len(confidences)
    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)

    bin_data = []
    ece = 0.0
    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        if start == end:
            continue
        bin_conf = confidences[start:end]
        bin_acc = accuracies[start:end]

        avg_conf = bin_conf.mean()
        avg_acc = bin_acc.mean()
        count = end - start

        ece += abs(avg_acc - avg_conf) * count / n
        bin_data.append(
            {
                "avg_confidence": float(avg_conf),
                "avg_accuracy": float(avg_acc),
                "count": int(count),
                "conf_min": float(bin_conf.min()),
                "conf_max": float(bin_conf.max()),
            }
        )

    return ece, bin_data


def compute_calibration(ctx: MetricsContext) -> MetricGroupResult:
    """Compute calibration metrics: ECE, reliability diagram, per-category ECE."""
    val_results = ctx.val_results
    result = MetricGroupResult()

    # Overall adaptive ECE.
    ece, bin_data = _adaptive_ece(val_results.scores, val_results.gt, val_results.est, n_bins=20)

    result.scalars.append(ScalarMetric(name="ece", value=ece))

    # Per-bin details DataFrame.
    rows = []
    for i, b in enumerate(bin_data):
        gap = b["avg_confidence"] - b["avg_accuracy"]
        rows.append(
            {
                "bin": i + 1,
                "conf_min": b["conf_min"],
                "conf_max": b["conf_max"],
                "avg_confidence": b["avg_confidence"],
                "avg_accuracy": b["avg_accuracy"],
                "gap": gap,
                "count": b["count"],
            }
        )
    result.dataframes.append(
        DataFrameResult(
            df=pd.DataFrame(rows),
            artifact_path="calibration/per_bin_details",
        )
    )

    # Reliability diagram figure.
    fig = _plot_reliability_diagram(ece, bin_data)
    result.figures.append(
        FigureResult(fig=fig, artifact_path="calibration/reliability_diagram.png")
    )

    # Per-category ECE.
    # val_results.classes is list[LabelId] (int|str); MERMAID always uses str.
    classes_str: list[str] = val_results.classes  # pyright: ignore[reportAssignmentType]
    ba_to_top = ctx.ba_to_top or build_ba_to_top(classes_str, ctx.ba_library)
    all_indices = list(range(len(val_results.gt)))
    groups = group_by_top_level(
        all_indices, val_results.gt, classes_str, ba_to_top, ctx.ba_library, min_samples=30
    )

    scores = np.asarray(val_results.scores)
    gt_arr = np.asarray(val_results.gt)
    est_arr = np.asarray(val_results.est)

    cat_rows = []
    for group in groups:
        idx = np.array(group["indices"])
        n = group["n_samples"]
        n_bins_cat = min(20, max(2, n // 10))
        cat_ece, _ = _adaptive_ece(scores[idx], gt_arr[idx], est_arr[idx], n_bins=n_bins_cat)
        cat_acc = float((est_arr[idx] == gt_arr[idx]).mean())
        cat_conf = float(scores[idx].mean())
        cat_rows.append(
            {
                "category": group["name"],
                "ece": cat_ece,
                "accuracy": cat_acc,
                "avg_confidence": cat_conf,
                "n_samples": n,
            }
        )

    cat_rows.sort(key=lambda r: r["ece"], reverse=True)
    result.dataframes.append(
        DataFrameResult(
            df=pd.DataFrame(cat_rows)
            if cat_rows
            else pd.DataFrame(
                columns=["category", "ece", "accuracy", "avg_confidence", "n_samples"]  # pyright: ignore[reportArgumentType]
            ),
            artifact_path="calibration/per_category_ece",
        )
    )

    return result


def _plot_reliability_diagram(
    ece: float,
    bin_data: list[dict[str, float | int]],
) -> Figure:
    """Plot reliability diagram with accuracy bars + overconfidence gap."""
    fig, ax = plt.subplots(figsize=(7, 5))

    confs = [b["avg_confidence"] for b in bin_data]
    accs = [b["avg_accuracy"] for b in bin_data]
    counts = [b["count"] for b in bin_data]
    widths = [b["conf_max"] - b["conf_min"] for b in bin_data]
    lefts = [b["conf_min"] for b in bin_data]

    # Accuracy bars.
    ax.bar(
        lefts,
        accs,
        width=widths,
        align="edge",
        color="#1976d2",
        edgecolor="white",
        alpha=0.8,
        label="Accuracy",
    )

    # Overconfidence gap overlay.
    for left, w, acc, conf in zip(lefts, widths, accs, confs, strict=False):
        if conf > acc:
            ax.bar(
                left,
                conf - acc,
                bottom=acc,
                width=w,
                align="edge",
                color="#d32f2f",
                alpha=0.3,
                edgecolor="white",
            )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.legend(
        handles=[
            Patch(facecolor="#1976d2", alpha=0.8, label="Accuracy"),
            Patch(facecolor="#d32f2f", alpha=0.3, label="Overconfidence gap"),
            Line2D([], [], color="k", linestyle="--", label="Perfect calibration"),
        ],
        loc="upper left",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram — ECE = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # Inset: sample count per bin.
    inset = ax.inset_axes((0.55, 0.05, 0.4, 0.25))
    inset.bar(range(len(counts)), counts, color="#666666", alpha=0.6)
    inset.set_ylabel("n", fontsize=8)
    inset.set_xlabel("Bin", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.set_title("Samples per bin", fontsize=8)

    plt.tight_layout()
    return fig
