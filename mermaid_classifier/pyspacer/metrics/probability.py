"""Probability-based metrics.

Metrics:
- Log loss (overall + per-category)

Requires val_proba and val_gt_labels in MetricsContext (pre-computed by
the coordinator from clf + dataset).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sklearn_log_loss

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


def compute_probability(ctx: MetricsContext) -> MetricGroupResult:
    """Compute probability-based metrics: log loss overall + per-category."""
    val_proba = ctx.val_proba
    val_gt_labels = ctx.val_gt_labels
    classes = ctx.clf.classes_

    result = MetricGroupResult()

    # Overall log loss.
    overall_ll = sklearn_log_loss(val_gt_labels, val_proba, labels=classes)
    result.scalars.append(ScalarMetric(name="log_loss", value=overall_ll))

    # Per-sample log loss: -log(p_true).
    classes_list = list(classes)
    gt_col_indices = [classes_list.index(g) for g in val_gt_labels]

    gt_col_indices_arr = np.array(gt_col_indices)
    p_true = val_proba[np.arange(len(val_proba)), gt_col_indices_arr]
    sample_losses = -np.log(np.clip(p_true, 1e-15, 1.0))

    # Group by top-level category.
    ba_to_top = ctx.ba_to_top or build_ba_to_top(classes_list, ctx.ba_library)

    groups = group_by_top_level(
        list(range(len(val_gt_labels))),
        gt_col_indices,
        classes_list,
        ba_to_top,
        ctx.ba_library,
        min_samples=30,
    )

    cat_rows = []
    for group in groups:
        idx = group["indices"]
        cat_losses = sample_losses[idx]
        cat_rows.append(
            {
                "category": group["name"],
                "log_loss": float(np.mean(cat_losses)),
                "n_samples": group["n_samples"],
            }
        )

    cat_rows.sort(key=lambda r: r["log_loss"], reverse=True)
    result.dataframes.append(
        DataFrameResult(
            df=pd.DataFrame(cat_rows)
            if cat_rows
            else pd.DataFrame(columns=["category", "log_loss", "n_samples"]),
            artifact_path="probability/per_category_log_loss",
        )
    )

    # Per-category log loss figure.
    if cat_rows:
        fig, ax = plt.subplots(figsize=(10, max(4, len(cat_rows) * 0.45)))
        try:
            names = [r["category"] for r in cat_rows]
            losses = [r["log_loss"] for r in cat_rows]
            counts = [r["n_samples"] for r in cat_rows]

            y_pos = range(len(names))
            bars = ax.barh(y_pos, losses, color="#d32f2f", edgecolor="white", alpha=0.85)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel("Log Loss (nats)")
            ax.set_title("Log Loss by Top-Level Category")
            ax.axvline(
                overall_ll,
                color="#1976d2",
                linestyle="--",
                linewidth=1.5,
                label=f"Overall: {overall_ll:.3f}",
            )
            ax.legend(loc="lower right")

            for bar, n in zip(bars, counts, strict=False):
                ax.text(
                    bar.get_width() + 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"n={n:,}",
                    va="center",
                    fontsize=9,
                    color="#444444",
                )

            ax.set_xlim(0, max(losses) * 1.25 if losses else 1)
            plt.tight_layout()
        except Exception:
            plt.close(fig)
            raise

        result.figures.append(
            FigureResult(fig=fig, artifact_path="probability/per_category_log_loss.png")
        )

    return result
