"""Ranking metrics.

Metrics:
- Top-K accuracy (k=1,3,5,10)
- Mean Reciprocal Rank (MRR)
- Per-category top-K breakdown
- Hierarchical top-K with taxonomic similarity

Requires val_proba, val_gt_labels, and clf in MetricsContext.
"""

from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mermaid_classifier.common.benthic_attributes import split_ba_gf
from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import (
    DataFrameResult,
    FigureResult,
    MetricGroupResult,
    ScalarMetric,
)
from mermaid_classifier.pyspacer.metrics._taxonomy_helpers import (
    build_ba_paths,
    build_ba_to_top,
    group_by_top_level,
    taxonomic_similarity,
)


class _TopKMRRResult(TypedDict):
    topk: dict[int, float]
    mrr: float
    ranks: NDArray[np.intp]
    sorted_indices: NDArray[np.intp]


def _compute_topk_mrr(
    proba: np.ndarray,
    gt_labels: list[str],
    classes: list[str],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> _TopKMRRResult:
    """Compute top-K accuracy and MRR from a probability matrix.

    Returns dict with 'topk' (dict k->accuracy), 'mrr' (float),
    'ranks' (1-indexed rank of true class per sample), and
    'sorted_indices' (argsort of -proba, reusable for hierarchical ranking).
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    sorted_indices = np.argsort(-proba, axis=1)

    n = len(gt_labels)
    ranks: NDArray[np.intp] = np.zeros(n, dtype=np.intp)
    for i, gt_label in enumerate(gt_labels):
        gt_idx = class_to_idx[gt_label]
        ranks[i] = int(np.where(sorted_indices[i] == gt_idx)[0][0]) + 1

    topk = {k: float(np.mean(ranks <= k)) for k in ks}
    mrr = float(np.mean(1.0 / ranks))
    return {"topk": topk, "mrr": mrr, "ranks": ranks, "sorted_indices": sorted_indices}


def compute_ranking(ctx: MetricsContext) -> MetricGroupResult:
    """Compute ranking metrics: top-K, MRR, per-category, hierarchical."""
    # Both are pre-computed by coordinator; it only calls this when val_proba is not None.
    assert ctx.val_proba is not None, "compute_ranking requires val_proba"
    assert ctx.val_gt_labels is not None, "compute_ranking requires val_gt_labels"
    val_proba: np.ndarray = ctx.val_proba
    val_gt_labels: list[str] = ctx.val_gt_labels  # pyright: ignore[reportAssignmentType]  # MERMAID always uses str labels
    classes: list[str] = list(ctx.clf.classes_)  # pyright: ignore[reportAssignmentType]  # MERMAID always uses str labels
    ba_library = ctx.ba_library

    result = MetricGroupResult()

    ks = (1, 3, 5, 10)
    ranking = _compute_topk_mrr(val_proba, val_gt_labels, classes, ks)

    # Overall scalars.
    for k in ks:
        result.scalars.append(ScalarMetric(name=f"top_{k}_accuracy", value=ranking["topk"][k]))
    result.scalars.append(ScalarMetric(name="mrr", value=ranking["mrr"]))

    # Per-category top-K.
    ba_to_top = ctx.ba_to_top or build_ba_to_top(classes, ba_library)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    gt_col_indices = [class_to_idx[g] for g in val_gt_labels]

    groups = group_by_top_level(
        list(range(len(val_gt_labels))),
        gt_col_indices,
        classes,
        ba_to_top,
        ba_library,
        min_samples=30,
    )

    ranks = ranking["ranks"]
    cat_results = []
    for group in groups:
        idx = group["indices"]
        ranks_arr = ranks[idx]
        row = {
            "category": group["name"],
            "n_samples": group["n_samples"],
            "mrr": float(np.mean(1.0 / ranks_arr)),
        }
        for k in ks:
            row[f"top_{k}"] = float(np.mean(ranks_arr <= k))
        cat_results.append(row)

    cat_results.sort(key=lambda r: r["top_1"], reverse=True)

    result.dataframes.append(
        DataFrameResult(
            df=pd.DataFrame(cat_results)
            if cat_results
            else pd.DataFrame(
                columns=["category", "top_1", "top_3", "top_5", "top_10", "mrr", "n_samples"]  # pyright: ignore[reportArgumentType]  # pandas stubs type columns as Axes|None
            ),
            artifact_path="ranking/per_category_topk",
        )
    )

    # Per-category top-K figure.
    if cat_results:
        fig, ax = plt.subplots(figsize=(12, max(4, len(cat_results) * 0.5)))
        try:
            bar_height = 0.18
            y_pos = np.arange(len(cat_results))
            colors = {1: "#1976d2", 3: "#388e3c", 5: "#f57c00", 10: "#7b1fa2"}

            for i, k in enumerate(ks):
                values = [r[f"top_{k}"] for r in cat_results]
                ax.barh(
                    y_pos + i * bar_height,
                    values,
                    bar_height,
                    label=f"Top-{k}",
                    color=colors[k],
                    alpha=0.85,
                )

            ax.set_yticks(y_pos + bar_height * 1.5)
            ax.set_yticklabels([r["category"] for r in cat_results])
            ax.invert_yaxis()
            ax.set_xlabel("Accuracy")
            ax.set_xlim(0, 1.05)
            ax.set_title("Top-K Accuracy by Top-Level Category")
            ax.legend(loc="lower right")
            plt.tight_layout()
        except Exception:
            plt.close(fig)
            raise

        result.figures.append(FigureResult(fig=fig, artifact_path="ranking/per_category_topk.png"))

    # Hierarchical top-K with taxonomic similarity.
    ba_paths = ctx.ba_paths or build_ba_paths(classes, ba_library)
    sorted_indices = ranking["sorted_indices"]
    class_ba_ids = [split_ba_gf(c)[0] for c in classes]
    gt_ba_ids = [split_ba_gf(g)[0] for g in val_gt_labels]

    hier_ks = (1, 3, 5, 10)
    thresholds = [1.0, 0.75, 0.5]
    n_samples = len(val_gt_labels)

    max_sim_at_k = {k: np.zeros(n_samples) for k in hier_ks}

    max_k = max(hier_ks)
    for i in range(n_samples):
        gt_ba = gt_ba_ids[i]
        top_indices = sorted_indices[i, :max_k]
        sims = [
            taxonomic_similarity(gt_ba, class_ba_ids[int(j)], ba_paths, ba_library)
            for j in top_indices
        ]
        for k in hier_ks:
            max_sim_at_k[k][i] = max(sims[:k])

    result.scalars.append(
        ScalarMetric(
            name="hierarchical_top_5_mean_similarity", value=float(np.mean(max_sim_at_k[5]))
        )
    )

    # Hierarchical top-K DataFrame.
    hier_rows = []
    for k in hier_ks:
        row = {
            "k": k,
            "mean_max_similarity": float(np.mean(max_sim_at_k[k])),
        }
        for t in thresholds:
            label = {1.0: "hit_exact", 0.75: "hit_sibling_0.75", 0.5: "hit_family_0.5"}[t]
            row[label] = float(np.mean(max_sim_at_k[k] >= t))
        hier_rows.append(row)

    result.dataframes.append(
        DataFrameResult(
            df=pd.DataFrame(hier_rows),
            artifact_path="ranking/hierarchical_topk",
        )
    )

    return result
