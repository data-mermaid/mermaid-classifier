"""Per-source validation metrics.

Breaks down validation metrics by the data source each annotation came
from (CoralNet source_id or MERMAID), so we can see how skewed the
headline numbers are by source composition. The headline accuracy on
heavily-imbalanced datasets is dominated by the largest source, while
deployment to a new site is closer to the median per-source accuracy.

Requires dataset (TrainingDataset) in MetricsContext, which provides
the `feature_loc_to_source` mapping populated during data loading.

Iteration order matches `compute_cover` (`cover.py:46-50`): walk
`dataset.labels.val.keys()` and treat each image's points as
contiguous in `val_results.gt`/`val_results.est`.
"""

import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)

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
    find_lca,
)


def compute_per_source(ctx: MetricsContext) -> MetricGroupResult:
    """Compute validation metrics grouped by data source."""
    val_results = ctx.val_results
    dataset = ctx.dataset
    classes = val_results.classes
    result = MetricGroupResult()

    feature_loc_to_source = getattr(dataset, 'feature_loc_to_source', None)
    if not feature_loc_to_source:
        # Older dataset instances (e.g. from re-evaluation paths) may not
        # have the per-image source map. Skip silently — coordinator logs
        # absence of the artifact group.
        return result

    # Build a per-val-index source array. Each image contributes
    # n_points entries; iteration order matches what evaluate_classifier
    # used when building val_results.
    source_per_index: list[str] = []
    images_per_source: Counter = Counter()
    for feature_loc in dataset.labels.val.keys():
        site, project_id = feature_loc_to_source[feature_loc]
        source_key = f"{site}:{project_id}"
        n_points = len(dataset.labels.val[feature_loc])
        source_per_index.extend([source_key] * n_points)
        images_per_source[source_key] += 1

    if len(source_per_index) != len(val_results.gt):
        # Defensive: order or counts drifted from evaluate_classifier.
        # Don't poison the run with a silently-wrong breakdown.
        raise ValueError(
            f"Per-source index count ({len(source_per_index)}) does not"
            f" match val_results length ({len(val_results.gt)})."
            " dataset.labels.val iteration order may have diverged from"
            " evaluate_classifier."
        )

    sources_arr = np.array(source_per_index)
    gt = np.array(val_results.gt)
    est = np.array(val_results.est)

    ba_paths = ctx.ba_paths or build_ba_paths(classes, ctx.ba_library)

    rows = []
    accuracies: list[float] = []
    for source_key in sorted(set(source_per_index)):
        mask = sources_arr == source_key
        gt_s = gt[mask]
        est_s = est[mask]
        n = int(mask.sum())

        if n == 0:
            continue

        site, source_id = source_key.split(':', 1)
        accuracy = float(accuracy_score(gt_s, est_s))
        accuracies.append(accuracy)

        try:
            # Sources with a single class in their val slice are
            # degenerate but not wrong (balanced_accuracy == accuracy
            # in that case). sklearn warns about the (1,1) confusion
            # matrix produced inside balanced_accuracy_score; silence
            # just that exact message so other UserWarnings still
            # surface.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "A single label was found in 'y_true' and "
                        "'y_pred'"
                    ),
                    category=UserWarning,
                )
                balanced_acc = float(balanced_accuracy_score(gt_s, est_s))
        except ValueError:
            balanced_acc = float('nan')

        prec, rec, f1, _ = precision_recall_fscore_support(
            gt_s, est_s, average='macro', zero_division=0)

        # Per-source cross-branch error rate, using the same definition
        # as taxonomic._compute_error_attribution (LCA == None ->
        # cross-branch).
        err_total = 0
        err_cross_branch = 0
        for gt_idx, est_idx in zip(gt_s, est_s):
            if gt_idx == est_idx:
                continue
            err_total += 1
            ba_gt, _ = split_ba_gf(classes[gt_idx])
            ba_est, _ = split_ba_gf(classes[est_idx])
            if find_lca(ba_gt, ba_est, ba_paths) is None:
                err_cross_branch += 1
        cross_branch_rate = (
            err_cross_branch / err_total if err_total > 0 else 0.0
        )

        rows.append({
            'source_key': source_key,
            'site': site,
            'source_id': source_id,
            'num_val_images': int(images_per_source[source_key]),
            'num_val_annotations': n,
            'accuracy': round(accuracy, 4),
            'balanced_accuracy': round(balanced_acc, 4),
            'f1_macro': round(float(f1), 4),
            'precision_macro': round(float(prec), 4),
            'recall_macro': round(float(rec), 4),
            'cross_branch_error_rate': round(cross_branch_rate, 4),
        })

    if not rows:
        return result

    df = pd.DataFrame(rows).sort_values(
        'num_val_annotations', ascending=False).reset_index(drop=True)

    result.dataframes.append(DataFrameResult(
        df=df, artifact_path='per_source/metrics'))

    result.scalars.extend([
        ScalarMetric(name='per_source/n_sources', value=float(len(rows))),
        ScalarMetric(
            name='per_source/min_accuracy', value=float(min(accuracies))),
        ScalarMetric(
            name='per_source/max_accuracy', value=float(max(accuracies))),
    ])

    if len(df) > 1:
        fig = _plot_accuracy_by_source(df)
        result.figures.append(FigureResult(
            fig=fig, artifact_path='per_source/accuracy_by_source.png'))

    return result


def _plot_accuracy_by_source(df: pd.DataFrame):
    """Horizontal grouped bar chart: accuracy + balanced_accuracy per source.

    Sources are ordered by val annotation count (largest at top), so the
    eye lands on the source that dominates headline accuracy first.
    """
    n = len(df)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * n + 1.5)))
    try:
        y = np.arange(n)
        bar_h = 0.4
        ax.barh(y - bar_h / 2, df['accuracy'], bar_h,
                color='#1976d2', label='Accuracy')
        ax.barh(y + bar_h / 2, df['balanced_accuracy'], bar_h,
                color='#d32f2f', label='Balanced Accuracy')

        ax.set_yticks(list(y))
        ax.set_yticklabels(df['source_key'], fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Score')
        ax.set_title(
            'Per-Source Validation Accuracy '
            '(sources ordered by val annotation count)')
        ax.invert_yaxis()
        ax.axvline(x=df['accuracy'].mean(), color='#1976d2',
                   linestyle=':', linewidth=0.8, alpha=0.6)
        ax.legend(loc='lower right', fontsize=9)

        # Annotate each row with annotation count for context.
        x_max = 1.0
        for i, count in enumerate(df['num_val_annotations']):
            ax.text(x_max + 0.005, i, f'n={count:,}',
                    va='center', ha='left', fontsize=8, color='#444')
        ax.set_xlim(0, x_max + 0.12)

        plt.tight_layout()
    except Exception:
        plt.close(fig)
        raise

    return fig
