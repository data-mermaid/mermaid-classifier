"""Per-image cover metrics.

Metrics:
- Per-class cover bias, RMSE, MAE, R-squared
- Aggregate summary scalars

Requires dataset (TrainingDataset) in MetricsContext.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import (
    DataFrameResult,
    FigureResult,
    MetricGroupResult,
    ScalarMetric,
)


def compute_cover(ctx: MetricsContext) -> MetricGroupResult:
    """Compute per-image cover reconstruction metrics."""
    val_results = ctx.val_results
    dataset = ctx.dataset
    classes = val_results.classes

    # Build per-image cover vectors from flat gt/est arrays.
    # evaluate_classifier iterates images in dict.keys() order,
    # with each image's points contiguous.
    all_classes = sorted(set(
        classes[i] for i in set(val_results.gt) | set(val_results.est)))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    n_classes = len(all_classes)

    gt_labels = [classes[i] for i in val_results.gt]
    est_labels = [classes[i] for i in val_results.est]

    n_images = len(list(dataset.labels.val.keys()))
    true_cover_matrix = np.zeros((n_images, n_classes))
    pred_cover_matrix = np.zeros((n_images, n_classes))
    offset = 0

    for img_idx, feature_loc in enumerate(dataset.labels.val.keys()):
        n_points = len(dataset.labels.val[feature_loc])
        img_gts = gt_labels[offset:offset + n_points]
        img_ests = est_labels[offset:offset + n_points]
        offset += n_points

        for label in img_gts:
            true_cover_matrix[img_idx, class_to_idx[label]] += 1
        for label in img_ests:
            pred_cover_matrix[img_idx, class_to_idx[label]] += 1
        true_cover_matrix[img_idx] /= n_points
        pred_cover_matrix[img_idx] /= n_points

    # Per-class metrics.
    errors = pred_cover_matrix - true_cover_matrix
    per_class_bias = errors.mean(axis=0)
    per_class_rmse = np.sqrt((errors ** 2).mean(axis=0))
    per_class_mae = np.abs(errors).mean(axis=0)

    per_class_r2 = np.full(n_classes, np.nan)
    for i in range(n_classes):
        true_col = true_cover_matrix[:, i]
        if true_col.std() > 0:
            per_class_r2[i] = r2_score(true_col, pred_cover_matrix[:, i])

    cover_df = pd.DataFrame({
        'bagf_id': all_classes,
        'bagf_name': [
            ctx.ba_library.bagf_id_to_name(c, ctx.gf_library)
            for c in all_classes
        ],
        'mean_true_cover_pct': true_cover_matrix.mean(axis=0) * 100,
        'bias_pct': per_class_bias * 100,
        'rmse_pct': per_class_rmse * 100,
        'mae_pct': per_class_mae * 100,
        'r_squared': per_class_r2,
    }).sort_values('mean_true_cover_pct', ascending=False)

    # Aggregate over classes with >0.5% mean cover.
    significant = cover_df[cover_df['mean_true_cover_pct'] > 0.5]

    result = MetricGroupResult()

    if len(significant) > 0:
        result.scalars.extend([
            ScalarMetric(
                name='cover_mean_abs_bias_pct',
                value=float(significant['bias_pct'].abs().mean())),
            ScalarMetric(
                name='cover_mean_rmse_pct',
                value=float(significant['rmse_pct'].mean())),
            ScalarMetric(
                name='cover_mean_mae_pct',
                value=float(significant['mae_pct'].mean())),
            ScalarMetric(
                name='cover_median_r_squared',
                value=float(significant['r_squared'].median())),
        ])
    else:
        result.scalars.extend([
            ScalarMetric(name='cover_mean_abs_bias_pct', value=0.0),
            ScalarMetric(name='cover_mean_rmse_pct', value=0.0),
            ScalarMetric(name='cover_mean_mae_pct', value=0.0),
            ScalarMetric(name='cover_median_r_squared', value=0.0),
        ])

    result.dataframes.append(DataFrameResult(
        df=cover_df, artifact_path='cover/per_class_cover_metrics'))

    # Bias bar chart for top classes by mean cover.
    top_n = min(20, len(significant))
    if top_n > 0:
        top_classes = significant.head(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            colors = [
                '#d32f2f' if b > 0 else '#1976d2'
                for b in top_classes['bias_pct']
            ]
            ax.barh(range(top_n), top_classes['bias_pct'], color=colors)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_classes['bagf_name'], fontsize=9)
            ax.set_xlabel('Cover Bias (%)')
            ax.set_title('Per-Class Cover Bias (top classes by mean cover)')
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.invert_yaxis()
            plt.tight_layout()
        except Exception:
            plt.close(fig)
            raise
        result.figures.append(FigureResult(
            fig=fig, artifact_path='cover/per_class_bias.png'))

    return result
