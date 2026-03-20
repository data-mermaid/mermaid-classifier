"""Classification metrics: confusion matrices, precision/recall/F1,
balanced accuracy, and Matthews Correlation Coefficient."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances
from spacer.data_classes import ValResults

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import (
    DataFrameResult,
    DictResult,
    FigureResult,
    MetricGroupResult,
    ScalarMetric,
)


def _hierarchical_class_order(val_results: ValResults) -> list[int]:
    """Compute class ordering via hierarchical clustering of prediction profiles.

    Clusters classes whose row-normalized confusion matrix rows (prediction
    distributions) are most similar, so the reordered matrix reveals
    block-diagonal structure.
    """
    n_classes = len(val_results.classes)
    if n_classes < 3:
        return list(range(n_classes))

    raw_cm = sklearn.metrics.confusion_matrix(
        y_true=val_results.gt,
        y_pred=val_results.est,
        labels=range(n_classes),
    )

    row_sums = raw_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_cm = raw_cm / row_sums

    dist_matrix = cosine_distances(norm_cm)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='average')
    return list(leaves_list(Z))


def _build_confusion_matrix(
    val_results: ValResults,
    normalize: bool,
    ba_library: 'BenthicAttributeLibrary',
    gf_library: 'GrowthFormLibrary',
    class_order: list[int],
) -> tuple[pd.DataFrame, plt.Figure]:
    """Build a confusion matrix DataFrame and matplotlib Figure.

    Returns (df, fig). Caller is responsible for closing the figure.
    """
    matrix = sklearn.metrics.confusion_matrix(
        y_true=val_results.gt,
        y_pred=val_results.est,
        labels=range(len(val_results.classes)),
        # 'true': values between 0 and 1 for each cell.
        # None: Each cell has a frequency.
        normalize='true' if normalize else None,
    )

    if normalize:
        # 0-to-1 values -> integer percents.
        matrix = np.int64(np.floor(matrix * 100))

    # Reorder rows and columns by hierarchical clustering order.
    matrix = matrix[np.ix_(class_order, class_order)]

    bagf_names = [
        ba_library.bagf_id_to_name(
            val_results.classes[class_index], gf_library)
        for class_index in class_order
    ]

    # Confusion matrix as a table.

    # To dataframe, labeling each column with a BA-GF combo.
    df = pd.DataFrame(data=matrix, columns=bagf_names)
    # Add column to label each row with a BA-GF combo.
    df.insert(loc=0, column='-', value=bagf_names)

    # Confusion matrix as a figure.

    # Create square figure, with size scaled to number of labels
    num_labels = len(bagf_names)
    fig_size = max(12, num_labels * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    try:
        # Matplotlib visualization of the confusion matrix.
        display = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=bagf_names)
        display.plot(
            ax=ax,
            cmap='Blues',
            # Prevent "100" displaying as "1e+02".
            values_format='d',
            # A color legend feels unnecessary here.
            colorbar=False,
        )

        # Move x-axis labels to the top.
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        # Rotate x-axis tick labels to prevent their texts from overlapping.
        label_font_size = max(8, min(12, 150 / num_labels))
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha='left',
            rotation_mode='anchor',
            fontsize=label_font_size,
        )
        # Match y-axis labels' font size with the x axis.
        plt.setp(
            ax.get_yticklabels(),
            fontsize=label_font_size,
        )

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
    except Exception:
        plt.close(fig)
        raise

    return df, fig


def compute_confusion_matrices(ctx: MetricsContext) -> MetricGroupResult:
    """Compute confusion matrices (frequency and percent-normalized)."""
    result = MetricGroupResult()

    class_order = _hierarchical_class_order(ctx.val_results)

    for normalize, filestem in [
        (False, 'confusion_matrix/frequencies'),
        (True, 'confusion_matrix/percents'),
    ]:
        try:
            df, fig = _build_confusion_matrix(
                ctx.val_results, normalize, ctx.ba_library, ctx.gf_library,
                class_order,
            )
        except Exception:
            for fig_result in result.figures:
                plt.close(fig_result.fig)
            raise
        result.dataframes.append(
            DataFrameResult(df=df, artifact_path=filestem))
        result.figures.append(
            FigureResult(fig=fig, artifact_path=filestem + '.png'))

    return result


def compute_precision_recall_f1(ctx: MetricsContext) -> MetricGroupResult:
    """Compute per-label and overall precision, recall, and F1."""
    val_results = ctx.val_results

    # Convert the valresults to a pandas dataframe.

    actual_annotations = pd.Categorical(
        [val_results.classes[i] for i in val_results.gt])
    predicted_annotations = pd.Categorical(
        [val_results.classes[i] for i in val_results.est])
    annotations_df = pd.DataFrame({
        'actual': actual_annotations,
        'predicted': predicted_annotations,
    })

    # Precision, recall, F1: per label

    per_label_metrics = []

    for label in val_results.classes:

        precision = sklearn.metrics.precision_score(
            annotations_df['actual'],
            annotations_df['predicted'],
            # This makes it produce single-label metrics.
            labels=[label],
            # For single-label, micro and macro are the same, so it doesn't
            # matter which we pass in for `average`.
            average='micro',
            # If any label is lacking true positives and false positives,
            # or true positives and false negatives, either precision or
            # recall may have zero in the denominator of the calculation.
            # So we define what value we use in that situation.
            zero_division=0.0,
        )
        recall = sklearn.metrics.recall_score(
            annotations_df['actual'],
            annotations_df['predicted'],
            labels=[label],
            average='micro',
            zero_division=0.0,
        )

        if precision + recall == 0.0:
            # Avoid division by zero.
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        per_label_metrics.append(dict(
            bagf_name=ctx.ba_library.bagf_id_to_name(label, ctx.gf_library),
            precision=ctx.format_func(precision),
            recall=ctx.format_func(recall),
            f1_score=ctx.format_func(f1_score),
            bagf_id=label,
        ))

    # Precision, recall, F1: overall

    overall_metrics = dict()

    # average='macro' calculates precision for each class and then averages
    # them, treating all classes equally irrespective of their frequency
    # in the dataset.
    # average='micro' does calculations globally without separating by class,
    # but in this case precision, recall, f1 score, and accuracy are all the
    # same result. And we already get accuracy from pyspacer. So, focus on
    # macro here.
    precision = sklearn.metrics.precision_score(
        actual_annotations,
        predicted_annotations,
        average='macro',
        zero_division=0.0,
    )
    recall = sklearn.metrics.recall_score(
        actual_annotations,
        predicted_annotations,
        average='macro',
        zero_division=0.0,
    )
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    overall_metrics |= dict(
        precision_macro=ctx.format_func(precision),
        recall_macro=ctx.format_func(recall),
        f1_macro=ctx.format_func(f1_score),
    )

    # Build result

    result = MetricGroupResult()

    # Scalars for MLflow log_metric().
    for name, value in overall_metrics.items():
        result.scalars.append(ScalarMetric(name=name, value=value))

    # Per-label metrics as a DataFrame artifact.
    result.dataframes.append(
        DataFrameResult(
            df=pd.DataFrame(per_label_metrics),
            artifact_path='metrics_per_label'))

    # Overall metrics as a dict artifact.
    result.dicts.append(
        DictResult(data=overall_metrics, artifact_path='metrics_overall.yaml'))

    return result


def compute_balanced_accuracy_mcc(ctx: MetricsContext) -> MetricGroupResult:
    """Compute balanced accuracy and Matthews Correlation Coefficient.

    Balanced accuracy is the macro-averaged recall (accounts for class
    imbalance). MCC is a balanced measure that considers all four quadrants
    of the confusion matrix.
    """
    gt_labels = [ctx.val_results.classes[i] for i in ctx.val_results.gt]
    est_labels = [ctx.val_results.classes[i] for i in ctx.val_results.est]

    balanced_acc = sklearn.metrics.balanced_accuracy_score(
        gt_labels, est_labels)
    mcc = sklearn.metrics.matthews_corrcoef(gt_labels, est_labels)

    return MetricGroupResult(
        scalars=[
            ScalarMetric(
                name='balanced_accuracy',
                value=ctx.format_func(balanced_acc)),
            ScalarMetric(
                name='mcc',
                value=ctx.format_func(mcc)),
        ],
    )
