from collections import Counter
from contextlib import contextmanager
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from spacer.data_classes import ValResults

from mermaid_classifier.common.benthic_attributes import split_ba_gf


@contextmanager
def make_confusion_matrix(
    val_results: ValResults,
    normalize: bool,
    ba_library: 'BenthicAttributeLibrary',
    gf_library: 'GrowthFormLibrary',
):
    """
    Make a confusion matrix out of the training evaluation results.
    Return it in dataframe and matplotlib figure forms.
    This is a context manager so that the figure can be closed to free memory.
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

    # Classes in order of their frequency in val_results true labels.
    class_indexes_in_freq_order = [
        class_index
        for class_index, _ in Counter(val_results.gt).most_common()]
    # Order columns by frequency.
    matrix = matrix[:, class_indexes_in_freq_order]
    # Order rows by frequency.
    matrix = matrix[class_indexes_in_freq_order, :]

    bagf_names = [
        ba_library.bagf_id_to_name(
            val_results.classes[class_index], gf_library)
        for class_index in class_indexes_in_freq_order
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

    yield df, fig

    # Close figure to free memory
    plt.close(fig)


def precision_recall_f1(
    val_results: ValResults,
    format_func: typing.Callable[[float], typing.Any],
    ba_library: 'BenthicAttributeLibrary',
    gf_library: 'GrowthFormLibrary',
) -> tuple[list[dict], dict]:
    """
    Given pyspacer ValResults and a metric-formatting function, return
    per-label and overall precision, recall, and f1 metrics.
    """

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
            bagf_name=ba_library.bagf_id_to_name(label, gf_library),
            precision=format_func(precision),
            recall=format_func(recall),
            f1_score=format_func(f1_score),
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
    f1_score = 2 * (precision * recall) / (precision + recall)

    overall_metrics |= dict(
        precision_macro=format_func(precision),
        recall_macro=format_func(recall),
        f1_macro=format_func(f1_score),
    )

    return per_label_metrics, overall_metrics


def cohens_kappa(val_results: ValResults) -> float:
    """Cohen's kappa: chance-corrected agreement between gt and est."""
    return sklearn.metrics.cohen_kappa_score(val_results.gt, val_results.est)


def matthews_corrcoef(val_results: ValResults) -> float:
    """Matthews Correlation Coefficient: robust multi-class metric."""
    return sklearn.metrics.matthews_corrcoef(val_results.gt, val_results.est)


def ba_level_accuracy(val_results: ValResults) -> float:
    """
    Accuracy after collapsing BA+GF classes to just BA (benthic attribute).
    Answers: ignoring growth form, how often is the benthic attribute correct?
    """
    # Map each class index to its BA-only string.
    ba_for_class = {}
    for idx, class_id in enumerate(val_results.classes):
        ba, _gf = split_ba_gf(str(class_id))
        ba_for_class[idx] = ba

    gt_ba = [ba_for_class[i] for i in val_results.gt]
    est_ba = [ba_for_class[i] for i in val_results.est]

    correct = sum(1 for g, e in zip(gt_ba, est_ba) if g == e)
    return correct / len(gt_ba)


def cover_bias(
    val_results: ValResults,
) -> tuple[dict[int, float], float]:
    """
    Per-class cover bias and mean absolute cover bias.

    For each class: (predicted_proportion - true_proportion).
    Returns (per_class_bias_dict, mean_abs_bias).
    per_class_bias_dict is keyed by class index.
    """
    n = len(val_results.gt)
    gt_counts = Counter(val_results.gt)
    est_counts = Counter(val_results.est)

    all_classes = set(gt_counts.keys()) | set(est_counts.keys())
    per_class_bias = {}
    for cls_idx in all_classes:
        true_prop = gt_counts.get(cls_idx, 0) / n
        pred_prop = est_counts.get(cls_idx, 0) / n
        per_class_bias[cls_idx] = pred_prop - true_prop

    mean_abs_bias = np.mean(np.abs(list(per_class_bias.values())))
    return per_class_bias, float(mean_abs_bias)


def expected_calibration_error(
    val_results: ValResults, n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Bins predictions by confidence (val_results.scores), then computes the
    weighted average of |accuracy - confidence| per bin.
    """
    scores = np.array(val_results.scores)
    correct = np.array(val_results.gt) == np.array(val_results.est)
    n = len(scores)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            # Include right boundary for the last bin.
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        bin_count = mask.sum()
        if bin_count == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = scores[mask].mean()
        ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def top_k_accuracy(
    val_results: ValResults, val_proba: np.ndarray, k: int = 3,
) -> float:
    """
    Top-k accuracy: fraction of samples where the true label is among
    the top k predicted classes.

    val_proba: shape (n_samples, n_classes), probability matrix aligned
               with val_results.classes ordering.
    """
    gt = np.array(val_results.gt)
    # For each sample, get the indices of the top-k classes.
    top_k_preds = np.argsort(val_proba, axis=1)[:, -k:]
    # Check if ground truth is in top-k for each sample.
    correct = np.array([gt[i] in top_k_preds[i] for i in range(len(gt))])
    return float(correct.mean())
