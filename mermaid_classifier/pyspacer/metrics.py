from collections import Counter
from contextlib import contextmanager
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from spacer.data_classes import ValResults


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
