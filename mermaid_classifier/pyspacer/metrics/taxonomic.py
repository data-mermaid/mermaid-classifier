"""Hierarchical taxonomy-aware metrics.

Metrics:
- Error attribution by LCA (lowest common ancestor)
- Top-level confusion matrix
- Growth form differentiation accuracy
"""

from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics

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
    find_lca,
)


def compute_taxonomic(ctx: MetricsContext) -> MetricGroupResult:
    """Compute all taxonomy-aware metrics."""
    result = MetricGroupResult()

    for partial in [
        _compute_error_attribution(ctx),
        _compute_top_level_confusion(ctx),
        _compute_gf_differentiation(ctx),
    ]:
        result.scalars.extend(partial.scalars)
        result.figures.extend(partial.figures)
        result.dataframes.extend(partial.dataframes)
        result.dicts.extend(partial.dicts)

    return result


def _compute_error_attribution(ctx: MetricsContext) -> MetricGroupResult:
    """Attribute misclassifications to their LCA in the taxonomy tree."""
    val_results = ctx.val_results
    classes = val_results.classes
    ba_library = ctx.ba_library

    ba_full_paths = ctx.ba_paths or build_ba_paths(classes, ba_library)

    def get_branch(ba_id):
        if ba_id in ba_full_paths:
            return ba_full_paths[ba_id][0]
        ancestors = ba_library.get_ancestor_ids(ba_id)
        return ancestors[0] if ancestors else ba_id

    # Count errors per LCA node.
    lca_error_counts: Counter = Counter()
    total_errors = 0

    for gt_idx, est_idx in zip(val_results.gt, val_results.est):
        if gt_idx == est_idx:
            continue
        total_errors += 1
        ba_gt, _ = split_ba_gf(classes[gt_idx])
        ba_est, _ = split_ba_gf(classes[est_idx])
        lca = find_lca(ba_gt, ba_est, ba_full_paths)
        lca_error_counts[lca] += 1

    result = MetricGroupResult()

    if total_errors == 0:
        result.scalars.extend([
            ScalarMetric(name='cross_branch_error_rate', value=0.0),
            ScalarMetric(name='within_branch_error_rate', value=0.0),
        ])
        result.dataframes.append(DataFrameResult(
            df=pd.DataFrame(columns=[
                'lca_node', 'lca_name', 'branch', 'error_count',
                'pct_of_errors', 'classes_in_subtree',
            ]),
            artifact_path='taxonomic/error_attribution',
        ))
        return result

    # Aggregate by branch.
    branch_errors: Counter = Counter()
    for lca_node, count in lca_error_counts.items():
        if lca_node is not None:
            branch_errors[get_branch(lca_node)] += count

    # Count model classes per LCA node.
    all_ba_ids = set()
    for bagf_id in classes:
        ba_id, _ = split_ba_gf(bagf_id)
        all_ba_ids.add(ba_id)

    lca_class_counts: dict[str, int] = {}
    for lca_node in lca_error_counts:
        if lca_node is None:
            continue
        descendants = ba_library.get_descendants(lca_node)
        desc_ids = {d['id'] for d in descendants} | {lca_node}
        lca_class_counts[lca_node] = len(desc_ids & all_ba_ids)

    cross_branch_count = lca_error_counts.get(None, 0)
    within_branch_count = total_errors - cross_branch_count

    result.scalars.extend([
        ScalarMetric(
            name='cross_branch_error_rate',
            value=cross_branch_count / total_errors),
        ScalarMetric(
            name='within_branch_error_rate',
            value=within_branch_count / total_errors),
    ])

    # Build DataFrame.
    rows = []
    for node, count in lca_error_counts.most_common():
        if node is None:
            rows.append({
                'lca_node': '(cross-branch)',
                'lca_name': '(cross-branch)',
                'branch': '',
                'error_count': count,
                'pct_of_errors': count / total_errors * 100,
                'classes_in_subtree': 0,
            })
        else:
            rows.append({
                'lca_node': node,
                'lca_name': ba_library.id_to_name(node),
                'branch': ba_library.id_to_name(get_branch(node)),
                'error_count': count,
                'pct_of_errors': count / total_errors * 100,
                'classes_in_subtree': lca_class_counts.get(node, 0),
            })
    result.dataframes.append(DataFrameResult(
        df=pd.DataFrame(rows),
        artifact_path='taxonomic/error_attribution',
    ))

    # Two-panel figure.
    fig = _plot_error_attribution(
        lca_error_counts, branch_errors, total_errors,
        cross_branch_count, ba_library, get_branch)
    result.figures.append(FigureResult(
        fig=fig, artifact_path='taxonomic/error_attribution.png'))

    return result


def _plot_error_attribution(
    lca_error_counts, branch_errors, total_errors,
    cross_branch_count, ba_library, get_branch,
):
    """Build the two-panel error attribution figure."""
    cmap = matplotlib.colormaps['tab20']

    branch_ids_sorted = sorted(
        branch_errors.keys(),
        key=lambda b: branch_errors[b], reverse=True)
    branch_color = {
        bid: cmap(i / max(len(branch_ids_sorted), 1))
        for i, bid in enumerate(branch_ids_sorted)
    }

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={'height_ratios': [1, 4]})
    fig.suptitle('Error Attribution by Taxonomy Node', fontsize=14, y=0.98)

    # Panel 1: Summary stacked horizontal bar.
    ax_bar = axes[0]
    left = 0
    legend_handles = []
    legend_labels = []
    for bid in branch_ids_sorted:
        frac = branch_errors[bid] / total_errors
        color = branch_color[bid]
        bar = ax_bar.barh(0, frac, left=left, color=color, edgecolor='white')
        if frac > 0.04:
            ax_bar.text(left + frac / 2, 0, f'{frac:.0%}',
                        ha='center', va='center', fontsize=10,
                        fontweight='bold')
        legend_handles.append(bar[0])
        legend_labels.append(
            f"{ba_library.id_to_name(bid)} ({branch_errors[bid]:,})")
        left += frac

    cross_frac = cross_branch_count / total_errors
    bar = ax_bar.barh(0, cross_frac, left=left, color='#999999',
                      edgecolor='white')
    if cross_frac > 0.04:
        ax_bar.text(left + cross_frac / 2, 0, f'{cross_frac:.0%}',
                    ha='center', va='center', fontsize=10, fontweight='bold')
    legend_handles.append(bar[0])
    legend_labels.append(f"Cross-branch ({cross_branch_count:,})")

    ax_bar.set_xlim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel('Fraction of all errors')
    ax_bar.legend(
        legend_handles, legend_labels,
        loc='lower center', bbox_to_anchor=(0.5, 1.05),
        ncol=min(len(legend_labels), 4), fontsize=9, frameon=False)

    # Panel 2: Top 15 LCA nodes horizontal bar chart.
    ax_detail = axes[1]

    top_lca = lca_error_counts.most_common(16)
    top_lca_filtered = []
    cross_entry = None
    for node, count in top_lca:
        if node is None:
            cross_entry = (node, count)
        else:
            top_lca_filtered.append((node, count))
        if len(top_lca_filtered) == 14 and cross_entry is not None:
            break
        if len(top_lca_filtered) == 15:
            break

    display_entries = []
    if cross_entry:
        display_entries.append(cross_entry)
    display_entries.extend(top_lca_filtered)
    display_entries.sort(key=lambda x: x[1])

    y_labels = []
    bar_colors = []
    bar_counts = []
    for node, count in display_entries:
        bar_counts.append(count)
        if node is None:
            y_labels.append('Cross-branch')
            bar_colors.append('#999999')
        else:
            name = ba_library.id_to_name(node)
            branch_id = get_branch(node)
            branch_name = ba_library.id_to_name(branch_id)
            if node == branch_id:
                y_labels.append(name)
            else:
                y_labels.append(f"{name}  ({branch_name})")
            bar_colors.append(branch_color.get(branch_id, '#cccccc'))

    y_pos = range(len(display_entries))
    bars = ax_detail.barh(y_pos, bar_counts, color=bar_colors,
                          edgecolor='white')
    ax_detail.set_yticks(list(y_pos))
    ax_detail.set_yticklabels(y_labels, fontsize=9)
    ax_detail.set_xlabel('Number of errors')

    for bar, count in zip(bars, bar_counts):
        pct = count / total_errors * 100
        ax_detail.text(
            bar.get_width() + max(bar_counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{count:,} ({pct:.1f}%)',
            ha='left', va='center', fontsize=9)

    ax_detail.set_xlim(0, max(bar_counts) * 1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _compute_top_level_confusion(ctx: MetricsContext) -> MetricGroupResult:
    """Build row-normalized confusion matrix at the top-level BA."""
    val_results = ctx.val_results
    classes = val_results.classes
    ba_library = ctx.ba_library

    ba_to_top = ctx.ba_to_top or build_ba_to_top(classes, ba_library)

    top_gt = []
    top_est = []
    for gt_idx, est_idx in zip(val_results.gt, val_results.est):
        gt_ba, _ = split_ba_gf(classes[gt_idx])
        est_ba, _ = split_ba_gf(classes[est_idx])
        top_gt.append(ba_to_top[gt_ba])
        top_est.append(ba_to_top[est_ba])

    # Order by frequency in gt, then append any IDs only in est.
    gt_counts = Counter(top_gt)
    top_ids_by_freq = [tid for tid, _ in gt_counts.most_common()]
    est_only = sorted(set(top_est) - set(top_ids_by_freq))
    top_ids_by_freq.extend(est_only)
    top_names = [ba_library.id_to_name(tid) for tid in top_ids_by_freq]
    id_to_idx = {tid: i for i, tid in enumerate(top_ids_by_freq)}

    n_top = len(top_ids_by_freq)
    cm = np.zeros((n_top, n_top), dtype=int)
    for g, e in zip(top_gt, top_est):
        cm[id_to_idx[g], id_to_idx[e]] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = np.int64(np.floor(cm / row_sums * 100))

    result = MetricGroupResult()

    # Figure.
    fig_size = max(8, n_top * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    try:
        disp = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix=cm_pct, display_labels=top_names)
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_title('Top-Level Confusion (row-normalized %)', pad=20)
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        label_fs = max(8, min(12, 150 / n_top))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='left',
                 rotation_mode='anchor', fontsize=label_fs)
        plt.setp(ax.get_yticklabels(), fontsize=label_fs)
        plt.tight_layout()
    except Exception:
        plt.close(fig)
        raise

    result.figures.append(FigureResult(
        fig=fig, artifact_path='taxonomic/top_level_confusion.png'))

    # DataFrame of off-diagonal confusions.
    confusions = []
    for i in range(n_top):
        for j in range(n_top):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    'true': top_names[i],
                    'predicted': top_names[j],
                    'row_normalized_pct': int(cm_pct[i, j]),
                    'sample_count': int(cm[i, j]),
                })
    confusions.sort(key=lambda x: x['row_normalized_pct'], reverse=True)
    result.dataframes.append(DataFrameResult(
        df=pd.DataFrame(confusions) if confusions else pd.DataFrame(
            columns=['true', 'predicted', 'row_normalized_pct', 'sample_count']),
        artifact_path='taxonomic/top_level_confusions',
    ))

    return result


def _compute_gf_differentiation(ctx: MetricsContext) -> MetricGroupResult:
    """Analyze growth form prediction accuracy."""
    val_results = ctx.val_results
    classes = val_results.classes
    gf_library = ctx.gf_library

    true_gf_names = []
    pred_gf_names = []
    ba_match_flags = []

    for gt_idx, est_idx in zip(val_results.gt, val_results.est):
        gt_ba, gt_gf = split_ba_gf(classes[gt_idx])
        est_ba, est_gf = split_ba_gf(classes[est_idx])

        gt_gf_name = gf_library.id_to_name(gt_gf) if gt_gf != '' else '(no GF)'
        est_gf_name = gf_library.id_to_name(est_gf) if est_gf != '' else '(no GF)'

        true_gf_names.append(gt_gf_name)
        pred_gf_names.append(est_gf_name)
        ba_match_flags.append(gt_ba == est_ba)

    true_gf_names_arr = np.array(true_gf_names)
    pred_gf_names_arr = np.array(pred_gf_names)
    ba_match_flags_arr = np.array(ba_match_flags)

    true_has_gf = true_gf_names_arr != '(no GF)'
    n_gf_relevant = true_has_gf.sum()

    result = MetricGroupResult()

    if n_gf_relevant == 0:
        result.scalars.extend([
            ScalarMetric(name='gf_accuracy_gf_relevant', value=0.0),
            ScalarMetric(name='within_ba_gf_accuracy', value=0.0),
        ])
        result.dataframes.append(DataFrameResult(
            df=pd.DataFrame(
                columns=['growth_form', 'precision', 'recall', 'f1',
                         'support']),
            artifact_path='taxonomic/gf_precision_recall_f1'))
        return result

    # GF accuracy among GF-relevant predictions.
    gf_relevant_correct = (
        true_gf_names_arr[true_has_gf] == pred_gf_names_arr[true_has_gf]
    ).sum()
    gf_relevant_acc = gf_relevant_correct / n_gf_relevant

    # Within-BA GF accuracy.
    ba_correct_and_has_gf = true_has_gf & ba_match_flags_arr
    n_ba_correct_gf_relevant = ba_correct_and_has_gf.sum()
    if n_ba_correct_gf_relevant > 0:
        within_ba_gf_correct = (
            true_gf_names_arr[ba_correct_and_has_gf]
            == pred_gf_names_arr[ba_correct_and_has_gf]
        ).sum()
        within_ba_gf_acc = within_ba_gf_correct / n_ba_correct_gf_relevant
    else:
        within_ba_gf_acc = float('nan')

    result.scalars.extend([
        ScalarMetric(name='gf_accuracy_gf_relevant', value=float(gf_relevant_acc)),
        ScalarMetric(name='within_ba_gf_accuracy', value=float(within_ba_gf_acc)),
    ])

    # Per-GF precision/recall/F1.
    true_gf_counts = Counter(true_gf_names_arr[true_has_gf])
    gf_row_labels = [name for name, _ in true_gf_counts.most_common()]
    gf_col_labels = gf_row_labels + ['(no GF)']

    t_filtered = true_gf_names_arr[true_has_gf]
    p_filtered = pred_gf_names_arr[true_has_gf]

    prec, rec, f1, support = sklearn.metrics.precision_recall_fscore_support(
        t_filtered, p_filtered, labels=gf_row_labels, zero_division=0)

    prf_df = pd.DataFrame({
        'growth_form': gf_row_labels,
        'precision': np.round(prec, 3),
        'recall': np.round(rec, 3),
        'f1': np.round(f1, 3),
        'support': support.astype(int),
    }).sort_values('support', ascending=False).reset_index(drop=True)

    result.dataframes.append(DataFrameResult(
        df=prf_df, artifact_path='taxonomic/gf_precision_recall_f1'))

    # GF confusion matrix figure.
    row_idx = {name: i for i, name in enumerate(gf_row_labels)}
    col_idx = {name: i for i, name in enumerate(gf_col_labels)}

    n_rows = len(gf_row_labels)
    n_cols = len(gf_col_labels)
    cm_raw = np.zeros((n_rows, n_cols), dtype=int)

    for t, p in zip(t_filtered, p_filtered):
        r = row_idx[t]
        c = col_idx.get(p, None)
        if c is not None:
            cm_raw[r, c] += 1

    row_sums = cm_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = np.int64(np.floor(cm_raw / row_sums * 100))

    fig, ax = plt.subplots(
        figsize=(max(10, n_cols * 0.9), max(6, n_rows * 0.55)))
    try:
        im = ax.imshow(cm_pct, cmap='Blues', aspect='auto')

        for i in range(n_rows):
            for j in range(n_cols):
                val = cm_pct[i, j]
                if val > 0:
                    text_color = 'white' if val > 50 else 'black'
                    ax.text(j, i, str(val), ha='center', va='center',
                            fontsize=9, color=text_color)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(gf_col_labels, fontsize=10)
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='left',
                 rotation_mode='anchor')
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(gf_row_labels, fontsize=10)
        ax.set_xlabel('Predicted Growth Form', fontsize=11)
        ax.set_ylabel('True Growth Form', fontsize=11)
        ax.set_title(
            'GF Confusion Matrix — row-normalized % (true label has GF)',
            pad=50)
        plt.tight_layout()
    except Exception:
        plt.close(fig)
        raise

    result.figures.append(FigureResult(
        fig=fig, artifact_path='taxonomic/gf_confusion.png'))

    return result
