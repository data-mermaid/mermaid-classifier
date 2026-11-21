"""Evaluation of a trained classifier on the held-out test groups.

The holdout images take no part in training, calibration or early
stopping, so scoring them gives an estimate that is not inflated by
shared sources. Points whose class the model never saw (a class can occur
only in the holdout) keep their prediction rows but stay out of the
accuracy-style metrics.
"""

import dataclasses
import logging
import typing

import duckdb
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from spacer.data_classes import ImageLabels, ValResults
from spacer.train_utils import evaluate_classifier

from mermaid_classifier.pyspacer.metrics import MetricsContext, MetricsCoordinator
from mermaid_classifier.pyspacer.metrics._logging import log_dataframe
from mermaid_classifier.pyspacer.options import Sites

logger = logging.getLogger(__name__)

HOLDOUT_PREFIX = "holdout/"
_POINT_KEY = ["bucket", "feature_vector", "row", "col"]


class _HoldoutDataset(typing.Protocol):
    holdout_labels: ImageLabels
    duck_conn: duckdb.DuckDBPyConnection
    feature_loc_to_source: dict[typing.Any, tuple[str, str]]
    _feature_path_to_s3_location: dict[str, tuple[str, str]]


@dataclasses.dataclass
class HoldoutEvaluation:
    # One row per evaluated point: the point key, site, gt_bagf, pred_bagf,
    # pred_score, in_model_classes.
    points: pd.DataFrame
    # Known-class points only, aligned with `known_labels` iteration order.
    # None when no holdout point has a class the model knows.
    val_results: ValResults | None
    known_labels: ImageLabels

    @property
    def n_unknown_class(self) -> int:
        return int((~self.points["in_model_classes"]).sum())


def evaluate_holdout(clf: typing.Any, dataset: _HoldoutDataset) -> HoldoutEvaluation:
    """Predict every holdout point with `clf` (the calibrated classifier)."""
    labels = dataset.holdout_labels
    classes = list(clf.classes_)
    class_set = set(classes)

    gts, ests, scores = evaluate_classifier(clf, labels)

    # Same iteration order as evaluate_classifier: images, then points.
    rows: list[dict[str, typing.Any]] = []
    known_labels = ImageLabels()
    for feature_loc in labels.keys():  # noqa: SIM118 -- ImageLabels.keys() is not a plain dict
        bucket, feature_vector = dataset._feature_path_to_s3_location[feature_loc.key]  # pyright: ignore[reportPrivateUsage]  # read-only lookup from local path to S3 location
        site = dataset.feature_loc_to_source[feature_loc][0]
        known_annotations = []
        for row, col, label in labels[feature_loc]:
            rows.append(
                {
                    "bucket": bucket,
                    "feature_vector": feature_vector,
                    "row": row,
                    "col": col,
                    "site": site,
                    "gt_bagf": label,
                    "in_model_classes": label in class_set,
                }
            )
            if label in class_set:
                known_annotations.append((row, col, label))
        if known_annotations:
            known_labels.add_image(feature_loc, known_annotations)

    if len(rows) != len(gts):
        raise ValueError(
            f"Holdout point count ({len(rows)}) does not match the number of"
            f" predictions ({len(gts)})."
        )
    points = pd.DataFrame(rows)
    points["pred_bagf"] = ests
    points["pred_score"] = scores

    known = points[points["in_model_classes"]]
    val_results = None
    if len(known) > 0:
        val_results = ValResults(
            scores=known["pred_score"].tolist(),
            gt=[classes.index(g) for g in known["gt_bagf"]],
            est=[classes.index(e) for e in known["pred_bagf"]],
            classes=classes,
        )
    return HoldoutEvaluation(points=points, val_results=val_results, known_labels=known_labels)


def holdout_annotations_df(
    duck_conn: duckdb.DuckDBPyConnection, points: pd.DataFrame
) -> pd.DataFrame:
    """Every holdout annotation row joined to its prediction."""
    duck_conn.register(
        "preds", points[[*_POINT_KEY, "pred_bagf", "pred_score", "in_model_classes"]]
    )
    join_on = " AND ".join(f"a.{c} = p.{c}" for c in _POINT_KEY)
    try:
        return duck_conn.execute(
            "SELECT a.* EXCLUDE (is_holdout, is_excluded),"
            " p.pred_bagf, p.pred_score, p.in_model_classes"
            " FROM (SELECT * FROM annotations WHERE training_set = 'holdout') a"
            f" LEFT JOIN preds p ON {join_on}"
        ).fetch_df()
    finally:
        duck_conn.unregister("preds")


def per_site_metrics(
    points: pd.DataFrame, format_func: typing.Callable[[float], typing.Any]
) -> dict[str, typing.Any]:
    """Accuracy and macro recall per site over known-class points.

    A site with no known-class rows is skipped.
    """
    metrics: dict[str, typing.Any] = {}
    known = points[points["in_model_classes"]]
    for site in Sites:
        rows = known[known["site"] == site.value]
        if len(rows) == 0:
            continue
        gt = rows["gt_bagf"].tolist()
        est = rows["pred_bagf"].tolist()
        metrics[f"{site.value}/accuracy"] = format_func(accuracy_score(gt, est))
        metrics[f"{site.value}/macro_recall"] = format_func(
            recall_score(gt, est, average="macro", labels=sorted(set(gt)), zero_division=0)  # pyright: ignore[reportArgumentType]
        )
    return metrics


def log_holdout_evaluation(
    *,
    dataset: typing.Any,
    clf_calibrated: typing.Any,
    predictor: typing.Any,
    ba_library: typing.Any,
    gf_library: typing.Any,
    format_func: typing.Callable[[float], typing.Any],
) -> None:
    """Score the holdout and log its metrics and artifacts under `holdout/`.

    Does nothing when the dataset has no holdout points.
    """
    if len(dataset.holdout_labels) == 0:
        return

    evaluation = evaluate_holdout(clf_calibrated, dataset)
    duck_conn = dataset.duck_conn

    log_dataframe(
        duck_conn,
        holdout_annotations_df(duck_conn, evaluation.points),
        "annotations_holdout",
    )
    mlflow.log_metric(HOLDOUT_PREFIX + "points_unknown_class", float(evaluation.n_unknown_class))
    for name, value in per_site_metrics(evaluation.points, format_func).items():
        mlflow.log_metric(HOLDOUT_PREFIX + name, value)

    if evaluation.val_results is None:
        logger.warning("No holdout point has a class the model knows; metrics skipped")
        return
    ctx = MetricsContext(
        val_results=evaluation.val_results,
        ba_library=ba_library,
        gf_library=gf_library,
        format_func=format_func,
        dataset=dataset,
        clf=predictor,
        eval_labels=evaluation.known_labels,
    )
    MetricsCoordinator(ctx, duck_conn=duck_conn, prefix=HOLDOUT_PREFIX).compute_and_log_all()
