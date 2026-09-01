"""Run the V1 classifier over pre-extracted features for review points."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mermaid_classifier.model_review.features import (
    PointKey,
    default_feature_loader,
    stack_features,
)
from mermaid_classifier.model_review.sample import ReviewPoint

__all__ = ["default_feature_loader", "predict_from_features", "predict_points"]


def predict_from_features(
    keys: Sequence[PointKey],
    features: NDArray[np.float32],
    predictor: Any,
) -> dict[PointKey, str]:
    """Top-1 BA::GF per point, from a matrix whose rows align with ``keys``."""
    labels = list(predictor.classes)
    proba = predictor.predict_proba(features)
    return {
        key: labels[int(np.argmax(row_proba))] for key, row_proba in zip(keys, proba, strict=True)
    }


def predict_points(
    points: list[ReviewPoint],
    classifier_location: str,
    load_features: Callable[[str, str], Any] = default_feature_loader,
    predictor: Any = None,
) -> dict[PointKey, str]:
    if predictor is None:
        from mermaid_classifier.pyspacer.annotation import resolve_classifier_artifact
        from mermaid_classifier.pyspacer.inference import load_predictor

        model_pt, model_json = resolve_classifier_artifact(classifier_location)
        predictor = load_predictor(model_pt, model_json)

    keys, features = stack_features(points, load_features)
    return predict_from_features(keys, features, predictor)
