"""Run the V1 classifier over pre-extracted features for review points."""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from mermaid_classifier.model_review.sample import ReviewPoint


def default_feature_loader(bucket: str, key: str):
    from spacer.data_classes import DataLocation, ImageFeatures

    return ImageFeatures.load(DataLocation("s3", bucket_name=bucket, key=key))


def predict_points(
    points: list[ReviewPoint],
    classifier_location: str,
    load_features: Callable[[str, str], Any] = default_feature_loader,
    predictor: Any = None,
) -> dict[tuple[str, int, int], str]:
    if predictor is None:
        from mermaid_classifier.pyspacer.annotation import resolve_classifier_artifact
        from mermaid_classifier.pyspacer.inference import load_predictor

        model_pt, model_json = resolve_classifier_artifact(classifier_location)
        predictor = load_predictor(model_pt, model_json)

    labels = list(predictor.classes)

    by_image: dict[tuple[str, str], list[ReviewPoint]] = defaultdict(list)
    for p in points:
        by_image[(p.bucket, p.feature_key)].append(p)

    results: dict[tuple[str, int, int], str] = {}
    for (bucket, key), img_points in by_image.items():
        feats = load_features(bucket, key)
        batch = np.vstack([feats.get_array((p.row, p.col)) for p in img_points])
        proba = predictor.predict_proba(batch)
        for p, row_proba in zip(img_points, proba, strict=True):
            i = int(np.argmax(row_proba))
            results[(p.image_id, p.row, p.col)] = labels[i]
    return results
