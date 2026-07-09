"""Build Label Studio tasks: image + fixed points + GT/V1 reference predictions.

Each point is rendered as a Label Studio keypoint region carrying a perRegion
Taxonomy label. That requires TWO result entries sharing one region id: a
``keypoint`` result (the geometry, bound to the KeyPoint control ``kp``) and a
``taxonomy`` result (the label path, bound to the Taxonomy control ``label``).
A plain ``keypointlabels`` result does NOT bind to a Taxonomy control, so the
label would not render (verified in the pilot).
"""

from collections.abc import Callable
from typing import Any

from mermaid_classifier.model_review.sample import ReviewPoint


def _point_results(
    idx: int, x: float, y: float, width: int, height: int, path: list[str]
) -> list[dict[str, Any]]:
    """A keypoint region + its perRegion taxonomy label, sharing one region id."""
    region_id = f"pt-{idx}"
    return [
        {
            "id": region_id,
            "type": "keypoint",
            "from_name": "kp",
            "to_name": "image",
            "original_width": width,
            "original_height": height,
            "value": {"x": x, "y": y, "width": 0.5},
        },
        {
            "id": region_id,
            "type": "taxonomy",
            "from_name": "label",
            "to_name": "image",
            "value": {"taxonomy": [path]},
        },
    ]


def build_task(
    image_id: str,
    points: list[ReviewPoint],
    v1_preds: dict[tuple[str, int, int], str],
    label_path: Callable[[str], list[str]],
    image_url: Callable[[str, str], str],
    image_size: Callable[[str, str], tuple[int, int]],
) -> dict[str, Any]:
    img_points = sorted([p for p in points if p.image_id == image_id], key=lambda p: (p.row, p.col))
    source_id = img_points[0].source_id
    width, height = image_size(source_id, image_id)

    original_points = []
    gt_result: list[dict[str, Any]] = []
    v1_result: list[dict[str, Any]] = []
    for idx, p in enumerate(img_points):
        x = p.col / width * 100
        y = p.row / height * 100
        v1_bagf = v1_preds[(p.image_id, p.row, p.col)]
        original_points.append({"row": p.row, "col": p.col, "gt": p.gt_bagf, "v1": v1_bagf})
        gt_result.extend(_point_results(idx, x, y, width, height, label_path(p.gt_bagf)))
        v1_result.extend(_point_results(idx, x, y, width, height, label_path(v1_bagf)))

    return {
        "data": {
            "image_url": image_url(source_id, image_id),
            "source": "coralnet",
            "image_id": image_id,
            "source_id": source_id,
            "original_points": original_points,
        },
        "predictions": [
            {"model_version": "ground-truth", "result": gt_result},
            {"model_version": "v1", "result": v1_result},
        ],
    }


def build_tasks(
    points: list[ReviewPoint],
    v1_preds: dict[tuple[str, int, int], str],
    label_path: Callable[[str], list[str]],
    image_url: Callable[[str, str], str],
    image_size: Callable[[str, str], tuple[int, int]],
) -> list[dict[str, Any]]:
    image_ids = sorted({p.image_id for p in points})
    return [
        build_task(image_id, points, v1_preds, label_path, image_url, image_size)
        for image_id in image_ids
    ]
