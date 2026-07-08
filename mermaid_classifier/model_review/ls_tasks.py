"""Build Label Studio tasks: image + fixed points + GT/V1 reference predictions."""

from collections.abc import Callable
from typing import Any

from mermaid_classifier.model_review.sample import ReviewPoint


def _keypoint_result(
    idx: int, x: float, y: float, width: int, height: int, label: str
) -> dict[str, Any]:
    return {
        "id": f"pt-{idx}",
        "type": "keypointlabels",
        "from_name": "label",
        "to_name": "image",
        "original_width": width,
        "original_height": height,
        "value": {"x": x, "y": y, "width": 0.5, "keypointlabels": [label]},
    }


def build_task(
    image_id: str,
    points: list[ReviewPoint],
    v1_preds: dict[tuple[str, int, int], str],
    label_name: Callable[[str], str],
    image_url: Callable[[str, str], str],
    image_size: Callable[[str, str], tuple[int, int]],
) -> dict[str, Any]:
    img_points = sorted([p for p in points if p.image_id == image_id], key=lambda p: (p.row, p.col))
    source_id = img_points[0].source_id
    width, height = image_size(source_id, image_id)

    original_points = []
    gt_result = []
    v1_result = []
    for idx, p in enumerate(img_points):
        x = p.col / width * 100
        y = p.row / height * 100
        v1_bagf = v1_preds[(p.image_id, p.row, p.col)]
        original_points.append({"row": p.row, "col": p.col, "gt": p.gt_bagf, "v1": v1_bagf})
        gt_result.append(_keypoint_result(idx, x, y, width, height, label_name(p.gt_bagf)))
        v1_result.append(_keypoint_result(idx, x, y, width, height, label_name(v1_bagf)))

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
    label_name: Callable[[str], str],
    image_url: Callable[[str, str], str],
    image_size: Callable[[str, str], tuple[int, int]],
) -> list[dict[str, Any]]:
    image_ids = sorted({p.image_id for p in points})
    return [
        build_task(image_id, points, v1_preds, label_name, image_url, image_size)
        for image_id in image_ids
    ]
