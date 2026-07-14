"""Build Label Studio tasks: image + fixed points + reference predictions.

Each point is two result entries sharing one region id: a ``keypointlabels``
result bound to the colored top-level control ``toplevel`` (the geometry + the
top-level colour) and a ``taxonomy`` result bound to ``label`` (the fine BA::GF
path). Each task carries three predictions: ``blank`` (unlabelled fixed points —
the reviewer's blind starting layer, set as the project model_version),
``ground-truth`` and ``v1`` (read-only reference tabs).
"""

from collections.abc import Callable
from typing import Any

from mermaid_classifier.model_review.ls_config import UNLABELED_TOPLEVEL
from mermaid_classifier.model_review.sample import ReviewPoint

# The reviewer's blind starting layer, exposed as a prediction and set as the
# project's model_version so LS copies it into each reviewer's annotation on open.
# Shown as a (read-only) tab, so the name must read clearly as "your start point".
BLANK_MODEL_VERSION = "Unlabelled Starting Set"


def _blank_keypoint(idx: int, x: float, y: float, width: int, height: int) -> dict[str, Any]:
    """A fixed keypoint with the grey Unlabeled top-level and NO taxonomy label.

    Used for the 'blank' prediction that seeds each reviewer's starting annotation
    so they label blind (points present, no label) rather than starting on V1.
    """
    return {
        "id": f"pt-{idx}",
        "type": "keypointlabels",
        "from_name": "toplevel",
        "to_name": "image",
        "original_width": width,
        "original_height": height,
        "value": {"x": x, "y": y, "width": 0.5, "keypointlabels": [UNLABELED_TOPLEVEL]},
    }


def _point_results(
    idx: int,
    x: float,
    y: float,
    width: int,
    height: int,
    toplevel: str,
    path: list[str],
) -> list[dict[str, Any]]:
    """A colored top-level keypoint region + its perRegion taxonomy label.

    Both entries share one region id: the ``keypointlabels`` result (bound to the
    ``toplevel`` control) carries the geometry and the top-level colour label; the
    ``taxonomy`` result (bound to ``label``) carries the fine BA::GF path.
    """
    region_id = f"pt-{idx}"
    return [
        {
            "id": region_id,
            "type": "keypointlabels",
            "from_name": "toplevel",
            "to_name": "image",
            "original_width": width,
            "original_height": height,
            "value": {"x": x, "y": y, "width": 0.5, "keypointlabels": [toplevel]},
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
    toplevel_name: Callable[[str], str],
    image_url: Callable[[str, str], str],
    image_size: Callable[[str, str], tuple[int, int]],
) -> dict[str, Any]:
    img_points = sorted([p for p in points if p.image_id == image_id], key=lambda p: (p.row, p.col))
    source_id = img_points[0].source_id
    width, height = image_size(source_id, image_id)

    original_points = []
    gt_result: list[dict[str, Any]] = []
    v1_result: list[dict[str, Any]] = []
    blank_result: list[dict[str, Any]] = []
    for idx, p in enumerate(img_points):
        x = p.col / width * 100
        y = p.row / height * 100
        v1_bagf = v1_preds[(p.image_id, p.row, p.col)]
        original_points.append({"row": p.row, "col": p.col, "gt": p.gt_bagf, "v1": v1_bagf})
        gt_result.extend(
            _point_results(
                idx, x, y, width, height, toplevel_name(p.gt_bagf), label_path(p.gt_bagf)
            )
        )
        v1_result.extend(
            _point_results(idx, x, y, width, height, toplevel_name(v1_bagf), label_path(v1_bagf))
        )
        blank_result.append(_blank_keypoint(idx, x, y, width, height))

    # "blank" is the reviewer's starting layer (set as the project's model_version):
    # fixed points, unlabelled, so they label blind. ground-truth/v1 are read-only
    # reference tabs they can toggle to afterwards.
    return {
        "data": {
            "image_url": image_url(source_id, image_id),
            "source": "coralnet",
            "image_id": image_id,
            "source_id": source_id,
            "original_points": original_points,
        },
        "predictions": [
            {"model_version": BLANK_MODEL_VERSION, "result": blank_result},
            {"model_version": "ground-truth", "result": gt_result},
            {"model_version": "v1", "result": v1_result},
        ],
    }


def build_tasks(
    points: list[ReviewPoint],
    v1_preds: dict[tuple[str, int, int], str],
    label_path: Callable[[str], list[str]],
    toplevel_name: Callable[[str], str],
    image_url: Callable[[str, str], str],
    image_size: Callable[[str, str], tuple[int, int]],
    image_set: str = "",
) -> list[dict[str, Any]]:
    image_ids = sorted({p.image_id for p in points})
    tasks = [
        build_task(image_id, points, v1_preds, label_path, toplevel_name, image_url, image_size)
        for image_id in image_ids
    ]
    for task in tasks:
        # Provenance: which image set this task belongs to (carried through export/synthesis).
        task["data"]["image_set"] = image_set
    return tasks
