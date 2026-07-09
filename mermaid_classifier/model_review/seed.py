"""Pre-seed one blank annotation per task x expert.

The blank layer is the fixed keypoints WITHOUT any taxonomy label — each expert
opens their own copy and assigns a label to every point via the Taxonomy tree.
A point the expert never labels stays a bare keypoint (no taxonomy result) and is
reported as unlabeled on export.
"""

import copy
from typing import Any


def blank_annotation_result(reference_result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the keypoint geometry from a reference result, dropping labels.

    The reference (a GT/V1 prediction) carries paired ``keypoint`` + ``taxonomy``
    entries per point; the blank expert layer keeps the ``keypoint`` regions so the
    fixed points are present, but drops the ``taxonomy`` labels so the expert fills
    them in independently.
    """
    return [copy.deepcopy(r) for r in reference_result if r.get("type") == "keypoint"]


def seed_all(client: Any, project_id: int, expert_user_ids: list[int]) -> int:
    """Create one blank annotation per (task, expert). SDK calls validated in the pilot.

    Uses the label_studio_sdk project API: list tasks, then create an annotation from
    the task's ground-truth prediction keypoint skeleton, attributed to each expert.
    """
    project = client.get_project(project_id)
    count = 0
    for task in project.get_tasks():
        gt_pred = next(
            (p for p in task.get("predictions", []) if p.get("model_version") == "ground-truth"),
            None,
        )
        if gt_pred is None:
            continue
        blank = blank_annotation_result(gt_pred["result"])
        for user_id in expert_user_ids:
            project.create_annotation(task["id"], result=blank, completed_by=user_id)
            count += 1
    return count
