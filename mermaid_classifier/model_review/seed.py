"""Pre-seed one blank annotation per task x expert.

Each expert opens their own copy of the fixed points, coloured grey (top-level
``Unlabeled``), and assigns a fine label to every point via the Taxonomy tree. A
point the expert never labels stays grey with no taxonomy result and is reported
as unlabeled on export.
"""

import copy
from typing import Any

from mermaid_classifier.model_review.ls_config import UNLABELED_TOPLEVEL


def blank_annotation_result(reference_result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the keypoint regions from a reference result but blank their labels.

    The reference (a GT/V1 prediction) carries paired ``keypointlabels`` (top-level
    colour) + ``taxonomy`` (fine) entries per point. The blank expert layer keeps
    the keypoint regions so the fixed points are present, resets the top-level label
    to the grey ``Unlabeled`` placeholder, and drops the taxonomy so the expert
    fills in the fine label independently.
    """
    blank: list[dict[str, Any]] = []
    for r in reference_result:
        if r.get("type") == "keypointlabels":
            region = copy.deepcopy(r)
            region["value"]["keypointlabels"] = [UNLABELED_TOPLEVEL]
            blank.append(region)
    return blank


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
