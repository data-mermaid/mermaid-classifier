"""Pre-seed one blank (Unlabeled) annotation per task x expert."""

import copy
from typing import Any

PLACEHOLDER = "Unlabeled"


def blank_annotation_result(reference_result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clone a prediction result but replace every label with the placeholder."""
    blank = copy.deepcopy(reference_result)
    for region in blank:
        region["value"]["keypointlabels"] = [PLACEHOLDER]
    return blank


def seed_all(client: Any, project_id: int, expert_user_ids: list[int]) -> int:
    """Create one blank annotation per (task, expert). SDK calls validated in the pilot.

    Uses the label_studio_sdk project API: list tasks, then create an annotation from
    the task's ground-truth prediction skeleton, attributed to each expert.
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
