"""Parse a Label Studio JSON export into expert labels + notes.

Each point is a keypoint region whose id is ``pt-<index>`` (matching the seeded
skeleton / ``original_points`` order) carrying a perRegion taxonomy label. We join
by that id rather than by position, and iterate ``original_points`` so every fixed
point yields exactly one ExpertLabel — a point the expert left unlabeled (no
taxonomy result, or a deleted keypoint) is reported as ``UNLABELED``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

UNLABELED = "UNLABELED"


def expert_name(completed_by: Any, user_map: dict[str, str] | None) -> str:
    """Resolve an annotation's ``completed_by`` to a stable expert identity.

    ``completed_by`` is the Label Studio account that made the annotation — usually a
    numeric user id, but some exports embed a ``{id, email, ...}`` object. Prefer an
    email (from the object, or from ``user_map`` id->email built from ``/api/users``);
    fall back to the raw id string so attribution is never lost.
    """
    if isinstance(completed_by, dict):
        return completed_by.get("email") or str(completed_by.get("id"))
    if user_map:
        return user_map.get(str(completed_by), str(completed_by))
    return str(completed_by)


@dataclass
class ExpertLabel:
    image_id: str
    expert: str
    row: int
    col: int
    bagf: str  # reconstructed BA_ID::GF_ID, or UNLABELED


@dataclass
class ExpertNote:
    image_id: str
    expert: str
    note: str


def parse_export(
    tasks: list[dict[str, Any]],
    path_to_bagf: Callable[[list[str]], str],
    user_map: dict[str, str] | None = None,
) -> tuple[list[ExpertLabel], list[ExpertNote]]:
    """Parse LS tasks-with-annotations. `path_to_bagf` maps a taxonomy name path
    (root->leaf, optional trailing growth form) to a BA_ID::GF_ID string.

    `user_map` (optional id->email, e.g. from `/api/users`) resolves each
    annotation's author to their email so results are attributable/filterable by
    reviewer; without it the raw account id is used.
    """
    labels: list[ExpertLabel] = []
    notes: list[ExpertNote] = []
    for task in tasks:
        image_id = task["data"]["image_id"]
        original = task["data"]["original_points"]
        for ann in task.get("annotations", []):
            expert = expert_name(ann.get("completed_by"), user_map)
            result = ann["result"]
            keypoint_ids = {
                r["id"] for r in result if r.get("type") in ("keypoint", "keypointlabels")
            }
            tax_by_id = {
                r["id"]: r["value"]["taxonomy"]
                for r in result
                if r.get("type") == "taxonomy" and r["value"].get("taxonomy")
            }
            for idx, point in enumerate(original):
                region_id = f"pt-{idx}"
                paths = tax_by_id.get(region_id)
                # point deleted or left unlabeled -> UNLABELED (never dropped)
                bagf = path_to_bagf(paths[0]) if region_id in keypoint_ids and paths else UNLABELED
                labels.append(
                    ExpertLabel(
                        image_id=image_id,
                        expert=expert,
                        row=point["row"],
                        col=point["col"],
                        bagf=bagf,
                    )
                )
            for region in result:
                if region.get("type") == "textarea" and region.get("from_name") == "notes":
                    text = region["value"]["text"]
                    joined = " ".join(text) if isinstance(text, list) else str(text)
                    if joined.strip():
                        notes.append(ExpertNote(image_id=image_id, expert=expert, note=joined))
    return labels, notes
