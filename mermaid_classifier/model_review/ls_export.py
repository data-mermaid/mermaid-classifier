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
) -> tuple[list[ExpertLabel], list[ExpertNote]]:
    """Parse LS tasks-with-annotations. `path_to_bagf` maps a taxonomy name path
    (root->leaf, optional trailing growth form) to a BA_ID::GF_ID string."""
    labels: list[ExpertLabel] = []
    notes: list[ExpertNote] = []
    for task in tasks:
        image_id = task["data"]["image_id"]
        original = task["data"]["original_points"]
        for ann in task.get("annotations", []):
            expert = str(ann.get("completed_by"))
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
