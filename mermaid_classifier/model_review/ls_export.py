"""Parse a Label Studio JSON export into expert labels + notes."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ExpertLabel:
    image_id: str
    expert: str
    row: int
    col: int
    bagf: str


@dataclass
class ExpertNote:
    image_id: str
    expert: str
    note: str


def parse_export(tasks: list[dict[str, Any]]) -> tuple[list[ExpertLabel], list[ExpertNote]]:
    labels: list[ExpertLabel] = []
    notes: list[ExpertNote] = []
    for task in tasks:
        image_id = task["data"]["image_id"]
        original = task["data"]["original_points"]
        for ann in task.get("annotations", []):
            expert = str(ann.get("completed_by"))
            keypoints = [r for r in ann["result"] if r.get("type") == "keypointlabels"]
            if len(keypoints) != len(original):
                raise ValueError(
                    f"annotation for image {image_id} by expert {expert} has "
                    f"{len(keypoints)} keypoints but {len(original)} original points"
                )
            for point, region in zip(original, keypoints, strict=True):
                labels.append(
                    ExpertLabel(
                        image_id=image_id,
                        expert=expert,
                        row=point["row"],
                        col=point["col"],
                        bagf=region["value"]["keypointlabels"][0],
                    )
                )
            for region in ann["result"]:
                if region.get("type") == "textarea" and region.get("from_name") == "notes":
                    text = region["value"]["text"]
                    joined = " ".join(text) if isinstance(text, list) else str(text)
                    if joined.strip():
                        notes.append(ExpertNote(image_id=image_id, expert=expert, note=joined))
    return labels, notes
