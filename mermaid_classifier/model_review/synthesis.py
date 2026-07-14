"""Synthesize expert / GT / V1 labels into the three comparisons (top-level)."""

from collections.abc import Callable
from itertools import combinations
from typing import Any

import pandas as pd

from mermaid_classifier.model_review.ls_export import ExpertLabel


def build_point_table(
    tasks: list[dict[str, Any]],
    expert_labels: list[ExpertLabel],
    roll: Callable[[str], str | None],
) -> pd.DataFrame:
    ref: dict[tuple[str, int, int], tuple[str | None, str | None]] = {}
    # Per-image provenance (image set + source + durable s3:// path), recorded next to
    # each annotation so the image is unambiguous when comparing.
    meta: dict[str, tuple[str, str, str]] = {}
    for task in tasks:
        data = task["data"]
        image_id = data["image_id"]
        meta[image_id] = (
            data.get("image_set", ""),
            data.get("source_id", ""),
            data.get("image_url", ""),
        )
        for p in data["original_points"]:
            ref[(image_id, p["row"], p["col"])] = (roll(p["gt"]), roll(p["v1"]))

    records = []
    for el in expert_labels:
        key = (el.image_id, el.row, el.col)
        gt_top, v1_top = ref.get(key, (None, None))
        image_set, source_id, image_url = meta.get(el.image_id, ("", "", ""))
        records.append(
            {
                "image_set": image_set,
                "image_id": el.image_id,
                "source_id": source_id,
                "image_url": image_url,
                "row": el.row,
                "col": el.col,
                "gt_top": gt_top,
                "v1_top": v1_top,
                "expert": el.expert,
                "expert_top": roll(el.bagf),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "image_set",
            "image_id",
            "source_id",
            "image_url",
            "row",
            "col",
            "gt_top",
            "v1_top",
            "expert",
            "expert_top",
        ],
    )


def _match_rate(a: list[Any], b: list[Any]) -> float:
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def v1_vs_gt_rate(tasks: list[dict[str, Any]], roll: Callable[[str], str | None]) -> float:
    """V1-vs-GT match rate over ALL distinct held-out points, independent of
    whether an expert reviewed them (GT/V1 come from the tasks, not the experts)."""
    seen: set[tuple[str, int, int]] = set()
    gts: list[str | None] = []
    v1s: list[str | None] = []
    for task in tasks:
        image_id = task["data"]["image_id"]
        for p in task["data"]["original_points"]:
            key = (image_id, p["row"], p["col"])
            if key in seen:
                continue
            seen.add(key)
            gts.append(roll(p["gt"]))
            v1s.append(roll(p["v1"]))
    return _match_rate(gts, v1s)


def agreement_summary(
    df: pd.DataFrame,
    tasks: list[dict[str, Any]],
    roll: Callable[[str], str | None],
) -> dict[str, float]:
    expert_tops = [
        expert_top
        for expert, expert_top in zip(df["expert"], df["expert_top"], strict=True)
        if expert is not None
    ]
    if expert_tops and all(t is None for t in expert_tops):
        raise ValueError(
            "no expert labels rolled up to a top-level category — check the "
            "name-vs-id mapping between the Label Studio config and the rollup"
        )

    # #1 V1 vs GT — over ALL held-out points (not limited to expert-reviewed ones)
    v1_vs_gt = v1_vs_gt_rate(tasks, roll)

    # #2 Expert vs GT — every expert label vs the point's GT
    expert_vs_gt = _match_rate(df["expert_top"].tolist(), df["gt_top"].tolist())

    # #3 Expert vs Expert — pairwise agreement on shared points
    agree = total = 0
    grouped = df.groupby(["image_id", "row", "col"])  # per point
    for _, g in grouped:
        labels = list(zip(g["expert"], g["expert_top"], strict=True))
        for (_, la), (_, lb) in combinations(labels, 2):
            if la is None or lb is None:
                continue
            total += 1
            if la == lb:
                agree += 1
    expert_vs_expert = (agree / total) if total else float("nan")

    return {
        "v1_vs_gt": v1_vs_gt,
        "expert_vs_gt": expert_vs_gt,
        "expert_vs_expert": expert_vs_expert,
    }
