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
    for task in tasks:
        image_id = task["data"]["image_id"]
        for p in task["data"]["original_points"]:
            ref[(image_id, p["row"], p["col"])] = (roll(p["gt"]), roll(p["v1"]))

    records = []
    for el in expert_labels:
        key = (el.image_id, el.row, el.col)
        gt_top, v1_top = ref.get(key, (None, None))
        records.append(
            {
                "image_id": el.image_id,
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
        columns=["image_id", "row", "col", "gt_top", "v1_top", "expert", "expert_top"],
    )


def _match_rate(a: list[Any], b: list[Any]) -> float:
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def agreement_summary(df: pd.DataFrame) -> dict[str, float]:
    # #1 V1 vs GT — one row per distinct point (dedupe experts)
    points = df.drop_duplicates(subset=["image_id", "row", "col"])
    v1_vs_gt = _match_rate(points["gt_top"].tolist(), points["v1_top"].tolist())

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
