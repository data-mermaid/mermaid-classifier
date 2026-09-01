"""Synthesize expert / GT / V1 / Beta labels into the comparison table (top-level)."""

from collections.abc import Callable
from itertools import combinations
from typing import Any, cast

import pandas as pd

from mermaid_classifier.model_review.ls_export import ExpertLabel


def build_point_table(
    tasks: list[dict[str, Any]],
    expert_labels: list[ExpertLabel],
    roll: Callable[[str], str | None],
) -> pd.DataFrame:
    ref: dict[tuple[str, int, int], tuple[str | None, str | None, str | None]] = {}
    # Per-image provenance (image set + site + source + durable s3:// path), recorded next
    # to each annotation so the image is unambiguous when comparing. `site` is what lets
    # agreement be read per data source rather than pooled.
    meta: dict[str, tuple[str, str, str, str]] = {}
    for task in tasks:
        data = task["data"]
        image_id = data["image_id"]
        meta[image_id] = (
            data.get("image_set", ""),
            data.get("source", ""),
            data.get("source_id", ""),
            data.get("image_url", ""),
        )
        for p in data["original_points"]:
            # A tasks file with no Beta reference layer leaves `beta` unset.
            beta = p.get("beta")
            ref[(image_id, p["row"], p["col"])] = (
                roll(p["gt"]),
                roll(p["v1"]),
                roll(beta) if beta else None,
            )

    records = []
    for el in expert_labels:
        key = (el.image_id, el.row, el.col)
        gt_top, v1_top, beta_top = ref.get(key, (None, None, None))
        image_set, site, source_id, image_url = meta.get(el.image_id, ("", "", "", ""))
        records.append(
            {
                "image_set": image_set,
                "site": site,
                "image_id": el.image_id,
                "source_id": source_id,
                "image_url": image_url,
                "row": el.row,
                "col": el.col,
                "gt_top": gt_top,
                "v1_top": v1_top,
                "beta_top": beta_top,
                "expert": el.expert,
                "expert_top": roll(el.bagf),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "image_set",
            "site",
            "image_id",
            "source_id",
            "image_url",
            "row",
            "col",
            "gt_top",
            "v1_top",
            "beta_top",
            "expert",
            "expert_top",
        ],
    )


def _match_rate(a: list[Any], b: list[Any]) -> float:
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def model_vs_gt_rate(
    tasks: list[dict[str, Any]], roll: Callable[[str], str | None], model_key: str
) -> float:
    """A reference model's match rate against GT over ALL distinct held-out points,
    independent of whether an expert reviewed them (both come from the tasks, not the
    experts). ``model_key`` is the ``original_points`` key: ``"v1"`` or ``"beta"``."""
    seen: set[tuple[str, int, int]] = set()
    gts: list[str | None] = []
    preds: list[str | None] = []
    for task in tasks:
        image_id = task["data"]["image_id"]
        for p in task["data"]["original_points"]:
            key = (image_id, p["row"], p["col"])
            if key in seen:
                continue
            seen.add(key)
            pred = p.get(model_key)
            gts.append(roll(p["gt"]))
            preds.append(roll(pred) if pred else None)
    return _match_rate(gts, preds)


def v1_vs_gt_rate(tasks: list[dict[str, Any]], roll: Callable[[str], str | None]) -> float:
    return model_vs_gt_rate(tasks, roll, "v1")


def beta_vs_gt_rate(tasks: list[dict[str, Any]], roll: Callable[[str], str | None]) -> float:
    return model_vs_gt_rate(tasks, roll, "beta")


def _rates(
    df: pd.DataFrame,
    tasks: list[dict[str, Any]],
    roll: Callable[[str], str | None],
) -> dict[str, float]:
    """The comparison rates over whatever slice of df/tasks is passed."""
    # #1 V1/Beta vs GT — over ALL held-out points (not limited to expert-reviewed ones)
    v1_vs_gt = v1_vs_gt_rate(tasks, roll)
    beta_vs_gt = beta_vs_gt_rate(tasks, roll)

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
        "beta_vs_gt": beta_vs_gt,
        "expert_vs_gt": expert_vs_gt,
        "expert_vs_expert": expert_vs_expert,
    }


def agreement_by_site(
    df: pd.DataFrame,
    tasks: list[dict[str, Any]],
    roll: Callable[[str], str | None],
) -> dict[str, dict[str, float]]:
    """The same rates, computed per data source.

    A site with no expert labels yields NaN rather than raising — the "nothing rolled up"
    guard belongs on the overall summary, not on one empty slice of it.
    """
    sites = sorted({task["data"].get("source", "") for task in tasks})
    return {
        site: _rates(
            # cast: pandas boolean indexing is typed as Series | DataFrame
            cast(pd.DataFrame, df.loc[df["site"] == site]),
            [t for t in tasks if t["data"].get("source", "") == site],
            roll,
        )
        for site in sites
    }


def image_counts_by_site(tasks: list[dict[str, Any]]) -> dict[str, int]:
    """Images per site in the built project — confirms the intended split actually landed."""
    counts: dict[str, int] = {}
    for task in tasks:
        site = task["data"].get("source", "")
        counts[site] = counts.get(site, 0) + 1
    return dict(sorted(counts.items()))


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

    return _rates(df, tasks, roll)
