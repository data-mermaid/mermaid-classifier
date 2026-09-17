"""Sorting region-mismatch events into what they are evidence of.

An event is a prediction that could not be confirmed in its image's region:
either out of region, or carrying an attribute with no recorded regions at
all. Each one lands in exactly one of three buckets.

`list_suspect` -- experts have annotated this (attribute, region) pair at
least `threshold` times across the corpus. Repeated human annotation is
evidence that the region list is incomplete, not that the model is wrong:
*Stypopodium* appears 605 times in the Western Indo-Pacific in ground truth,
which is a broadly distributed genus rather than 605 mistakes.

`unknown_list` -- the predicted attribute has no recorded regions, so the
question cannot be answered from the data at hand.

`model_error` -- everything else.

There is no fourth bucket for an ambiguous site. All 12 MEOW region polygons
measured mutually disjoint (none of the 66 pairs intersect, and their areas
sum exactly to their union), so the upstream first-intersecting-polygon
assignment is deterministic except exactly on a shared boundary.

Triage annotates; it never filters. No label is dropped from a headline rate
because of its bucket, and `region_list_suspects` leaves with the counts that
make it a taxonomy bug report for the data team rather than an exclusion list.
"""

import dataclasses
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum

import pandas as pd

from mermaid_classifier.region_eval.metrics import ScoredPoints

DEFAULT_LIST_SUSPECT_THRESHOLD = 5

EVENT_COLUMNS = (
    "point_position",
    "image_id",
    "image_region_id",
    "gt_label",
    "pred_label",
    "pred_attribute_id",
    "allowed_region_ids",
    "ground_truth_count",
    "bucket",
)
BUCKET_COUNT_COLUMNS = ("bucket", "n")
REGION_LIST_SUSPECT_COLUMNS = ("attribute_id", "region_id", "n_ground_truth", "n_predicted")


@dataclasses.dataclass(frozen=True)
class _Event:
    """One region-mismatch event, before it becomes a table row."""

    point_position: int
    image_id: str
    image_region_id: str
    gt_label: str
    pred_label: str
    pred_attribute_id: str
    allowed_region_ids: tuple[str, ...]
    ground_truth_count: int
    bucket: "TriageBucket"


class TriageBucket(StrEnum):
    """What one event is evidence of. Ordered most to least actionable upstream."""

    LIST_SUSPECT = "list_suspect"
    UNKNOWN_LIST = "unknown_list"
    MODEL_ERROR = "model_error"


@dataclasses.dataclass(frozen=True)
class TriageResult:
    n_unrecorded_region_excluded: int
    threshold: int
    events: pd.DataFrame
    bucket_counts: pd.DataFrame
    region_list_suspects: pd.DataFrame


def triage_events(
    points: ScoredPoints,
    *,
    ground_truth_counts: Mapping[tuple[str, str], int],
    threshold: int = DEFAULT_LIST_SUSPECT_THRESHOLD,
) -> TriageResult:
    """Bucket every region-mismatch event in a scored slice.

    `ground_truth_counts` maps (benthic attribute id, region id) to the number
    of confirmed human annotations of that pair corpus-wide. A pair the
    mapping omits counts as zero, which is a `model_error` rather than a
    missing row.

    `points` arrives already filtered of unrecorded image regions, and the
    count of what was dropped travels through to the result.
    """
    if threshold < 1:
        raise ValueError(f"threshold must be at least 1, got {threshold}")

    events: list[_Event] = []
    for position in range(points.n_points):
        out_of_region = bool(points.pred_out_of_region[position])
        region_unknown = bool(points.pred_region_unknown[position])
        if not (out_of_region or region_unknown):
            continue
        attribute_id = points.pred_attribute_ids[position]
        region_id = points.image_region_ids[position]
        annotations = ground_truth_counts.get((attribute_id, region_id), 0)
        events.append(
            _Event(
                point_position=position,
                image_id=points.image_ids[position],
                image_region_id=region_id,
                gt_label=points.gt_labels[position],
                pred_label=points.pred_labels[position],
                pred_attribute_id=attribute_id,
                allowed_region_ids=tuple(sorted(points.pred_allowed[position])),
                ground_truth_count=annotations,
                bucket=_bucket(region_unknown, annotations, threshold),
            )
        )

    return TriageResult(
        n_unrecorded_region_excluded=points.n_unrecorded_region_excluded,
        threshold=threshold,
        events=_frame([dataclasses.asdict(event) for event in events], EVENT_COLUMNS),
        bucket_counts=_bucket_counts(events),
        region_list_suspects=_region_list_suspects(events),
    )


def _bucket(region_unknown: bool, annotations: int, threshold: int) -> TriageBucket:
    """The one bucket an event belongs to.

    An attribute with no recorded regions is checked first: no annotation
    count can make a list suspect when there is no list.
    """
    if region_unknown:
        return TriageBucket.UNKNOWN_LIST
    if annotations >= threshold:
        return TriageBucket.LIST_SUSPECT
    return TriageBucket.MODEL_ERROR


def _bucket_counts(events: Sequence[_Event]) -> pd.DataFrame:
    """Every bucket, including the ones that caught nothing."""
    counts: Counter[TriageBucket] = Counter(event.bucket for event in events)
    return _frame(
        [{"bucket": bucket, "n": counts[bucket]} for bucket in TriageBucket],
        BUCKET_COUNT_COLUMNS,
    )


def _region_list_suspects(events: Sequence[_Event]) -> pd.DataFrame:
    """Attribute-region pairs the ground truth supports but the region list omits."""
    predictions: Counter[tuple[str, str]] = Counter()
    annotations: dict[tuple[str, str], int] = {}
    for event in events:
        if event.bucket is not TriageBucket.LIST_SUSPECT:
            continue
        pair = (event.pred_attribute_id, event.image_region_id)
        predictions[pair] += 1
        annotations[pair] = event.ground_truth_count

    ranked = sorted(
        ((annotations[pair], pair[0], pair[1], count) for pair, count in predictions.items()),
        key=lambda entry: (-entry[0], entry[1], entry[2]),
    )
    return _frame(
        [
            {
                "attribute_id": attribute_id,
                "region_id": region_id,
                "n_ground_truth": n_ground_truth,
                "n_predicted": n_predicted,
            }
            for n_ground_truth, attribute_id, region_id, n_predicted in ranked
        ],
        REGION_LIST_SUSPECT_COLUMNS,
    )


def _frame(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    """A DataFrame that keeps its columns even with no rows."""
    if not rows:
        return pd.DataFrame(columns=pd.Index(columns))
    return pd.DataFrame(list(rows), columns=pd.Index(columns))
