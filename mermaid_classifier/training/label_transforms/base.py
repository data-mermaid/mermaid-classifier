"""Transform abstract base class and TransformPlan dataclass.

A Transform decides — based on per-class counts in the *training* split —
which BAGF classes to drop or remap, then applies that decision uniformly
to all three splits (train/ref/val). The decision is captured in a
TransformPlan: an explicit, returned record of every per-class action,
which replaces the implicit module-level state the old rare-policy used
for MLflow logging.

The shared BALibrary / GFLibrary Protocols are duck-typed minimally so
unit tests can inject in-memory fakes without touching the MERMAID API.
"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import ClassVar, Protocol

from spacer.data_classes import ImageLabels
from spacer.task_utils import TrainingTaskLabels


class BALibrary(Protocol):
    """Minimal interface used by transforms. Implemented by the real
    ``BenthicAttributeLibrary`` and by test fakes."""

    by_id: dict[str, dict]                     # ba_id -> {'id', 'name', 'parent', ...}

    def get_ancestor_ids(self, ba_id: str) -> list[str]: ...


class GFLibrary(Protocol):
    """Minimal interface used by transforms. Used for human-readable
    names in MLflow logs; transform math treats GF IDs as opaque."""

    by_id: dict[str, str]                      # gf_id -> name


# Sentinel returned by TransformPlan.decide for classes that the transform
# left untouched. Distinguished from "drop" and "remap" so the caller can
# branch cleanly on the three cases without sentinel-string magic.
KEEP: str = "keep"
DROP: str = "drop"
REMAP: str = "remap"


@dataclasses.dataclass
class TransformPlan:
    """Per-class decisions produced by ``Transform.plan``.

    ``actions`` maps source BAGF -> "keep" | "drop". Classes absent from
    ``actions`` are implicitly kept (callers can use plan.decide() to
    abstract over both cases). ``remap`` maps source BAGF -> target BAGF
    for classes that get relabelled (used by merge transforms). A class
    listed in ``remap`` overrides any "keep"/"drop" entry in ``actions``.

    The plan is **pure data**: serialisable, comparable, loggable. The
    same plan can be replayed on a fresh ImageLabels later (e.g. for
    re-evaluation) without re-deriving from counts.
    """

    transform_name: str
    actions: dict[str, str] = dataclasses.field(default_factory=dict)
    remap: dict[str, str] = dataclasses.field(default_factory=dict)
    # Snapshot of training-split counts at planning time, so the artifact
    # logged to MLflow records why each decision was made without forcing
    # the reader back to the dataset.
    train_counts: dict[str, int] = dataclasses.field(default_factory=dict)

    def decide(self, bagf: str) -> tuple[str, str | None]:
        """Resolve the action for one source BAGF.

        Returns ``(KEEP, None)``, ``(DROP, None)``, or ``(REMAP, target)``.
        """
        if bagf in self.remap:
            return REMAP, self.remap[bagf]
        action = self.actions.get(bagf, KEEP)
        if action == DROP:
            return DROP, None
        return KEEP, None

    def to_log_records(self) -> list[dict]:
        """One row per source class, suitable for a CSV/DataFrame artifact."""
        rows: list[dict] = []
        all_classes = set(self.train_counts) | set(self.actions) | set(self.remap)
        for bagf in sorted(all_classes):
            action, target = self.decide(bagf)
            rows.append({
                "transform": self.transform_name,
                "bagf_id": bagf,
                "count": int(self.train_counts.get(bagf, 0)),
                "action": action,
                "target_bagf_id": target or "",
            })
        return rows


class Transform(ABC):
    """Base class for label transformations.

    Subclasses implement ``plan`` to decide what to do with each class.
    The base class provides a default ``apply`` that walks each split's
    annotations and emits a fresh ``ImageLabels`` with rewritten/dropped
    tuples — overridable for transforms that need bespoke iteration.
    """

    name: ClassVar[str]                       # set by @register_transform

    @abstractmethod
    def plan(
        self,
        train_class_counts: dict[str, int],
        ba_library: BALibrary,
        gf_library: GFLibrary,
    ) -> TransformPlan:
        """Decide drop/remap for each class based on training counts.

        Pure: must not mutate inputs. Must not depend on order.
        """
        ...

    def apply(
        self,
        labels: TrainingTaskLabels,
        plan: TransformPlan,
    ) -> TrainingTaskLabels:
        """Apply ``plan`` uniformly to train/ref/val.

        Default: rewrite each (row, col, bagf) tuple per the plan's
        decide() result; rebuild a fresh ImageLabels per split. Splits
        whose entire annotation set was dropped are left empty (caller
        is responsible for downstream validation).
        """
        for set_name in ("train", "ref", "val"):
            labels[set_name] = self._apply_to_split(labels[set_name], plan)
        return labels

    @staticmethod
    def _apply_to_split(
        split: ImageLabels,
        plan: TransformPlan,
    ) -> ImageLabels:
        out = ImageLabels()
        for feature_loc in split.keys():
            kept: list[tuple[int, int, str]] = []
            for row, col, bagf in split[feature_loc]:
                action, target = plan.decide(bagf)
                if action == DROP:
                    continue
                if action == REMAP:
                    kept.append((row, col, target))
                else:
                    kept.append((row, col, bagf))
            if kept:
                out.add_image(feature_loc, kept)
        return out
