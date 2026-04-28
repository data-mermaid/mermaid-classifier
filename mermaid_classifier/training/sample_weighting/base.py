"""Strategy abstract base class.

A Strategy maps per-class sample counts (BA+GF combo string -> count)
to per-class weights (BA+GF combo string -> float). Concrete strategies
implement ``compute_raw_weights`` with their own algorithm; the base
class wraps the call with shared behaviour:

  - validate strategy returns a strictly positive weight for every key
    in ``class_counts``;
  - apply the optional ``weight_ratio_cap`` to bound how much a single
    class can dominate the loss.

Drop/merge of rare classes lives in
``mermaid_classifier.training.label_transforms`` and is applied to the
data pipeline (training labels, validation labels, classifier classes_)
before any weighting runs. This module sees only the kept label set.

Why a Protocol-style ``ba_library`` instead of importing
``BenthicAttributeLibrary`` directly: that class hits the MERMAID API
on construction, so unit tests must inject a fake. We rely on duck
typing on the two methods we use: ``by_id`` (dict) and
``get_ancestor_ids``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol


class BALibrary(Protocol):
    """Minimal interface used by strategies. Implemented by the real
    ``BenthicAttributeLibrary`` and by test fakes."""

    by_id: dict[str, dict]                   # ba_id -> {'id', 'name', 'parent', ...}

    def get_ancestor_ids(self, ba_id: str) -> list[str]: ...


class GFLibrary(Protocol):
    """Minimal interface used by strategies. We currently only use
    GF for human-readable names in MLflow logs; the strategy math
    treats the GF ID string itself as the unit."""

    by_id: dict[str, str]                    # gf_id -> name


class Strategy(ABC):
    """Base class for sample-weighting strategies.

    Subclasses implement ``compute_raw_weights``. The base class enforces
    the positivity contract and applies the optional weight ratio cap.

    Constructor arguments:
        alpha            -- smoothing exponent in [0, 1] (algorithm-specific
                            meaning; see each subclass docstring).
        weight_ratio_cap -- optional cap on max:min ratio of weights,
                            applied as a "lower the ceiling" clip in
                            ``compute()``. ``None`` disables capping.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        weight_ratio_cap: float | None = None,
    ):
        self.alpha = float(alpha)
        self.weight_ratio_cap = (
            None if weight_ratio_cap is None else float(weight_ratio_cap)
        )

    @abstractmethod
    def compute_raw_weights(
        self,
        class_counts: dict[str, int],
        ba_library: BALibrary,
        gf_library: GFLibrary,
    ) -> dict[str, float]:
        """Return raw per-class weights, indexed by BA+GF combo string.

        Implementations must return a strictly positive weight for every
        key in ``class_counts``.
        """
        ...

    def compute(
        self,
        class_counts: dict[str, int],
        ba_library: BALibrary,
        gf_library: GFLibrary,
    ) -> dict[str, float]:
        if not class_counts:
            return {}

        raw = self.compute_raw_weights(class_counts, ba_library, gf_library)

        if set(raw) != set(class_counts):
            missing = set(class_counts) - set(raw)
            extra = set(raw) - set(class_counts)
            raise RuntimeError(
                f"Strategy {type(self).__name__} returned weights with"
                f" mismatched keys vs class_counts."
                f" missing={sorted(missing)!r} extra={sorted(extra)!r}"
            )

        final: dict[str, float] = {}
        for label, weight in raw.items():
            if weight <= 0:
                raise RuntimeError(
                    f"Strategy {type(self).__name__} returned non-positive"
                    f" raw weight {weight!r} for class {label!r}."
                )
            final[label] = float(weight)

        # Apply max:min ratio cap.
        if self.weight_ratio_cap is not None and len(final) >= 2:
            ceiling = min(final.values()) * self.weight_ratio_cap
            for label, w in final.items():
                if w > ceiling:
                    final[label] = ceiling

        return final


def split_alpha_safe_inverse(count: int, alpha: float) -> float:
    """count^(-alpha) with a tiny floor on count so alpha=1 with
    count=0 doesn't blow up. count=0 shouldn't occur in practice (since
    class_counts is built from observed labels), but the floor makes the
    function total."""
    return max(int(count), 1) ** (-float(alpha))


def split_alpha_softmax_normalize(values: dict[Any, float]) -> dict[Any, float]:
    """Normalize a dict of positive values so they sum to 1. Used by
    sibling-share computations. Robust to a single key (returns 1.0)."""
    total = sum(values.values())
    if total <= 0:
        # Degenerate; fall back to uniform.
        n = len(values)
        return {k: 1.0 / n for k in values}
    return {k: v / total for k, v in values.items()}
