"""Strategy abstract base class.

A Strategy maps per-class sample counts (BA+GF combo string -> count)
to per-class weights (BA+GF combo string -> float). Concrete strategies
implement ``compute_raw_weights`` with their own algorithm; the base
class wraps the call with shared behaviour:

  - rare-class policy (drop / keep / merge) applied as a post-step on
    the raw weights, so each Strategy only worries about its math.
  - per-class "action" tracking (kept / zeroed) for MLflow logging.

The plan originally placed the rare-class policy as a separate
data-pipeline filter in the ``TrainingDataset``. We instead apply
"drop" as **zero-weighting**: rare classes get weight = 0, which
makes the per-sample CE term for those classes zero and zeroes the
direct gradient w.r.t. the model's logits *for that class*.

Caveat: this is *not* the same as removing rare-class samples from
the dataset. The zero-weighted class's logit still appears in the
softmax denominator for every other class's CE term, so a small
residual gradient flows back through shared layers. For experimental
purposes (downweighting rare classes to reduce their training
influence) this is fine; for full exclusion of a class from training,
filter the data instead.

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


# Module-level state for the rare-action inspector. Set by Strategy.compute().
# Keyed by BA+GF label string (the same keys as class_counts). Cleared and
# rewritten on each Strategy.compute() call so callers can inspect per-class
# actions without changing the public return type of compute().
#
# Not thread-safe: concurrent compute() calls on different strategies will
# clobber each other's actions. The current pipeline is single-threaded
# (TrainingRunner.run is sequential, _compute_class_weights returns before
# the trainer fires); if parallel runs land in the same process, this needs
# to move to a thread-local or be returned as a structured result instead.
_LAST_RARE_ACTIONS: dict[str, str] = {}


def rare_action_for_class(bagf_label: str) -> str:
    """Return the rare-class action ('kept' | 'zeroed') applied to this
    class on the most recent ``Strategy.compute()`` call.

    Returns 'kept' if the class wasn't seen on the most recent call
    (e.g. if you ask about a class that wasn't in the input counts).
    """
    return _LAST_RARE_ACTIONS.get(bagf_label, "kept")


class Strategy(ABC):
    """Base class for sample-weighting strategies.

    Subclasses implement ``compute_raw_weights``. The base class handles
    the rare-class policy and bookkeeping for logging.

    Constructor arguments:
        alpha        -- smoothing exponent in [0, 1] (algorithm-specific
                        meaning; see each subclass docstring).
        min_count    -- threshold below which a class is "rare".
        rare_policy  -- "drop" | "keep" | "merge".
                        "merge" raises NotImplementedError because a
                        proper merge requires relabelling training samples
                        up the BA tree, which lives in the data pipeline.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        min_count: int = 10,
        rare_policy: str = "drop",
    ):
        self.alpha = float(alpha)
        self.min_count = int(min_count)
        self.rare_policy = str(rare_policy)

    @abstractmethod
    def compute_raw_weights(
        self,
        class_counts: dict[str, int],
        ba_library: BALibrary,
        gf_library: GFLibrary,
    ) -> dict[str, float]:
        """Return raw per-class weights, indexed by BA+GF combo string.

        Implementations must return a strictly positive weight for every
        key in ``class_counts``. The base class will apply the rare-class
        policy on top.
        """
        ...

    def compute(
        self,
        class_counts: dict[str, int],
        ba_library: BALibrary,
        gf_library: GFLibrary,
    ) -> dict[str, float]:
        if self.rare_policy == "merge":
            raise NotImplementedError(
                "rare_policy='merge' is not implemented. Merging rare"
                " classes up the BA tree requires relabelling samples in"
                " the data pipeline (TrainingDataset), which is a larger"
                " change than the loss-level weighting in this module."
                " Use 'drop' (zero-weighting in CE loss) or 'keep' for"
                " now, or curate a label_rollup_spec_csv to merge labels"
                " before weighting."
            )

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

        # Apply rare-class policy.
        actions: dict[str, str] = {}
        final: dict[str, float] = {}
        for label, weight in raw.items():
            if weight <= 0:
                raise RuntimeError(
                    f"Strategy {type(self).__name__} returned non-positive"
                    f" raw weight {weight!r} for class {label!r}."
                )
            if class_counts[label] < self.min_count and self.rare_policy == "drop":
                final[label] = 0.0
                actions[label] = "zeroed"
            else:
                final[label] = float(weight)
                actions[label] = "kept"

        # Stash actions for MLflow logging (read via rare_action_for_class).
        _LAST_RARE_ACTIONS.clear()
        _LAST_RARE_ACTIONS.update(actions)

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
