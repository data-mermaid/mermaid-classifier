"""MergeRareClassesToBAParent — relabel rare classes to a viable BA ancestor.

For each BAGF whose training count is below ``min_count``, walk up the
BA hierarchy and find the *lowest* non-root ancestor whose merged
``(ancestor, gf)`` count would meet the threshold. Plan a remap to
that ancestor's BAGF; if no non-root ancestor satisfies the threshold,
fall back to dropping the class (per the user's choice — avoids
quietly folding rare classes into top-level bins like "Hard coral"
just to satisfy a count).

This is the data-pipeline implementation that ``rare_policy='merge'``
on the old ``SampleWeightingOptions`` only stubbed: it raised
``NotImplementedError`` because merging requires relabelling samples,
not just zeroing loss weights.

Algorithm: greedy, classes processed in count-ascending order. For
each rare class, walk parent-first up the chain (excluding the root)
and pick the first ancestor where the running ``(ancestor, gf)``
count plus this class's count meets ``min_count``. The running counts
are updated after each successful merge so a later class can ride on
the cumulative buildup at a shared ancestor. Classes that fail every
ancestor short of the root are dropped.
"""
from __future__ import annotations

from mermaid_classifier.common.benthic_attributes import (
    combine_ba_gf,
    split_ba_gf,
)
from mermaid_classifier.training.label_transforms.base import (
    DROP,
    BALibrary,
    GFLibrary,
    Transform,
    TransformPlan,
)
from mermaid_classifier.training.label_transforms.registry import (
    register_transform,
)


@register_transform("merge_rare")
class MergeRareClassesToBAParent(Transform):
    """Merge rare classes up the BA hierarchy; drop if no viable ancestor."""

    def __init__(self, min_count: int = 10):
        if int(min_count) < 1:
            raise ValueError(
                f"min_count must be >= 1, got {min_count!r}"
            )
        self.min_count = int(min_count)

    def plan(
        self,
        train_class_counts: dict[str, int],
        ba_library: BALibrary,
        gf_library: GFLibrary,
    ) -> TransformPlan:
        actions: dict[str, str] = {}
        remap: dict[str, str] = {}

        # Parse and group counts by (ba, gf).
        parsed: dict[str, tuple[str, str]] = {}
        for bagf in train_class_counts:
            parsed[bagf] = split_ba_gf(bagf)

        # Running effective counts at each (ba, gf) point. Updated as
        # merges are planned so later iterations see the buildup.
        eff: dict[tuple[str, str], int] = {}
        for bagf, count in train_class_counts.items():
            eff[parsed[bagf]] = count

        rare_bagfs = sorted(
            (b for b, c in train_class_counts.items() if c < self.min_count),
            key=lambda b: (train_class_counts[b], b),
        )

        for rare_bagf in rare_bagfs:
            ba_id, gf_id = parsed[rare_bagf]

            # If this class's effective count has already crossed the
            # threshold (because other rare siblings merged into the
            # same point), nothing to do.
            if eff.get((ba_id, gf_id), 0) >= self.min_count:
                continue

            ancestors_root_first = ba_library.get_ancestor_ids(ba_id)
            # Walk parent-first, but exclude the root (the topmost ancestor
            # whose own parent is None — get_ancestor_ids returns it as
            # ancestors_root_first[0]).
            if len(ancestors_root_first) <= 1:
                # No non-root ancestors available: either the rare class
                # is at depth 1 (only the root above it) or it's a root
                # itself. Drop.
                actions[rare_bagf] = DROP
                continue
            parent_first_chain = list(reversed(ancestors_root_first))[:-1]

            my_count = train_class_counts[rare_bagf]
            merged_target_ba: str | None = None
            for anc in parent_first_chain:
                if eff.get((anc, gf_id), 0) + my_count >= self.min_count:
                    merged_target_ba = anc
                    break

            if merged_target_ba is None:
                actions[rare_bagf] = DROP
                continue

            target_bagf = combine_ba_gf(merged_target_ba, gf_id)
            remap[rare_bagf] = target_bagf
            eff[(merged_target_ba, gf_id)] = (
                eff.get((merged_target_ba, gf_id), 0) + my_count
            )
            eff[(ba_id, gf_id)] = 0

        return TransformPlan(
            transform_name=self.name,
            actions=actions,
            remap=remap,
            train_counts=dict(train_class_counts),
        )
