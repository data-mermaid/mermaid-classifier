"""Per-group quota allocation for ``within_class='source_spread'``.

Within one class, ``water_fill`` splits the class's target row count
across the groups (CoralNet sources / MERMAID projects) that hold it, so
that no single group dominates the class.
"""

from __future__ import annotations

from collections.abc import Hashable


def water_fill[G: Hashable](group_counts: dict[G, int], target_n: int) -> dict[G, int]:
    """Split ``target_n`` rows across groups, as evenly as group sizes allow.

    Finds the largest per-group cap L with ``sum(min(n_g, L)) <= target_n``
    and gives each group ``min(n_g, L)``. The remainder (fewer rows than
    there are groups above L) goes one row each to the groups still below
    their size, in ascending sorted-key order, so the result does not
    depend on dict iteration order. A ``target_n`` at or above the total
    keeps every group whole.

    Group keys must be sortable.
    """
    if not group_counts:
        return {}
    total = sum(group_counts.values())
    if target_n >= total:
        return dict(group_counts)
    target_n = max(target_n, 0)

    # Largest L in [0, max n] with sum(min(n, L)) <= target_n (monotone in L).
    lo, hi = 0, max(group_counts.values())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in group_counts.values()) <= target_n:
            lo = mid
        else:
            hi = mid - 1
    level = lo

    quotas = {g: min(n, level) for g, n in group_counts.items()}
    remainder = target_n - sum(quotas.values())
    for g in sorted(group_counts):  # pyright: ignore[reportCallIssue, reportArgumentType]  # keys are sortable by contract
        if remainder == 0:
            break
        if quotas[g] < group_counts[g]:
            quotas[g] += 1
            remainder -= 1
    return quotas
