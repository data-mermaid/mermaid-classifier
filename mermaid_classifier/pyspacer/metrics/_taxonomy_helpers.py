"""Shared taxonomy helpers used by multiple metric groups."""

from __future__ import annotations

import typing
from collections import defaultdict

# LabelId is `int | str` in pyspacer; MERMAID always uses str labels.
# These helpers are typed str-only since they parse BAGF strings internally.
from mermaid_classifier.common.benthic_attributes import split_ba_gf

if typing.TYPE_CHECKING:
    from mermaid_classifier.common.benthic_attributes import (
        BenthicAttributeLibrary,
    )


def top_level_ancestor(ba_id: str, ba_library: BenthicAttributeLibrary) -> str:
    """Return the root ancestor of a BA node, or itself if it is a root."""
    ancestors = ba_library.get_ancestor_ids(ba_id)  # root-first
    return ancestors[0] if ancestors else ba_id


def build_ba_to_top(
    classes: list[str],
    ba_library: BenthicAttributeLibrary,
) -> dict[str, str]:
    """Map each BA ID (extracted from BAGF class IDs) to its top-level ancestor."""
    ba_to_top: dict[str, str] = {}
    for bagf_id in classes:
        ba_id, _ = split_ba_gf(bagf_id)
        if ba_id not in ba_to_top:
            ba_to_top[ba_id] = top_level_ancestor(ba_id, ba_library)
    return ba_to_top


def build_ba_paths(
    classes: list[str],
    ba_library: BenthicAttributeLibrary,
) -> dict[str, list[str]]:
    """Map each BA ID to its root-to-leaf path [root, ..., parent, self]."""
    ba_paths: dict[str, list[str]] = {}
    for bagf_id in classes:
        ba_id, _ = split_ba_gf(bagf_id)
        if ba_id not in ba_paths:
            ba_paths[ba_id] = ba_library.get_ancestor_ids(ba_id) + [ba_id]
    return ba_paths


def find_lca(
    ba_a: str,
    ba_b: str,
    ba_paths: dict[str, list[str]],
) -> str | None:
    """Walk both root-to-leaf paths in parallel, return last matching node.

    Returns None if the two paths diverge at position 0 (different top-level).
    """
    path_a = ba_paths[ba_a]
    path_b = ba_paths[ba_b]
    lca = None
    for a, b in zip(path_a, path_b, strict=False):
        if a == b:
            lca = a
        else:
            break
    return lca


def taxonomic_similarity(
    ba_a: str,
    ba_b: str,
    ba_paths: dict[str, list[str]],
    ba_library: BenthicAttributeLibrary,
) -> float:
    """Fraction of taxonomic path shared between two BAs.

    Returns 1.0 for exact match, ~0.75 for siblings, down to 0.0 for
    unrelated top-level categories.
    """
    if ba_a == ba_b:
        return 1.0
    lca = find_lca(ba_a, ba_b, ba_paths)
    if lca is None:
        return 0.0
    shared_depth = len(ba_library.get_ancestor_ids(lca)) + 1
    max_depth = max(len(ba_paths[ba_a]), len(ba_paths[ba_b]))
    return shared_depth / max_depth


def group_by_top_level(
    sample_indices: list[int],
    gt_indices: list[int],
    classes: list[str],
    ba_to_top: dict[str, str],
    ba_library: BenthicAttributeLibrary,
    min_samples: int = 30,
) -> list[dict[str, typing.Any]]:
    """Group samples by ground-truth top-level BA category.

    Returns list of dicts with keys: top_ba_id, name, indices, n_samples.
    Categories with fewer than min_samples are excluded.
    """
    category_indices: dict[str, list[int]] = defaultdict(list)
    for i in sample_indices:
        gt_ba, _ = split_ba_gf(classes[gt_indices[i]])
        top_ba = ba_to_top[gt_ba]
        category_indices[top_ba].append(i)

    groups = []
    for top_ba_id, indices in category_indices.items():
        if len(indices) < min_samples:
            continue
        groups.append(
            {
                "top_ba_id": top_ba_id,
                "name": ba_library.id_to_name(top_ba_id),
                "indices": indices,
                "n_samples": len(indices),
            }
        )
    return groups
