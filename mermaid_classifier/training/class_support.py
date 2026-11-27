"""Per-(class, group) point counts from a built dataset's annotations.

Kept apart from `holdout_selection` so that module stays free of data access.
Takes any object with a `duck_conn`; imports no pyspacer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def class_group_counts(dataset: Any) -> pd.DataFrame:
    """
    Count annotation points per class and group.

    Columns class_key (`<ba>::<gf>`, gf empty kept as ''), site, project_id, n,
    sorted by class_key, site, project_id. MERMAID rows are included as read.
    Without a project lookup they all share project 'all', so
    `select_holdout_groups.py` drops them.
    """
    rows = dataset.duck_conn.execute(
        "SELECT benthic_attribute_id || '::' || growth_form_id AS class_key,"
        " site, project_id, COUNT(*) AS n"
        " FROM annotations GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"
    ).fetchall()
    return pd.DataFrame(
        {
            "class_key": pd.Series([str(r[0]) for r in rows], dtype=object),
            "site": pd.Series([str(r[1]) for r in rows], dtype=object),
            "project_id": pd.Series([str(r[2]) for r in rows], dtype=object),
            "n": pd.Series([int(r[3]) for r in rows], dtype="int64"),
        },
    )


def provenance_path(counts_path: Path) -> Path:
    """The provenance sidecar beside a support-counts CSV: `<stem>.provenance.json`."""
    return counts_path.with_suffix(".provenance.json")


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Dotted field paths to leaf values; a list of named dicts is keyed by each `name`."""
    if isinstance(value, dict):
        items = [(str(k), v) for k, v in value.items()]  # pyright: ignore[reportUnknownVariableType]
    elif isinstance(value, list) and all(isinstance(v, dict) and "name" in v for v in value):
        items = [(str(v["name"]), v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    else:
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, child in items:
        flat.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
    return flat


def lineage_mismatches(recorded: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Each field whose recorded value differs from the expected one, as
    `<field>: recorded <value>, expected <value>`; a field absent on one side reads None."""
    flat_recorded, flat_expected = _flatten(recorded), _flatten(expected)
    return [
        f"{field}: recorded {flat_recorded.get(field)!r}, expected {flat_expected.get(field)!r}"
        for field in sorted(flat_recorded.keys() | flat_expected.keys())
        if flat_recorded.get(field) != flat_expected.get(field)
    ]
