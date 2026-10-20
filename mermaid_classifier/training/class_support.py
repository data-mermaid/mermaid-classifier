"""Per-(class, group) point counts from a built dataset's annotations.

Kept apart from `holdout_selection` so that module stays free of data access.
Takes any object with a `duck_conn`; imports no pyspacer.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def class_group_counts(dataset: Any) -> pd.DataFrame:
    """
    Count annotation points per class and group.

    Columns class_key (`<ba>::<gf>`, gf empty kept as ''), site, project_id, n,
    sorted by class_key, site, project_id. MERMAID rows are included as read.
    Without a project lookup they all share project 'all', so
    `select_holdout_groups.py` drops them and `count_class_support.py` refuses
    the config.
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
