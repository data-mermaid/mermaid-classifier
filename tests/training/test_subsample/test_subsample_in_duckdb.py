"""End-to-end determinism test for the subsampling SQL.

This is the actual acceptance test for the bug-fix: two independent
DuckDB connections (mimicking two parallel sweep processes) with
different thread counts must select the same set of rows from the same
input table.

The test recreates the SQL applied by ``TrainingDataset._apply_subsample``
without spinning up the full TrainingDataset pipeline, so it stays fast
and self-contained. If the SQL in train.py changes, mirror the update
here.
"""
from __future__ import annotations

import unittest

import duckdb
import pandas as pd

from mermaid_classifier.training.subsample import (
    SubsampleOptions,
    compute_per_class_targets,
)


def _build_synthetic_annotations(seed: int = 0) -> pd.DataFrame:
    """1000-row synthetic annotations table over 10 BA+GF classes
    with a skewed (50%/30%/...) per-class distribution.

    Each row gets a unique (site, project_id, image_id, row, col) tuple
    so the deterministic ORDER BY produces a stable order.
    """
    rng = pd.Series(range(1000), name="i")
    # Skewed distribution: class 0 has 500 rows, class 1 has 200,
    # class 2 has 100, classes 3-9 split the rest.
    weights = [500, 200, 100, 50, 50, 30, 30, 20, 10, 10]
    assert sum(weights) == 1000
    rows = []
    idx = 0
    for cls_i, w in enumerate(weights):
        for j in range(w):
            rows.append({
                "site": "coralnet",
                "project_id": "p1",
                "image_id": f"img_{idx:05d}",
                "row": j,
                "col": j,
                "benthic_attribute_id": f"ba_{cls_i:02d}",
                "growth_form_id": f"gf_{cls_i:02d}",
                "feature_vector": f"feat_{idx}",
            })
            idx += 1
    df = pd.DataFrame(rows)
    # Shuffle so the natural row order is NOT the deterministic one.
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _apply_sql(conn: duckdb.DuckDBPyConnection,
               annotations: pd.DataFrame,
               targets: dict[tuple[str, str], int]) -> set[str]:
    """Apply the same SQL as TrainingDataset._apply_subsample and
    return the set of surviving image_ids."""
    conn.register("_input_annotations", annotations)
    conn.execute(
        "CREATE OR REPLACE TABLE annotations AS"
        " SELECT * FROM _input_annotations"
    )
    targets_df = pd.DataFrame(
        [
            {
                "benthic_attribute_id": ba,
                "growth_form_id": gf,
                "target_n": int(n),
            }
            for (ba, gf), n in targets.items()
        ]
    )
    conn.register("_subsample_targets", targets_df)
    conn.execute(
        "CREATE OR REPLACE TABLE annotations AS"
        " WITH numbered AS ("
        "   SELECT *,"
        "          ROW_NUMBER() OVER ("
        "              PARTITION BY benthic_attribute_id, growth_form_id"
        "              ORDER BY site, project_id, image_id, row, col"
        "          ) AS _rn"
        "   FROM annotations"
        " )"
        " SELECT n.* EXCLUDE (_rn)"
        " FROM numbered n"
        " JOIN _subsample_targets t USING ("
        "     benthic_attribute_id, growth_form_id"
        " )"
        " WHERE n._rn <= t.target_n"
    )
    conn.unregister("_subsample_targets")
    conn.unregister("_input_annotations")
    rows = conn.execute("SELECT image_id FROM annotations").fetchall()
    return {r[0] for r in rows}


class SubsampleSqlDeterminismTest(unittest.TestCase):
    def setUp(self):
        self.annotations = _build_synthetic_annotations()
        opts = SubsampleOptions(
            strategy="stratified", total_annotations=200,
        )
        counts = (
            self.annotations.groupby(
                ["benthic_attribute_id", "growth_form_id"]
            ).size().to_dict()
        )
        self.targets = compute_per_class_targets(opts, counts)

    def _new_conn(self, threads: int) -> duckdb.DuckDBPyConnection:
        c = duckdb.connect(":memory:")
        # Run with different thread counts to exercise parallel scan
        # ordering. The whole point of this fix is invariance under
        # thread count.
        c.execute(f"SET threads = {threads}")
        return c

    def test_two_connections_select_identical_rows(self):
        # Same input, different connections, different thread counts.
        # If the SQL is non-deterministic, these sets will diverge.
        rows_a = _apply_sql(self._new_conn(threads=1), self.annotations,
                            self.targets)
        rows_b = _apply_sql(self._new_conn(threads=4), self.annotations,
                            self.targets)
        self.assertEqual(rows_a, rows_b)

    def test_repeated_run_in_same_connection_is_stable(self):
        conn = self._new_conn(threads=4)
        rows_a = _apply_sql(conn, self.annotations, self.targets)
        rows_b = _apply_sql(conn, self.annotations, self.targets)
        self.assertEqual(rows_a, rows_b)

    def test_per_class_counts_match_targets(self):
        conn = self._new_conn(threads=2)
        _apply_sql(conn, self.annotations, self.targets)
        rows = conn.execute(
            "SELECT benthic_attribute_id, growth_form_id, COUNT(*)"
            " FROM annotations GROUP BY 1, 2 ORDER BY 1, 2"
        ).fetchall()
        realized = {(ba, gf): n for ba, gf, n in rows}
        for cls, target in self.targets.items():
            with self.subTest(cls=cls):
                self.assertEqual(realized.get(cls, 0), target)

    def test_soft_balanced_strategy_deterministic_under_threads(self):
        # Same end-to-end determinism guarantee as stratified, applied
        # to the new soft_balanced allocator.
        opts = SubsampleOptions(
            strategy="soft_balanced",
            total_annotations=200,
            balance_alpha=0.5,
        )
        counts = (
            self.annotations.groupby(
                ["benthic_attribute_id", "growth_form_id"]
            ).size().to_dict()
        )
        targets = compute_per_class_targets(opts, counts)
        rows_a = _apply_sql(self._new_conn(threads=1), self.annotations,
                            targets)
        rows_b = _apply_sql(self._new_conn(threads=4), self.annotations,
                            targets)
        self.assertEqual(rows_a, rows_b)

    def test_balanced_strategy_caps_at_available(self):
        # Use 'balanced' with target_per_class > class size for some
        # classes -> they should be kept in full, not oversampled.
        opts = SubsampleOptions(
            strategy="balanced", target_per_class=40,
        )
        counts = (
            self.annotations.groupby(
                ["benthic_attribute_id", "growth_form_id"]
            ).size().to_dict()
        )
        targets = compute_per_class_targets(opts, counts)
        conn = self._new_conn(threads=2)
        rows = _apply_sql(conn, self.annotations, targets)
        # Class 9 only has 10 rows; balanced should NOT produce more
        # than 10 surviving rows for it.
        n_class_9 = conn.execute(
            "SELECT COUNT(*) FROM annotations"
            " WHERE benthic_attribute_id = 'ba_09'"
        ).fetchone()[0]
        self.assertEqual(n_class_9, 10)
        # Common class capped at 40.
        n_class_0 = conn.execute(
            "SELECT COUNT(*) FROM annotations"
            " WHERE benthic_attribute_id = 'ba_00'"
        ).fetchone()[0]
        self.assertEqual(n_class_0, 40)
        # Sanity: total survivors equal target sum.
        self.assertEqual(len(rows), sum(targets.values()))


if __name__ == "__main__":
    unittest.main()
