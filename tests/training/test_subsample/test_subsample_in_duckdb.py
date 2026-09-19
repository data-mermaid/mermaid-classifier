"""Determinism acceptance test for the subsampling step.

Two independent DuckDB connections (mimicking two parallel sweep processes)
running at different thread counts must select the same rows from the same
input table. The assertion runs against ``TrainingDataset._apply_subsample``
itself, so a change to its SQL -- losing the deterministic ORDER BY inside
the ROW_NUMBER() window, say -- fails here.
"""

from __future__ import annotations

import unittest

import pandas as pd
from pyspacer.test_train import override_settings
from pyspacer.test_training_dataset_pipeline import _make_dataset

from mermaid_classifier.training.subsample import SubsampleOptions

# Skewed class distribution: one class is half the table, the tail is thin.
_CLASS_WEIGHTS = [500, 200, 100, 50, 50, 30, 30, 20, 10, 10]

_ROW_KEY = ["site", "project_id", "image_id", "row", "col"]


def _synthetic_annotations(seed: int = 0) -> pd.DataFrame:
    """1000 rows over 10 classes, each with a unique annotation key.

    Shuffled, so the natural row order is not the deterministic one --
    without a stable ORDER BY the two connections would diverge.
    """
    rows = []
    idx = 0
    for cls_i, weight in enumerate(_CLASS_WEIGHTS):
        for j in range(weight):
            rows.append(
                {
                    "site": "coralnet",
                    "project_id": "p1",
                    "image_id": f"img_{idx:05d}",
                    "row": j,
                    "col": j,
                    "label_id": "100",
                    "benthic_attribute_id": f"ba_{cls_i:02d}",
                    "growth_form_id": f"gf_{cls_i:02d}",
                    "bucket": "my-bucket",
                    "feature_vector": f"feat_{idx}.fv",
                }
            )
            idx += 1
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


class SubsampleDeterminismTest(unittest.TestCase):
    def setUp(self):
        self.override = override_settings(aws_anonymous="True")
        self.override.__enter__()
        self.addCleanup(self.override.__exit__, None, None, None)

    def _surviving_keys(self, threads: int, opts: SubsampleOptions) -> list[tuple]:
        dataset = _make_dataset(self)
        df = _synthetic_annotations()  # noqa: F841 — named in the DuckDB SQL below
        dataset.duck_conn.execute("CREATE OR REPLACE TABLE annotations AS SELECT * FROM df")
        dataset.duck_conn.execute(f"SET threads TO {threads}")

        dataset._apply_subsample(opts)

        return dataset.duck_conn.execute(
            f"SELECT {', '.join(_ROW_KEY)} FROM annotations ORDER BY {', '.join(_ROW_KEY)}"
        ).fetchall()

    def test_thread_count_does_not_change_the_selected_rows(self):
        opts = SubsampleOptions(strategy="stratified", total_annotations=300)

        one_thread = self._surviving_keys(1, opts)
        eight_threads = self._surviving_keys(8, opts)

        self.assertEqual(len(one_thread), 300)
        self.assertEqual(one_thread, eight_threads)


if __name__ == "__main__":
    unittest.main()
