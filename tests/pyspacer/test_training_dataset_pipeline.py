"""Characterization tests for the TrainingDataset in-DuckDB pipeline.

Rather than driving the full TrainingDataset.__init__ (which reads from S3),
these tests use the NoInitDataset pattern to characterize the core in-DuckDB
pipeline steps forward from a seeded synthetic ``annotations`` table:

  rollup (LabelRollupSpec.roll_up_in_duckdb)
  → filter (LabelFilter.filter_in_duckdb)
  → subsample (_apply_subsample)
  → prep + split (prep_annotations_for_pyspacer → preprocess_labels)
  → tag rows (add_training_set_names)

``download_features_parallel`` is mocked throughout so no S3 access occurs.

Sub-steps characterized
-----------------------
- rollup_in_duckdb: row count + BA/GF values change as expected
- filter_in_duckdb: excluded rows removed
- _apply_subsample: row count drops to per-class cap; audit df populated
- prep_annotations_for_pyspacer: returns TrainingTaskLabels with .train/.ref/.val;
  total points across splits equals input count; split respects ref_val_ratios
- add_training_set_names: annotations table gains training_set column with
  values in {train, ref, val}; no NULLs

Sub-steps NOT characterized (with reason)
------------------------------------------
- set_train_summary_stats: requires ``get_benthic_attribute_library()`` and
  ``get_growth_form_library()``, which hit the MERMAID API. Mocking them
  would involve patching the entire library lookup layer; that work is out
  of scope for this characterization pass.
- The full TrainingDataset.__init__ end-to-end: reads CoralNet CSVs and a
  MERMAID Parquet from S3 via DuckDB, which is impractical to run offline.
"""

import shutil
import tempfile
import unittest
from io import StringIO
from unittest import mock

import pandas as pd

from mermaid_classifier.pyspacer.label_specs import LabelFilter, LabelRollupSpec
from mermaid_classifier.pyspacer.options import DatasetOptions
from mermaid_classifier.training.subsample import SubsampleOptions

# Reuse scaffolding from the existing test module.
from pyspacer.test_train import NoInitDataset, override_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_PER_CLASS = 10
_BA_IDS = ["ba_a", "ba_b", "ba_c"]


def _make_dataset(test_case: unittest.TestCase) -> NoInitDataset:
    """Return a NoInitDataset with all attributes needed by the pipeline methods.

    The temp feature dir is removed via the test case's cleanup so the suite
    stays hermetic and doesn't leak directories.
    """
    dataset = NoInitDataset()
    dataset._feature_dir = tempfile.mkdtemp()
    test_case.addCleanup(shutil.rmtree, dataset._feature_dir, ignore_errors=True)
    dataset.profiled_sections = []
    dataset._feature_path_to_s3_location = {}
    dataset.feature_loc_to_source = {}
    dataset.options = DatasetOptions(ref_val_ratios=(0.1, 0.1))
    # artifacts is already set by NoInitDataset.__init__ (Artifacts())
    return dataset


def _seed_annotations(dataset: NoInitDataset) -> None:
    """Seed a deterministic 30-row annotations table over 3 BA classes."""
    rows = []
    for i, ba in enumerate(_BA_IDS):
        for j in range(_N_PER_CLASS):
            rows.append(
                {
                    "image_id": f"img_{i}_{j}",
                    "row": j,
                    "col": j,
                    "label_id": "100",
                    "benthic_attribute_id": ba,
                    "growth_form_id": "",
                    "site": "mermaid",
                    "bucket": "my-bucket",
                    "project_id": f"p{i}",
                    "feature_vector": f"{ba}/img_{j}.fv",
                }
            )
    df = pd.DataFrame(rows)  # noqa: F841 — referenced by name in DuckDB SQL
    dataset.duck_conn.execute("CREATE OR REPLACE TABLE annotations AS SELECT * FROM df")


_ROLLUP_CSV = "from_ba_id,from_gf_id,to_ba_id,to_gf_id\nba_a,,ba_top,\n"
_FILTER_CSV = "ba_id,gf_id\nba_top,\nba_b,\n"


# ---------------------------------------------------------------------------
# 1. Rollup step
# ---------------------------------------------------------------------------


class RollupStepTest(unittest.TestCase):
    """Characterize LabelRollupSpec.roll_up_in_duckdb on the pipeline table."""

    def setUp(self):
        self.override = override_settings(aws_anonymous="True")
        self.override.__enter__()
        self.dataset = _make_dataset(self)
        _seed_annotations(self.dataset)

    def tearDown(self):
        self.override.__exit__(None, None, None)

    def test_rollup_changes_ba_id_for_mapped_class(self):
        """ba_a maps to ba_top; ba_b and ba_c are unchanged."""
        r = LabelRollupSpec(StringIO(_ROLLUP_CSV))
        r.roll_up_in_duckdb(self.dataset.duck_conn, "annotations")

        ba_ids = {
            row[0]
            for row in self.dataset.duck_conn.execute(
                "SELECT DISTINCT benthic_attribute_id FROM annotations"
            ).fetchall()
        }
        self.assertIn("ba_top", ba_ids, msg="ba_a should have been rolled up to ba_top")
        self.assertNotIn("ba_a", ba_ids, msg="ba_a should be gone after rollup")
        self.assertIn("ba_b", ba_ids)
        self.assertIn("ba_c", ba_ids)

    def test_rollup_preserves_row_count(self):
        """Rollup does not drop rows."""
        r = LabelRollupSpec(StringIO(_ROLLUP_CSV))
        r.roll_up_in_duckdb(self.dataset.duck_conn, "annotations")

        count = self.dataset.duck_conn.execute("SELECT count(*) FROM annotations").fetchone()[0]
        self.assertEqual(count, 30)

    def test_rollup_temp_column_absent(self):
        """bagf_id temp column must not survive after rollup."""
        r = LabelRollupSpec(StringIO(_ROLLUP_CSV))
        r.roll_up_in_duckdb(self.dataset.duck_conn, "annotations")

        cols = [row[0] for row in self.dataset.duck_conn.execute("DESCRIBE annotations").fetchall()]
        self.assertNotIn("bagf_id", cols)


# ---------------------------------------------------------------------------
# 2. Filter step
# ---------------------------------------------------------------------------


class FilterStepTest(unittest.TestCase):
    """Characterize LabelFilter.filter_in_duckdb on the pipeline table."""

    def setUp(self):
        self.override = override_settings(aws_anonymous="True")
        self.override.__enter__()
        self.dataset = _make_dataset(self)
        _seed_annotations(self.dataset)

    def tearDown(self):
        self.override.__exit__(None, None, None)

    def test_filter_removes_excluded_class(self):
        """ba_c is not in the inclusion filter → should be removed."""
        f = LabelFilter(StringIO(_FILTER_CSV), inclusion=True)
        f.filter_in_duckdb(self.dataset.duck_conn, "annotations")

        ba_ids = {
            row[0]
            for row in self.dataset.duck_conn.execute(
                "SELECT DISTINCT benthic_attribute_id FROM annotations"
            ).fetchall()
        }
        self.assertNotIn("ba_c", ba_ids, msg="ba_c should be filtered out")
        self.assertIn("ba_b", ba_ids)

    def test_filter_reduces_row_count(self):
        """The filter CSV keeps ba_b only (ba_top not in original table) → 10 rows."""
        f = LabelFilter(StringIO(_FILTER_CSV), inclusion=True)
        f.filter_in_duckdb(self.dataset.duck_conn, "annotations")

        count = self.dataset.duck_conn.execute("SELECT count(*) FROM annotations").fetchone()[0]
        self.assertEqual(count, 10)


# ---------------------------------------------------------------------------
# 3. Subsample step
# ---------------------------------------------------------------------------


class SubsampleStepTest(unittest.TestCase):
    """Characterize _apply_subsample on a seeded annotations table."""

    def setUp(self):
        self.override = override_settings(aws_anonymous="True")
        self.override.__enter__()
        self.dataset = _make_dataset(self)
        _seed_annotations(self.dataset)

    def tearDown(self):
        self.override.__exit__(None, None, None)

    def test_subsample_reduces_row_count_to_target(self):
        """balanced subsample with total_annotations=6 → 2 rows per class."""
        opts = SubsampleOptions(strategy="balanced", total_annotations=6)
        self.dataset._apply_subsample(opts)

        count = self.dataset.duck_conn.execute("SELECT count(*) FROM annotations").fetchone()[0]
        self.assertEqual(count, 6)

    def test_subsample_caps_per_class(self):
        """Each class should have exactly 2 rows after balanced subsample."""
        opts = SubsampleOptions(strategy="balanced", total_annotations=6)
        self.dataset._apply_subsample(opts)

        per_class = self.dataset.duck_conn.execute(
            "SELECT benthic_attribute_id, count(*) FROM annotations GROUP BY 1 ORDER BY 1"
        ).fetchall()
        for _ba, count in per_class:
            with self.subTest(ba=_ba):
                self.assertEqual(count, 2)

    def test_subsample_audit_df_populated(self):
        """_subsample_audit_df and _subsample_realized_total are set after the call."""
        opts = SubsampleOptions(strategy="balanced", total_annotations=6)
        self.dataset._apply_subsample(opts)

        self.assertIsNotNone(self.dataset._subsample_audit_df)
        self.assertEqual(self.dataset._subsample_realized_total, 6)

    def test_subsample_audit_df_has_correct_columns(self):
        opts = SubsampleOptions(strategy="balanced", total_annotations=6)
        self.dataset._apply_subsample(opts)

        df = self.dataset._subsample_audit_df
        self.assertIn("pre_count", df.columns)
        self.assertIn("target_n", df.columns)
        self.assertIn("realized_n", df.columns)

    def test_subsample_no_op_when_annotations_empty(self):
        """_apply_subsample on an empty table logs a warning and returns cleanly."""
        self.dataset.duck_conn.execute("DELETE FROM annotations")
        opts = SubsampleOptions(strategy="balanced", total_annotations=6)
        with self.assertLogs(logger="train", level="WARNING"):
            self.dataset._apply_subsample(opts)

        count = self.dataset.duck_conn.execute("SELECT count(*) FROM annotations").fetchone()[0]
        self.assertEqual(count, 0)


# ---------------------------------------------------------------------------
# 4. prep_annotations_for_pyspacer + split
# ---------------------------------------------------------------------------


class PrepAnnotationsTest(unittest.TestCase):
    """Characterize prep_annotations_for_pyspacer and the train/ref/val split."""

    def setUp(self):
        self.override = override_settings(aws_anonymous="True", download_max_workers=1)
        self.override.__enter__()
        self.dataset = _make_dataset(self)
        _seed_annotations(self.dataset)

    def tearDown(self):
        self.override.__exit__(None, None, None)

    def _call_prep(self):
        with mock.patch(
            "mermaid_classifier.pyspacer.dataset.download_features_parallel",
            return_value=set(),
        ):
            return self.dataset.prep_annotations_for_pyspacer()

    def test_returns_training_task_labels(self):
        """prep_annotations_for_pyspacer returns an object with .train/.ref/.val."""
        labels = self._call_prep()
        self.assertTrue(
            hasattr(labels, "train") and hasattr(labels, "ref") and hasattr(labels, "val")
        )

    def test_total_labels_equal_input_count(self):
        """Sum of train+ref+val label counts equals the number of annotations."""
        labels = self._call_prep()
        total = labels.train.label_count + labels.ref.label_count + labels.val.label_count
        # 30 annotations, 3 classes × 10 each — all survive stratified split.
        self.assertEqual(total, 30)

    def test_split_is_non_empty_across_all_sets(self):
        """With 3 classes × 10 points each, all three splits are non-empty."""
        labels = self._call_prep()
        self.assertGreater(labels.train.label_count, 0)
        self.assertGreater(labels.ref.label_count, 0)
        self.assertGreater(labels.val.label_count, 0)

    def test_ref_val_are_smaller_than_train(self):
        """With default (0.1, 0.1) ratios, ref and val are smaller than train."""
        labels = self._call_prep()
        self.assertLess(labels.ref.label_count, labels.train.label_count)
        self.assertLess(labels.val.label_count, labels.train.label_count)

    def test_all_three_classes_in_train(self):
        """All three BA classes should appear in the train split."""
        labels = self._call_prep()
        classes = labels.train.classes_set
        self.assertIn("ba_a::", classes)
        self.assertIn("ba_b::", classes)
        self.assertIn("ba_c::", classes)

    def test_feature_path_to_s3_location_populated(self):
        """After prep, _feature_path_to_s3_location maps local paths → S3 keys."""
        self._call_prep()
        self.assertGreater(len(self.dataset._feature_path_to_s3_location), 0)


# ---------------------------------------------------------------------------
# 5. add_training_set_names
# ---------------------------------------------------------------------------


class AddTrainingSetNamesTest(unittest.TestCase):
    """Characterize add_training_set_names populating the training_set column."""

    def setUp(self):
        self.override = override_settings(aws_anonymous="True", download_max_workers=1)
        self.override.__enter__()
        self.dataset = _make_dataset(self)
        _seed_annotations(self.dataset)

    def tearDown(self):
        self.override.__exit__(None, None, None)

    def test_training_set_column_values_in_expected_set(self):
        """After add_training_set_names, all training_set values are train/ref/val."""
        with mock.patch(
            "mermaid_classifier.pyspacer.dataset.download_features_parallel",
            return_value=set(),
        ):
            labels = self.dataset.prep_annotations_for_pyspacer()
        self.dataset.labels = labels
        self.dataset.add_training_set_names()

        distinct = {
            row[0]
            for row in self.dataset.duck_conn.execute(
                "SELECT DISTINCT training_set FROM annotations"
            ).fetchall()
        }
        self.assertEqual(distinct, {"train", "ref", "val"})

    def test_no_null_training_set(self):
        """Every row should have a non-NULL training_set after the call."""
        with mock.patch(
            "mermaid_classifier.pyspacer.dataset.download_features_parallel",
            return_value=set(),
        ):
            labels = self.dataset.prep_annotations_for_pyspacer()
        self.dataset.labels = labels
        self.dataset.add_training_set_names()

        null_count = self.dataset.duck_conn.execute(
            "SELECT count(*) FROM annotations WHERE training_set IS NULL"
        ).fetchone()[0]
        self.assertEqual(null_count, 0)

    def test_train_ref_val_counts_match_labels(self):
        """Row counts per set in the table match the label counts from prep."""
        with mock.patch(
            "mermaid_classifier.pyspacer.dataset.download_features_parallel",
            return_value=set(),
        ):
            labels = self.dataset.prep_annotations_for_pyspacer()
        self.dataset.labels = labels
        self.dataset.add_training_set_names()

        counts = {
            row[0]: row[1]
            for row in self.dataset.duck_conn.execute(
                "SELECT training_set, count(*) FROM annotations GROUP BY training_set"
            ).fetchall()
        }
        self.assertEqual(counts["train"], labels.train.label_count)
        self.assertEqual(counts["ref"], labels.ref.label_count)
        self.assertEqual(counts["val"], labels.val.label_count)


if __name__ == "__main__":
    unittest.main()
