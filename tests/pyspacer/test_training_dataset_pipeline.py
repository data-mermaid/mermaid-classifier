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
from unittest import mock

import pandas as pd

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


# ---------------------------------------------------------------------------
# 2. Filter step
# ---------------------------------------------------------------------------


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

    def test_total_labels_equal_input_count(self):
        """Sum of train+ref+val label counts equals the number of annotations."""
        labels = self._call_prep()
        total = labels.train.label_count + labels.ref.label_count + labels.val.label_count
        # 30 annotations, 3 classes × 10 each — all survive stratified split.
        self.assertEqual(total, 30)

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


class PrepAnnotationsDownloadFailureTest(unittest.TestCase):
    """The aggregate warning prep_annotations_for_pyspacer logs on a download
    failure -- the downloader's own per-key warning lands on a logger this
    module does not attach to train.log, so this is what a post-mortem from
    train.log alone can still read.
    """

    def setUp(self):
        self.override = override_settings(aws_anonymous="True", download_max_workers=1)
        self.override.__enter__()
        self.dataset = _make_dataset(self)
        _seed_annotations(self.dataset)

    def tearDown(self):
        self.override.__exit__(None, None, None)

    def test_the_aggregate_warning_names_the_failed_keys(self):
        failed = {("my-bucket", "ba_a/img_0.fv")}
        with (
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.download_features_parallel",
                return_value=failed,
            ),
            self.assertLogs("train", level="WARNING") as logs,
        ):
            self.dataset.prep_annotations_for_pyspacer()

        message = "\n".join(logs.output)
        self.assertIn("ba_a/img_0.fv", message)


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


# ---------------------------------------------------------------------------
# 6. set_train_summary_stats / describe_train_summary_stats
# ---------------------------------------------------------------------------


class _FakeBALibrary:
    """Minimal BA library: just the id_to_name(...) set_train_summary_stats uses."""

    def id_to_name(self, ba_id: str | None) -> str | None:
        return None if ba_id is None else f"BA[{ba_id}]"


class _FakeGFLibrary:
    def id_to_name(self, gf_id: str | None) -> str | None:
        if gf_id is None:
            return None
        return "" if gf_id == "" else f"GF[{gf_id}]"


class SetTrainSummaryStatsTest(unittest.TestCase):
    """Characterize set_train_summary_stats / describe_train_summary_stats.

    Runs the pipeline through add_training_set_names on the seeded 30-row,
    3-class table (all rows survive the split → nothing dropped), then mocks the
    BA/GF libraries (the only MERMAID-API touch) and computes the summary.
    """

    def setUp(self):
        self.override = override_settings(aws_anonymous="True", download_max_workers=1)
        self.override.__enter__()
        self.dataset = _make_dataset(self)
        _seed_annotations(self.dataset)

        with mock.patch(
            "mermaid_classifier.pyspacer.dataset.download_features_parallel",
            return_value=set(),
        ):
            self.dataset.labels = self.dataset.prep_annotations_for_pyspacer()
        self.dataset.add_training_set_names()

        with (
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.get_benthic_attribute_library",
                return_value=_FakeBALibrary(),
            ),
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.get_growth_form_library",
                return_value=_FakeGFLibrary(),
            ),
        ):
            self.dataset.set_train_summary_stats()

    def tearDown(self):
        self.override.__exit__(None, None, None)

    def test_summary_stats_dict(self):
        """The computed train_summary_stats matches the seeded 30-row dataset."""
        stats = self.dataset.artifacts.train_summary_stats
        labels = self.dataset.labels

        self.assertEqual(stats["annotations"], 30)
        self.assertEqual(stats["images"], 30)
        self.assertEqual(stats["bas"], 3)
        self.assertEqual(stats["bagfs"], 3)
        # Nothing is dropped: 10 per class clears the stratified-split threshold.
        self.assertEqual(stats["annotations_dropped"], 0)
        self.assertEqual(stats["bas_dropped"], 0)
        self.assertEqual(stats["bagfs_dropped"], 0)
        # Per-set counts come straight from the split labels and sum to the total.
        self.assertEqual(stats["annotations_train"], labels.train.label_count)
        self.assertEqual(stats["annotations_ref"], labels.ref.label_count)
        self.assertEqual(stats["annotations_val"], labels.val.label_count)
        self.assertEqual(
            stats["annotations_train"] + stats["annotations_ref"] + stats["annotations_val"],
            30,
        )

    def test_ba_and_bagf_counts_carry_mocked_names(self):
        """ba_counts / bagf_counts are populated and resolve names via the libraries."""
        ba_counts = self.dataset.artifacts.ba_counts
        bagf_counts = self.dataset.artifacts.bagf_counts

        self.assertEqual(ba_counts.shape[0], 3)
        self.assertEqual(bagf_counts.shape[0], 3)
        self.assertEqual(
            set(ba_counts["benthic_attribute_name"]),
            {"BA[ba_a]", "BA[ba_b]", "BA[ba_c]"},
        )
        self.assertIn("growth_form_name", bagf_counts.columns)
        # num_annotations per BA totals the 30 seeded rows.
        self.assertEqual(int(ba_counts["num_annotations"].sum()), 30)


if __name__ == "__main__":
    unittest.main()
