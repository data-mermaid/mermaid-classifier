"""TrainingDataset steps for source-grouped holdout: the MERMAID
image->project lookup, point dedupe, holdout tagging, and the
add_training_set_names key-uniqueness guard.

Uses a real in-memory DuckDB; S3 and the BA/GF libraries are stubbed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb
import pandas as pd
from support.dataset import make_dataset
from support.extractor import make_extractor_spec
from support.settings import override_settings

from mermaid_classifier.pyspacer.dataset import TrainingDataset
from mermaid_classifier.pyspacer.extraction import SIDECAR_FILENAME
from mermaid_classifier.pyspacer.options import DatasetOptions

TEST_BUCKET = "test-mermaid-bucket"
MERMAID_PREFIX = f"s3://{TEST_BUCKET}/mermaid/"


def _write_parquet(directory: str, name: str, df: pd.DataFrame) -> str:
    path = str(Path(directory) / name)
    conn = duckdb.connect()
    conn.register("df", df)
    conn.execute(f"COPY (SELECT * FROM df) TO '{path}' (FORMAT parquet)")
    conn.close()
    return path


def _mermaid_df(image_ids: list[str], ba_id: str = "ba_a", points: int = 1) -> pd.DataFrame:
    """`points` annotations per image, at (n, n) for n in 1..points."""
    ids = [i for i in image_ids for _ in range(points)]
    positions = [n for _ in image_ids for n in range(1, points + 1)]
    return pd.DataFrame(
        {
            "image_id": ids,
            "row": positions,
            "col": positions,
            "benthic_attribute_id": [ba_id] * len(ids),
            "growth_form_id": [""] * len(ids),
        }
    )


def _lookup_df(mapping: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": list(mapping),
            "project_id": list(mapping.values()),
            "project_name": [f"name-{p}" for p in mapping.values()],
            "site_id": [f"site-{p}" for p in mapping.values()],
        }
    )


class MermaidProjectLookupTest(unittest.TestCase):
    def _read(
        self,
        mermaid_df: pd.DataFrame,
        lookup_df: pd.DataFrame | None,
        lookup_allowed: int = 0,
        missing_allowed: int = 0,
    ):
        dataset = make_dataset(self)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        mermaid_path = _write_parquet(tmp.name, "mermaid.parquet", mermaid_df)
        lookup_path = (
            None if lookup_df is None else _write_parquet(tmp.name, "lookup.parquet", lookup_df)
        )
        dataset.options = DatasetOptions(mermaid_project_lookup_uri=lookup_path)
        with override_settings(
            aws_anonymous="True",
            mermaid_annotations_parquet_pattern=mermaid_path,
            mermaid_project_lookup_percent_unmatched_allowed=lookup_allowed,
            training_inputs_percent_missing_allowed=missing_allowed,
        ):
            dataset.read_mermaid_data()
        return dataset

    def _load(
        self,
        mermaid_df: pd.DataFrame,
        lookup_df: pd.DataFrame,
        lookup_allowed: int = 0,
        missing_allowed: int = 0,
        excluded_images: tuple[str, ...] = (),
        included_labels: tuple[str, ...] = (),
    ) -> TrainingDataset:
        """Run the filtered-annotations load, where the unmatched check lives."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        mermaid_path = _write_parquet(tmp.name, "mermaid.parquet", mermaid_df)
        lookup_path = _write_parquet(tmp.name, "lookup.parquet", lookup_df)
        excluded_csv = included_csv = None
        if excluded_images:
            excluded_csv = str(Path(tmp.name) / "excluded.csv")
            Path(excluded_csv).write_text("image_id\n" + "".join(f"{i}\n" for i in excluded_images))
        if included_labels:
            included_csv = str(Path(tmp.name) / "included.csv")
            Path(included_csv).write_text(
                "ba_id,gf_id\n" + "".join(f"{b},\n" for b in included_labels)
            )
        options = DatasetOptions(
            include_mermaid=True,
            mermaid_project_lookup_uri=lookup_path,
            excluded_images_csv=excluded_csv,
            included_labels_csv=included_csv,
        )
        with (
            override_settings(
                aws_anonymous="True",
                mermaid_annotations_parquet_pattern=mermaid_path,
                mermaid_project_lookup_percent_unmatched_allowed=lookup_allowed,
                training_inputs_percent_missing_allowed=missing_allowed,
            ),
            mock.patch("mermaid_classifier.pyspacer.dataset.S3FileSystem"),
        ):
            dataset = TrainingDataset.filtered_annotations_only(options)
        self.addCleanup(dataset.cleanup)
        return dataset

    def _projects(self, dataset) -> dict[str, str]:
        return dict(
            dataset.duck_conn.execute("SELECT image_id, project_id FROM annotations").fetchall()
        )

    def test_matched_images_get_their_project(self):
        dataset = self._read(_mermaid_df(["i1", "i2"]), _lookup_df({"i1": "pA", "i2": "pB"}))
        self.assertEqual(self._projects(dataset), {"i1": "pA", "i2": "pB"})

    def test_unmatched_images_are_unknown_within_tolerance(self):
        dataset = self._load(_mermaid_df(["i1", "i2"]), _lookup_df({"i1": "pA"}), lookup_allowed=60)
        self.assertEqual(self._projects(dataset), {"i1": "pA", "i2": "unknown"})

    def test_unmatched_over_tolerance_raises(self):
        with self.assertRaisesRegex(RuntimeError, "project lookup"):
            self._load(_mermaid_df(["i1", "i2"]), _lookup_df({"i1": "pA"}), lookup_allowed=10)

    def test_single_unmatched_image_raises_despite_generous_missing_fv_tolerance(self):
        images = [f"i{n}" for n in range(10)]
        with self.assertRaisesRegex(
            RuntimeError, "Regenerate the lookup.*MERMAID_PROJECT_LOOKUP_PERCENT_UNMATCHED_ALLOWED"
        ):
            self._load(
                _mermaid_df(images),
                _lookup_df(dict.fromkeys(images[:-1], "pA")),
                missing_allowed=10,
            )

    def test_raising_lookup_tolerance_allows_single_unmatched_image(self):
        images = [f"i{n}" for n in range(10)]
        dataset = self._load(
            _mermaid_df(images),
            _lookup_df(dict.fromkeys(images[:-1], "pA")),
            lookup_allowed=10,
        )
        self.assertEqual(self._projects(dataset)["i9"], "unknown")

    def test_image_in_two_projects_raises(self):
        lookup = pd.concat([_lookup_df({"i1": "pA"}), _lookup_df({"i1": "pB"})])
        with self.assertRaisesRegex(ValueError, "more than one project"):
            self._read(_mermaid_df(["i1"]), lookup)

    def test_repeated_identical_lookup_rows_do_not_duplicate_annotations(self):
        lookup = pd.concat([_lookup_df({"i1": "pA"}), _lookup_df({"i1": "pA"})])
        dataset = self._read(_mermaid_df(["i1"]), lookup)
        count = dataset.duck_conn.execute("SELECT count(*) FROM annotations").fetchall()[0][0]
        self.assertEqual(count, 1)

    def test_no_lookup_keeps_project_all(self):
        dataset = self._read(_mermaid_df(["i1", "i2"]), None)
        self.assertEqual(self._projects(dataset), {"i1": "all", "i2": "all"})

    def test_read_keeps_unmatched_images_unknown_without_checking(self):
        dataset = self._read(_mermaid_df(["i1", "i2"]), _lookup_df({"i1": "pA"}))
        self.assertEqual(self._projects(dataset), {"i1": "pA", "i2": "unknown"})

    def test_excluded_unmatched_image_does_not_raise(self):
        dataset = self._load(
            _mermaid_df(["i1", "i2"]), _lookup_df({"i1": "pA"}), excluded_images=("i2",)
        )
        self.assertEqual(self._projects(dataset), {"i1": "pA"})

    def test_unmatched_image_that_is_not_excluded_raises(self):
        with self.assertRaisesRegex(RuntimeError, "project lookup"):
            self._load(
                _mermaid_df(["i1", "i2", "i3"]),
                _lookup_df({"i1": "pA"}),
                excluded_images=("i3",),
            )

    def test_unmatched_image_with_all_labels_filtered_out_does_not_raise(self):
        mermaid = pd.concat(
            [_mermaid_df(["i1"]), _mermaid_df(["i2"], ba_id="ba_b")], ignore_index=True
        )
        dataset = self._load(mermaid, _lookup_df({"i1": "pA"}), included_labels=("ba_a",))
        self.assertEqual(self._projects(dataset), {"i1": "pA"})

    def test_denominator_counts_only_surviving_images(self):
        # 1 unmatched of 2 survivors is 50%; counting the excluded image too
        # would read 1 of 3 and slip under a 40% tolerance.
        with self.assertRaisesRegex(RuntimeError, r"\(1 of 2\)"):
            self._load(
                _mermaid_df(["i1", "i2", "i3"]),
                _lookup_df({"i1": "pA", "i3": "pA"}),
                lookup_allowed=40,
                excluded_images=("i3",),
            )


def _seed_points(dataset, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)  # noqa: F841 -- DuckDB scans by name
    dataset.duck_conn.execute("CREATE OR REPLACE TABLE annotations AS SELECT * FROM df")


def _pt(image: str, row: int, col: int, ba: str, gf: str = "", label_id: str = "1") -> dict:
    return {
        "image_id": image,
        "row": row,
        "col": col,
        "label_id": label_id,
        "benthic_attribute_id": ba,
        "growth_form_id": gf,
        "site": "coralnet",
        "bucket": "b",
        "project_id": "1",
        "feature_vector": f"fv/{image}",
    }


class DedupePointsTest(unittest.TestCase):
    def setUp(self):
        override = override_settings(aws_anonymous="True")
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)
        self.dataset = make_dataset(self)
        _seed_points(
            self.dataset,
            [
                _pt("i1", 1, 1, "ba_a"),
                _pt("i1", 1, 1, "ba_a", label_id="2"),  # exact BA+GF duplicate
                _pt("i1", 1, 1, "ba_a", label_id="3"),  # and a third copy
                _pt("i1", 2, 2, "ba_a"),
                _pt("i1", 2, 2, "ba_b"),  # conflicting labels
                _pt("i1", 3, 3, "ba_a", gf="g1"),
                _pt("i1", 3, 3, "ba_a", gf="g2"),  # conflict on growth form only
                _pt("i2", 1, 1, "ba_a"),  # same row/col, other image: unrelated
            ],
        )
        self.dataset._dedupe_points()

    def _remaining(self) -> list[tuple]:
        return self.dataset.duck_conn.execute(
            "SELECT image_id, row, col, benthic_attribute_id, label_id"
            " FROM annotations ORDER BY image_id, row, col"
        ).fetchall()

    def test_duplicates_collapse_to_one_row_conflicts_are_dropped(self):
        self.assertEqual(
            self._remaining(),
            [("i1", 1, 1, "ba_a", "1"), ("i2", 1, 1, "ba_a", "1")],
        )

    def test_counts_are_recorded(self):
        self.assertEqual(
            self.dataset._dedupe_stats,
            {"annotations_duplicate_rows_removed": 2, "points_label_conflicts_dropped": 2},
        )

    def test_no_op_without_duplicates(self):
        _seed_points(self.dataset, [_pt("i1", 1, 1, "ba_a"), _pt("i1", 2, 2, "ba_b")])
        self.dataset._dedupe_points()
        self.assertEqual(len(self._remaining()), 2)
        self.assertEqual(
            self.dataset._dedupe_stats,
            {"annotations_duplicate_rows_removed": 0, "points_label_conflicts_dropped": 0},
        )

    def test_selection_is_independent_of_input_order(self):
        rows = [_pt("i1", 1, 1, "ba_a", label_id=str(n)) for n in (5, 3, 9)]
        for ordering in (rows, list(reversed(rows))):
            _seed_points(self.dataset, ordering)
            self.dataset._dedupe_points()
            self.assertEqual(self._remaining(), [("i1", 1, 1, "ba_a", "3")])


class TrainingSetNamesUniquenessTest(unittest.TestCase):
    def test_duplicate_point_key_raises_before_joining(self):
        override = override_settings(aws_anonymous="True")
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)
        dataset = make_dataset(self)
        _seed_points(dataset, [_pt("i1", 1, 1, "ba_a"), _pt("i1", 1, 1, "ba_a", label_id="2")])
        with self.assertRaisesRegex(ValueError, "not unique"):
            dataset.add_training_set_names()


class _FakeS3:
    def __init__(self, present: set[str], sidecars: dict[str, bytes]) -> None:
        self._present = present
        self._sidecars = sidecars

    def find(self, path: str) -> list[str]:  # noqa: ARG002
        return sorted(self._present)

    def exists(self, key: str) -> bool:
        return key in self._sidecars

    def cat_file(self, key: str) -> bytes:
        return self._sidecars[key]


class _FakeBALibrary:
    def id_to_name(self, ba_id: str | None) -> str | None:
        return None if ba_id is None else f"BA[{ba_id}]"


class _FakeGFLibrary:
    def id_to_name(self, gf_id: str | None) -> str | None:
        return None if gf_id is None else gf_id


class HoldoutTaggingInitTest(unittest.TestCase):
    """Real TrainingDataset.__init__: tagging happens after dedupe, test
    groups leave the table when a CV fold is held out, and default runs
    keep their artifact schemas."""

    # Four projects, two images each, five points per image.
    IMAGES = {f"img_p{p}{n}": f"p{p}" for p in "ABCD" for n in (1, 2)}

    def _build(
        self,
        holdout_csv_text: str | None,
        holdout_fold: str = "test",
        *,
        with_lookup: bool = True,
        excluded_images: tuple[str, ...] = (),
    ) -> TrainingDataset:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)

        # img_pA1's point (1, 1) appears twice; img_pA2's point (1, 1) is
        # labeled both ba_a and ba_b.
        mermaid = pd.concat(
            [
                _mermaid_df([i for i in self.IMAGES if i[-2] in "AB"], points=5),
                _mermaid_df([i for i in self.IMAGES if i[-2] in "CD"], ba_id="ba_b", points=5),
                _mermaid_df(["img_pA1"]),
                _mermaid_df(["img_pA2"], ba_id="ba_b"),
            ],
            ignore_index=True,
        )
        mermaid_path = _write_parquet(tmp.name, "mermaid.parquet", mermaid)
        lookup_path = _write_parquet(tmp.name, "lookup.parquet", _lookup_df(self.IMAGES))

        holdout_csv = None
        if holdout_csv_text is not None:
            holdout_csv = str(Path(tmp.name) / "holdout.csv")
            Path(holdout_csv).write_text(holdout_csv_text)
        excluded_csv = None
        if excluded_images:
            excluded_csv = str(Path(tmp.name) / "excluded.csv")
            Path(excluded_csv).write_text("image_id\n" + "".join(f"{i}\n" for i in excluded_images))

        present = {f"{TEST_BUCKET}/mermaid/{i}_featurevector" for i in self.IMAGES}
        sidecar = {
            MERMAID_PREFIX + SIDECAR_FILENAME: json.dumps(make_extractor_spec(8).to_dict()).encode()
        }
        options = DatasetOptions(
            include_mermaid=True,
            mermaid_project_lookup_uri=lookup_path if with_lookup else None,
            holdout_groups_csv=holdout_csv,
            excluded_images_csv=excluded_csv,
            holdout_fold=holdout_fold,
            ref_val_ratios=(0.1, 0.1),
        )
        with (
            override_settings(
                aws_anonymous="True",
                mermaid_train_data_bucket=TEST_BUCKET,
                mermaid_annotations_parquet_pattern=mermaid_path,
                download_max_workers=1,
            ),
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.S3FileSystem",
                return_value=_FakeS3(present, sidecar),
            ),
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.download_features_parallel",
                return_value=set(),
            ),
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.get_benthic_attribute_library",
                return_value=_FakeBALibrary(),
            ),
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.get_growth_form_library",
                return_value=_FakeGFLibrary(),
            ),
        ):
            dataset = TrainingDataset(options)
        self.addCleanup(dataset.cleanup)
        return dataset

    def _images(self, dataset: TrainingDataset, where: str) -> set[str]:
        return {
            r[0]
            for r in dataset.duck_conn.execute(
                f"SELECT DISTINCT image_id FROM annotations WHERE {where}"
            ).fetchall()
        }

    def test_default_run_dedupes_and_has_no_holdout(self):
        dataset = self._build(None)
        # 40 points, minus img_pA2's conflicting one; img_pA1's duplicate
        # collapses to one row.
        self.assertEqual(
            dataset.duck_conn.execute("SELECT count(*) FROM annotations").fetchall()[0][0], 39
        )
        self.assertEqual(self._images(dataset, "is_holdout OR is_excluded"), set())
        stats = dataset.artifacts.train_summary_stats
        self.assertEqual(stats["annotations_duplicate_rows_removed"], 1)
        self.assertEqual(stats["points_label_conflicts_dropped"], 1)

    def test_fold_test_tags_test_groups_as_holdout(self):
        dataset = self._build("site,group_id,fold\nmermaid,pA,test\nmermaid,pB,0\n")
        self.assertEqual(self._images(dataset, "is_holdout"), {"img_pA1", "img_pA2"})
        self.assertEqual(self._images(dataset, "is_excluded"), set())

    def test_fold_k_removes_test_groups_and_tags_fold_k(self):
        dataset = self._build(
            "site,group_id,fold\nmermaid,pA,test\nmermaid,pC,0\nmermaid,pB,1\n",
            holdout_fold="0",
        )
        remaining = self._images(dataset, "TRUE")
        self.assertFalse({"img_pA1", "img_pA2"} & remaining)
        self.assertEqual(self._images(dataset, "is_holdout"), {"img_pC1", "img_pC2"})
        self.assertEqual(self._images(dataset, "is_excluded"), set())

    def test_mermaid_groups_without_a_project_lookup_are_rejected(self):
        # Without the lookup every MERMAID row is project 'all', so no
        # listed MERMAID project could match, even one in an inactive fold.
        with self.assertRaisesRegex(ValueError, "mermaid_project_lookup_uri is unset"):
            self._build("site,group_id,fold\nmermaid,pA,1\n", with_lookup=False)

    def test_test_group_emptied_by_image_exclusion_warns_instead_of_raising(self):
        with self.assertLogs("mermaid_classifier.pyspacer.label_specs", level="WARNING") as logs:
            dataset = self._build(
                "site,group_id,fold\nmermaid,pA,test\nmermaid,pB,0\n",
                excluded_images=("img_pA1", "img_pA2"),
            )
        self.assertIn("(mermaid, pA)", "\n".join(logs.output))
        self.assertEqual(self._images(dataset, "is_holdout"), set())

    def test_val_artifact_schema_is_unchanged_by_tag_columns(self):
        dataset = self._build(None)
        columns = [r[0] for r in dataset.duck_conn.execute("DESCRIBE annotations").fetchall()]
        self.assertIn("is_holdout", columns)
        self.assertNotIn("is_holdout", dataset.get_annotations("all").columns)
        self.assertNotIn("is_excluded", dataset.get_annotations("all").columns)


if __name__ == "__main__":
    unittest.main()
