"""TrainingDataset steps for source-grouped holdout: the MERMAID
image->project lookup, point dedupe, holdout tagging, the holdout cap's
effect on the feature sync, and the add_training_set_names key-uniqueness guard.

The dataset versions are real segdata-spec tables in an S3 fake
(segdata_fixtures); DuckDB runs for real.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
from segdata_spec.manifest import FeatureSetPin, Pin
from support.dataset import make_dataset
from support.settings import override_settings

from mermaid_classifier.pyspacer.dataset import TrainingDataset, resolve_groups
from mermaid_classifier.pyspacer.options import DatasetOptions, PinnedDataset
from pyspacer.segdata_fixtures import (
    FakeS3,
    Image,
    label_row,
    load_training_dataset,
    publish_dataset,
    publish_feature_set,
    publish_taxonomy,
    write_project_lookup,
)

# CoralNet label 100 maps to ba-1; 200 to ba-2.
CORALNET_MAPPING = [
    {
        "provider_id": str(label_id),
        "provider_label": f"label {label_id}",
        "benthic_attribute_id": ba,
        "benthic_attribute_name": ba,
        "growth_form_id": None,
        "growth_form_name": None,
    }
    for label_id, ba in ((100, "ba-1"), (200, "ba-2"))
]
CORALNET_LABELS = [label_row(100), label_row(200)]
# MERMAID label 1 is ba_a, 2 is ba_b, neither with a growth form.
MERMAID_LABELS = [label_row(1, "ba_a", ""), label_row(2, "ba_b", "")]


def _unsynced(pin: Pin, labels: str) -> PinnedDataset:
    """A pin whose feature set is never read, for tests that stop at read_datasets."""
    features = FeatureSetPin(dataset=pin, family="efficientnet-v1", manifest_sha256="0")
    return PinnedDataset(pin=pin, features=features, labels=labels)  # pyright: ignore[reportArgumentType]


def _source(fake: FakeS3, pin: Pin, image_ids: list[str], **kwargs: Any) -> PinnedDataset:
    return PinnedDataset(pin=pin, features=publish_feature_set(fake, pin, image_ids), **kwargs)


class _Harness(unittest.TestCase):
    def setUp(self):
        self.fake = FakeS3()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write(self, name: str, text: str) -> str:
        path = self.tmp / name
        path.write_text(text)
        return str(path)

    def lookup(self, image_projects: list[tuple[str, str]]) -> str:
        return write_project_lookup(str(self.tmp / "lookup.parquet"), image_projects)

    def publish_mermaid(self, points: list[tuple[str, int, int, int]]) -> tuple[Pin, Pin]:
        images = [Image(image_id, "mermaid") for image_id in dict.fromkeys(p[0] for p in points)]
        pin = publish_dataset(
            self.fake, "mermaid", images, points, MERMAID_LABELS, source_type="mermaid"
        )
        return pin, publish_taxonomy(self.fake, {"gf-x": "X"})

    def load(self, options: DatasetOptions, **settings: Any) -> TrainingDataset:
        overrides = {
            "feature_cache_dir": str(self.tmp / "cache"),
            "download_max_workers": 1,
            "training_inputs_percent_missing_allowed": 0,
            **settings,
        }
        dataset = load_training_dataset(self.fake, options, CORALNET_MAPPING, **overrides)
        self.addCleanup(dataset.cleanup)
        return dataset


class MermaidProjectLookupTest(_Harness):
    """read_datasets gives each MERMAID row its image's project from the lookup."""

    def _read(
        self,
        mermaid_images: list[str],
        lookup: list[tuple[str, str]],
        coralnet_images: tuple[Image, ...] = (),
        lookup_allowed: int = 0,
        missing_allowed: int = 0,
    ):
        pin, taxonomy = self.publish_mermaid([(i, 1, 1, 1) for i in mermaid_images])
        sources = [_unsynced(pin, "mermaid")]
        if coralnet_images:
            cn_points = [(image.image_id, 1, 1, 100) for image in coralnet_images]
            cn_pin = publish_dataset(
                self.fake, "coralnet-raw", list(coralnet_images), cn_points, CORALNET_LABELS
            )
            sources.append(_unsynced(cn_pin, "coralnet"))
        dataset = make_dataset(self)
        dataset.options = DatasetOptions(
            datasets=sources, taxonomy=taxonomy, mermaid_project_lookup_uri=self.lookup(lookup)
        )
        dataset.groups = resolve_groups(dataset.options)
        with (
            override_settings(
                mermaid_project_lookup_percent_unmatched_allowed=lookup_allowed,
                training_inputs_percent_missing_allowed=missing_allowed,
            ),
            mock.patch(
                "mermaid_classifier.common.benthic_attributes.CoralNetMermaidMapping._download_mapping",
                return_value=CORALNET_MAPPING,
            ),
        ):
            dataset.read_datasets(self.fake)
        return dataset

    def _projects(self, dataset) -> dict[str, str]:
        return dict(
            dataset.duck_conn.execute("SELECT image_id, project_id FROM annotations").fetchall()
        )

    def test_mermaid_rows_take_the_lookup_project_and_coralnet_rows_keep_their_source(self):
        dataset = self._read(
            ["m1", "m2"], [("m1", "P1"), ("m2", "P2")], coralnet_images=(Image("c1", "s12"),)
        )
        self.assertEqual(self._projects(dataset), {"m1": "P1", "m2": "P2", "c1": "12"})
        stats = dataset.artifacts.mermaid_project_stats
        self.assertEqual(stats["project_id"].tolist(), ["P1", "P2"])

    def test_unmatched_images_are_unknown_within_tolerance(self):
        dataset = self._read(["m1", "m2"], [("m1", "P1")], lookup_allowed=60)
        self.assertEqual(self._projects(dataset), {"m1": "P1", "m2": "unknown"})

    def test_unmatched_over_tolerance_raises(self):
        with self.assertRaisesRegex(RuntimeError, "project lookup"):
            self._read(["m1", "m2"], [("m1", "P1")], lookup_allowed=10)

    def test_single_unmatched_image_raises_despite_generous_missing_fv_tolerance(self):
        images = [f"m{n}" for n in range(10)]
        with self.assertRaisesRegex(
            RuntimeError, "Regenerate the lookup.*MERMAID_PROJECT_LOOKUP_PERCENT_UNMATCHED_ALLOWED"
        ):
            self._read(images, [(i, "P1") for i in images[:-1]], missing_allowed=10)

    def test_raising_lookup_tolerance_allows_single_unmatched_image(self):
        images = [f"m{n}" for n in range(10)]
        dataset = self._read(images, [(i, "P1") for i in images[:-1]], lookup_allowed=10)
        self.assertEqual(self._projects(dataset)["m9"], "unknown")

    def test_image_in_two_projects_raises(self):
        with self.assertRaisesRegex(ValueError, "more than one project"):
            self._read(["m1"], [("m1", "P1"), ("m1", "P2")])

    def test_repeated_identical_lookup_rows_do_not_duplicate_annotations(self):
        dataset = self._read(["m1"], [("m1", "P1"), ("m1", "P1")])
        count = dataset.duck_conn.execute("SELECT count(*) FROM annotations").fetchall()[0][0]
        self.assertEqual(count, 1)


def _seed_points(dataset, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)  # noqa: F841 -- DuckDB scans by name
    dataset.duck_conn.execute("CREATE OR REPLACE TABLE annotations AS SELECT * FROM df")


def _pt(
    image: str, row: int, col: int, ba: str, gf: str = "", label_id: str = "1", dataset: str = "d1"
) -> dict:
    return {
        "dataset": dataset,
        "image_id": image,
        "row": row,
        "col": col,
        "label_id": label_id,
        "benthic_attribute_id": ba,
        "growth_form_id": gf,
        "site": "coralnet",
        "bucket": "b",
        "project_id": "1",
    }


class DedupePointsTest(unittest.TestCase):
    def setUp(self):
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
                _pt("i1", 1, 1, "ba_a", dataset="d2"),  # same point, other dataset
            ],
        )
        self.dataset._dedupe_points()

    def _remaining(self) -> list[tuple]:
        return self.dataset.duck_conn.execute(
            "SELECT dataset, image_id, row, col, benthic_attribute_id, label_id"
            " FROM annotations ORDER BY dataset, image_id, row, col"
        ).fetchall()

    def test_duplicates_collapse_conflicts_drop_and_datasets_stay_apart(self):
        self.assertEqual(
            self._remaining(),
            [
                ("d1", "i1", 1, 1, "ba_a", "1"),
                ("d1", "i2", 1, 1, "ba_a", "1"),
                ("d2", "i1", 1, 1, "ba_a", "1"),
            ],
        )

    def test_counts_are_recorded(self):
        self.assertEqual(
            self.dataset._dedupe_stats,
            {"annotations_duplicate_rows_removed": 2, "points_label_conflicts_dropped": 2},
        )

    def test_selection_is_independent_of_input_order(self):
        rows = [_pt("i1", 1, 1, "ba_a", label_id=str(n)) for n in (5, 3, 9)]
        for ordering in (rows, list(reversed(rows))):
            _seed_points(self.dataset, ordering)
            self.dataset._dedupe_points()
            self.assertEqual(self._remaining(), [("d1", "i1", 1, 1, "ba_a", "3")])


class TrainingSetNamesUniquenessTest(unittest.TestCase):
    def test_duplicate_point_key_raises_before_joining(self):
        dataset = make_dataset(self)
        rows = [_pt("i1", 1, 1, "ba_a"), _pt("i1", 1, 1, "ba_a", label_id="2")]
        _seed_points(dataset, [{**row, "feature_vector": "fv/i1"} for row in rows])
        with self.assertRaisesRegex(ValueError, "not unique"):
            dataset.add_training_set_names()


class HoldoutTaggingInitTest(_Harness):
    """Real TrainingDataset.__init__: tagging happens after dedupe, test
    groups leave the table when a CV fold is held out, and default runs
    keep their artifact schemas."""

    # Four projects, two images each.
    IMAGES = {f"img_p{p}{n}": f"p{p}" for p in "ABCD" for n in (1, 2)}

    def _build(
        self,
        holdout_csv_text: str | None,
        holdout_fold: str = "test",
        *,
        with_lookup: bool = True,
        excluded_images: tuple[str, ...] = (),
    ) -> TrainingDataset:
        # Five points per image: label 1 (ba_a) in pA/pB, 2 (ba_b) in pC/pD.
        # img_pA1's point (1, 1) appears twice; img_pA2's point (1, 1) is
        # labeled both ba_a and ba_b.
        points = [
            (image, n, n, 1 if project in ("pA", "pB") else 2)
            for image, project in self.IMAGES.items()
            for n in range(1, 6)
        ] + [("img_pA1", 1, 1, 1), ("img_pA2", 1, 1, 2)]
        pin, taxonomy = self.publish_mermaid(points)
        holdout_csv = (
            None if holdout_csv_text is None else self.write("holdout.csv", holdout_csv_text)
        )
        excluded_csv = None
        if excluded_images:
            excluded_csv = self.write(
                "excluded.csv", "image_id\n" + "".join(f"{i}\n" for i in excluded_images)
            )
        options = DatasetOptions(
            datasets=[_source(self.fake, pin, list(self.IMAGES), labels="mermaid")],
            taxonomy=taxonomy,
            feature_extractor_weights="s3://test-bucket/efficientnet.pt",
            mermaid_project_lookup_uri=self.lookup(list(self.IMAGES.items()))
            if with_lookup
            else None,
            holdout_groups_csv=holdout_csv,
            excluded_images_csv=excluded_csv,
            holdout_fold=holdout_fold,
        )
        return self.load(options)

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
        self.assertEqual([key for key in self.fake.feature_gets() if "img_pA" in key], [])

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


class CoralNetHoldoutTest(_Harness):
    """CoralNet holdout groups: the groups_csv cross-check and the cap before sync."""

    TRAIN = [Image("t1", "s1"), Image("t2", "s1"), Image("t3", "s1")]
    HOLDOUT = [Image("h1", "s2"), Image("h2", "s2"), Image("h3", "s2")]

    def _options(self, holdout_csv: str, **kwargs: Any) -> DatasetOptions:
        images = self.TRAIN + self.HOLDOUT
        # Ten points per image, alternating labels 100 and 200: five per class.
        points = [(i.image_id, n, n, 100 if n % 2 else 200) for i in images for n in range(10)]
        pin = publish_dataset(self.fake, "coralnet-raw", images, points, CORALNET_LABELS)
        source = _source(
            self.fake,
            pin,
            [i.image_id for i in images],
            labels="coralnet",
            groups_csv=kwargs.pop("groups_csv", None),
        )
        return DatasetOptions(
            datasets=[source],
            feature_extractor_weights="s3://test-bucket/efficientnet.pt",
            holdout_groups_csv=self.write("holdout.csv", holdout_csv),
            **kwargs,
        )

    def test_a_holdout_group_the_groups_csv_leaves_out_is_refused_by_name(self):
        options = self._options(
            "site,group_id,fold\ncoralnet,99,test\n",
            groups_csv=self.write("groups.csv", "s1\ns2\n"),
        )
        with self.assertRaisesRegex(ValueError, "s99"):
            self.load(options)

    def test_a_filtered_out_group_in_an_inactive_fold_only_warns(self):
        options = self._options(
            "site,group_id,fold\ncoralnet,99,1\n",
            groups_csv=self.write("groups.csv", "s1\ns2\n"),
        )
        with self.assertLogs("train", level="WARNING") as logs:
            self.load(options)
        self.assertTrue([line for line in logs.output if "s99" in line], logs.output)

    def test_only_the_holdout_images_the_cap_keeps_are_synced(self):
        # Each holdout image has five points per class, so a cap of five keeps
        # one image: the first in md5(seed || dataset/image_id) order.
        options = self._options("site,group_id,fold\ncoralnet,2,test\n", holdout_max_per_class=5)
        kept = min(
            ("h1", "h2", "h3"),
            key=lambda image: hashlib.md5(f"0coralnet-raw/{image}".encode()).hexdigest(),
        )

        dataset = self.load(options)

        synced = sorted(key.rsplit("/", 1)[-1] for key in self.fake.feature_gets())
        self.assertEqual(
            synced, sorted(f"{image}.featurevector" for image in ("t1", "t2", "t3", kept))
        )
        self.assertEqual(dataset.holdout_labels.label_count, 10)


if __name__ == "__main__":
    unittest.main()
