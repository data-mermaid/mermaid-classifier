"""TrainingDataset._apply_subsample in source_spread mode, its bypass of
holdout and val rows, and the per-class holdout and val caps.

Uses a real in-memory DuckDB over a seeded annotations table.
"""

from __future__ import annotations

import collections
import hashlib
import unittest

import pandas as pd
from support.dataset import NoInitDataset, make_dataset
from support.settings import override_settings

from mermaid_classifier.training.subsample import SubsampleOptions

# (site, project_id, is_holdout) -> (class, images, points per image)
_GROUPS = [
    (("coralnet", "s1", False), "ba_a", 2, 3),  # 6 rows
    (("coralnet", "s2", False), "ba_a", 5, 10),  # 50 rows
    (("coralnet", "s3", False), "ba_a", 5, 10),  # 50 rows
    (("coralnet", "h1", True), "ba_a", 4, 10),  # 40 holdout rows
    (("coralnet", "h1", True), "ba_b", 1, 5),  # 5 holdout rows
]


def _seed(dataset: NoInitDataset) -> None:
    rows = []
    for (site, project, holdout), ba, n_images, points in _GROUPS:
        for i in range(n_images):
            for p in range(points):
                rows.append(
                    {
                        "site": site,
                        "project_id": project,
                        "image_id": f"{project}_{ba}_img{i}",
                        "row": p,
                        "col": p,
                        "label_id": "100",
                        "benthic_attribute_id": ba,
                        "growth_form_id": "",
                        "bucket": "my-bucket",
                        "feature_vector": f"{project}/{ba}_img{i}.fv",
                        "is_holdout": holdout,
                        "is_excluded": False,
                    }
                )
    df = pd.DataFrame(rows)  # noqa: F841 -- DuckDB scans by name
    dataset.duck_conn.execute("CREATE OR REPLACE TABLE annotations AS SELECT * FROM df")


def _md5_ranked_images(project: str, ba: str, n_images: int, seed: int = 0) -> list[str]:
    """The project's images in md5(seed || feature_vector) order: a digest
    every DuckDB version computes the same way."""
    images = [f"{project}_{ba}_img{i}" for i in range(n_images)]
    return sorted(
        images,
        key=lambda image: hashlib.md5(
            f"{seed}{project}/{image.removeprefix(project + '_')}.fv".encode()
        ).hexdigest(),
    )


class _Base(unittest.TestCase):
    def setUp(self):
        override = override_settings(aws_anonymous="True")
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)
        self.dataset = make_dataset(self)
        _seed(self.dataset)

    def _rows(self, where: str) -> list[tuple]:
        return self.dataset.duck_conn.execute(
            f"SELECT project_id, image_id FROM annotations WHERE {where}"
        ).fetchall()

    def _per_image(self, where: str) -> dict[str, collections.Counter]:
        per_project: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for project, image in self._rows(where):
            per_project[project][image] += 1
        return per_project


class SourceSpreadTest(_Base):
    # One class, 106 non-holdout rows over groups of 6 / 50 / 50, target 30:
    # water-fill level 12 gives quotas 6 / 12 / 12.
    opts = SubsampleOptions(strategy="balanced", total_annotations=30, within_class="source_spread")

    def test_realized_rows_per_group_equal_the_water_fill_quotas(self):
        self.dataset._apply_subsample(self.opts)

        realized = {p: sum(c.values()) for p, c in self._per_image("NOT is_holdout").items()}
        self.assertEqual(realized, {"s1": 6, "s2": 12, "s3": 12})

    def test_whole_images_are_taken_before_a_partial_one(self):
        self.dataset._apply_subsample(self.opts)

        per_project = self._per_image("NOT is_holdout")
        # s1: both 3-point images are kept whole.
        self.assertEqual(sorted(per_project["s1"].values()), [3, 3])
        # s2/s3: quota 12 over 10-point images = one whole image + 2 points.
        for project in ("s2", "s3"):
            self.assertEqual(sorted(per_project[project].values(), reverse=True), [10, 2])

    def test_images_are_taken_in_md5_order(self):
        self.dataset._apply_subsample(self.opts)

        whole, partial = _md5_ranked_images("s2", "ba_a", 5)[:2]
        self.assertEqual(self._per_image("project_id = 's2'")["s2"], {whole: 10, partial: 2})

    def test_the_seed_changes_which_images_are_taken(self):
        self.dataset._apply_subsample(self.opts)
        first = set(self._rows("project_id = 's2'"))

        _seed(self.dataset)
        other = SubsampleOptions(
            strategy="balanced", total_annotations=30, within_class="source_spread", seed=1
        )
        self.dataset._apply_subsample(other)
        second = set(self._rows("project_id = 's2'"))

        self.assertNotEqual(first, second)

    def test_group_audit_lists_available_quota_and_realized(self):
        self.dataset._apply_subsample(self.opts)

        audit = self.dataset._subsample_group_audit_df
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(
            list(audit.columns),
            [
                "benthic_attribute_id",
                "growth_form_id",
                "site",
                "project_id",
                "available",
                "quota",
                "realized",
            ],
        )
        by_project = audit.set_index("project_id")
        self.assertEqual(by_project["available"].to_dict(), {"s1": 6, "s2": 50, "s3": 50})
        self.assertEqual(by_project["quota"].to_dict(), {"s1": 6, "s2": 12, "s3": 12})
        self.assertEqual(by_project["realized"].to_dict(), {"s1": 6, "s2": 12, "s3": 12})

    def test_per_class_audit_keeps_its_schema_and_counts_non_holdout_rows(self):
        self.dataset._apply_subsample(self.opts)

        audit = self.dataset._subsample_audit_df
        assert audit is not None
        self.assertEqual(
            list(audit.columns),
            [
                "benthic_attribute_id",
                "growth_form_id",
                "pre_count",
                "target_n",
                "realized_n",
                "target_multiplier",
            ],
        )
        self.assertEqual(audit.iloc[0].to_dict()["pre_count"], 106)
        self.assertEqual(audit.iloc[0].to_dict()["realized_n"], 30)


class MultiplierTest(_Base):
    def setUp(self):
        super().setUp()
        # A second non-holdout class so the balanced cap is total // 2.
        rows = [
            {
                "site": "coralnet",
                "project_id": "s4",
                "image_id": f"s4_img{i}",
                "row": p,
                "col": p,
                "label_id": "101",
                "benthic_attribute_id": "ba_c",
                "growth_form_id": "",
                "bucket": "my-bucket",
                "feature_vector": f"s4/img{i}.fv",
                "is_holdout": False,
                "is_excluded": False,
            }
            for i in range(20)
            for p in range(10)
        ]
        extra = pd.DataFrame(rows)  # noqa: F841 -- DuckDB scans by name
        self.dataset.duck_conn.execute("INSERT INTO annotations BY NAME SELECT * FROM extra")

    def _count(self, ba: str) -> int:
        return self.dataset.duck_conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE benthic_attribute_id = ? AND NOT is_holdout",
            [ba],
        ).fetchone()[0]

    def _opts(self, multiplier: float, within_class: str) -> SubsampleOptions:
        return SubsampleOptions(
            strategy="balanced",
            total_annotations=60,
            within_class=within_class,  # pyright: ignore[reportArgumentType]
            class_target_multipliers={"ba_a::": multiplier},
        )

    def test_boosted_class_realizes_the_multiple_of_the_cap(self):
        self.dataset._apply_subsample(self._opts(3.0, "ordered"))
        # cap = 60 // 2 = 30; ba_a has 106 rows, so 3 * 30 = 90.
        self.assertEqual(self._count("ba_a"), 90)
        self.assertEqual(self._count("ba_c"), 30)

    def test_boosted_class_realizes_the_multiple_in_source_spread_mode(self):
        self.dataset._apply_subsample(self._opts(3.0, "source_spread"))
        self.assertEqual(self._count("ba_a"), 90)

    def test_boost_is_clamped_to_the_available_rows(self):
        self.dataset._apply_subsample(self._opts(10.0, "ordered"))
        self.assertEqual(self._count("ba_a"), 106)

    def test_audit_records_the_multiplier_per_class(self):
        self.dataset._apply_subsample(self._opts(3.0, "ordered"))
        audit = self.dataset._subsample_audit_df
        assert audit is not None
        self.assertEqual(
            audit.set_index("benthic_attribute_id")["target_multiplier"].to_dict(),
            {"ba_a": 3.0, "ba_c": 1.0},
        )


class OrderedModeTest(_Base):
    def test_ordered_has_no_group_audit(self):
        self.dataset._apply_subsample(SubsampleOptions(strategy="balanced", total_annotations=30))
        self.assertIsNone(self.dataset._subsample_group_audit_df)

    def test_ordered_keeps_the_first_rows_by_key(self):
        self.dataset._apply_subsample(SubsampleOptions(strategy="balanced", total_annotations=30))

        per_project = self._per_image("NOT is_holdout")
        # Head of (site, project_id, ...) order: all of s1 (6), then 24 of s2.
        self.assertEqual({p: sum(c.values()) for p, c in per_project.items()}, {"s1": 6, "s2": 24})


class HoldoutBypassTest(_Base):
    def test_subsample_leaves_holdout_rows_untouched(self):
        for within_class in ("ordered", "source_spread"):
            with self.subTest(within_class=within_class):
                _seed(self.dataset)
                self.dataset._apply_subsample(
                    SubsampleOptions(
                        strategy="balanced", total_annotations=30, within_class=within_class
                    )
                )
                holdout = self.dataset.duck_conn.execute(
                    "SELECT count(*) FROM annotations WHERE is_holdout"
                ).fetchall()[0][0]
                kept = self.dataset.duck_conn.execute(
                    "SELECT count(*) FROM annotations WHERE NOT is_holdout"
                ).fetchall()[0][0]
                self.assertEqual(holdout, 45)
                # Holdout ba_b does not count as a class for the budget: only
                # ba_a is non-holdout, so its target is the whole 30.
                self.assertEqual(kept, 30)


class HoldoutCapTest(_Base):
    def _holdout_counts(self) -> dict[str, int]:
        return dict(
            self.dataset.duck_conn.execute(
                "SELECT benthic_attribute_id, count(*) FROM annotations WHERE is_holdout GROUP BY 1"
            ).fetchall()
        )

    def test_caps_each_class_and_leaves_smaller_classes_whole(self):
        self.dataset._cap_holdout_rows(15)

        self.assertEqual(self._holdout_counts(), {"ba_a": 15, "ba_b": 5})

    def test_takes_whole_images_first(self):
        self.dataset._cap_holdout_rows(15)

        per_image = collections.Counter(
            image for _, image in self._rows("is_holdout AND benthic_attribute_id = 'ba_a'")
        )
        self.assertEqual(sorted(per_image.values(), reverse=True), [10, 5])

    def test_images_are_taken_in_md5_order(self):
        self.dataset._cap_holdout_rows(15)

        whole, partial = _md5_ranked_images("h1", "ba_a", 4)[:2]
        per_image = collections.Counter(
            image for _, image in self._rows("is_holdout AND benthic_attribute_id = 'ba_a'")
        )
        self.assertEqual(per_image, {whole: 10, partial: 5})

    def test_never_touches_non_holdout_rows(self):
        self.dataset._cap_holdout_rows(1)

        kept = self.dataset.duck_conn.execute(
            "SELECT count(*) FROM annotations WHERE NOT is_holdout"
        ).fetchall()[0][0]
        self.assertEqual(kept, 106)

    def test_selection_is_stable_across_thread_counts(self):
        selections = []
        for threads in (1, 8):
            _seed(self.dataset)
            self.dataset.duck_conn.execute(f"SET threads TO {threads}")
            self.dataset._cap_holdout_rows(15)
            selections.append(
                sorted(
                    self.dataset.duck_conn.execute(
                        "SELECT image_id, row, col FROM annotations WHERE is_holdout"
                    ).fetchall()
                )
            )
        self.assertEqual(selections[0], selections[1])

    def test_audit_lists_available_and_realized_per_class(self):
        self.dataset._cap_holdout_rows(15)

        audit = self.dataset._holdout_audit_df
        assert audit is not None
        self.assertEqual(
            list(audit.columns),
            ["benthic_attribute_id", "growth_form_id", "available", "realized"],
        )
        by_class = audit.set_index("benthic_attribute_id")
        self.assertEqual(by_class["available"].to_dict(), {"ba_a": 40, "ba_b": 5})
        self.assertEqual(by_class["realized"].to_dict(), {"ba_a": 15, "ba_b": 5})

    def test_no_holdout_rows_is_a_no_op(self):
        self.dataset.duck_conn.execute("DELETE FROM annotations WHERE is_holdout")

        self.dataset._cap_holdout_rows(15)

        self.assertIsNone(self.dataset._holdout_audit_df)

    def test_table_without_holdout_column_is_a_no_op(self):
        self.dataset.duck_conn.execute("ALTER TABLE annotations DROP COLUMN is_holdout")

        self.dataset._cap_holdout_rows(15)

        self.assertIsNone(self.dataset._holdout_audit_df)


def _add_val_rows(dataset: NoInitDataset, groups: list[tuple[str, str, int, int]]) -> None:
    """Add an is_val column (FALSE for the seeded rows) and val rows:
    (project, class, images, points per image)."""
    dataset.duck_conn.execute("ALTER TABLE annotations ADD COLUMN is_val BOOLEAN DEFAULT FALSE")
    rows = [
        {
            "site": "coralnet",
            "project_id": project,
            "image_id": f"{project}_{ba}_img{i}",
            "row": p,
            "col": p,
            "label_id": "100",
            "benthic_attribute_id": ba,
            "growth_form_id": "",
            "bucket": "my-bucket",
            "feature_vector": f"{project}/{ba}_img{i}.fv",
            "is_holdout": False,
            "is_excluded": False,
            "is_val": True,
        }
        for project, ba, n_images, points in groups
        for i in range(n_images)
        for p in range(points)
    ]
    extra = pd.DataFrame(rows)  # noqa: F841 -- DuckDB scans by name
    dataset.duck_conn.execute("INSERT INTO annotations BY NAME SELECT * FROM extra")


class ValBypassTest(_Base):
    def test_subsample_leaves_val_rows_untouched_and_val_classes_do_not_count(self):
        for within_class in ("ordered", "source_spread"):
            with self.subTest(within_class=within_class):
                _seed(self.dataset)
                # 20 val rows of ba_a, and 5 of ba_v, a class only val has.
                _add_val_rows(self.dataset, [("v1", "ba_a", 2, 10), ("v1", "ba_v", 1, 5)])
                self.dataset._apply_subsample(
                    SubsampleOptions(
                        strategy="balanced", total_annotations=30, within_class=within_class
                    )
                )
                counts = dict(
                    self.dataset.duck_conn.execute(
                        "SELECT CASE WHEN is_val THEN 'val' WHEN is_holdout THEN 'holdout'"
                        " ELSE 'train' END, count(*) FROM annotations GROUP BY 1"
                    ).fetchall()
                )
                # ba_v counting toward K would halve ba_a's target to 15.
                self.assertEqual(counts, {"val": 25, "holdout": 45, "train": 30})


class ValCapTest(_Base):
    def setUp(self):
        super().setUp()
        _add_val_rows(self.dataset, [("v1", "ba_a", 4, 10), ("v1", "ba_b", 1, 5)])

    def _val_per_image(self, ba: str) -> collections.Counter:
        return collections.Counter(
            image for _, image in self._rows(f"is_val AND benthic_attribute_id = '{ba}'")
        )

    def test_caps_each_class_taking_whole_images_first(self):
        self.dataset._cap_val_rows(15)

        self.assertEqual(sorted(self._val_per_image("ba_a").values(), reverse=True), [10, 5])
        self.assertEqual(sum(self._val_per_image("ba_b").values()), 5)
        audit = self.dataset._val_audit_df
        assert audit is not None
        by_class = audit.set_index("benthic_attribute_id")
        self.assertEqual(by_class["available"].to_dict(), {"ba_a": 40, "ba_b": 5})
        self.assertEqual(by_class["realized"].to_dict(), {"ba_a": 15, "ba_b": 5})

    def test_never_touches_holdout_or_train_rows(self):
        self.dataset._cap_val_rows(1)

        self.assertEqual(len(self._rows("is_holdout")), 45)
        self.assertEqual(len(self._rows("NOT is_holdout AND NOT is_val")), 106)

    def test_holdout_rows_are_unchanged_when_the_val_cap_follows(self):
        def holdout_rows() -> list[tuple]:
            return sorted(
                self.dataset.duck_conn.execute(
                    "SELECT image_id, row, col FROM annotations WHERE is_holdout"
                ).fetchall()
            )

        self.dataset._cap_holdout_rows(15)
        holdout_only = holdout_rows()
        _seed(self.dataset)
        _add_val_rows(self.dataset, [("v1", "ba_a", 4, 10)])
        self.dataset._cap_holdout_rows(15)
        self.dataset._cap_val_rows(15)

        self.assertEqual(holdout_rows(), holdout_only)
        self.assertEqual(len(self._rows("is_holdout")), 20)


if __name__ == "__main__":
    unittest.main()
