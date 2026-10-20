"""TrainingDataset's split step: split_mode, holdout labels, and the
training_set tags and stats that follow from them.

Uses a real in-memory DuckDB; feature downloads and the BA/GF libraries
are stubbed.
"""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd
from support.dataset import make_dataset
from support.settings import override_settings

from mermaid_classifier.pyspacer.options import DatasetOptions
from mermaid_classifier.pyspacer.splitting import split_by_image

_CLASSES = ("ba_a", "ba_b", "ba_c")
_IMAGES_PER_CLASS = 20
_POINTS_PER_IMAGE = 3
_HOLDOUT_IMAGES = ("hold_0", "hold_1")


def _rows(
    include_holdout: bool,
    extra: tuple[tuple[str, str, bool], ...] = (),
    val: tuple[tuple[str, str, str], ...] = (),
) -> list[dict[str, object]]:
    """`extra` adds (image, ba, is_holdout) images to the base dataset;
    `val` adds (image, ba, project) images tagged is_val."""
    rows = []

    def add(image: str, ba: str, project: str, is_holdout: bool, is_val: bool = False) -> None:
        for n in range(_POINTS_PER_IMAGE):
            rows.append(
                {
                    "image_id": image,
                    "row": n,
                    "col": n,
                    "label_id": "1",
                    "benthic_attribute_id": ba,
                    "growth_form_id": "",
                    "site": "coralnet",
                    "bucket": "b",
                    "project_id": project,
                    "feature_vector": f"fv/{image}",
                    "is_holdout": is_holdout,
                    "is_excluded": False,
                    "is_val": is_val,
                }
            )

    for ba in _CLASSES:
        for i in range(_IMAGES_PER_CLASS):
            add(f"{ba}_{i}", ba, "p1", False)
    if include_holdout:
        for image in _HOLDOUT_IMAGES:
            add(image, "ba_a", "p2", True)
    for image, ba, is_holdout in extra:
        add(image, ba, "p3", is_holdout)
    for image, ba, project in val:
        add(image, ba, project, False, is_val=True)
    return rows


class _FakeLibrary:
    def id_to_name(self, value: str | None) -> str | None:
        return value


class _SplitCase(unittest.TestCase):
    def _prepare(
        self,
        split_mode: str,
        include_holdout: bool = True,
        extra: tuple[tuple[str, str, bool], ...] = (),
        val: tuple[tuple[str, str, str], ...] = (),
        options: DatasetOptions | None = None,
    ):
        override = override_settings(aws_anonymous="True", download_max_workers=1)
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)
        dataset = make_dataset(self)
        dataset.options = options or DatasetOptions(
            ref_val_ratios=(0.1, 0.1),
            split_mode=split_mode,  # pyright: ignore[reportArgumentType]
        )
        df = pd.DataFrame(_rows(include_holdout, extra, val))  # noqa: F841 -- DuckDB scans by name
        dataset.duck_conn.execute("CREATE OR REPLACE TABLE annotations AS SELECT * FROM df")
        with (
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.download_features_parallel",
                return_value=set(),
            ) as download,
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.get_benthic_attribute_library",
                return_value=_FakeLibrary(),
            ),
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.get_growth_form_library",
                return_value=_FakeLibrary(),
            ),
        ):
            dataset.labels = dataset.prep_annotations_for_pyspacer()
            dataset.add_training_set_names()
            dataset.set_train_summary_stats()
        self.downloaded = set(download.call_args.args[0])
        return dataset


class ImageSplitDatasetTest(_SplitCase):
    def _sets_per_image(self, dataset) -> dict[str, set[str | None]]:
        result: dict[str, set[str | None]] = {}
        for image, training_set in dataset.duck_conn.execute(
            "SELECT image_id, training_set FROM annotations"
        ).fetchall():
            result.setdefault(image, set()).add(training_set)
        return result

    def test_images_mode_keeps_each_image_in_one_set(self):
        dataset = self._prepare("images")
        sets = self._sets_per_image(dataset)
        self.assertTrue(all(len(s) == 1 for s in sets.values()), sets)
        used = set().union(*sets.values())
        self.assertEqual(used, {"train", "ref", "val", "holdout"})

    def test_points_mode_splits_images_across_sets(self):
        dataset = self._prepare("points")
        sets = self._sets_per_image(dataset)
        self.assertTrue(any(len(s) > 1 for s in sets.values()))

    def test_holdout_rows_go_only_to_holdout_labels(self):
        for split_mode in ("points", "images"):
            with self.subTest(split_mode=split_mode):
                dataset = self._prepare(split_mode)
                sets = self._sets_per_image(dataset)
                for image in _HOLDOUT_IMAGES:
                    self.assertEqual(sets[image], {"holdout"})
                self.assertFalse(
                    [i for i, s in sets.items() if i not in _HOLDOUT_IMAGES and "holdout" in s]
                )
                self.assertEqual(
                    dataset.holdout_labels.label_count,
                    len(_HOLDOUT_IMAGES) * _POINTS_PER_IMAGE,
                )
                self.assertIn(("b", "fv/hold_0"), self.downloaded)

    def test_points_mode_split_ignores_holdout_rows(self):
        def non_holdout_tags(dataset) -> list[tuple]:
            return dataset.duck_conn.execute(
                "SELECT feature_vector, row, col, training_set FROM annotations"
                " WHERE NOT is_holdout ORDER BY ALL"
            ).fetchall()

        with_holdout = non_holdout_tags(self._prepare("points", include_holdout=True))
        without_holdout = non_holdout_tags(self._prepare("points", include_holdout=False))
        self.assertEqual(with_holdout, without_holdout)

    def test_without_holdout_rows_holdout_labels_is_empty(self):
        dataset = self._prepare("images", include_holdout=False)
        self.assertEqual(dataset.holdout_labels.label_count, 0)

    def test_stats_count_holdout_separately_from_dropped(self):
        dataset = self._prepare("images")
        stats = dataset.artifacts.train_summary_stats
        holdout_points = len(_HOLDOUT_IMAGES) * _POINTS_PER_IMAGE
        self.assertEqual(stats["annotations_holdout"], holdout_points)
        self.assertEqual(stats["annotations_dropped"], 0)
        self.assertIn(f"{holdout_points} holdout", dataset.describe_train_summary_stats())

        ba_a = dataset.artifacts.ba_counts.set_index("benthic_attribute_id").loc["ba_a"]
        self.assertEqual((ba_a["holdout"], ba_a["dropped"]), (holdout_points, 0))
        bagf_a = dataset.artifacts.bagf_counts.set_index("benthic_attribute_id").loc["ba_a"]
        self.assertEqual((bagf_a["holdout"], bagf_a["dropped"]), (holdout_points, 0))

        projects = dataset.compute_project_stats(has_training_sets=True).set_index("project_id")
        self.assertEqual(
            (projects.loc["p2", "holdout"], projects.loc["p2", "dropped"]), (holdout_points, 0)
        )
        self.assertEqual(projects.loc["p1", "holdout"], 0)


class SplitDropStatsTest(_SplitCase):
    # Two training images: too few for the split, which drops the class.
    # Its holdout image must not hide that drop.
    RARE = (("rare_0", "ba_rare", False), ("rare_1", "ba_rare", False), ("rare_h", "ba_rare", True))

    def test_class_dropped_from_split_counts_as_dropped_despite_holdout_rows(self):
        dataset = self._prepare("images", extra=self.RARE)
        stats = dataset.artifacts.train_summary_stats
        self.assertEqual(
            (stats["bas_dropped"], stats["bagfs_dropped"], stats["bagfs_dropped_by_split"]),
            (1, 1, 1),
        )
        self.assertEqual(stats["annotations_dropped"], 2 * _POINTS_PER_IMAGE)

    def test_split_drop_of_a_class_in_three_images_is_a_warning(self):
        real_split = split_by_image

        def split_without_ba_c_in_ref(*args, **kwargs):
            split = real_split(*args, **kwargs)
            split.ref = split.ref.filter_classes(split.ref.classes_set - {"ba_c::"})
            return split

        with (
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.split_by_image",
                side_effect=split_without_ba_c_in_ref,
            ),
            self.assertLogs("train", level="WARNING") as logs,
        ):
            dataset = self._prepare("images")
        self.assertIn("ba_c", "\n".join(logs.output))
        self.assertEqual(dataset.artifacts.train_summary_stats["bagfs_dropped_by_split"], 1)

    def test_split_drop_of_a_class_in_two_images_is_logged_by_name(self):
        with self.assertLogs("train", level="INFO") as logs:
            self._prepare("images", extra=self.RARE)
        drop_logs = [line for line in logs.output if "ba_rare" in line]
        self.assertTrue(drop_logs)
        self.assertTrue(all(line.startswith("INFO") for line in drop_logs), drop_logs)


# One val group per class, two images each, in their own project.
_VAL_IMAGES = tuple((f"val_{ba}_{i}", ba, "pv") for ba in _CLASSES for i in range(2))


def _grouped_val_options() -> DatasetOptions:
    return DatasetOptions(
        val_groups_csv="val_groups.csv", split_mode="images", ref_val_ratios=(0.1, 0.0)
    )


class GroupedValSplitTest(_SplitCase):
    def _prepare_grouped(self, val=_VAL_IMAGES, extra=()):
        return self._prepare("images", val=val, extra=extra, options=_grouped_val_options())

    def _images_by_set(self, dataset) -> dict[str | None, set[str]]:
        result: dict[str | None, set[str]] = {}
        for image, training_set in dataset.duck_conn.execute(
            "SELECT DISTINCT image_id, training_set FROM annotations"
        ).fetchall():
            result.setdefault(training_set, set()).add(image)
        return result

    def test_holdout_val_and_split_rows_are_disjoint(self):
        dataset = self._prepare_grouped()
        by_set = self._images_by_set(dataset)
        val_images = {image for image, _, _ in _VAL_IMAGES}
        self.assertEqual(by_set["val"], val_images)
        self.assertEqual(by_set["holdout"], set(_HOLDOUT_IMAGES))
        split_images = by_set["train"] | by_set["ref"]
        self.assertEqual(len(split_images), len(_CLASSES) * _IMAGES_PER_CLASS)
        self.assertFalse(split_images & (val_images | set(_HOLDOUT_IMAGES)))
        self.assertNotIn(None, by_set)
        self.assertEqual(dataset.labels.val.label_count, len(val_images) * _POINTS_PER_IMAGE)
        self.assertIn(("b", "fv/val_ba_a_0"), self.downloaded)

    def test_val_source_is_recorded_in_the_summary_stats(self):
        grouped = self._prepare_grouped().artifacts.train_summary_stats
        self.assertEqual(grouped["annotations_val_source"], "groups")
        split = self._prepare("images").artifacts.train_summary_stats
        self.assertEqual(split["annotations_val_source"], "split")

    def test_no_val_rows_raises(self):
        with self.assertRaisesRegex(ValueError, "no annotation is in a val group"):
            self._prepare_grouped(val=())

    def test_model_class_without_val_points_is_a_warning(self):
        val = tuple(v for v in _VAL_IMAGES if v[1] != "ba_c")
        with self.assertLogs("train", level="WARNING") as logs:
            self._prepare_grouped(val=val)
        self.assertTrue([line for line in logs.output if "ba_c" in line], logs.output)

    def test_split_drop_threshold_is_two_images(self):
        # With no val share a class in two images can reach train and ref,
        # so its drop is a warning, not the info a rarer class gets.
        real_split = split_by_image

        def split_without_ba_two_in_ref(*args, **kwargs):
            split = real_split(*args, **kwargs)
            split.ref = split.ref.filter_classes(split.ref.classes_set - {"ba_two::"})
            return split

        two = (("two_0", "ba_two", False), ("two_1", "ba_two", False))
        with (
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.split_by_image",
                side_effect=split_without_ba_two_in_ref,
            ),
            self.assertLogs("train", level="INFO") as logs,
        ):
            self._prepare_grouped(extra=two)
        drop_logs = [line for line in logs.output if "ba_two" in line and "Split drops" in line]
        self.assertTrue(drop_logs, logs.output)
        self.assertTrue(all(line.startswith("WARNING") for line in drop_logs), drop_logs)

    def test_options_reject_inconsistent_val_settings(self):
        cases = {
            "points split with val groups": {
                "val_groups_csv": "v.csv",
                "split_mode": "points",
                "ref_val_ratios": (0.1, 0.0),
            },
            "val share with val groups": {
                "val_groups_csv": "v.csv",
                "split_mode": "images",
                "ref_val_ratios": (0.1, 0.1),
            },
            "no val share without val groups": {"ref_val_ratios": (0.1, 0.0)},
        }
        for name, kwargs in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                DatasetOptions(**kwargs)  # pyright: ignore[reportArgumentType]


class EvalGroupsDisjointTest(unittest.TestCase):
    """The post-split guard: no group may feed two of train+ref, val and holdout."""

    def _check(self, sets_by_group: dict[str, tuple[str, ...]], options: DatasetOptions) -> None:
        override = override_settings(aws_anonymous="True")
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)
        dataset = make_dataset(self)
        dataset.options = options
        df = pd.DataFrame(  # noqa: F841 -- DuckDB scans by name
            [
                {"site": "coralnet", "project_id": group, "training_set": training_set}
                for group, sets in sets_by_group.items()
                for training_set in sets
            ]
        )
        dataset.duck_conn.execute("CREATE OR REPLACE TABLE annotations AS SELECT * FROM df")
        dataset._assert_eval_groups_disjoint()

    def test_group_in_train_and_val_is_rejected_only_with_val_groups(self):
        groups = {"p1": ("train", "ref", "val"), "p2": ("train",)}
        self._check(groups, DatasetOptions(split_mode="images"))
        with self.assertRaisesRegex(RuntimeError, r"\(coralnet, p1\)"):
            self._check(groups, _grouped_val_options())

    def test_group_in_holdout_and_another_set_is_rejected_in_every_mode(self):
        for options in (DatasetOptions(), _grouped_val_options()):
            for other in ("train", "val"):
                with (
                    self.subTest(val_groups=options.val_groups_csv, other=other),
                    self.assertRaisesRegex(RuntimeError, r"\(coralnet, p1\)"),
                ):
                    self._check({"p1": ("holdout", other), "p2": ("train",)}, options)

    def test_dropped_rows_and_disjoint_groups_pass(self):
        groups = {"p1": ("train", "ref", None), "p2": ("val", None), "p3": ("holdout",)}
        self._check(groups, _grouped_val_options())


if __name__ == "__main__":
    unittest.main()
