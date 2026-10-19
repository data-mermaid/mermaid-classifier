"""split_by_image: the image-grouped stratified train/ref/val split."""

from __future__ import annotations

import random
import unittest

from spacer.data_classes import DataLocation, ImageLabels
from spacer.messages import TrainingTaskLabels
from spacer.task_utils import preprocess_labels

from mermaid_classifier.pyspacer.splitting import split_by_image


def _labels(images: dict[str, list[str]]) -> ImageLabels:
    """One point per listed label, at (n, n) for the label's position n."""
    return ImageLabels(
        {
            DataLocation("filesystem", key): [(n, n, label) for n, label in enumerate(points)]
            for key, points in images.items()
        }
    )


def _background_dataset(
    images_per_class: dict[str, int], points_per_image: int = 4, prefix: str = "img"
) -> dict[str, list[str]]:
    """Each image holds one point of its own class plus `bg` points.

    `bg` is in every image, so each image's rarest class is its own class.
    """
    images: dict[str, list[str]] = {}
    for label, n_images in images_per_class.items():
        for i in range(n_images):
            images[f"{prefix}/{label}/{i}"] = [label] + ["bg"] * (points_per_image - 1)
    return images


def _set_keys(split: TrainingTaskLabels) -> dict[str, set[DataLocation]]:
    return {name: set(split[name].keys()) for name in ("train", "ref", "val")}


class SplitByImageTest(unittest.TestCase):
    IMAGES_PER_CLASS = {"c3": 3, "c4": 4, "c10": 10, "c40": 40, "c200": 200}

    def setUp(self):
        self.images = _background_dataset(self.IMAGES_PER_CLASS)
        self.split = split_by_image(_labels(self.images), (0.1, 0.1), seed=0)

    def test_no_image_is_in_two_sets(self):
        keys = _set_keys(self.split)
        self.assertFalse(keys["train"] & keys["ref"])
        self.assertFalse(keys["train"] & keys["val"])
        self.assertFalse(keys["ref"] & keys["val"])
        self.assertEqual(len(keys["train"]) + len(keys["ref"]) + len(keys["val"]), len(self.images))

    def test_every_kept_class_is_in_train_ref_and_val(self):
        expected = {*self.IMAGES_PER_CLASS, "bg"}
        for name in ("train", "ref", "val"):
            with self.subTest(set=name):
                self.assertEqual(self.split[name].classes_set, expected)

    def test_output_passes_pyspacer_preprocessing_unchanged(self):
        before = {name: self.split[name].label_count for name in ("train", "ref", "val")}
        processed = preprocess_labels(self.split)
        after = {name: processed[name].label_count for name in ("train", "ref", "val")}
        self.assertEqual(after, before)


class RareClassDropTest(unittest.TestCase):
    def test_classes_in_fewer_than_three_images_are_dropped_and_logged(self):
        images = _background_dataset({"c2": 2, "c3": 3, "c50": 50})
        # c2 has three points, but only across two images.
        images["img/c2/0"].append("c2")
        with self.assertLogs("mermaid_classifier.pyspacer.splitting", level="INFO") as logs:
            split = split_by_image(_labels(images), (0.1, 0.1), seed=0)

        for name in ("train", "ref", "val"):
            self.assertNotIn("c2", split[name].classes_set)
            self.assertIn("c3", split[name].classes_set)
        # The c2 images keep their bg points.
        total = sum(split[name].label_count for name in ("train", "ref", "val"))
        self.assertEqual(total, sum(len(p) for p in images.values()) - 3)
        self.assertIn("c2", "\n".join(logs.output))


class SetSizeTest(unittest.TestCase):
    def _assert_sizes(self, images: dict[str, list[str]], ratios: tuple[float, float]) -> None:
        split = split_by_image(_labels(images), ratios, seed=3)
        n = sum(len(p) for p in images.values())
        ref_ratio, val_ratio = ratios
        self.assertAlmostEqual(split.ref.label_count / n, ref_ratio, delta=0.01)
        self.assertAlmostEqual(split.val.label_count / n, val_ratio, delta=0.01)

    def test_sizes_follow_the_ratios(self):
        rng = random.Random(1)
        images = {
            f"img/{i}": [rng.choice("abcdefgh") for _ in range(rng.randint(1, 8))]
            for i in range(3000)
        }
        self._assert_sizes(images, (0.15, 0.05))

    def test_ref_is_not_capped_at_5000_points(self):
        # 70,000 points, so 10% is 7,000: above PySpacer's 5,000 ref cap.
        images = _background_dataset(
            {f"c{k}": 700 for k in range(10)}, points_per_image=10, prefix="big"
        )
        split = split_by_image(_labels(images), (0.1, 0.1), seed=0)
        self.assertGreater(split.ref.label_count, 5000)
        self._assert_sizes(images, (0.1, 0.1))


class SeedTest(unittest.TestCase):
    def setUp(self):
        self.labels = _labels(_background_dataset({"a": 30, "b": 60, "c": 90}))

    def test_same_seed_gives_the_same_split(self):
        first = _set_keys(split_by_image(self.labels, (0.1, 0.1), seed=7))
        second = _set_keys(split_by_image(self.labels, (0.1, 0.1), seed=7))
        self.assertEqual(first, second)

    def test_different_seed_gives_a_different_split(self):
        first = _set_keys(split_by_image(self.labels, (0.1, 0.1), seed=7))
        other = _set_keys(split_by_image(self.labels, (0.1, 0.1), seed=8))
        self.assertNotEqual(first, other)

    def test_stable_key_decides_the_order_not_the_location(self):
        # Same images under two different local roots: the split must
        # follow the stable key, not the download path.
        def relocated(root: str) -> ImageLabels:
            return ImageLabels(
                {
                    DataLocation("filesystem", f"{root}/{loc.key}"): self.labels[loc]
                    for loc in self.labels.keys()  # noqa: SIM118 -- ImageLabels is not a dict
                }
            )

        def strip_root(loc: DataLocation) -> str:
            return loc.key.split("/", 1)[1]

        splits = [
            split_by_image(relocated(root), (0.1, 0.1), seed=7, stable_key=strip_root)
            for root in ("tmpA", "tmpB")
        ]
        self.assertEqual(
            *[
                {name: {strip_root(k) for k in keys} for name, keys in _set_keys(s).items()}
                for s in splits
            ]
        )


class InvalidRatiosTest(unittest.TestCase):
    def test_ratios_leaving_no_train_set_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "ratios"):
            split_by_image(_labels(_background_dataset({"a": 5})), (0.5, 0.5), seed=0)


if __name__ == "__main__":
    unittest.main()
