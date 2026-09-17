"""Unit tests for region_eval/features.py.

Feature files are real npz archives written into a temp dir, in the layout the
extractor emits: `meta` (version, n_points, dim), `rows`/`cols` as uint16, and
`feat` as float64. The loader seam is a callable that reads those files, so no
test here reaches S3.

Every fixture vector is filled with `row * 1000 + col`, which makes alignment
checkable by hand: if the cache ever paired a point's metadata with a
neighbouring point's vector, the first cell would not match the (row, col) the
metadata carries.
"""

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from mermaid_classifier.region_eval.features import (
    build_feature_cache,
    read_feature_file,
    write_feature_cache,
)

REGION = "3c3c3c3c-0000-4000-8000-000000000002"
LABEL = "4d4d4d4d-0000-4000-8000-000000000001::"
DIM = 8


def _cell(row: int, col: int) -> float:
    """The value every cell of one point's fixture vector carries."""
    return float(row * 1000 + col)


def _feature_bytes(points: list[tuple[int, int]], *, dim: int = DIM) -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        meta=np.array([1, len(points), dim], dtype=np.int64),
        rows=np.array([row for row, _ in points], dtype=np.uint16),
        cols=np.array([col for _, col in points], dtype=np.uint16),
        feat=np.array([[_cell(row, col)] * dim for row, col in points], dtype=np.float64),
    )
    return buffer.getvalue()


def _probe_rows(points_by_image: dict[str, list[tuple[int, int]]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for image_id, points in points_by_image.items():
        for position, (row, col) in enumerate(points):
            rows.append(
                {
                    "image_id": image_id,
                    "point_id": f"{image_id}-p{position}",
                    "row": row,
                    "col": col,
                    "gt_label": LABEL,
                    "region_id": REGION,
                    "held_out": image_id != "ta1",
                }
            )
    return pd.DataFrame(rows)


class FeatureCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _write_feature_file(self, image_id: str, points: list[tuple[int, int]]) -> None:
        (self.root / f"{image_id}.npz").write_bytes(_feature_bytes(points))

    def _loader(self, image_id: str) -> bytes:
        return (self.root / f"{image_id}.npz").read_bytes()

    def _assert_aligned(self, cache) -> None:
        for position in range(len(cache.image_ids)):
            self.assertEqual(
                cache.features[position, 0],
                _cell(int(cache.rows[position]), int(cache.cols[position])),
                f"vector {position} does not belong to its metadata",
            )

    def test_a_point_absent_from_the_feature_file_is_dropped_and_counted(self):
        self._write_feature_file("i1", [(10, 20), (50, 60)])
        probe = _probe_rows({"i1": [(10, 20), (30, 40), (50, 60)]})

        cache = build_feature_cache(probe, self._loader, feature_dim=DIM, workers=2)

        self.assertEqual(cache.n_points_requested, 3)
        self.assertEqual(cache.n_points_missing_row_col, 1)
        self.assertEqual(cache.n_points_missing_image, 0)
        self.assertEqual(cache.features.shape, (2, DIM))
        self.assertEqual(list(cache.point_ids), ["i1-p0", "i1-p2"])
        self._assert_aligned(cache)

    def test_an_image_with_no_feature_file_is_skipped_and_counted(self):
        self._write_feature_file("i1", [(10, 20), (30, 40)])
        probe = _probe_rows({"i1": [(10, 20), (30, 40)], "gone": [(70, 80), (90, 100)]})

        cache = build_feature_cache(probe, self._loader, feature_dim=DIM, workers=2)

        self.assertEqual(cache.n_points_missing_image, 2)
        self.assertEqual(cache.n_points_missing_row_col, 0)
        self.assertEqual(cache.missing_image_ids, ("gone",))
        self.assertEqual(list(cache.image_ids), ["i1", "i1"])
        self._assert_aligned(cache)

    def test_surviving_rows_keep_their_probe_metadata(self):
        self._write_feature_file("ta1", [(10, 20)])
        self._write_feature_file("i2", [(30, 40)])
        probe = _probe_rows({"ta1": [(10, 20)], "i2": [(30, 40)]})

        cache = build_feature_cache(probe, self._loader, feature_dim=DIM, workers=2)

        self.assertEqual(list(cache.image_ids), ["ta1", "i2"])
        self.assertEqual(list(cache.gt_labels), [LABEL, LABEL])
        self.assertEqual(list(cache.region_ids), [REGION, REGION])
        self.assertEqual(list(cache.held_out), [False, True])
        self._assert_aligned(cache)

    def test_features_are_cast_to_float32(self):
        self._write_feature_file("i1", [(10, 20)])
        cache = build_feature_cache(
            _probe_rows({"i1": [(10, 20)]}), self._loader, feature_dim=DIM, workers=2
        )
        self.assertEqual(cache.features.dtype, np.float32)

    def test_every_point_missing_leaves_an_empty_but_shaped_matrix(self):
        probe = _probe_rows({"gone": [(10, 20)]})
        cache = build_feature_cache(probe, self._loader, feature_dim=DIM, workers=2)
        self.assertEqual(cache.features.shape, (0, DIM))
        self.assertEqual(cache.n_points_missing_image, 1)

    def test_a_cached_shard_restarts_the_build_without_the_loader(self):
        self._write_feature_file("i1", [(10, 20), (50, 60)])
        probe = _probe_rows({"i1": [(10, 20), (30, 40), (50, 60)], "gone": [(70, 80)]})
        shards = self.root / "shards"

        first = build_feature_cache(
            probe, self._loader, feature_dim=DIM, workers=2, shard_dir=shards
        )

        def refuse(image_id: str) -> bytes:
            raise AssertionError(f"loader called for {image_id} despite a cached shard")

        second = build_feature_cache(probe, refuse, feature_dim=DIM, workers=2, shard_dir=shards)

        self.assertEqual(list(second.point_ids), list(first.point_ids))
        self.assertEqual(second.n_points_missing_row_col, first.n_points_missing_row_col)
        self.assertEqual(second.n_points_missing_image, first.n_points_missing_image)
        self.assertEqual(second.missing_image_ids, first.missing_image_ids)
        np.testing.assert_array_equal(second.features, first.features)

    def test_a_shard_built_for_other_points_is_rebuilt_rather_than_restored(self):
        """Positions in a shard index one run's probe rows. Rebuilding into a
        directory whose shards came from a different selection -- another seed,
        another target size, a refreshed export -- would otherwise lay the old
        run's vectors on the new run's rows: every point scored with another
        image's features while carrying this image's ground truth, and every
        missing count reading zero.
        """
        first_points = {f"i{index}": [(10 * index, 20 * index)] for index in range(1, 5)}
        second_points = {f"i{index}": [(10 * index, 20 * index)] for index in range(5, 9)}
        for points_by_image in (first_points, second_points):
            for image_id, points in points_by_image.items():
                self._write_feature_file(image_id, points)
        shards = self.root / "shards"

        build_feature_cache(
            _probe_rows(first_points),
            self._loader,
            feature_dim=DIM,
            workers=2,
            batch_size=2,
            shard_dir=shards,
        )
        second = build_feature_cache(
            _probe_rows(second_points),
            self._loader,
            feature_dim=DIM,
            workers=2,
            batch_size=2,
            shard_dir=shards,
        )

        self.assertEqual(list(second.image_ids), ["i5", "i6", "i7", "i8"])
        self.assertEqual(second.n_points_missing_row_col, 0)
        self.assertEqual(second.n_points_missing_image, 0)
        self._assert_aligned(second)

    def test_a_shard_written_without_a_points_key_is_rebuilt(self):
        """A shard left by a build that predates the key cannot say which
        points it holds, so restoring it is the same gamble as restoring a
        stale one.
        """
        probe = _probe_rows({"i1": [(10, 20)], "i2": [(30, 40)]})
        for image_id, points in (("i1", [(10, 20)]), ("i2", [(30, 40)])):
            self._write_feature_file(image_id, points)
        shards = self.root / "shards"
        build_feature_cache(
            probe, self._loader, feature_dim=DIM, workers=2, batch_size=1, shard_dir=shards
        )
        stored = np.load(shards / "batch_00000.npz", allow_pickle=False)
        np.savez_compressed(
            shards / "batch_00000.npz",
            **{name: stored[name] for name in stored.files if name != "points_key"},
        )
        fetched: list[str] = []

        def recording(image_id: str) -> bytes:
            fetched.append(image_id)
            return self._loader(image_id)

        cache = build_feature_cache(
            probe, recording, feature_dim=DIM, workers=2, batch_size=1, shard_dir=shards
        )

        self.assertEqual(fetched, ["i1"], "the keyless shard is the only batch downloaded again")
        self.assertEqual(list(cache.image_ids), ["i1", "i2"])
        self._assert_aligned(cache)

    def test_a_feature_file_that_does_not_parse_raises_rather_than_counting_it_missing(self):
        """A truncated archive that read as a missing image would shrink the
        probe silently: the missing-image count absorbs it, and the score is
        taken on fewer points than it says. Only a fetch is forgiving.
        """
        self._write_feature_file("i1", [(10, 20)])
        whole = _feature_bytes([(30, 40)])
        (self.root / "i2.npz").write_bytes(whole[: len(whole) // 2])
        probe = _probe_rows({"i1": [(10, 20)], "i2": [(30, 40)]})

        with self.assertRaises(zipfile.BadZipFile):
            build_feature_cache(probe, self._loader, feature_dim=DIM, workers=2)

    def test_batching_covers_every_image(self):
        images = {f"i{index}": [(10 * index, 20 * index)] for index in range(1, 8)}
        for image_id, points in images.items():
            self._write_feature_file(image_id, points)

        cache = build_feature_cache(
            _probe_rows(images), self._loader, feature_dim=DIM, workers=2, batch_size=2
        )

        self.assertEqual(len(cache.image_ids), 7)
        self.assertEqual(cache.n_points_missing_row_col, 0)
        self._assert_aligned(cache)

    def test_a_feature_file_of_the_wrong_dimension_raises(self):
        (self.root / "i1.npz").write_bytes(_feature_bytes([(10, 20)], dim=DIM + 1))
        with self.assertRaises(ValueError):
            build_feature_cache(
                _probe_rows({"i1": [(10, 20)]}), self._loader, feature_dim=DIM, workers=2
            )

    def test_written_cache_round_trips_its_arrays(self):
        self._write_feature_file("i1", [(10, 20)])
        cache = build_feature_cache(
            _probe_rows({"i1": [(10, 20)]}), self._loader, feature_dim=DIM, workers=2
        )
        path = self.root / "probe_features.npz"

        write_feature_cache(cache, path)

        stored = np.load(path, allow_pickle=False)
        np.testing.assert_array_equal(stored["features"], cache.features)
        self.assertEqual(list(stored["point_id"]), ["i1-p0"])
        self.assertEqual(list(stored["held_out"]), [True])


class ReadFeatureFileTest(unittest.TestCase):
    def test_rows_cols_and_features_come_back_aligned(self):
        rows, cols, feat = read_feature_file(_feature_bytes([(10, 20), (30, 40)]))
        self.assertEqual(list(rows), [10, 30])
        self.assertEqual(list(cols), [20, 40])
        self.assertEqual(feat.shape, (2, DIM))
        self.assertEqual(feat[1, 0], _cell(30, 40))

    def test_a_truncated_archive_raises(self):
        buffer = io.BytesIO()
        np.savez(
            buffer,
            meta=np.array([1, 2, DIM], dtype=np.int64),
            rows=np.array([10, 30], dtype=np.uint16),
            cols=np.array([20], dtype=np.uint16),
            feat=np.zeros((2, DIM), dtype=np.float64),
        )
        with self.assertRaises(ValueError):
            read_feature_file(buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
