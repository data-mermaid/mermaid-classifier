"""Unit tests for region_eval/features.py.

Feature files are real npz archives written straight into the temp download
directory `build_feature_cache` reads from, in the layout the extractor
emits: `meta` (version, n_points, dim), `rows`/`cols` as uint16, and `feat` as
float64. Pre-populating that directory is the injection point: the real
`download_features_parallel` sees every file already on disk and never
touches S3. A test simulating an image with no feature file at all patches
`download_features_parallel` instead, since only S3 would otherwise say so.

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
from unittest import mock

import numpy as np
import pandas as pd

from mermaid_classifier.region_eval.features import (
    DEFAULT_FEATURE_BUCKET,
    DEFAULT_FEATURE_PREFIX,
    DEFAULT_FEATURE_SUFFIX,
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
        (self.root / f"{image_id}{DEFAULT_FEATURE_SUFFIX}").write_bytes(_feature_bytes(points))

    def _no_s3(self):
        """Patch the downloader for a case an image legitimately has no file.

        Without this, `build_feature_cache` would ask the real downloader to
        fetch it, which reaches for S3.
        """
        return mock.patch(
            "mermaid_classifier.region_eval.features.download_features_parallel",
            return_value=set(),
        )

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

        cache = build_feature_cache(probe, self.root, feature_dim=DIM, workers=2)

        self.assertEqual(cache.n_points_requested, 3)
        self.assertEqual(cache.n_points_missing_row_col, 1)
        self.assertEqual(cache.n_points_missing_image, 0)
        self.assertEqual(cache.features.shape, (2, DIM))
        self.assertEqual(list(cache.point_ids), ["i1-p0", "i1-p2"])
        self._assert_aligned(cache)

    def test_an_image_with_no_feature_file_is_skipped_and_counted(self):
        self._write_feature_file("i1", [(10, 20), (30, 40)])
        probe = _probe_rows({"i1": [(10, 20), (30, 40)], "gone": [(70, 80), (90, 100)]})

        with self._no_s3():
            cache = build_feature_cache(probe, self.root, feature_dim=DIM, workers=2)

        self.assertEqual(cache.n_points_missing_image, 2)
        self.assertEqual(cache.n_points_missing_row_col, 0)
        self.assertEqual(cache.missing_image_ids, ("gone",))
        self.assertEqual(list(cache.image_ids), ["i1", "i1"])
        self._assert_aligned(cache)

    def test_surviving_rows_keep_their_probe_metadata(self):
        self._write_feature_file("ta1", [(10, 20)])
        self._write_feature_file("i2", [(30, 40)])
        probe = _probe_rows({"ta1": [(10, 20)], "i2": [(30, 40)]})

        cache = build_feature_cache(probe, self.root, feature_dim=DIM, workers=2)

        self.assertEqual(list(cache.image_ids), ["ta1", "i2"])
        self.assertEqual(list(cache.gt_labels), [LABEL, LABEL])
        self.assertEqual(list(cache.region_ids), [REGION, REGION])
        self.assertEqual(list(cache.held_out), [False, True])
        self._assert_aligned(cache)

    def test_features_are_cast_to_float32(self):
        self._write_feature_file("i1", [(10, 20)])
        cache = build_feature_cache(
            _probe_rows({"i1": [(10, 20)]}), self.root, feature_dim=DIM, workers=2
        )
        self.assertEqual(cache.features.dtype, np.float32)

    def test_every_point_missing_leaves_an_empty_but_shaped_matrix(self):
        probe = _probe_rows({"gone": [(10, 20)]})
        with self._no_s3():
            cache = build_feature_cache(probe, self.root, feature_dim=DIM, workers=2)
        self.assertEqual(cache.features.shape, (0, DIM))
        self.assertEqual(cache.n_points_missing_image, 1)

    def test_a_feature_file_that_does_not_parse_raises_rather_than_counting_it_missing(self):
        """A truncated archive that read as a missing image would shrink the
        probe silently: the missing-image count absorbs it, and the score is
        taken on fewer points than it says. Only a fetch is forgiving.
        """
        self._write_feature_file("i1", [(10, 20)])
        whole = _feature_bytes([(30, 40)])
        (self.root / f"i2{DEFAULT_FEATURE_SUFFIX}").write_bytes(whole[: len(whole) // 2])
        probe = _probe_rows({"i1": [(10, 20)], "i2": [(30, 40)]})

        with self.assertRaises(zipfile.BadZipFile):
            build_feature_cache(probe, self.root, feature_dim=DIM, workers=2)

    def test_a_feature_file_of_the_wrong_dimension_raises(self):
        (self.root / f"i1{DEFAULT_FEATURE_SUFFIX}").write_bytes(
            _feature_bytes([(10, 20)], dim=DIM + 1)
        )
        with self.assertRaises(ValueError):
            build_feature_cache(
                _probe_rows({"i1": [(10, 20)]}), self.root, feature_dim=DIM, workers=2
            )

    def test_written_cache_stores_n_points_requested(self):
        self._write_feature_file("i1", [(10, 20)])
        rows = _probe_rows({"i1": [(10, 20)], "gone": [(30, 40)]}).assign(
            benthic_attribute_id="ba1",
            benthic_attribute_name="BA1",
            growth_form_id="gf1",
            growth_form_name="GF1",
            region_name="Region",
            site_id="",
        )
        with self._no_s3():
            cache = build_feature_cache(rows, self.root, feature_dim=DIM, workers=2)
        path = self.root / "probe_features.npz"

        write_feature_cache(cache, rows, path)

        stored = np.load(path, allow_pickle=False)
        self.assertEqual(int(stored["n_points_requested"]), 2)

    def test_a_failed_download_is_counted_apart_from_a_missing_file(self):
        """A throttled or credential-expired transfer must not read as the
        file never having existed: only the image download_features_parallel
        actually reports failed lands in the download-failure bucket, so the
        two counts partition the same missing-image total differently.
        """
        self._write_feature_file("i1", [(10, 20)])
        probe = _probe_rows({"i1": [(10, 20)], "gone": [(30, 40)], "throttled": [(50, 60)]})
        failed_key = (
            DEFAULT_FEATURE_BUCKET,
            f"{DEFAULT_FEATURE_PREFIX}throttled{DEFAULT_FEATURE_SUFFIX}",
        )
        with mock.patch(
            "mermaid_classifier.region_eval.features.download_features_parallel",
            return_value={failed_key},
        ):
            cache = build_feature_cache(probe, self.root, feature_dim=DIM, workers=2)

        self.assertEqual(cache.n_points_missing_download_failed, 1)
        self.assertEqual(cache.download_failed_image_ids, ("throttled",))
        self.assertEqual(cache.n_points_missing_image, 1)
        self.assertEqual(cache.missing_image_ids, ("gone",))

    def test_a_download_failure_is_logged_apart_from_a_missing_file(self):
        probe = _probe_rows({"gone": [(30, 40)], "throttled": [(50, 60)]})
        failed_key = (
            DEFAULT_FEATURE_BUCKET,
            f"{DEFAULT_FEATURE_PREFIX}throttled{DEFAULT_FEATURE_SUFFIX}",
        )
        with (
            mock.patch(
                "mermaid_classifier.region_eval.features.download_features_parallel",
                return_value={failed_key},
            ),
            self.assertLogs("mermaid_classifier.region_eval.features", level="WARNING") as logs,
        ):
            build_feature_cache(probe, self.root, feature_dim=DIM, workers=2)

        messages = "\n".join(logs.output)
        self.assertIn("throttled", messages)
        self.assertIn("download", messages.lower())

    def test_an_image_id_containing_a_path_separator_raises_without_downloading(self):
        probe = _probe_rows({"a/b": [(10, 20)]})

        with (
            mock.patch(
                "mermaid_classifier.region_eval.features.download_features_parallel"
            ) as mock_download,
            self.assertRaises(ValueError),
        ):
            build_feature_cache(probe, self.root, feature_dim=DIM, workers=2)

        mock_download.assert_not_called()

    def test_a_bare_dotdot_image_id_raises_without_downloading(self):
        probe = _probe_rows({"..": [(10, 20)]})

        with (
            mock.patch(
                "mermaid_classifier.region_eval.features.download_features_parallel"
            ) as mock_download,
            self.assertRaises(ValueError),
        ):
            build_feature_cache(probe, self.root, feature_dim=DIM, workers=2)

        mock_download.assert_not_called()

    def test_a_well_formed_uuid_image_id_still_works(self):
        image_id = "5e5e5e5e-0000-4000-8000-000000000003"
        self._write_feature_file(image_id, [(10, 20)])
        probe = _probe_rows({image_id: [(10, 20)]})

        cache = build_feature_cache(probe, self.root, feature_dim=DIM, workers=2)

        self.assertEqual(list(cache.image_ids), [image_id])
        self._assert_aligned(cache)

    def test_written_cache_round_trips_its_arrays(self):
        self._write_feature_file("i1", [(10, 20)])
        # write_feature_cache hashes the full PROBE_COLUMNS schema, wider than
        # the METADATA_COLUMNS build_feature_cache itself needs.
        rows = _probe_rows({"i1": [(10, 20)]}).assign(
            benthic_attribute_id="ba1",
            benthic_attribute_name="BA1",
            growth_form_id="gf1",
            growth_form_name="GF1",
            region_name="Region",
            site_id="",
        )
        cache = build_feature_cache(rows, self.root, feature_dim=DIM, workers=2)
        path = self.root / "probe_features.npz"

        write_feature_cache(cache, rows, path)

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
