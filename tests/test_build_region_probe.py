"""Tests for scripts/build_region_probe.py.

`PublishProbeTest` stubs the S3 client directly, so a probe version's upload
is asserted key by key rather than through the network. `BuildRegionProbeMainTest`
runs the real local build end to end -- annotation selection, the frozen
snapshots, the manifest -- with only the two network-facing seams
(`fetch_source`, the benthic-attribute/growth-form/region libraries) replaced,
and checks that omitting --publish leaves `publish_probe` untouched while
giving it never touches S3 either way. `HeldOutImagesCsvTest` round-trips
`held_out_images_csv`'s output through the real `ImageExclusionFilter` that
training reads it with, rather than asserting the CSV's text.
"""

import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

import duckdb
import pandas as pd

from mermaid_classifier.pyspacer.label_specs import ImageExclusionFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_region_probe as brp  # noqa: E402

TROPICAL_ATLANTIC = "9a9a9a9a-0000-4000-8000-000000000001"
BA_ROOT = "9b9b9b9b-0000-4000-8000-000000000000"
BA_CORAL = "9b9b9b9b-0000-4000-8000-000000000001"

_ALL_PUBLISHED_FILES = (
    brp.PROBE_POINTS_FILE,
    brp.PROBE_HELD_OUT_IMAGES_FILE,
    brp.PROBE_REGIONS_FILE,
    brp.PROBE_COUNTS_FILE,
    brp.PROBE_NAMES_FILE,
    brp.PROBE_ANCESTRY_FILE,
    brp.PROBE_MANIFEST_FILE,
    brp.PROBE_FEATURES_FILE,
)


class _FakeBenthicAttributeLibrary:
    """Just enough of BenthicAttributeLibrary for one probe build: one
    attribute, its regions, and a one-level ancestry."""

    def __init__(self):
        self.by_id = {BA_CORAL: {"name": "Some Coral"}}
        self.region_ids_by_id = {BA_CORAL: frozenset({TROPICAL_ATLANTIC})}

    def get_ancestor_ids(self, attribute_id: str) -> list[str]:
        return [BA_ROOT] if attribute_id == BA_CORAL else []


class _FakeChoiceLibrary:
    """A stand-in for GrowthFormLibrary/RegionLibrary: an id-to-name map."""

    def __init__(self, by_id: dict[str, str]):
        self.by_id = by_id


def _annotations() -> pd.DataFrame:
    """Two Tropical Atlantic images, in the shape read_annotations emits.

    Tropical Atlantic is the default census region, so both images enter the
    probe in full regardless of eligibility, which keeps this fixture minimal.
    """
    rows: list[dict[str, object]] = []
    for image_index in range(2):
        for point_index in range(2):
            rows.append(
                {
                    "image_id": f"img{image_index}",
                    "point_id": f"img{image_index}-p{point_index}",
                    "row": 10 * point_index,
                    "col": 20 * point_index,
                    "benthic_attribute_id": BA_CORAL,
                    "benthic_attribute_name": "Some Coral",
                    "growth_form_id": "",
                    "growth_form_name": "",
                    "region_id": TROPICAL_ATLANTIC,
                    "region_name": "Tropical Atlantic",
                }
            )
    return pd.DataFrame(rows)


class PublishProbeTest(unittest.TestCase):
    """`publish_probe` against a stubbed S3 client -- no network involved."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = Path(self.tmp.name) / "out"
        self.out_dir.mkdir()
        for name in _ALL_PUBLISHED_FILES:
            (self.out_dir / name).write_bytes(name.encode())

    def _client(self, contents: list[dict[str, str]] | None = None) -> mock.Mock:
        client = mock.Mock()
        found = contents or []
        client.list_objects_v2.return_value = {"KeyCount": len(found), "Contents": found}
        return client

    def test_uploads_every_expected_file_and_only_those(self):
        client = self._client()
        with mock.patch.object(brp.boto3, "client", return_value=client) as make_client:
            uris = brp.publish_probe(
                self.out_dir, "s3://bucket/prefix/v1/", region_name="us-east-1"
            )

        make_client.assert_called_once_with("s3", region_name="us-east-1")
        client.list_objects_v2.assert_called_once_with(Bucket="bucket", Prefix="prefix/v1/")

        uploaded_keys = {call.args[2] for call in client.upload_file.call_args_list}
        expected_keys = {f"prefix/v1/{name}" for name in _ALL_PUBLISHED_FILES}
        self.assertEqual(uploaded_keys, expected_keys)
        self.assertEqual(client.upload_file.call_count, len(_ALL_PUBLISHED_FILES))
        self.assertEqual(set(uris), {f"s3://bucket/{key}" for key in expected_keys})

    def test_a_skip_features_build_uploads_no_feature_cache(self):
        (self.out_dir / brp.PROBE_FEATURES_FILE).unlink()
        client = self._client()
        with mock.patch.object(brp.boto3, "client", return_value=client):
            uris = brp.publish_probe(
                self.out_dir, "s3://bucket/prefix/v1/", region_name="us-east-1"
            )

        uploaded_keys = {call.args[2] for call in client.upload_file.call_args_list}
        expected_keys = {f"prefix/v1/{name}" for name in _ALL_PUBLISHED_FILES[:-1]}
        self.assertEqual(uploaded_keys, expected_keys)
        self.assertNotIn(f"prefix/v1/{brp.PROBE_FEATURES_FILE}", uploaded_keys)
        self.assertEqual(len(uris), len(_ALL_PUBLISHED_FILES) - 1)

    def test_a_prefix_missing_its_trailing_slash_still_nests_under_the_version(self):
        client = self._client()
        with mock.patch.object(brp.boto3, "client", return_value=client):
            brp.publish_probe(self.out_dir, "s3://bucket/prefix/v1", region_name="us-east-1")

        uploaded_keys = {call.args[2] for call in client.upload_file.call_args_list}
        self.assertIn(f"prefix/v1/{brp.PROBE_POINTS_FILE}", uploaded_keys)

    def test_refuses_when_an_object_already_exists_and_names_it(self):
        conflict_key = f"prefix/v1/{brp.PROBE_POINTS_FILE}"
        client = self._client(contents=[{"Key": conflict_key}])
        with (
            mock.patch.object(brp.boto3, "client", return_value=client),
            self.assertRaises(FileExistsError) as ctx,
        ):
            brp.publish_probe(self.out_dir, "s3://bucket/prefix/v1/", region_name="us-east-1")

        self.assertIn(conflict_key, str(ctx.exception))
        client.upload_file.assert_not_called()


class BuildRegionProbeMainTest(unittest.TestCase):
    """`main()`'s local build, with only its two network seams replaced."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.parquet"
        _annotations().to_parquet(self.source, index=False)

        patches = [
            mock.patch.object(
                brp, "get_benthic_attribute_library", return_value=_FakeBenthicAttributeLibrary()
            ),
            mock.patch.object(brp, "get_growth_form_library", return_value=_FakeChoiceLibrary({})),
            mock.patch.object(
                brp,
                "get_region_library",
                return_value=_FakeChoiceLibrary({TROPICAL_ATLANTIC: "Tropical Atlantic"}),
            ),
            mock.patch.object(brp, "fetch_source", side_effect=self._fake_fetch_source),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _fake_fetch_source(self, uri: str, destination: Path, *, region_name: str) -> str:
        shutil.copy(self.source, destination)
        return "etag-fixture"

    def test_omitting_publish_writes_locally_and_calls_publish_probe_never(self):
        out_dir = self.root / "out"
        with mock.patch.object(brp, "publish_probe") as publish:
            rc = brp.main(["--out-dir", str(out_dir), "--skip-features"])

        self.assertEqual(rc, 0)
        publish.assert_not_called()
        for name in _ALL_PUBLISHED_FILES[:-1]:  # --skip-features never writes the npz
            self.assertTrue((out_dir / name).exists(), f"{name} was not written")
        self.assertFalse((out_dir / brp.PROBE_FEATURES_FILE).exists())

    def test_publish_forwards_the_built_directory_and_the_given_uri(self):
        out_dir = self.root / "out"
        with mock.patch.object(brp, "publish_probe", return_value=[]) as publish:
            rc = brp.main(
                [
                    "--out-dir",
                    str(out_dir),
                    "--skip-features",
                    "--publish",
                    "s3://bucket/prefix/v9/",
                ]
            )

        self.assertEqual(rc, 0)
        publish.assert_called_once_with(
            out_dir, "s3://bucket/prefix/v9/", region_name=brp.DEFAULT_REGION
        )


class HeldOutImagesCsvTest(unittest.TestCase):
    """`held_out_images_csv`'s output, fed back through the real
    `ImageExclusionFilter` that training reads it with."""

    def _filtered_ids(self, csv_text: str, image_ids: list[str]) -> set[str]:
        """Every id in `image_ids` that survives `csv_text`'s exclusion list."""
        conn = duckdb.connect()
        seed = pd.DataFrame({"image_id": image_ids})  # noqa: F841 — referenced by name in DuckDB SQL
        conn.execute("CREATE TABLE annotations AS SELECT * FROM seed")

        exclusion = ImageExclusionFilter(StringIO(csv_text))
        exclusion.filter_in_duckdb(conn, "annotations")

        return {row[0] for row in conn.execute("SELECT image_id FROM annotations").fetchall()}

    def test_excludes_exactly_the_held_out_images_and_no_others(self):
        rows = pd.DataFrame(
            {
                "image_id": ["img1", "img1", "img2", "img3", "img3"],
                "held_out": [True, True, False, True, True],
            }
        )
        csv_text = brp.held_out_images_csv(rows)

        remaining = self._filtered_ids(csv_text, ["img1", "img2", "img3"])

        self.assertEqual(remaining, {"img2"})

    def test_no_held_out_rows_writes_a_header_only_csv_that_excludes_nothing(self):
        rows = pd.DataFrame({"image_id": ["img1", "img2"], "held_out": [False, False]})
        csv_text = brp.held_out_images_csv(rows)

        exclusion = ImageExclusionFilter(StringIO(csv_text))

        self.assertTrue(exclusion.is_empty())
        remaining = self._filtered_ids(csv_text, ["img1", "img2"])
        self.assertEqual(remaining, {"img1", "img2"})


if __name__ == "__main__":
    unittest.main()
