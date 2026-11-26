"""Unit tests for scripts/build_feature_set.py.

S3 is a hand-written in-memory fake that honours IfMatch / IfNoneMatch the
way the dataset bucket does, and the dataset version is a real segdata-spec
manifest and tables, so sync_tables / open_tables / claim_extractor run for
real. pyspacer is replaced through sys.modules: the EfficientNet forward pass
is covered upstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

import pyarrow as pa
from botocore.exceptions import ClientError
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import ExtractorMismatch, PinMismatch
from segdata_spec.manifest import (
    EtlInfo,
    FileEntry,
    ImageStore,
    Manifest,
    Pin,
    SourceInfo,
    Stats,
    dump_manifest,
    manifest_sha256,
)
from segdata_spec.publish import table_bytes
from segdata_spec.schema import (
    IMAGES_SCHEMA,
    LABEL_COUNTS_SCHEMA,
    LABELS_SCHEMA,
    POINTS_SCHEMA,
    conform,
)
from support.paths import add_scripts_to_path

add_scripts_to_path()

import build_feature_set as bfs  # noqa: E402
import launch_processing as lp  # noqa: E402

BUCKET = DATASET_BUCKET
DATASET = "coralnet"
VERSION = "20261124-cfc389d"
IMAGE_PREFIX = "images/coralnet/"
FAMILY = "efficientnet-b0-v1"
FEATURE_PREFIX = f"features/coralnet/{FAMILY}/"
SPEC_KEY = f"{FEATURE_PREFIX}_extractor_spec.json"
EXTRACTOR = {
    "class_path": "spacer.extractors.efficientnet.EfficientNetExtractor",
    "crop_size": 224,
    "feature_dim": 1280,
    "weights_uri": "s3://test-bucket/efficientnet.pt",
    "weights_sha256": "0" * 64,
}


# ---- fake S3 ---------------------------------------------------------


def _client_error(code: str, status: int, op: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        op,
    )


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _Paginator:
    def __init__(self, fake: FakeS3):
        self._fake = fake

    def paginate(self, *, Bucket: str, Prefix: str):
        keys = sorted(k for b, k in self._fake.objects if b == Bucket and k.startswith(Prefix))
        yield {"Contents": [{"Key": k} for k in keys]} if keys else {}


class FakeS3:
    """The boto3 client surface the script and segdata-spec touch.

    ``calls`` logs every get and put as ``(op, key, kwargs)`` in order.
    ``put_conflicts`` names keys whose conditional put fails with 412 even
    though the fake holds no object there, as a concurrent writer would.
    """

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.etag_overrides: dict[tuple[str, str], str] = {}
        self.put_conflicts: set[str] = set()
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def seed(self, key: str, data: bytes) -> None:
        self.objects[(BUCKET, key)] = data

    def etag(self, key: str) -> str:
        return (
            self.etag_overrides.get((BUCKET, key))
            or hashlib.md5(self.objects[(BUCKET, key)]).hexdigest()
        )

    def get_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get", Key, dict(kwargs)))
        if (Bucket, Key) not in self.objects:
            raise _client_error("NoSuchKey", 404, "GetObject")
        if_match = kwargs.get("IfMatch")
        if if_match is not None and if_match.strip('"') != self.etag(Key):
            raise _client_error("PreconditionFailed", 412, "GetObject")
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def put_object(self, *, Bucket: str, Key: str, Body: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("put", Key, dict(kwargs)))
        exists = (Bucket, Key) in self.objects or Key in self.put_conflicts
        if kwargs.get("IfNoneMatch") == "*" and exists:
            raise _client_error("PreconditionFailed", 412, "PutObject")
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()
        return {}

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self)

    def keys_called(self, op: str) -> list[str]:
        return [key for o, key, _ in self.calls if o == op]


# ---- a published dataset version --------------------------------------


def _publish(fake: FakeS3, images: dict[str, tuple[str, str]], points: list[tuple[str, int, int]]):
    """Seed a points dataset version; ``images`` maps image_id to (key, group).

    Each image's bytes are its id, so its ETag is the MD5 of the id.
    Returns the version's Pin.
    """
    ids = list(images)
    for image_id, (key, _group) in images.items():
        fake.seed(IMAGE_PREFIX + key, image_id.encode())
    n = len(ids)
    images_table = conform(
        pa.table(
            {
                "image_id": ids,
                "key": [images[i][0] for i in ids],
                "etag": [fake.etag(IMAGE_PREFIX + images[i][0]) for i in ids],
                "size": [len(i) for i in ids],
                "crc64nvme": pa.nulls(n, pa.string()),
                "media_type": ["image/jpeg"] * n,
                "width": [100] * n,
                "height": [100] * n,
                "group": [images[i][1] for i in ids],
                "mask_key": pa.nulls(n, pa.string()),
                "mask_etag": pa.nulls(n, pa.string()),
                "mask_size": pa.nulls(n, pa.int64()),
                "mask_crc64nvme": pa.nulls(n, pa.string()),
                "source_fingerprint": ["fp"] * n,
                "source_name": pa.nulls(n, pa.string()),
                "attrs": pa.nulls(n, pa.string()),
            }
        ),
        IMAGES_SCHEMA,
    )
    points_table = conform(
        pa.table(
            {
                "image_id": [p[0] for p in points],
                "row": [p[1] for p in points],
                "col": [p[2] for p in points],
                "label_id": [1] * len(points),
                "annotator": ["a"] * len(points),
            }
        ),
        POINTS_SCHEMA,
    )
    labels = conform(
        pa.table(
            {
                "label_id": [1],
                "name": ["Coral"],
                "mermaid_benthic_attribute_id": [None],
                "mermaid_benthic_attribute": [None],
                "mermaid_growth_form": [None],
                "mapping_source": ["identity"],
            }
        ),
        LABELS_SCHEMA,
    )
    counts = conform(
        pa.table({"image_id": ids, "label_id": [1] * n, "n": [1] * n}), LABEL_COUNTS_SCHEMA
    )
    tables = {
        "images.parquet": images_table,
        "points.parquet": points_table,
        "labels.parquet": labels,
        "stats/label_counts.parquet": counts,
    }
    prefix = f"datasets/{DATASET}/{VERSION}/"
    files = {}
    for name, table in tables.items():
        data = table_bytes(table)
        fake.seed(prefix + name, data)
        files[name] = FileEntry(
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), rows=table.num_rows
        )
    manifest = Manifest(
        schema_version=1,
        dataset=DATASET,
        version=VERSION,
        kind="points",
        created_at=datetime(2026, 11, 24, tzinfo=UTC),
        image_store=ImageStore(bucket=BUCKET, prefix=IMAGE_PREFIX, owner="segdata"),
        etl=EtlInfo(
            package="segdata-etl",
            command="test",
            git_sha="cfc389d",
            git_dirty=False,
            uv_lock_sha256="l" * 64,
        ),
        source=SourceInfo(type="test", uri="s3://src/", revision="r1"),
        files=files,
        stats=Stats(n_images=n, n_points=len(points), n_groups=1, image_bytes=0, dropped={}),
    )
    manifest_bytes = dump_manifest(manifest)
    fake.seed(prefix + "manifest.json", manifest_bytes)
    return Pin(name=DATASET, version=VERSION, manifest_sha256=manifest_sha256(manifest_bytes))


# ---- harness ----------------------------------------------------------


class FakeProgressWriter:
    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def write(self, s: str) -> None:
        self.records.append(json.loads(s))

    def flush(self) -> None:
        pass


class FakeErrorWriter:
    def __init__(self):
        self.rows: list[list[str]] = []

    def writerow(self, row: list[str]) -> None:
        self.rows.append(row)


class RunHarness(unittest.TestCase):
    """Runs bfs.run against a FakeS3 with pyspacer stubbed out.

    The stub extract_features records each message and writes
    ``FV:<job_token>`` to the message's filesystem feature location.
    """

    def setUp(self):
        self.fake = FakeS3()
        self.cache = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.cache, ignore_errors=True))
        self.extract_msgs: list[dict[str, Any]] = []
        self.progress = FakeProgressWriter()
        self.errors = FakeErrorWriter()

    def _extract(self, msg: dict[str, Any]) -> None:
        image_bytes = Path(msg["image_loc"]["key"]).read_bytes()
        self.extract_msgs.append({**msg, "image_bytes": image_bytes})
        Path(msg["feature_loc"]["key"]).write_bytes(f"FV:{msg['job_token']}".encode())

    def run_script(self, pin: Pin, **overrides: Any) -> bfs.RunCounters:
        kwargs: dict[str, Any] = {
            "client": self.fake,
            "pin": pin,
            "bucket": BUCKET,
            "family": FAMILY,
            "cache_dir": self.cache,
            "groups": None,
            "shard": None,
            "extractor": "STUB",
            "extractor_spec": EXTRACTOR,
            "dry_run": False,
            "max_io_workers": 2,
            "progress_writer": self.progress,
            "error_writer": self.errors,
        }
        kwargs.update(overrides)
        with mock.patch.dict(
            sys.modules,
            {
                "spacer.tasks": mock.MagicMock(extract_features=self._extract),
                "spacer.data_classes": mock.MagicMock(DataLocation=lambda **kw: dict(kw)),
                "spacer.messages": mock.MagicMock(ExtractFeaturesMsg=lambda **kw: dict(kw)),
            },
        ):
            return bfs.run(**kwargs)

    def image_gets(self) -> list[str]:
        return [k for k in self.fake.keys_called("get") if k.startswith(IMAGE_PREFIX)]

    def feature_puts(self) -> list[str]:
        return [
            k
            for k in self.fake.keys_called("put")
            if k.startswith(FEATURE_PREFIX) and k != SPEC_KEY
        ]


# ---- tests ------------------------------------------------------------


class ClaimExtractorTest(RunHarness):
    def test_the_family_is_claimed_before_any_image_is_read(self):
        pin = _publish(self.fake, {"i1": ("s1/i1.jpg", "s1")}, [("i1", 5, 6)])

        self.run_script(pin)

        ops = [(op, key) for op, key, _ in self.fake.calls]
        claim = ops.index(("put", SPEC_KEY))
        first_image = ops.index(("get", IMAGE_PREFIX + "s1/i1.jpg"))
        self.assertLess(claim, first_image)
        self.assertEqual(json.loads(self.fake.objects[(BUCKET, SPEC_KEY)]), EXTRACTOR)

    def test_a_family_claimed_by_another_extractor_reads_and_writes_nothing(self):
        pin = _publish(self.fake, {"i1": ("s1/i1.jpg", "s1")}, [("i1", 5, 6)])
        self.fake.seed(SPEC_KEY, json.dumps({**EXTRACTOR, "weights_sha256": "f" * 64}).encode())

        with self.assertRaises(ExtractorMismatch):
            self.run_script(pin)

        self.assertEqual(self.image_gets(), [])
        self.assertEqual(self.feature_puts(), [])
        self.assertEqual(self.extract_msgs, [])


class ExtractImageTest(RunHarness):
    def test_an_image_without_a_family_file_is_extracted_at_its_points_and_put_once(self):
        pin = _publish(
            self.fake,
            {"i1": ("s1/i1.jpg", "s1"), "i2": ("s1/i2.jpg", "s1")},
            # Unsorted, with a duplicate (row, col), interleaved with another image's point.
            [("i1", 30, 4), ("i2", 1, 1), ("i1", 2, 9), ("i1", 30, 4), ("i1", 2, 3)],
        )

        counters = self.run_script(pin)

        gets = {k: kw for op, k, kw in self.fake.calls if op == "get"}
        self.assertEqual(
            gets[IMAGE_PREFIX + "s1/i1.jpg"]["IfMatch"], hashlib.md5(b"i1").hexdigest()
        )
        by_token = {m["job_token"]: m for m in self.extract_msgs}
        self.assertEqual(by_token["i1"]["rowcols"], [(2, 3), (2, 9), (30, 4)])
        self.assertEqual(by_token["i1"]["image_bytes"], b"i1")
        self.assertEqual(by_token["i1"]["image_loc"]["storage_type"], "filesystem")
        self.assertEqual(by_token["i1"]["feature_loc"]["storage_type"], "filesystem")
        puts = {k: kw for op, k, kw in self.fake.calls if op == "put"}
        key = f"{FEATURE_PREFIX}i1.featurevector"
        self.assertEqual(puts[key], {"IfNoneMatch": "*", "ChecksumAlgorithm": "CRC64NVME"})
        self.assertEqual(self.fake.objects[(BUCKET, key)], b"FV:i1")
        self.assertEqual(counters.images_ok, 2)

    def test_an_image_whose_family_file_exists_is_skipped_without_a_get(self):
        pin = _publish(
            self.fake,
            {"i1": ("s1/i1.jpg", "s1"), "i2": ("s1/i2.jpg", "s1")},
            [("i1", 1, 1), ("i2", 2, 2)],
        )
        self.fake.seed(f"{FEATURE_PREFIX}i1.featurevector", b"old")

        counters = self.run_script(pin)

        self.assertEqual(self.image_gets(), [IMAGE_PREFIX + "s1/i2.jpg"])
        self.assertEqual(self.fake.objects[(BUCKET, f"{FEATURE_PREFIX}i1.featurevector")], b"old")
        self.assertEqual((counters.images_ok, counters.images_skipped), (1, 1))

    def test_a_png_image_is_read_from_its_exact_key(self):
        pin = _publish(self.fake, {"i7": ("s2/i7.png", "s2")}, [("i7", 1, 1)])

        self.run_script(pin)

        self.assertEqual(self.image_gets(), [IMAGE_PREFIX + "s2/i7.png"])
        self.assertEqual(self.feature_puts(), [f"{FEATURE_PREFIX}i7.featurevector"])

    def test_an_image_with_no_points_is_skipped_without_a_get_or_put(self):
        pin = _publish(
            self.fake, {"i1": ("s1/i1.jpg", "s1"), "i2": ("s1/i2.jpg", "s1")}, [("i1", 1, 1)]
        )

        counters = self.run_script(pin)

        self.assertEqual(self.image_gets(), [IMAGE_PREFIX + "s1/i1.jpg"])
        self.assertEqual(self.feature_puts(), [f"{FEATURE_PREFIX}i1.featurevector"])
        records = {r["image_id"]: r for r in self.progress.records}
        self.assertEqual(
            (records["i2"]["outcome"], records["i2"]["reason"]), ("skipped", "no_points")
        )
        self.assertEqual(counters.images_failed, 0)


class FailureTest(RunHarness):
    def test_a_changed_image_fails_alone_and_the_run_exits_1(self):
        pin = _publish(
            self.fake,
            {"i1": ("s1/i1.jpg", "s1"), "i2": ("s1/i2.jpg", "s1")},
            [("i1", 1, 1), ("i2", 2, 2)],
        )
        # The object at i1's key no longer matches the ETag images.parquet recorded.
        self.fake.seed(IMAGE_PREFIX + "s1/i1.jpg", b"replaced")

        counters = self.run_script(pin)

        self.assertEqual(self.feature_puts(), [f"{FEATURE_PREFIX}i2.featurevector"])
        self.assertEqual((counters.images_ok, counters.images_failed), (1, 1))
        self.assertEqual([row[1] for row in self.errors.rows], ["i1"])
        self.assertEqual(counters.exit_code(), 1)

    def test_a_feature_put_that_finds_the_file_present_counts_as_skipped(self):
        pin = _publish(self.fake, {"i1": ("s1/i1.jpg", "s1")}, [("i1", 1, 1)])
        self.fake.put_conflicts.add(f"{FEATURE_PREFIX}i1.featurevector")

        counters = self.run_script(pin)

        self.assertEqual(
            (counters.images_ok, counters.images_skipped, counters.images_failed), (0, 1, 0)
        )
        self.assertEqual(self.errors.rows, [])
        self.assertEqual(counters.exit_code(), 0)

    def test_a_wrong_manifest_sha_stops_before_any_claim_or_image_get(self):
        pin = _publish(self.fake, {"i1": ("s1/i1.jpg", "s1")}, [("i1", 1, 1)])
        wrong = Pin(name=pin.name, version=pin.version, manifest_sha256="e" * 64)

        with self.assertRaises(PinMismatch):
            self.run_script(wrong)

        self.assertEqual(self.fake.keys_called("put"), [])
        self.assertEqual(self.image_gets(), [])


class UnsafeImageIdTest(RunHarness):
    def test_an_image_id_that_is_not_a_plain_file_name_stops_before_any_claim(self):
        for image_id in ("../evil", "a/b", "..", ".hidden", ""):
            with self.subTest(image_id=image_id):
                self.fake = FakeS3()
                self.cache = Path(tempfile.mkdtemp())
                self.addCleanup(shutil.rmtree, self.cache, ignore_errors=True)
                pin = _publish(
                    self.fake,
                    {"ok1": ("s1/ok1.jpg", "s1"), image_id: ("s1/bad.jpg", "s1")},
                    [("ok1", 1, 1), (image_id, 2, 2)],
                )

                with self.assertRaises(ValueError):
                    self.run_script(pin)

                self.assertEqual(self.fake.keys_called("put"), [])
                self.assertEqual(self.image_gets(), [])


class DryRunTest(RunHarness):
    def test_a_dry_run_claims_reads_and_writes_nothing_and_counts_the_planned_images(self):
        pin = _publish(
            self.fake,
            {"i1": ("s1/i1.jpg", "s1"), "i2": ("s1/i2.jpg", "s1")},
            [("i1", 1, 1), ("i2", 2, 2)],
        )

        counters = self.run_script(pin, dry_run=True)

        self.assertEqual(self.fake.keys_called("put"), [])
        self.assertEqual(self.image_gets(), [])
        self.assertEqual(self.extract_msgs, [])
        self.assertEqual((counters.images_ok, counters.images_failed), (2, 0))


class ParseShardTest(unittest.TestCase):
    def test_a_valid_shard_parses_to_index_and_count(self):
        self.assertEqual(bfs.parse_shard("0/4"), (0, 4))
        self.assertEqual(bfs.parse_shard("3/4"), (3, 4))

    def test_an_out_of_range_or_malformed_shard_is_refused(self):
        for text in ("4/4", "-1/4", "0/0", "x", "1"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                bfs.parse_shard(text)

    def test_every_shard_the_launcher_hands_out_parses(self):
        from mermaid_classifier.sagemaker.launcher_config import ShardConfig

        shard = ShardConfig(workers=4, per_worker_arg="--shard")

        parsed = [bfs.parse_shard(item) for [item] in lp.worker_items(shard, Path("unused"))]

        self.assertEqual(parsed, [(0, 4), (1, 4), (2, 4), (3, 4)])


class ResolveDeviceTest(unittest.TestCase):
    def _mock_torch(self, mps: bool, cuda: bool):
        return mock.patch.dict(
            sys.modules,
            {
                "torch": mock.MagicMock(
                    cuda=mock.MagicMock(is_available=mock.MagicMock(return_value=cuda)),
                    backends=mock.MagicMock(
                        mps=mock.MagicMock(is_available=mock.MagicMock(return_value=mps)),
                    ),
                ),
            },
        )

    def test_auto_prefers_mps(self):
        with self._mock_torch(mps=True, cuda=True):
            self.assertEqual(bfs.resolve_device("auto"), "mps")

    def test_auto_falls_back_to_cuda(self):
        with self._mock_torch(mps=False, cuda=True):
            self.assertEqual(bfs.resolve_device("auto"), "cuda")

    def test_auto_falls_back_to_cpu(self):
        with self._mock_torch(mps=False, cuda=False):
            self.assertEqual(bfs.resolve_device("auto"), "cpu")

    def test_explicit_mps_when_unavailable_raises(self):
        with self._mock_torch(mps=False, cuda=False), self.assertRaises(RuntimeError):
            bfs.resolve_device("mps")

    def test_an_available_explicit_device_is_returned_unchanged(self):
        """The auto branches all name a device of their own, so only this one
        proves --device cuda runs on cuda rather than being downgraded."""
        with self._mock_torch(mps=False, cuda=True):
            self.assertEqual(bfs.resolve_device("cuda"), "cuda")


class SelectImagesTest(unittest.TestCase):
    def _images(self, groups: dict[str, str]) -> pa.Table:
        return pa.table({"image_id": list(groups), "group": list(groups.values())})

    def test_groups_keep_only_those_groups_images(self):
        table = self._images({"a": "g1", "b": "g2", "c": "g3", "d": "g1"})

        selected = bfs.select_images(table, groups={"g1", "g3"}, shard=None)

        self.assertEqual(selected.column("image_id").to_pylist(), ["a", "c", "d"])

    def test_shards_are_disjoint_and_cover_the_selection(self):
        ids = [f"img{i}" for i in range(12)]
        table = self._images(dict.fromkeys(ids, "g"))

        shards = [
            bfs.select_images(table, groups=None, shard=(i, 3)).column("image_id").to_pylist()
            for i in range(3)
        ]

        for i in range(3):
            for j in range(i + 1, 3):
                with self.subTest(i=i, j=j):
                    self.assertEqual(set(shards[i]) & set(shards[j]), set())
        self.assertEqual(sorted(sum(shards, [])), sorted(ids))
        self.assertTrue(all(shards), "12 ids over 3 shards leave none empty")
        # Worker processes compute shards independently, so the assignment is pinned
        # to sha256(image_id)[:8] big-endian mod n, not to anything process-seeded.
        self.assertEqual(
            [bfs.shard_of(f"img{i}", 3) for i in range(12)], [1, 1, 2, 1, 2, 2, 0, 2, 1, 0, 1, 0]
        )

    def test_a_group_selection_that_names_nothing_present_raises(self):
        table = self._images({"a": "g1", "b": "g2"})

        for groups in ("g9", "g1,g9", "", " , "):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                args = argparse.Namespace(groups=groups, groups_csv=None)
                bfs.select_images(table, bfs.groups_from_args(args), shard=None)


if __name__ == "__main__":
    unittest.main()
