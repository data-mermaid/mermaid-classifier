"""Published segdata-spec versions in a hand-written S3 fake, for dataset loading tests.

The manifests, tables, feature-set manifest and index are the real segdata-spec
models and schemas, so sync_tables / sync_feature_set / open_* run for real
against them. Every object read is logged, so a test can assert which feature
files were fetched and that nothing listed a prefix.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from segdata_spec import DATASET_BUCKET
from segdata_spec.manifest import (
    EtlInfo,
    FeatureSetManifest,
    FeatureSetPin,
    FeatureSetStats,
    FileEntry,
    ImageStore,
    Manifest,
    Pin,
    SourceInfo,
    Stats,
    TaxonomyManifest,
    dump_manifest,
    manifest_sha256,
)
from segdata_spec.publish import table_bytes
from segdata_spec.schema import (
    BENTHIC_ATTRIBUTES_SCHEMA,
    FEATURE_INDEX_SCHEMA,
    GROWTH_FORMS_SCHEMA,
    IMAGES_SCHEMA,
    LABEL_COUNTS_SCHEMA,
    LABELS_SCHEMA,
    POINTS_SCHEMA,
    conform,
)

BUCKET = DATASET_BUCKET
VERSION = "20261124-cfc389d"
FAMILY = "efficientnet-v1"
CREATED = datetime(2026, 11, 24, tzinfo=UTC)
ETL = EtlInfo(
    package="segdata-etl", command="test", git_sha="cfc389d", git_dirty=False, uv_lock_sha256="0"
)
EXTRACTOR = {
    "class_path": "spacer.extractors.efficientnet.EfficientNetExtractor",
    "crop_size": 224,
    "feature_dim": 1280,
    "weights_uri": "s3://test-bucket/efficientnet.pt",
    "weights_sha256": "0" * 64,
}


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


class FakeS3:
    """The boto3 S3 client surface segdata-spec reads with.

    ``gets`` logs every requested key in order; ``listings`` logs every
    listing call, which training must never make.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.gets: list[str] = []
        self.listings: list[str] = []

    def seed(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    def get_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        self.gets.append(Key)
        if Bucket != BUCKET or Key not in self.objects:
            raise _client_error("NoSuchKey", 404, "GetObject")
        if_match = kwargs.get("IfMatch")
        if if_match is not None and if_match.strip('"') != _md5(self.objects[Key]):
            raise _client_error("PreconditionFailed", 412, "GetObject")
        return {"Body": _Body(self.objects[Key])}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.listings.append(str(kwargs.get("Prefix")))
        return {}

    def get_paginator(self, name: str) -> Any:
        self.listings.append(name)
        raise AssertionError("training must not list S3")

    def feature_gets(self) -> list[str]:
        return [key for key in self.gets if key.endswith(".featurevector")]


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _seed_tables(fake: FakeS3, prefix: str, tables: Mapping[str, pa.Table]) -> dict[str, FileEntry]:
    files = {}
    for name, table in tables.items():
        data = table_bytes(table)
        fake.seed(prefix + name, data)
        files[name] = FileEntry(
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), rows=table.num_rows
        )
    return files


@dataclass(frozen=True)
class Image:
    image_id: str
    group: str
    attrs: dict[str, Any] | None = None


def label_row(
    label_id: int, ba_id: str | None = None, gf_name: str | None = None
) -> dict[str, Any]:
    """A labels.parquet row; ``ba_id`` None is an unmapped (CoralNet) label."""
    return {
        "label_id": label_id,
        "name": f"label {label_id}",
        "mermaid_benthic_attribute_id": ba_id,
        "mermaid_benthic_attribute": None if ba_id is None else f"BA {ba_id}",
        "mermaid_growth_form": gf_name,
        "mapping_source": "identity",
    }


def publish_dataset(
    fake: FakeS3,
    name: str,
    images: Sequence[Image],
    points: Sequence[tuple[str, int, int, int]],
    labels: Sequence[dict[str, Any]],
    source_type: str = "coralnet-scrape",
) -> Pin:
    """Seed a points dataset version; ``points`` rows are (image_id, row, col, label_id)."""
    n = len(images)
    images_table = conform(
        pa.table(
            {
                "image_id": [i.image_id for i in images],
                "key": [f"{i.group}/{i.image_id}.jpg" for i in images],
                "etag": ["e"] * n,
                "size": [1] * n,
                "crc64nvme": pa.nulls(n, pa.string()),
                "media_type": ["image/jpeg"] * n,
                "width": [100] * n,
                "height": [100] * n,
                "group": [i.group for i in images],
                "mask_key": pa.nulls(n, pa.string()),
                "mask_etag": pa.nulls(n, pa.string()),
                "mask_size": pa.nulls(n, pa.int64()),
                "mask_crc64nvme": pa.nulls(n, pa.string()),
                "source_fingerprint": ["fp"] * n,
                "source_name": pa.nulls(n, pa.string()),
                "attrs": [None if i.attrs is None else json.dumps(i.attrs) for i in images],
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
                "label_id": [p[3] for p in points],
                "annotator": ["a"] * len(points),
            }
        ),
        POINTS_SCHEMA,
    )
    labels_table = conform(pa.Table.from_pylist(list(labels)), LABELS_SCHEMA)
    counts = conform(
        pa.table({"image_id": [], "label_id": [], "n": []}).cast(LABEL_COUNTS_SCHEMA),
        LABEL_COUNTS_SCHEMA,
    )
    prefix = f"datasets/{name}/{VERSION}/"
    files = _seed_tables(
        fake,
        prefix,
        {
            "images.parquet": images_table,
            "points.parquet": points_table,
            "labels.parquet": labels_table,
            "stats/label_counts.parquet": counts,
        },
    )
    manifest = Manifest(
        schema_version=1,
        dataset=name,
        version=VERSION,
        kind="points",
        created_at=CREATED,
        image_store=ImageStore(bucket=BUCKET, prefix=f"images/{name}/", owner="segdata"),
        etl=ETL,
        source=SourceInfo(type=source_type, uri="s3://src/", revision="r1"),
        files=files,
        stats=Stats(
            n_images=n,
            n_points=len(points),
            n_groups=len({i.group for i in images}),
            image_bytes=0,
            dropped={},
        ),
    )
    data = dump_manifest(manifest)
    fake.seed(prefix + "manifest.json", data)
    return Pin(name=name, version=VERSION, manifest_sha256=manifest_sha256(data))


def feature_key(pin: Pin, image_id: str) -> str:
    return f"features/{pin.name}/{FAMILY}/{image_id}.featurevector"


def publish_feature_set(
    fake: FakeS3,
    pin: Pin,
    image_ids: Iterable[str],
    extractor: Mapping[str, Any] = EXTRACTOR,
    n_missing: int = 0,
) -> FeatureSetPin:
    """Seed a feature file per image (its bytes are ``FV:<image_id>``) and the set's index."""
    rows = []
    for image_id in image_ids:
        data = f"FV:{image_id}".encode()
        key = feature_key(pin, image_id)
        fake.seed(key, data)
        rows.append(
            {
                "image_id": image_id,
                "key": key,
                "etag": _md5(data),
                "size": len(data),
                "crc64nvme": None,
                "n_points": 1,
            }
        )
    index = conform(pa.Table.from_pylist(rows, schema=FEATURE_INDEX_SCHEMA), FEATURE_INDEX_SCHEMA)
    prefix = f"datasets/{pin.name}/{pin.version}/features/{FAMILY}/"
    files = _seed_tables(fake, prefix, {"index.parquet": index})
    manifest = FeatureSetManifest(
        schema_version=1,
        dataset=pin,
        family=FAMILY,
        extractor=dict(extractor),
        files=files,
        stats=FeatureSetStats(n_images=len(rows), n_missing=n_missing),
        etl=ETL,
        created_at=CREATED,
    )
    data = dump_manifest(manifest)
    fake.seed(prefix + "manifest.json", data)
    return FeatureSetPin(dataset=pin, family=FAMILY, manifest_sha256=manifest_sha256(data))


def publish_taxonomy(fake: FakeS3, growth_forms: Mapping[str, str]) -> Pin:
    """Seed a mermaid-benthic taxonomy; ``growth_forms`` maps id to name."""
    prefix = f"taxonomy/mermaid-benthic/{VERSION}/"
    files = _seed_tables(
        fake,
        prefix,
        {
            "benthic_attributes.parquet": conform(
                pa.table({"id": ["ba-x"], "name": ["X"], "parent_id": [None]}),
                BENTHIC_ATTRIBUTES_SCHEMA,
            ),
            "growth_forms.parquet": conform(
                pa.table({"id": list(growth_forms), "name": list(growth_forms.values())}),
                GROWTH_FORMS_SCHEMA,
            ),
        },
    )
    manifest = TaxonomyManifest(
        schema_version=1,
        name="mermaid-benthic",
        version=VERSION,
        created_at=CREATED,
        etl=ETL,
        source=SourceInfo(type="test", uri="s3://src/", revision="r1"),
        files=files,
    )
    data = dump_manifest(manifest)
    fake.seed(prefix + "manifest.json", data)
    return Pin(name="mermaid-benthic", version=VERSION, manifest_sha256=manifest_sha256(data))


def write_project_lookup(path: str, image_projects: Sequence[tuple[str, str]]) -> str:
    """Write a MERMAID image -> project lookup parquet from (image_id, project_id) rows."""
    pq.write_table(
        pa.table(
            {
                "image_id": [image_id for image_id, _ in image_projects],
                "project_id": [project for _, project in image_projects],
                "project_name": [f"name-{project}" for _, project in image_projects],
                "site_id": [f"site-{project}" for _, project in image_projects],
            }
        ),
        path,
    )
    return path


class _NameLibrary:
    def id_to_name(self, value: str | None) -> str | None:
        return value


def load_training_dataset(
    fake: FakeS3,
    options: Any,
    coralnet_mapping: Sequence[Mapping[str, Any]],
    extractor: Mapping[str, Any] = EXTRACTOR,
    filtered_only: bool = False,
    **settings: Any,
) -> Any:
    """Run a real TrainingDataset.__init__ against ``fake``.

    ``filtered_only`` runs TrainingDataset.filtered_annotations_only instead.

    ``coralnet_mapping`` stands in for the MERMAID API's CoralNet label mapping,
    ``extractor`` for the spec read off ``options.feature_extractor_weights``, and
    ``settings`` overrides the pydantic settings for the duration of the load.
    """
    from support.settings import override_settings

    from mermaid_classifier.pyspacer.dataset import TrainingDataset
    from mermaid_classifier.pyspacer.inference import ExtractorSpec

    with (
        override_settings(**settings),
        mock.patch(
            "mermaid_classifier.common.benthic_attributes.CoralNetMermaidMapping._download_mapping",
            return_value=list(coralnet_mapping),
        ),
        mock.patch("mermaid_classifier.pyspacer.dataset.make_s3_client", return_value=fake),
        mock.patch(
            "mermaid_classifier.pyspacer.dataset.extractor_spec_from_weights",
            return_value=ExtractorSpec.from_dict(dict(extractor)),
        ),
        mock.patch(
            "mermaid_classifier.pyspacer.dataset.get_benthic_attribute_library",
            return_value=_NameLibrary(),
        ),
        mock.patch(
            "mermaid_classifier.pyspacer.dataset.get_growth_form_library",
            return_value=_NameLibrary(),
        ),
    ):
        if filtered_only:
            return TrainingDataset.filtered_annotations_only(options)
        return TrainingDataset(options)
