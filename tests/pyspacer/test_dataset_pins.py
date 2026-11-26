"""TrainingDataset reads points and feature vectors from pinned dataset versions.

The versions are real segdata-spec manifests and tables in an S3 fake
(segdata_fixtures), so sync and open run for real. The MERMAID API (label
mapping, BA/GF name libraries) and the weights hash are the only stubs.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from botocore.exceptions import ConnectionClosedError
from segdata_spec.errors import ChecksumMismatch
from segdata_spec.manifest import FeatureSetPin, Pin
from segdata_spec.sync import sync_feature_set
from support.dataset import make_dataset
from urllib3.exceptions import ProtocolError

from mermaid_classifier.pyspacer.dataset import TrainingDataset, resolve_groups
from mermaid_classifier.pyspacer.options import DatasetOptions, PinnedDataset
from pyspacer.segdata_fixtures import (
    BUCKET,
    EXTRACTOR,
    FakeS3,
    Image,
    feature_key,
    label_row,
    load_training_dataset,
    publish_dataset,
    publish_feature_set,
    publish_taxonomy,
)

# CoralNet label 100 maps to a BA+GF, 200 to a BA alone; 300 is unmapped.
CORALNET_MAPPING = [
    {
        "provider_id": "100",
        "provider_label": "Acropora branching",
        "benthic_attribute_id": "ba-1",
        "benthic_attribute_name": "Acropora",
        "growth_form_id": "gf-1",
        "growth_form_name": "Branching",
    },
    {
        "provider_id": "200",
        "provider_label": "Sand",
        "benthic_attribute_id": "ba-2",
        "benthic_attribute_name": "Sand",
        "growth_form_id": None,
        "growth_form_name": None,
    },
]
CORALNET_LABELS = [label_row(100), label_row(200), label_row(300)]
CORALNET_IMAGES = [
    Image("i1", "s7", {"source_id": 7}),
    Image("i2", "s9", {"source_id": 9}),
]


def _mapping_patch():
    return mock.patch(
        "mermaid_classifier.common.benthic_attributes.CoralNetMermaidMapping._download_mapping",
        return_value=CORALNET_MAPPING,
    )


def _source(pin: Pin, features: FeatureSetPin | None = None, **kwargs: Any) -> PinnedDataset:
    features = features or FeatureSetPin(dataset=pin, family="efficientnet-v1", manifest_sha256="0")
    return PinnedDataset(pin=pin, features=features, **kwargs)


def _annotation_rows(dataset: TrainingDataset) -> list[tuple]:
    return dataset.duck_conn.execute(
        "SELECT image_id, row, col, benthic_attribute_id, growth_form_id, site, project_id"
        " FROM annotations ORDER BY image_id, row"
    ).fetchall()


class ReadDatasetsTest(unittest.TestCase):
    """read_datasets turns each pinned version's points into annotation rows."""

    def setUp(self):
        self.fake = FakeS3()
        self.dataset = make_dataset(self)

    def _read(self, *sources: PinnedDataset, taxonomy: Pin | None = None) -> None:
        self.dataset.options = DatasetOptions(datasets=list(sources), taxonomy=taxonomy)
        self.dataset.groups = resolve_groups(self.dataset.options)
        with _mapping_patch():
            self.dataset.read_datasets(self.fake)

    def test_coralnet_points_map_through_the_coralnet_mapping(self):
        pin = publish_dataset(
            self.fake,
            "coralnet-raw",
            CORALNET_IMAGES,
            [("i1", 10, 20, 100), ("i1", 30, 40, 200), ("i2", 50, 60, 100), ("i2", 70, 80, 300)],
            CORALNET_LABELS,
        )
        self._read(_source(pin, labels="coralnet"))

        self.assertEqual(
            _annotation_rows(self.dataset),
            [
                ("i1", 10, 20, "ba-1", "gf-1", "coralnet", "7"),
                ("i1", 30, 40, "ba-2", "", "coralnet", "7"),
                ("i2", 50, 60, "ba-1", "gf-1", "coralnet", "9"),
            ],
        )
        self.assertEqual(self.dataset.artifacts.unmapped_labels["label_id"].tolist(), ["300"])
        self.assertEqual(self.dataset.coralnet_source_ids, ["7", "9"])

    def test_mermaid_points_take_the_ba_and_the_taxonomy_growth_form_id(self):
        pin = publish_dataset(
            self.fake,
            "mermaid",
            [Image("m1", "mermaid")],
            [("m1", 1, 2, 1), ("m1", 3, 4, 2), ("m1", 5, 6, 3)],
            [label_row(1, "ba-x", "Massive"), label_row(2, "ba-y", ""), label_row(3, "ba-z")],
        )
        taxonomy = publish_taxonomy(self.fake, {"gf-massive": "Massive", "gf-other": "Other"})
        self._read(_source(pin, labels="mermaid"), taxonomy=taxonomy)

        self.assertEqual(
            _annotation_rows(self.dataset),
            [
                ("m1", 1, 2, "ba-x", "gf-massive", "mermaid", "all"),
                ("m1", 3, 4, "ba-y", "", "mermaid", "all"),
                ("m1", 5, 6, "ba-z", "", "mermaid", "all"),
            ],
        )

    def test_a_growth_form_name_absent_from_the_taxonomy_is_refused_by_name(self):
        pin = publish_dataset(
            self.fake,
            "mermaid",
            [Image("m1", "mermaid")],
            [("m1", 1, 2, 1)],
            [label_row(1, "ba-x", "Blobby")],
        )
        taxonomy = publish_taxonomy(self.fake, {"gf-massive": "Massive"})
        with self.assertRaisesRegex(ValueError, "Blobby"):
            self._read(_source(pin, labels="mermaid"), taxonomy=taxonomy)

    def test_a_labels_kind_that_contradicts_labels_parquet_is_refused(self):
        mermaid_labels = [label_row(1, "ba-x", "")]
        cases = {
            "coralnet on MERMAID ids": ("coralnet", mermaid_labels),
            "mermaid without MERMAID ids": ("mermaid", CORALNET_LABELS),
        }
        taxonomy = publish_taxonomy(self.fake, {"gf-massive": "Massive"})
        for case, (kind, labels) in cases.items():
            with self.subTest(case=case):
                fake_points = [("i1", 1, 2, labels[0]["label_id"])]
                pin = publish_dataset(self.fake, f"d-{kind}", CORALNET_IMAGES, fake_points, labels)
                with self.assertRaisesRegex(ValueError, f"d-{kind}"):
                    self._read(_source(pin, labels=kind), taxonomy=taxonomy)

    def test_coralnet_labels_on_a_dataset_not_scraped_from_coralnet_are_refused(self):
        pin = publish_dataset(
            self.fake,
            "other-points",
            CORALNET_IMAGES,
            [("i1", 1, 2, 100)],
            CORALNET_LABELS,
            source_type="mermaid-confirmed-annotations",
        )
        with self.assertRaisesRegex(ValueError, "other-points.*mermaid-confirmed-annotations"):
            self._read(_source(pin, labels="coralnet"))


class _FullLoad(unittest.TestCase):
    """Runs a real TrainingDataset.__init__ against the S3 fake."""

    def setUp(self):
        self.fake = FakeS3()
        self.cache = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.cache, ignore_errors=True)

    def publish_coralnet(self, images: list[Image]) -> Pin:
        # Ten points per image, alternating two mapped labels, so the split has
        # enough of each class for train, ref and val.
        points = [
            (image.image_id, n, n, 100 if n % 2 else 200) for image in images for n in range(10)
        ]
        return publish_dataset(self.fake, "coralnet-raw", images, points, CORALNET_LABELS)

    def load(self, *sources: PinnedDataset, **settings: Any) -> TrainingDataset:
        options = DatasetOptions(
            datasets=list(sources),
            feature_extractor_weights="s3://test-bucket/efficientnet.pt",
        )
        overrides = {
            "feature_cache_dir": str(self.cache),
            "download_max_workers": 2,
            "training_inputs_percent_missing_allowed": 0,
            **settings,
        }
        dataset = load_training_dataset(self.fake, options, CORALNET_MAPPING, **overrides)
        self.addCleanup(dataset.cleanup)
        return dataset


THREE_IMAGES = [Image("i1", "s7"), Image("i2", "s9"), Image("i3", "s9")]


class GroupSelectionTest(_FullLoad):
    def test_groups_csv_keeps_only_its_groups_and_syncs_exactly_their_features(self):
        pin = self.publish_coralnet(THREE_IMAGES)
        features = publish_feature_set(self.fake, pin, ["i1", "i2", "i3"])
        groups_csv = self.cache / "groups.csv"
        groups_csv.write_text("s9\n\n")

        dataset = self.load(_source(pin, features, labels="coralnet", groups_csv=str(groups_csv)))

        image_ids = dataset.duck_conn.execute(
            "SELECT DISTINCT image_id FROM annotations ORDER BY image_id"
        ).fetchall()
        self.assertEqual(image_ids, [("i2",), ("i3",)])
        self.assertEqual(
            sorted(self.fake.feature_gets()), [feature_key(pin, "i2"), feature_key(pin, "i3")]
        )


class GroupsCsvRefusedTest(_FullLoad):
    def _load_with_groups(self, text: str) -> None:
        pin = self.publish_coralnet(THREE_IMAGES)
        features = publish_feature_set(self.fake, pin, ["i1", "i2", "i3"])
        groups_csv = self.cache / "groups.csv"
        groups_csv.write_text(text)
        self.load(_source(pin, features, labels="coralnet", groups_csv=str(groups_csv)))

    def test_a_groups_csv_of_only_blank_lines_is_refused(self):
        with self.assertRaisesRegex(ValueError, "names no dataset group"):
            self._load_with_groups("\n  \n\n")

    def test_a_group_absent_from_the_version_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            self._load_with_groups("s9\ns99\n")
        self.assertIn("s99", str(caught.exception))
        self.assertNotIn("s9'", str(caught.exception))


class FeaturePathsTest(_FullLoad):
    def test_labels_point_at_the_cached_feature_files_and_nothing_lists_s3(self):
        pin = self.publish_coralnet(THREE_IMAGES)
        features = publish_feature_set(self.fake, pin, ["i1", "i2", "i3"])

        dataset = self.load(_source(pin, features, labels="coralnet"))

        locations = [
            *dataset.labels.train.keys(),
            *dataset.labels.ref.keys(),
            *dataset.labels.val.keys(),
        ]
        cached = {
            str(self.cache / BUCKET / feature_key(pin, image_id)) for image_id in ("i1", "i2", "i3")
        }
        self.assertEqual({loc.key for loc in locations}, cached)
        self.assertEqual({Path(key).read_bytes() for key in cached}, {b"FV:i1", b"FV:i2", b"FV:i3"})
        self.assertEqual(self.fake.listings, [])


class MissingFeaturesTest(_FullLoad):
    """One of three selected images (33%) is absent from the feature index."""

    def _publish(self) -> PinnedDataset:
        pin = self.publish_coralnet(THREE_IMAGES)
        features = publish_feature_set(self.fake, pin, ["i1", "i3"], n_missing=1)
        return _source(pin, features, labels="coralnet")

    def test_under_the_allowance_the_image_is_dropped_with_a_warning(self):
        source = self._publish()
        with self.assertLogs("train", level="WARNING") as logs:
            dataset = self.load(source, training_inputs_percent_missing_allowed=34)

        image_ids = dataset.duck_conn.execute(
            "SELECT DISTINCT image_id FROM annotations ORDER BY image_id"
        ).fetchall()
        self.assertEqual(image_ids, [("i1",), ("i3",)])
        self.assertIn("coralnet-raw/i2", "\n".join(logs.output))

    def test_over_the_allowance_the_run_stops(self):
        source = self._publish()
        with self.assertRaisesRegex(RuntimeError, "coralnet-raw/i2"):
            self.load(source, training_inputs_percent_missing_allowed=33)
        self.assertEqual(self.fake.feature_gets(), [])

    def test_the_project_lookup_tolerance_does_not_loosen_it(self):
        source = self._publish()
        with self.assertRaisesRegex(RuntimeError, "coralnet-raw/i2"):
            self.load(
                source,
                training_inputs_percent_missing_allowed=33,
                mermaid_project_lookup_percent_unmatched_allowed=100,
            )


class FeatureSyncRetryTest(_FullLoad):
    """sync_feature_vectors retries transient network errors and nothing else."""

    def _load_failing(self, errors: list[Exception]) -> tuple[Any, mock.MagicMock]:
        """Load with the feature download (image_ids selected) raising `errors` in order."""
        pin = self.publish_coralnet(THREE_IMAGES)
        features = publish_feature_set(self.fake, pin, ["i1", "i2", "i3"])
        pending = list(errors)

        def flaky(*args: Any, **kwargs: Any) -> Any:
            if kwargs["image_ids"] and pending:
                raise pending.pop(0)
            return sync_feature_set(*args, **kwargs)

        patch = mock.patch(
            "mermaid_classifier.pyspacer.dataset.sync_feature_set", side_effect=flaky
        )
        delay = mock.patch("mermaid_classifier.pyspacer.dataset.SYNC_RETRY_DELAY_SECONDS", 0)
        with patch as spy, delay:
            dataset = self.load(_source(pin, features, labels="coralnet"))
        return dataset, spy

    def _downloads(self, spy: mock.MagicMock) -> int:
        return sum(1 for call in spy.call_args_list if call.kwargs["image_ids"])

    def test_a_transient_error_is_retried_once_and_the_load_completes(self):
        error = ConnectionClosedError(endpoint_url="https://s3")
        with self.assertLogs("train", level="WARNING") as logs:
            dataset, spy = self._load_failing([error])

        self.assertEqual(self._downloads(spy), 2)
        self.assertIn("coralnet-raw", dataset._feature_sets)
        self.assertIn("coralnet-raw", "\n".join(logs.output))

    def test_a_raw_urllib3_error_is_retried(self):
        _, spy = self._load_failing([ProtocolError("connection broken")])
        self.assertEqual(self._downloads(spy), 2)

    def test_an_integrity_error_propagates_after_one_download_call(self):
        calls = []
        pin = self.publish_coralnet(THREE_IMAGES)
        features = publish_feature_set(self.fake, pin, ["i1", "i2", "i3"])

        def corrupt(*args: Any, **kwargs: Any) -> Any:
            if kwargs["image_ids"]:
                calls.append(1)
                raise ChecksumMismatch("bad bytes")
            return sync_feature_set(*args, **kwargs)

        with (
            mock.patch("mermaid_classifier.pyspacer.dataset.sync_feature_set", corrupt),
            self.assertRaises(ChecksumMismatch),
        ):
            self.load(_source(pin, features, labels="coralnet"))
        self.assertEqual(len(calls), 1)

    def test_a_persistent_transient_error_is_raised_after_three_attempts(self):
        raised = [ConnectionClosedError(endpoint_url=f"https://s3/{n}") for n in range(5)]
        errors = list(raised)
        pin = self.publish_coralnet(THREE_IMAGES)
        features = publish_feature_set(self.fake, pin, ["i1", "i2", "i3"])
        spy = mock.MagicMock(side_effect=lambda *a, **k: _raise_or_sync(errors, *a, **k))
        delay = mock.patch("mermaid_classifier.pyspacer.dataset.SYNC_RETRY_DELAY_SECONDS", 0)
        with (
            mock.patch("mermaid_classifier.pyspacer.dataset.sync_feature_set", spy),
            delay,
            self.assertRaises(ConnectionClosedError) as caught,
        ):
            self.load(_source(pin, features, labels="coralnet"))

        self.assertIs(caught.exception, raised[2])
        self.assertEqual(self._downloads(spy), 3)

    def test_the_sync_is_logged_with_the_dataset_name_and_image_count(self):
        pin = self.publish_coralnet(THREE_IMAGES)
        features = publish_feature_set(self.fake, pin, ["i1", "i2", "i3"])
        with self.assertLogs("train", level="INFO") as logs:
            self.load(_source(pin, features, labels="coralnet"))

        syncing = [
            m for m in logs.output if "INFO" in m and "coralnet-raw" in m and "3 images" in m
        ]
        self.assertEqual(len(syncing), 1)
        self.assertTrue(
            any("INFO" in m and "Synced features for coralnet-raw" in m for m in logs.output)
        )

    def test_a_retry_resumes_from_the_cache_instead_of_refetching_features(self):
        pin = self.publish_coralnet(THREE_IMAGES)
        features = publish_feature_set(self.fake, pin, ["i1", "i2", "i3"])
        real_get = self.fake.get_object
        fetched = []

        def drop_after_two_features(**kwargs: Any) -> Any:
            if kwargs["Key"].endswith(".featurevector"):
                if len(fetched) == 2:
                    fetched.append(None)
                    raise ConnectionClosedError(endpoint_url="https://s3")
                fetched.append(kwargs["Key"])
            return real_get(**kwargs)

        delay = mock.patch("mermaid_classifier.pyspacer.dataset.SYNC_RETRY_DELAY_SECONDS", 0)
        with (
            mock.patch.object(self.fake, "get_object", side_effect=drop_after_two_features),
            delay,
            self.assertLogs("train", level="WARNING"),
        ):
            self.load(_source(pin, features, labels="coralnet"), download_max_workers=1)

        keys = self.fake.feature_gets()
        self.assertEqual(len(keys), 3)
        self.assertEqual(len(set(keys)), 3)


def _raise_or_sync(errors: list[Exception], *args: Any, **kwargs: Any) -> Any:
    if kwargs["image_ids"]:
        raise errors.pop(0)
    return sync_feature_set(*args, **kwargs)


class LabelIntegrityTest(_FullLoad):
    def test_a_point_label_absent_from_labels_parquet_is_refused_by_dataset(self):
        taxonomy = publish_taxonomy(self.fake, {"gf-massive": "Massive"})
        cases = {
            "mermaid": ([label_row(1, "ba-x", "")], "mermaid"),
            "coralnet": ([label_row(100)], "coralnet"),
        }
        dataset = make_dataset(self)
        for kind, (labels, labels_kind) in cases.items():
            with self.subTest(kind=kind):
                pin = publish_dataset(
                    self.fake,
                    f"d-{kind}",
                    [Image("x1", "mermaid" if kind == "mermaid" else "s7")],
                    [("x1", 1, 2, labels[0]["label_id"]), ("x1", 3, 4, 99)],
                    labels,
                )
                dataset.options = DatasetOptions(
                    datasets=[_source(pin, labels=labels_kind)], taxonomy=taxonomy
                )
                dataset.groups = resolve_groups(dataset.options)
                with (
                    _mapping_patch(),
                    self.assertRaisesRegex(ValueError, rf"d-{kind}.*\[99\]"),
                ):
                    dataset.read_datasets(self.fake)


class RefusedBeforeFeatureDownloadTest(_FullLoad):
    def test_an_extractor_that_differs_from_the_declared_weights_is_refused(self):
        pin = self.publish_coralnet(THREE_IMAGES)
        other = {**EXTRACTOR, "weights_sha256": "f" * 64}
        features = publish_feature_set(self.fake, pin, ["i1", "i2", "i3"], extractor=other)

        with self.assertRaisesRegex(ValueError, "record a different extractor from"):
            self.load(_source(pin, features, labels="coralnet"))
        self.assertEqual(self.fake.feature_gets(), [])

    def test_only_the_set_that_differs_is_named(self):
        same = publish_dataset(
            self.fake, "same", [Image("a1", "s7")], [("a1", 1, 1, 100)], CORALNET_LABELS
        )
        other = publish_dataset(
            self.fake, "other", [Image("b1", "s7")], [("b1", 1, 1, 100)], CORALNET_LABELS
        )
        same_features = publish_feature_set(self.fake, same, ["a1"])
        other_features = publish_feature_set(
            self.fake, other, ["b1"], extractor={**EXTRACTOR, "weights_sha256": "f" * 64}
        )

        with self.assertRaises(ValueError) as caught:
            self.load(
                _source(same, same_features, labels="coralnet"),
                _source(other, other_features, labels="coralnet"),
            )
        self.assertIn("other feature set efficientnet-v1", str(caught.exception))
        self.assertNotIn("same feature set", str(caught.exception))
        self.assertEqual(self.fake.feature_gets(), [])

    def test_no_declared_extractor_weights_is_refused(self):
        pin = self.publish_coralnet(THREE_IMAGES)
        features = publish_feature_set(self.fake, pin, ["i1", "i2", "i3"])
        options = DatasetOptions(datasets=[_source(pin, features, labels="coralnet")])

        with self.assertRaisesRegex(ValueError, "declares no dataset.feature_extractor_weights"):
            load_training_dataset(
                self.fake, options, CORALNET_MAPPING, feature_cache_dir=str(self.cache)
            )
        self.assertEqual(self.fake.feature_gets(), [])


if __name__ == "__main__":
    unittest.main()
