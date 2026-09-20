"""Tests for the extractor-provenance value type and its derivation.

The point of ExtractorSpec is that the identity of the image -> feature-vector
transform is read off the thing that performs it, and re-checked wherever that
transform is performed again. These tests exercise both halves against the
real pyspacer extractor rather than a stub.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from spacer.data_classes import DataLocation
from spacer.extractors import EfficientNetExtractor
from support.calibrated_model import make_calibrated_model
from support.extractor import make_extractor_spec

from mermaid_classifier.common.s3_utils import sha256_of_uri
from mermaid_classifier.pyspacer.extraction import extractor_spec_from_weights
from mermaid_classifier.pyspacer.inference import (
    MANIFEST_KEY,
    ExtractorMismatchError,
    ExtractorSpec,
    ManifestError,
    export_artifact,
)

STOCK_CLASS_PATH = "spacer.extractors.efficientnet.EfficientNetExtractor"


def _stock_extractor() -> EfficientNetExtractor:
    """Geometry is a class property, so this costs no weight load."""
    return EfficientNetExtractor(
        data_locations={"weights": DataLocation("filesystem", "unused.pt")}
    )


class SerializationTest(unittest.TestCase):
    def test_round_trips(self):
        spec = make_extractor_spec(1280)
        self.assertEqual(ExtractorSpec.from_dict(spec.to_dict()), spec)

    def test_rejects_missing_key(self):
        data = make_extractor_spec().to_dict()
        del data["weights_sha256"]
        with self.assertRaisesRegex(ValueError, "weights_sha256"):
            ExtractorSpec.from_dict(data)

    def test_rejects_unknown_key(self):
        data = make_extractor_spec().to_dict()
        data["normalisation"] = "imagenet"
        with self.assertRaisesRegex(ValueError, "normalisation"):
            ExtractorSpec.from_dict(data)

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError):
            ExtractorSpec.from_dict(["not", "an", "object"])


class IdentityTest(unittest.TestCase):
    def test_same_bytes_at_a_different_uri_is_the_same_extractor(self):
        a = make_extractor_spec(weights_uri="s3://one/efficientnet.pt")
        b = make_extractor_spec(weights_uri="s3://two/efficientnet_weights.pt")
        self.assertNotEqual(a, b)
        self.assertEqual(a.identity(), b.identity())

    def test_different_bytes_is_a_different_extractor(self):
        a = make_extractor_spec()
        b = make_extractor_spec(weights_sha256="f" * 64)
        self.assertNotEqual(a.identity(), b.identity())


class FromExtractorTest(unittest.TestCase):
    def test_reads_geometry_off_the_extractor(self):
        spec = ExtractorSpec.from_extractor(
            _stock_extractor(), weights_uri="s3://b/w.pt", weights_sha256="a" * 64
        )
        self.assertEqual(spec.class_path, STOCK_CLASS_PATH)
        self.assertEqual(spec.crop_size, EfficientNetExtractor.CROP_SIZE)
        self.assertEqual(spec.feature_dim, 1280)

    def test_a_throughput_subclass_records_the_stock_class(self):
        # build_feature_bucket.py wraps the extractor to pick a device and a
        # batch size. That is the same feature function, so the spec has to
        # name the stock class or serving could never match it.
        class _DeviceCaching(EfficientNetExtractor):
            pass

        spec = ExtractorSpec.from_extractor(
            _DeviceCaching(data_locations={"weights": DataLocation("filesystem", "unused.pt")}),
            weights_uri="s3://b/w.pt",
            weights_sha256="a" * 64,
        )
        self.assertEqual(spec.class_path, STOCK_CLASS_PATH)


class CheckExtractorTest(unittest.TestCase):
    def test_accepts_the_extractor_it_describes(self):
        extractor = _stock_extractor()
        spec = ExtractorSpec.from_extractor(
            extractor, weights_uri="s3://b/w.pt", weights_sha256="a" * 64
        )
        spec.check_extractor(extractor)  # no raise

    def test_rejects_a_different_crop(self):
        spec = make_extractor_spec(1280, crop_size=64)
        with self.assertRaisesRegex(ExtractorMismatchError, "crop_size"):
            spec.check_extractor(_stock_extractor())

    def test_rejects_a_different_feature_width(self):
        spec = make_extractor_spec(1280)
        with self.assertRaisesRegex(ExtractorMismatchError, "feature_dim"):
            spec.check_feature_dim(512)


class FromManifestTest(unittest.TestCase):
    def test_reads_the_block(self):
        spec = make_extractor_spec(1280)
        self.assertEqual(ExtractorSpec.from_manifest({MANIFEST_KEY: spec.to_dict()}), spec)

    def test_refuses_a_manifest_without_the_block(self):
        # A pre-contract artifact cannot say what produced its features, and
        # guessing is the failure this whole contract exists to remove.
        with self.assertRaisesRegex(ManifestError, MANIFEST_KEY):
            ExtractorSpec.from_manifest({"schema_version": 1})

    def test_refuses_a_malformed_block(self):
        with self.assertRaises(ManifestError):
            ExtractorSpec.from_manifest({MANIFEST_KEY: {"crop_size": 224}})


class Sha256OfUriTest(unittest.TestCase):
    def test_matches_hashlib_over_the_same_bytes(self):
        payload = b"efficientnet weights stand-in" * 1000
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "weights.pt"
            path.write_bytes(payload)
            self.assertEqual(sha256_of_uri(str(path)), hashlib.sha256(payload).hexdigest())

    def test_spans_more_than_one_read_chunk(self):
        payload = b"x" * (3 * (1 << 20) + 17)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "weights.pt"
            path.write_bytes(payload)
            self.assertEqual(sha256_of_uri(str(path)), hashlib.sha256(payload).hexdigest())


class ExtractorSpecFromWeightsTest(unittest.TestCase):
    def test_derives_geometry_and_hashes_the_real_bytes(self):
        payload = b"weights bytes"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "efficientnet.pt"
            path.write_bytes(payload)
            spec = extractor_spec_from_weights(str(path))

        self.assertEqual(spec.class_path, STOCK_CLASS_PATH)
        self.assertEqual(spec.crop_size, 224)
        self.assertEqual(spec.feature_dim, 1280)
        self.assertEqual(spec.weights_sha256, hashlib.sha256(payload).hexdigest())


class ExportRequiresAMatchingExtractorTest(unittest.TestCase):
    def test_refuses_a_spec_that_does_not_fit_the_head(self):
        # The head takes whatever the extractor emits; recording an extractor
        # of a different width would describe a pairing that cannot run.
        model, X = make_calibrated_model()
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ExtractorMismatchError):
            export_artifact(model, d, X, extractor=make_extractor_spec(X.shape[1] + 1))

    def test_manifest_carries_the_spec_verbatim(self):
        model, X = make_calibrated_model()
        spec = make_extractor_spec(X.shape[1])
        with tempfile.TemporaryDirectory() as d:
            export_artifact(model, d, X, extractor=spec)
            manifest = json.loads((Path(d) / "model.json").read_text())
        self.assertEqual(ExtractorSpec.from_manifest(manifest), spec)
        # schema_version is untouched: the block is additive, so mermaid-api's
        # Classifier.register keeps reading the manifest unchanged.
        self.assertEqual(manifest["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
