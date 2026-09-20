"""Tests for resolving which extractor produced a run's feature vectors.

A head fitted across two feature spaces is wrong in a way no later gate
detects -- the artifact loads, the parity gate passes, and every score comes
from a distribution the head never saw. These tests pin the point where that
is refused.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from support.dataset import make_dataset
from support.extractor import make_extractor_spec
from support.settings import override_settings

from mermaid_classifier.pyspacer.extraction import SIDECAR_FILENAME

MERMAID_PREFIX = "s3://mermaid-bucket/mermaid/"
CORALNET_PREFIX = "s3://coralnet-bucket/"


class _FakeS3:
    """Stands in for the s3fs handle: resolve_extractor_spec calls only
    exists() and cat_file()."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    def exists(self, key: str) -> bool:
        return key in self._objects

    def cat_file(self, key: str) -> bytes:
        return self._objects[key]


def _sidecar(prefix: str, spec) -> tuple[str, bytes]:
    return prefix + SIDECAR_FILENAME, json.dumps(spec.to_dict()).encode()


class FeatureSourcePrefixesTest(unittest.TestCase):
    def test_lists_only_the_sources_in_play(self):
        dataset = make_dataset(self)
        with override_settings(
            mermaid_train_data_bucket="mermaid-bucket",
            coralnet_train_data_bucket="coralnet-bucket",
        ):
            dataset.options.include_mermaid = True
            dataset.options.coralnet_manifest_uri = None
            self.assertEqual(dataset.feature_source_prefixes(), [MERMAID_PREFIX])

            dataset.options.coralnet_manifest_uri = "s3://bucket/manifest.parquet"
            self.assertEqual(dataset.feature_source_prefixes(), [MERMAID_PREFIX, CORALNET_PREFIX])


class ResolveExtractorSpecTest(unittest.TestCase):
    def _dataset(self, objects, declared_weights=None):
        dataset = make_dataset(self)
        dataset.s3 = _FakeS3(objects)
        dataset.options.feature_extractor_weights = declared_weights
        return dataset

    def test_matching_sidecars_resolve_to_that_extractor(self):
        spec = make_extractor_spec(1280)
        objects = dict([_sidecar(MERMAID_PREFIX, spec), _sidecar(CORALNET_PREFIX, spec)])
        dataset = self._dataset(objects)
        self.assertEqual(dataset.resolve_extractor_spec([MERMAID_PREFIX, CORALNET_PREFIX]), spec)

    def test_disagreeing_sidecars_stop_the_run(self):
        mermaid = make_extractor_spec(1280)
        coralnet = make_extractor_spec(1280, weights_sha256="f" * 64)
        objects = dict([_sidecar(MERMAID_PREFIX, mermaid), _sidecar(CORALNET_PREFIX, coralnet)])
        dataset = self._dataset(objects)
        with self.assertRaises(ValueError) as ctx:
            dataset.resolve_extractor_spec([MERMAID_PREFIX, CORALNET_PREFIX])
        # Both sources named, so the operator can see which to re-extract.
        self.assertIn(MERMAID_PREFIX, str(ctx.exception))
        self.assertIn(CORALNET_PREFIX, str(ctx.exception))

    def test_a_prefix_without_a_sidecar_falls_back_to_the_declaration(self):
        # CoralNet's public bucket: we read it, we cannot write a sidecar to
        # it, so the run's own config is the only record.
        spec = make_extractor_spec(1280)
        dataset = self._dataset({}, declared_weights="s3://cfg/efficientnet.pt")
        with mock.patch(
            "mermaid_classifier.pyspacer.dataset.extractor_spec_from_weights",
            return_value=spec,
        ):
            self.assertEqual(dataset.resolve_extractor_spec([CORALNET_PREFIX]), spec)

    def test_no_sidecar_and_no_declaration_stops_the_run(self):
        dataset = self._dataset({}, declared_weights=None)
        with self.assertRaises(ValueError) as ctx:
            dataset.resolve_extractor_spec([CORALNET_PREFIX])
        self.assertIn(CORALNET_PREFIX, str(ctx.exception))
        self.assertIn("feature_extractor_weights", str(ctx.exception))

    def test_a_declaration_contradicting_a_sidecar_stops_the_run(self):
        # The config names weights the bucket was not built with. Without this
        # check the declaration would be believed for the sources that have no
        # sidecar, and silently wrong for the one that does.
        sidecar_spec = make_extractor_spec(1280)
        declared_spec = make_extractor_spec(1280, weights_sha256="f" * 64)
        objects = dict([_sidecar(MERMAID_PREFIX, sidecar_spec)])
        dataset = self._dataset(objects, declared_weights="s3://cfg/other.pt")
        with (
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.extractor_spec_from_weights",
                return_value=declared_spec,
            ),
            self.assertRaises(ValueError) as ctx,
        ):
            dataset.resolve_extractor_spec([MERMAID_PREFIX])
        self.assertIn("dataset.feature_extractor_weights", str(ctx.exception))

    def test_the_same_bytes_at_a_different_uri_is_not_a_conflict(self):
        # The release copies the extractor to a per-version key, so a run's
        # declaration and a bucket's sidecar can legitimately name different
        # paths to one file.
        sidecar_spec = make_extractor_spec(1280, weights_uri="s3://bucket/efficientnet.pt")
        declared_spec = make_extractor_spec(
            1280, weights_uri="s3://cfg/classifier/v1/efficientnet_weights.pt"
        )
        objects = dict([_sidecar(MERMAID_PREFIX, sidecar_spec)])
        dataset = self._dataset(objects, declared_weights=declared_spec.weights_uri)
        with mock.patch(
            "mermaid_classifier.pyspacer.dataset.extractor_spec_from_weights",
            return_value=declared_spec,
        ):
            resolved = dataset.resolve_extractor_spec([MERMAID_PREFIX])
        # The producer's own record wins: it names where the bytes were used.
        self.assertEqual(resolved, sidecar_spec)


if __name__ == "__main__":
    unittest.main()
