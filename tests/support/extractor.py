"""An ExtractorSpec for artifact tests.

`export_artifact` requires the identity of the extractor that produced the
features a head was fitted to. Tests fit on synthetic vectors, so they need a
spec whose `feature_dim` matches the fixture's width rather than production's
1280 — the export refuses a spec that disagrees with the head's input width.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from mermaid_classifier.pyspacer.inference import ExtractorSpec

# Stands in for a digest. A test that exercises hashing computes a real one.
PLACEHOLDER_SHA256 = "0" * 64


def make_extractor_spec(feature_dim: int = 8, **overrides: Any) -> ExtractorSpec:
    """A spec matching `feature_dim`-wide features, with fields overridable."""
    spec = ExtractorSpec(
        class_path="spacer.extractors.efficientnet.EfficientNetExtractor",
        crop_size=224,
        feature_dim=feature_dim,
        weights_uri="s3://test-bucket/efficientnet.pt",
        weights_sha256=PLACEHOLDER_SHA256,
    )
    return dataclasses.replace(spec, **overrides) if overrides else spec
