"""ExtractorSpec: the identity of the image -> feature-vector transform.

A head is fitted to feature vectors, never to pixels, so the extractor that
produced those vectors is part of the model's contract. This value type
carries that identity from the process that extracts features, through
training and export, into the released manifest, where the release gate and
the serving lane re-check it against the extractor they are about to use.

Stdlib only, so the [inference] lane can read the manifest without pulling
pyspacer or boto3. Extractor objects are duck-typed for the same reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mermaid_classifier.pyspacer.inference import (
    ExtractorMismatchError,
    ManifestError,
)

# Top-level key in model.json. Additive: schema_version stays 1, and
# consumers that read only the keys they know (mermaid-api's
# Classifier.register) are unaffected.
MANIFEST_KEY = "feature_extraction"

_PYSPACER_NAMESPACE = "spacer."


@dataclass(frozen=True)
class ExtractorSpec:
    """What produced a feature vector: which extractor, which weights."""

    class_path: str
    crop_size: int
    feature_dim: int
    weights_uri: str
    weights_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> ExtractorSpec:
        """Parse strictly: unknown, missing, or wrongly-typed keys raise."""
        if not isinstance(data, dict):
            raise ValueError(f"{MANIFEST_KEY} must be a JSON object; got {type(data).__name__}")
        fields = {f: data.get(f) for f in cls.__dataclass_fields__}
        missing = sorted(f for f, v in fields.items() if v is None)
        if missing:
            raise ValueError(f"{MANIFEST_KEY} missing key(s): {', '.join(missing)}")
        extra = sorted(set(data) - set(cls.__dataclass_fields__))
        if extra:
            raise ValueError(f"{MANIFEST_KEY} has unknown key(s): {', '.join(extra)}")
        return cls(
            class_path=str(fields["class_path"]),
            crop_size=int(fields["crop_size"]),  # pyright: ignore[reportArgumentType]  # JSON value
            feature_dim=int(fields["feature_dim"]),  # pyright: ignore[reportArgumentType]  # JSON value
            weights_uri=str(fields["weights_uri"]),
            weights_sha256=str(fields["weights_sha256"]),
        )

    @classmethod
    def from_extractor(
        cls, extractor: Any, *, weights_uri: str, weights_sha256: str
    ) -> ExtractorSpec:
        """Read the geometry off a live extractor rather than restating it.

        class_path resolves to the nearest pyspacer class in the MRO, so a
        local throughput subclass (device placement, batch size) records the
        stock class whose feature function it implements. verify_device_numerics
        is what holds that equivalence.
        """
        return cls(
            class_path=_pyspacer_class_path(type(extractor)),
            crop_size=int(extractor.CROP_SIZE),
            feature_dim=int(extractor.feature_dim),
            weights_uri=weights_uri,
            weights_sha256=weights_sha256,
        )

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> ExtractorSpec:
        """Read the block out of a parsed model.json.

        Raises ManifestError when absent: an artifact cut before this contract
        existed cannot say which extractor trained it, and guessing is the
        failure mode the contract exists to remove.
        """
        if MANIFEST_KEY not in manifest:
            raise ManifestError(
                f"model.json has no {MANIFEST_KEY!r} block, so the extractor that"
                " produced its training features is unrecorded. Artifacts cut"
                " before this contract cannot be served by this code."
            )
        try:
            return cls.from_dict(manifest[MANIFEST_KEY])
        except ValueError as exc:
            raise ManifestError(f"malformed {MANIFEST_KEY} in model.json: {exc}") from exc

    def check_extractor(self, extractor: Any) -> None:
        """Raise ExtractorMismatchError unless `extractor` matches this spec.

        Geometry only — the weights hash is checked where the bytes are
        fetched, not here.
        """
        actual = _pyspacer_class_path(type(extractor))
        mismatches: list[str] = []
        if actual != self.class_path:
            mismatches.append(f"class_path: trained with {self.class_path}, runtime has {actual}")
        if int(extractor.CROP_SIZE) != self.crop_size:
            mismatches.append(
                f"crop_size: trained with {self.crop_size}, runtime has {extractor.CROP_SIZE}"
            )
        if mismatches:
            raise ExtractorMismatchError(
                "Extractor does not match the one the model was trained against — "
                + "; ".join(mismatches)
            )

    def check_feature_dim(self, feature_dim: int) -> None:
        """Raise ExtractorMismatchError unless extracted features are the width
        the head was fitted to."""
        if int(feature_dim) != self.feature_dim:
            raise ExtractorMismatchError(
                f"feature_dim: trained with {self.feature_dim}, extraction produced {feature_dim}"
            )

    def identity(self) -> tuple[str, int, int, str]:
        """What makes two specs the same feature function: the extractor and
        the weight bytes.

        ``weights_uri`` is excluded. The same bytes are reachable at more than
        one key -- a producer's own record and a run's declaration can name
        different paths to one file without the features differing.
        """
        return (self.class_path, self.crop_size, self.feature_dim, self.weights_sha256)

    def describe(self) -> str:
        """One-line form for error messages naming two specs."""
        return (
            f"{self.class_path} crop={self.crop_size} dim={self.feature_dim}"
            f" weights={self.weights_uri} sha256={self.weights_sha256[:12]}…"
        )


def _pyspacer_class_path(cls: type) -> str:
    """Dotted path of the nearest class in the `spacer.` namespace, falling
    back to the class itself for an extractor defined entirely outside it."""
    for klass in cls.__mro__:
        if klass.__module__.startswith(_PYSPACER_NAMESPACE):
            return f"{klass.__module__}.{klass.__name__}"
    return f"{cls.__module__}.{cls.__name__}"
