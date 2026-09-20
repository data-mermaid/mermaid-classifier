"""Building the feature extractor, and describing what it produced.

The image -> feature-vector step runs in a different process and at a
different time from training, so a run has to be told which extractor made
its inputs. These helpers are the one place that turns a weights URI into an
extractor and into the `ExtractorSpec` that travels with the model, so the
script that writes feature vectors and the pipeline that consumes them
describe them the same way.

Heavy imports are function-local: naming a weights file must not cost the
torch import, which both spacer's extractor package and the inference
subpackage pull in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mermaid_classifier.common.s3_utils import is_s3_uri, parse_s3_uri, sha256_of_uri

if TYPE_CHECKING:
    from spacer.data_classes import DataLocation

    from mermaid_classifier.pyspacer.inference import ExtractorSpec

# Written at the root of a feature-vector prefix by the job that produced it.
SIDECAR_FILENAME = "_extractor_spec.json"


def weights_data_location(uri: str) -> DataLocation:
    """Parse an s3://bucket/key URI or filesystem path into a DataLocation."""
    # Deferred so a caller that only needs to name the weights does not pay
    # for spacer's extractor package, which imports torch.
    from spacer.data_classes import DataLocation

    if is_s3_uri(uri):
        bucket, key = parse_s3_uri(uri)
        return DataLocation(storage_type="s3", key=key, bucket_name=bucket)
    return DataLocation(storage_type="filesystem", key=uri)


def extractor_spec_from_weights(weights_uri: str) -> ExtractorSpec:
    """Describe the stock extractor loaded from `weights_uri`.

    Geometry comes off the extractor class, which costs nothing — no weights
    are read to answer CROP_SIZE or feature_dim. The one fetch is the hash of
    the weights object itself.
    """
    from spacer.extractors import EfficientNetExtractor

    from mermaid_classifier.pyspacer.inference import ExtractorSpec

    extractor = EfficientNetExtractor(
        data_locations={"weights": weights_data_location(weights_uri)}
    )
    return ExtractorSpec.from_extractor(
        extractor,
        weights_uri=weights_uri,
        weights_sha256=sha256_of_uri(weights_uri),
    )
