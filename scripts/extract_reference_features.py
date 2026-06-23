"""Stack REAL EfficientNet feature vectors from pyspacer feature files into an
(N, feature_dim) float32 .npy for the portable-artifact live parity gate.

The live gate must run on real features (random vectors sit in flat softmax
regions and under-exercise the per-class calibration tails). pyspacer's
production pipeline already emits ImageFeatures (.featurevector) files; this
tool concatenates their per-point vectors.

Usage (local or s3):
  uv run python scripts/extract_reference_features.py \\
      --out reference_features.npy \\
      s3://bucket/key1.featurevector s3://bucket/key2.featurevector
  uv run python scripts/extract_reference_features.py \\
      --out reference_features.npy /path/to/*.featurevector

Then run the live gate:
  PORTABLE_ARTIFACT_LIVE_MODEL=s3://bucket/model.pkl \\
  PORTABLE_ARTIFACT_LIVE_FEATURES=reference_features.npy \\
  cd tests && uv run --no-sync python -m unittest -v \\
      pyspacer.test_portable_artifact.LiveModelParityTest
"""
from __future__ import annotations

import argparse
from urllib.parse import urlparse

import numpy as np
from spacer.data_classes import DataLocation, ImageFeatures


def _data_location(loc: str) -> DataLocation:
    uri = urlparse(loc)
    if uri.scheme == "s3":
        return DataLocation(
            "s3", bucket_name=uri.netloc, key=uri.path.strip("/"))
    return DataLocation("filesystem", key=loc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument(
        "features", nargs="+",
        help="pyspacer feature files (.featurevector), local paths or s3:// URIs",
    )
    args = ap.parse_args()

    vectors: list[list[float]] = []
    for loc in args.features:
        feats = ImageFeatures.load(_data_location(loc))
        for pf in feats.point_features:
            vectors.append(pf.data)

    X = np.asarray(vectors, dtype=np.float32)
    if X.ndim != 2:
        raise SystemExit(f"expected a 2-D feature matrix; got shape {X.shape}")
    np.save(args.out, X)
    print(f"wrote {X.shape[0]} real feature vectors "
          f"(dim {X.shape[1]}) to {args.out}")


if __name__ == "__main__":
    main()
