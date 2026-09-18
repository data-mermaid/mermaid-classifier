"""Build the frozen region probe set and cache its features, locally.

Selects a fixed slice of the MERMAID confirmed-annotation export (every
Tropical Atlantic image plus seeded proportional samples of the two
Indo-Pacific regions), freezes the benthic-attribute region map, display names
and taxonomic ancestry alongside it, and downloads each selected point's
feature vector into one aligned matrix. Scoring a model on the probe afterwards
touches neither S3 nor the MERMAID API.

Everything is written under --out-dir. Nothing is uploaded unless --publish
names an s3://bucket/prefix/ to copy the build to; publishing a probe version
is a separate, deliberate step and --publish carries no default.

Each published version is immutable; a new build requires a new version prefix.
Run: AWS_PROFILE=wcs-admin uv run python scripts/build_region_probe.py --out-dir region_probe/v2 --publish s3://dev-datamermaid-sm-sources/region_probe/v2/

Outputs (in --out-dir):
    probe_points.parquet   one row per probe annotation point, PROBE_COLUMNS
    held_out_images.csv    distinct image ids from held-out probe rows, ready
                            for training's excluded_images_csv option
    ba_regions.json        the frozen benthic-attribute -> region-ids snapshot
    ba_region_counts.json  corpus-wide confirmed annotations per (attribute, region)
    names.json             display names for benthic attributes, growth forms, regions
    ba_ancestry.json       each benthic attribute's root-to-leaf ancestry path
    manifest.json          provenance, realized counts
    probe_features.npz     features[N,1280] float32 + aligned point metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import boto3

from mermaid_classifier.common.benthic_attributes import (
    BenthicAttributeLibrary,
    get_benthic_attribute_library,
    get_growth_form_library,
    get_region_library,
)
from mermaid_classifier.common.s3_utils import parse_s3_uri
from mermaid_classifier.region_eval.features import (
    DEFAULT_FEATURE_BUCKET,
    DEFAULT_FEATURE_PREFIX,
    DEFAULT_WORKERS,
    build_feature_cache,
    write_feature_cache,
)
from mermaid_classifier.region_eval.probe_set import (
    DEFAULT_CENSUS_REGION_NAMES,
    DEFAULT_REGION_FLOOR,
    DEFAULT_TARGET_IMAGES,
    PROBE_ANCESTRY_FILE,
    PROBE_COUNTS_FILE,
    PROBE_FEATURES_FILE,
    PROBE_FILE_NAMES,
    PROBE_HELD_OUT_IMAGES_FILE,
    PROBE_MANIFEST_FILE,
    PROBE_NAMES_FILE,
    PROBE_POINTS_FILE,
    PROBE_REGIONS_FILE,
    NameSnapshot,
    ProbeSelectionOptions,
    ancestry_snapshot_json,
    build_manifest,
    build_probe_set,
    ground_truth_counts_json,
    held_out_images_csv,
    name_snapshot_json,
    read_annotations,
    region_snapshot_json,
)

logger = logging.getLogger("build_region_probe")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_URI = "s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet"
DEFAULT_REGION = "us-east-1"


def builder_git_sha() -> str:
    """The commit the probe was built from, or "unknown" outside a checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def frozen_names(library: BenthicAttributeLibrary) -> NameSnapshot:
    """Display names for everything a region report renders.

    Frozen with the probe for the same reason the region map is: a name
    resolved at scoring time would make a published artifact depend on a later
    state of the taxonomy than the score it annotates.
    """
    return NameSnapshot(
        benthic_attributes={
            attribute_id: str(result["name"]) for attribute_id, result in library.by_id.items()
        },
        growth_forms=dict(get_growth_form_library().by_id),
        regions=dict(get_region_library().by_id),
    )


def frozen_ancestry(library: BenthicAttributeLibrary) -> dict[str, list[str]]:
    """Each benthic attribute's root-to-leaf path, root first and ending in itself.

    Two attributes share an ancestor exactly when their paths agree at the
    root, which is the condition `within_branch_share` reads.
    """
    return {
        attribute_id: [*library.get_ancestor_ids(attribute_id), attribute_id]
        for attribute_id in library.by_id
    }


def fetch_source(uri: str, destination: Path, *, region_name: str) -> str | None:
    """Download the annotation parquet and return the object's ETag."""
    bucket, key = parse_s3_uri(uri)
    client = boto3.client("s3", region_name=region_name)
    etag = client.head_object(Bucket=bucket, Key=key).get("ETag")
    client.download_file(bucket, key, str(destination))
    return etag


def publish_probe(out_dir: Path, uri: str, *, region_name: str = DEFAULT_REGION) -> list[str]:
    """Upload every file the build wrote under `out_dir` to `uri`.

    Refuses when the prefix already holds any object: a published probe
    version is immutable, so two scores can claim a shared baseline only when
    neither has silently overwritten the other. `probe_features.npz` uploads
    only when `out_dir` carries one, which a `--skip-features` build never
    writes.

    Known limitation: the emptiness check above is check-then-act, not
    atomic, so two `--publish` runs racing on the same fresh prefix can both
    observe it empty and both proceed, interleaving files from two builds
    under one version. A per-object conditional write
    (`ExtraArgs={"IfNoneMatch": "*"}` on `upload_file`) would close that
    window, but the installed s3transfer rejects `IfNoneMatch` before any
    request reaches S3 -- it is not in `TransferManager.ALLOWED_UPLOAD_ARGS`
    -- so `upload_file` cannot carry it. This tool is built for one operator
    publishing one version at a time; two people, or two automated runs,
    cutting the same version simultaneously is the race this leaves open.
    """
    bucket, prefix = parse_s3_uri(uri)
    prefix = prefix.rstrip("/") + "/"
    client = boto3.client("s3", region_name=region_name)

    existing = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    found = [item["Key"] for item in existing.get("Contents", [])]
    if found:
        raise FileExistsError(
            f"s3://{bucket}/{prefix} already holds {found[0]!r}; a published probe"
            " version is immutable. Publish to a new version prefix instead."
        )

    names = [name for name in PROBE_FILE_NAMES if name != PROBE_FEATURES_FILE]
    if (out_dir / PROBE_FEATURES_FILE).exists():
        names.append(PROBE_FEATURES_FILE)

    uploaded: list[str] = []
    for name in names:
        key = f"{prefix}{name}"
        client.upload_file(str(out_dir / name), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "region_probe")
    parser.add_argument("--source-uri", default=DEFAULT_SOURCE_URI)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-images", type=int, default=DEFAULT_TARGET_IMAGES)
    parser.add_argument("--region-floor", type=int, default=DEFAULT_REGION_FLOOR)
    parser.add_argument(
        "--census-region",
        action="append",
        dest="census_regions",
        help="region taken in full and left in training (repeatable);"
        f" default {sorted(DEFAULT_CENSUS_REGION_NAMES)}",
    )
    parser.add_argument("--feature-bucket", default=DEFAULT_FEATURE_BUCKET)
    parser.add_argument("--feature-prefix", default=DEFAULT_FEATURE_PREFIX)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--aws-region", default=DEFAULT_REGION)
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="select and freeze the points only; download no feature vectors",
    )
    parser.add_argument(
        "--publish",
        default=None,
        metavar="S3_URI",
        help="s3://bucket/prefix/ to upload every file under --out-dir to, once"
        " the build is complete; refuses if the prefix already holds anything."
        " No default -- publishing a probe version is a deliberate,"
        " one-time act, never a fallback destination.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        local_source = Path(staging) / "annotations.parquet"
        logger.info("downloading %s", args.source_uri)
        etag = fetch_source(args.source_uri, local_source, region_name=args.aws_region)
        annotations = read_annotations(local_source)
    logger.info("source rows: %d", len(annotations))

    library = get_benthic_attribute_library()
    region_ids_by_attribute = library.region_ids_by_id
    (out_dir / PROBE_REGIONS_FILE).write_text(region_snapshot_json(region_ids_by_attribute))

    names = frozen_names(library)
    ancestry = frozen_ancestry(library)
    (out_dir / PROBE_NAMES_FILE).write_text(name_snapshot_json(names))
    (out_dir / PROBE_ANCESTRY_FILE).write_text(ancestry_snapshot_json(ancestry))

    options = ProbeSelectionOptions(
        seed=args.seed,
        target_images=args.target_images,
        region_floor=args.region_floor,
        census_region_names=(
            frozenset(args.census_regions) if args.census_regions else DEFAULT_CENSUS_REGION_NAMES
        ),
    )
    probe = build_probe_set(
        annotations, region_ids_by_attribute=region_ids_by_attribute, options=options
    )
    probe.rows.to_parquet(out_dir / PROBE_POINTS_FILE, index=False)
    (out_dir / PROBE_HELD_OUT_IMAGES_FILE).write_text(held_out_images_csv(probe.rows))
    # Frozen for the same reason as the region map: scoring reads these counts
    # against a fixed threshold, so a corpus that grows afterwards must not
    # move a published score.
    (out_dir / PROBE_COUNTS_FILE).write_text(ground_truth_counts_json(probe.ground_truth_counts))

    manifest = build_manifest(
        probe,
        source_uri=args.source_uri,
        source_etag=etag,
        source_row_count=len(annotations),
        builder_git_sha=builder_git_sha(),
        names=names,
        ancestry=ancestry,
    )
    for stratum in probe.strata:
        logger.info(
            "  %-24s %-7s %5d/%-5d images, %6d points, held_out=%s",
            stratum.region_name,
            stratum.policy,
            stratum.n_images,
            stratum.n_eligible_images,
            stratum.n_points,
            stratum.held_out,
        )

    if not args.skip_features:
        with tempfile.TemporaryDirectory() as downloads:
            cache = build_feature_cache(
                probe.rows,
                Path(downloads),
                bucket=args.feature_bucket,
                prefix=args.feature_prefix,
                workers=args.workers,
            )
        write_feature_cache(cache, probe.rows, out_dir / PROBE_FEATURES_FILE)
        manifest["features"] = {
            "n_points_requested": cache.n_points_requested,
            "n_points_cached": int(cache.features.shape[0]),
            "n_points_missing_row_col": cache.n_points_missing_row_col,
            "n_points_missing_image": cache.n_points_missing_image,
            "n_images_missing": len(cache.missing_image_ids),
            "feature_bucket": args.feature_bucket,
            "feature_prefix": args.feature_prefix,
            "feature_dim": int(cache.features.shape[1]),
        }
        logger.info(
            "cached %d/%d points (%d missing (row,col), %d in %d missing image(s))",
            cache.features.shape[0],
            cache.n_points_requested,
            cache.n_points_missing_row_col,
            cache.n_points_missing_image,
            len(cache.missing_image_ids),
        )

    (out_dir / PROBE_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    logger.info(
        "probe %s: %s images, %s points, content_hash=%s",
        manifest["probe_version"],
        manifest["n_images"],
        manifest["n_points"],
        probe.content_hash,
    )
    logger.info("wrote %s", out_dir)

    if args.publish:
        for uri in publish_probe(out_dir, args.publish, region_name=args.aws_region):
            logger.info("published %s", uri)

    return 0


if __name__ == "__main__":
    sys.exit(main())
