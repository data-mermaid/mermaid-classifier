"""Score published model artifacts against the frozen region probe, locally.

Loads the probe points, their frozen benthic-attribute region map, the frozen
corpus-wide annotation counts triage reads, the frozen display names and
taxonomic ancestry, and the cached feature matrix from --probe-dir, runs each
--model through the production loader, and writes one report directory per
model under --out-dir (summary.csv, the per-region/label/direction tables,
decisions.csv and manifest.json). A probe carrying no counts file leaves
triage reading the probe's own ground truth, a lower bound that manifest.json
records; one carrying no names renders ids, and one carrying no ancestry
leaves the within-branch share uncomputed. Both are recorded there too.

A model path is either a local directory holding model.pt + model.json, or the
s3:// prefix of a released version. --probe-dir is likewise a local directory
or the s3://bucket/prefix/ a probe was published to. The feature cache is
downloaded and written back when --probe-dir has none. Nothing is uploaded:
publishing a score is a deliberate step of its own.

Run: AWS_PROFILE=wcs-admin uv run python scripts/evaluate_region_probe.py \
        --probe-dir region_probe/v1 \
        --model v1=s3://mermaid-config/classifier/v1 \
        --out-dir region_eval/v1

Outputs (in --out-dir):
    <model>/summary.csv           one row per metric, each with its denominator
    <model>/per_region.csv        per-region rates and accuracy
    <model>/per_label.csv         region-discriminating classes by incident count
    <model>/per_direction.csv     one row per (image region, excluded region)
    <model>/direction_matrix.csv  the same counts as a matrix
    <model>/confusion.csv         commonest (region, truth, prediction) triples
    <model>/region_list_suspects.csv  pairs the ground truth supports upstream
    <model>/decisions.csv         the statistics that choose between mitigations
    <model>/manifest.json         probe hashes, model paths, drift diagnostic
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import boto3

from mermaid_classifier.common.s3_utils import is_s3_uri, parse_s3_uri
from mermaid_classifier.region_eval.features import (
    DEFAULT_FEATURE_BUCKET,
    DEFAULT_FEATURE_PREFIX,
    DEFAULT_WORKERS,
)
from mermaid_classifier.region_eval.metrics import RegionMetricsOptions
from mermaid_classifier.region_eval.score import (
    PROBE_FEATURES_FILE,
    ModelScore,
    load_probe,
    score_model,
    write_report,
)
from mermaid_classifier.region_eval.triage import DEFAULT_LIST_SUSPECT_THRESHOLD

logger = logging.getLogger("evaluate_region_probe")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGION = "us-east-1"
ARTIFACT_FILES = ("model.pt", "model.json")


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split a --model NAME=PATH argument.

    Only the first "=" separates the two: an S3 key may carry more, and
    splitting on the last would silently score the wrong object.
    """
    name, separator, path = spec.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(f"--model must be NAME=PATH, got {spec!r}")
    return name.strip(), path.strip()


def resolve_artifact(path: str, staging: Path, *, region_name: str) -> tuple[Path, Path]:
    """The local model.pt and model.json for a directory or s3:// prefix."""
    if not path.startswith("s3://"):
        local = Path(path)
        resolved = tuple(local / name for name in ARTIFACT_FILES)
        missing = [str(candidate) for candidate in resolved if not candidate.exists()]
        if missing:
            raise FileNotFoundError(f"model artifact incomplete: {', '.join(missing)}")
        return resolved[0], resolved[1]

    bucket, prefix = parse_s3_uri(path)
    staging.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3", region_name=region_name)
    for name in ARTIFACT_FILES:
        client.download_file(bucket, f"{prefix.rstrip('/')}/{name}", str(staging / name))
    return staging / "model.pt", staging / "model.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        metavar="NAME=PATH",
        help="a model to score; PATH is a local dir or s3:// prefix holding"
        " model.pt + model.json (repeatable, and repeats are scored paired)",
    )
    parser.add_argument(
        "--probe-dir",
        required=True,
        help="a local directory or an s3://bucket/prefix/ holding a probe"
        " scripts/build_region_probe.py wrote (and --publish may have copied to S3)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=RegionMetricsOptions().alpha)
    parser.add_argument("--n-resamples", type=int, default=RegionMetricsOptions().n_resamples)
    parser.add_argument("--seed", type=int, default=RegionMetricsOptions().seed)
    parser.add_argument(
        "--target-margin",
        type=float,
        default=None,
        help="half-width each interval is checked against; cells wider than it"
        " are flagged imprecise rather than read as comparisons",
    )
    parser.add_argument("--triage-threshold", type=int, default=DEFAULT_LIST_SUSPECT_THRESHOLD)
    parser.add_argument("--feature-bucket", default=DEFAULT_FEATURE_BUCKET)
    parser.add_argument("--feature-prefix", default=DEFAULT_FEATURE_PREFIX)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--aws-region", default=DEFAULT_REGION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    specs = [parse_model_spec(spec) for spec in args.models]

    if not is_s3_uri(args.probe_dir):
        features_path = Path(args.probe_dir) / PROBE_FEATURES_FILE
        if not features_path.exists():
            logger.info("no %s; downloading feature vectors", features_path)
    with tempfile.TemporaryDirectory() as downloads:
        probe = load_probe(
            args.probe_dir,
            download_dir=Path(downloads),
            bucket=args.feature_bucket,
            prefix=args.feature_prefix,
            workers=args.workers,
            region_name=args.aws_region,
        )
    logger.info(
        "probe %s: %d cached point(s), content_hash=%s",
        probe.manifest.get("probe_version", "?"),
        probe.features.n_points,
        probe.content_hash,
    )

    options = RegionMetricsOptions(
        alpha=args.alpha,
        n_resamples=args.n_resamples,
        seed=args.seed,
        target_margin=args.target_margin,
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scores: list[ModelScore] = []
    with tempfile.TemporaryDirectory() as staging:
        for name, path in specs:
            model_pt, model_json = resolve_artifact(
                path, Path(staging) / name, region_name=args.aws_region
            )
            logger.info("scoring %s from %s", name, path)
            score = score_model(
                name,
                model_pt_path=model_pt,
                model_json_path=model_json,
                probe=probe,
                options=options,
                triage_threshold=args.triage_threshold,
            )
            write_report(score, out_dir / name)
            scores.append(score)
            overall = score.metrics.overall
            logger.info(
                "  out of region %.3f%% [%.3f%%, %.3f%%] (%d of %d);"
                " ground-truth floor %.3f%% [%.3f%%, %.3f%%] (%d of %d)",
                overall.oor_rate.rate * 100,
                overall.oor_rate.ci_low * 100,
                overall.oor_rate.ci_high * 100,
                overall.oor_rate.k,
                overall.oor_rate.n,
                overall.gt_oor_rate.rate * 100,
                overall.gt_oor_rate.ci_low * 100,
                overall.gt_oor_rate.ci_high * 100,
                overall.gt_oor_rate.k,
                overall.gt_oor_rate.n,
            )
            if score.drift.get("status") != "computed":
                logger.warning("  region list drift not computed: %s", score.drift.get("reason"))

    logger.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
