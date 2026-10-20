"""THE place to change which images go into a model-review project.

To make a NEW image set:
  1. Add an ImageSet below (at minimum with a new `name` — each distinct name
     creates its own new Label Studio project and existing projects are left
     untouched) and point IMAGE_SET at it. Keep earlier sets so they stay
     reproducible.
  2. Rerun:  uv run --extra training --with pillow \\
                python -m mermaid_classifier.model_review.cli create-project --token <token>

Every field below is a knob that defines "which images" (and which model they are
compared against). Nothing else in the pipeline needs editing.

The sample is stratified by site: each `SiteSpec` in `sites` contributes a share of
`n_images` proportional to its `weight`, and each share is drawn independently.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteSpec:
    """One site's share of the sample, and where its data lives in S3."""

    site: str  # "coralnet" | "mermaid" — matches the val CSV's `site` column
    weight: float  # relative share of n_images (largest-remainder allocation)
    manifest_uri: str  # parquet holding the full ground-truth grid per image
    feature_bucket: str  # bucket holding the per-image .featurevector files
    image_bucket: str  # bucket holding the display images
    image_prefix: str  # S3 prefix for this site's LS source storage (scopes the images)
    image_bucket_region: str  # region of image_bucket (for the LS S3 storage config)
    image_key_template: str  # S3 key template; uses {source_id}, {image_id}, {ext}
    image_key_extensions: tuple[str, ...] = ()  # {ext} probe order; () = template is exact


@dataclass(frozen=True)
class ImageSet:
    name: str  # LS project title (Label Studio caps it at 50 chars); stamped into each task
    classifier: str  # reviewed model: S3 dir / MLflow id / local dir (auto-downloaded)
    beta_classifier: str  # Beta pickle; scored in an isolated scikit-learn 1.1.3 subprocess
    heldout_csv: str  # val split defining test-set eligibility (image had >=1 val point)
    v1_rollup_csv: str  # rollup mapping ground truth into the model's label set
    n_images: int  # how many images to sample, across all sites
    min_points: int  # only sample images with >= this many ground-truth points
    seed: int  # RNG seed — change it to draw a different random sample
    sites: tuple[SiteSpec, ...]  # the strata, and each one's share of n_images


CORALNET_SITE = SiteSpec(
    site="coralnet",
    weight=2.0,
    manifest_uri=(
        "s3://dev-datamermaid-sm-sources/etl-outputs/coralnet/top108_full/"
        "coralnet_classifier_manifest_top108_full.parquet"
    ),
    feature_bucket="2605-coralnet-public-sources",
    image_bucket="dev-datamermaid-sm-sources",
    image_prefix="coralnet-public-images/",
    image_bucket_region="us-east-1",
    image_key_template="coralnet-public-images/s{source_id}/images/{image_id}.jpg",
)

# MERMAID display images and feature vectors share the one `mermaid/` prefix. Every
# image carries a 25-point 5x5 grid, and ~0.4% have no display object — hence the
# extension probe, which also doubles as the "is this image displayable" check.
MERMAID_SITE = SiteSpec(
    site="mermaid",
    weight=1.0,
    manifest_uri="s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet",
    feature_bucket="coral-reef-training",
    image_bucket="coral-reef-training",
    image_prefix="mermaid/",
    image_bucket_region="us-east-1",
    image_key_template="mermaid/{image_id}.{ext}",
    image_key_extensions=("png", "jpg", "jpeg"),
)

# The coarse-label-set model trained on every CoralNet source, so its ground-truth
# grids come from the `all` manifest rather than `top108_full`.
CORALNET_ALL_SITE = SiteSpec(
    site="coralnet",
    weight=2.0,
    manifest_uri=(
        "s3://dev-datamermaid-sm-sources/etl-outputs/coralnet/all/"
        "coralnet_classifier_manifest_all.parquet"
    ),
    feature_bucket="2605-coralnet-public-sources",
    image_bucket="dev-datamermaid-sm-sources",
    image_prefix="coralnet-public-images/",
    image_bucket_region="us-east-1",
    image_key_template="coralnet-public-images/s{source_id}/images/{image_id}.jpg",
)

# weight 2:1 over 20 images -> 13 CoralNet + 7 MERMAID.
V1_IMAGE_SET = ImageSet(
    name="Model Review (20 images) + Beta + Comparison",
    classifier="s3://mermaid-config/classifier/v2/",
    # The model deployed to the MERMAID API. It shares V1's EfficientNet extractor, so
    # the same pre-extracted feature vectors score both. Repo-root-relative, like
    # heldout_csv. Diverging extractors would make Beta's tab plausible but wrong, and
    # nothing in code detects that — see beta_infer.py.
    beta_classifier="../models/beta_model/classifier.pkl",
    heldout_csv="../reports/model_benchmark/data/v1_annotations_val.csv",
    v1_rollup_csv="sagemaker/configs/coralnet_top108_full/rollups.csv",
    n_images=20,
    min_points=15,
    seed=1,
    sites=(CORALNET_SITE, MERMAID_SITE),
)

# The coarse-label-set model (mermaid-classifier PR #111): MLflow run
# 5d0d74a696614c7f87bcc8d173988a60, 94 classes with species and region-specific
# labels rolled up. Unreleased, so it is loaded by MLflow model id. The held-out CSV
# is that run's `annotations_val.csv` artifact. The "v1" tab shows this model.
COARSE_IMAGE_SET = ImageSet(
    name="Model Review coarse (20 img) + Beta + Comparison",
    classifier="m-2e523eb2d73242d79cfcbe9f6634de33",
    beta_classifier="../models/beta_model/classifier.pkl",
    heldout_csv="../reports/model_benchmark/data/coarse_annotations_val.csv",
    v1_rollup_csv="sagemaker/configs/coralnet_all_plus_mermaid_coarse/rollups.csv",
    n_images=20,
    min_points=15,
    seed=2,
    sites=(CORALNET_ALL_SITE, MERMAID_SITE),
)

# The min-support model (mermaid-classifier PR #115): MLflow run
# 31c89e1bdc8b4c699161be40ba701632, 80 classes, classes below 500 points or 10 sources
# rolled up. The held-out CSV is that run's source-grouped `annotations_holdout.csv`;
# neither this model nor Beta trained on those sources. The "v1" tab shows this model.
MINSUPPORT_IMAGE_SET = ImageSet(
    name="Model Review minsup (20 img) + Beta + Comparison",
    classifier="m-3201af06083c47519f5dd2a1653e96c9",
    beta_classifier="../models/beta_model/classifier.pkl",
    heldout_csv="../reports/model_benchmark/data/minsupport_annotations_holdout.csv",
    v1_rollup_csv="sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped_minsupport/rollups.csv",
    n_images=20,
    min_points=15,
    seed=3,
    sites=(CORALNET_ALL_SITE, MERMAID_SITE),
)

IMAGE_SET = MINSUPPORT_IMAGE_SET
