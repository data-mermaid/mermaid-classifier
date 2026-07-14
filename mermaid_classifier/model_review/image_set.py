"""THE place to change which images go into a model-review project.

To make a NEW image set:
  1. Edit the values in IMAGE_SET below (at minimum change `name` — each distinct
     name creates its own new Label Studio project and existing projects are left
     untouched).
  2. Rerun:  uv run --extra training --with pillow \\
                python -m mermaid_classifier.model_review.cli create-project --token <token>

Every field below is a knob that defines "which images" (and which model they are
compared against). Nothing else in the pipeline needs editing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageSet:
    name: str  # LS project title; also stamped into every task's data
    classifier: str  # reviewed model: S3 dir / MLflow id / local dir (auto-downloaded)
    heldout_csv: str  # val split defining test-set eligibility (image had >=1 val point)
    manifest_uri: str  # CoralNet manifest parquet: the full ground-truth grid per image
    feature_bucket: str  # bucket holding per-image .featurevector files
    image_bucket: str  # bucket holding the display .jpg images
    image_prefix: str  # S3 prefix for the LS source storage (scopes the images)
    image_bucket_region: str  # region of image_bucket (for the LS S3 storage config)
    image_key_template: str  # S3 key template; uses {source_id} and {image_id}
    v1_rollup_csv: str  # rollup mapping ground truth into the model's label set
    n_images: int  # how many images to sample
    min_points: int  # only sample images with >= this many ground-truth points
    seed: int  # RNG seed — change it to draw a different random sample


# First test run reproduces today's setup exactly, EXCEPT n_images=20 (was 50).
IMAGE_SET = ImageSet(
    name="model-review",
    classifier="s3://mermaid-config/classifier/v2/",
    heldout_csv="../reports/model_benchmark/data/v1_annotations_val.csv",
    manifest_uri=(
        "s3://dev-datamermaid-sm-sources/etl-outputs/coralnet/top108_full/"
        "coralnet_classifier_manifest_top108_full.parquet"
    ),
    feature_bucket="2605-coralnet-public-sources",
    image_bucket="dev-datamermaid-sm-sources",
    image_prefix="coralnet-public-images/",
    image_bucket_region="us-east-1",
    image_key_template="coralnet-public-images/s{source_id}/images/{image_id}.jpg",
    v1_rollup_csv="sagemaker/configs/coralnet_top108_full/rollups.csv",
    n_images=20,
    min_points=15,
    seed=1,
)
