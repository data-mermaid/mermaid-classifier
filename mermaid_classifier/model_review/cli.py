"""CLI to build Label Studio tasks and synthesize exported results."""

import argparse
import csv
import json
from collections.abc import Callable
from typing import Any

from mermaid_classifier.common.benthic_attributes import (
    BenthicAttributeLibrary,
    GrowthFormLibrary,
    combine_ba_gf,
    get_benthic_attribute_library,
    get_growth_form_library,
    split_ba_gf,
)
from mermaid_classifier.model_review import (
    ls_config,
    ls_export,
    ls_tasks,
    sample,
    synthesis,
    v1_infer,
)
from mermaid_classifier.model_review.ls_config import UNLABELED_TOPLEVEL
from mermaid_classifier.model_review.rollup import load_toplevel, make_rollup_fn


def make_toplevel_name() -> Callable[[str], str]:
    """BA_ID::GF_ID -> its top-level category NAME (for the colored keypoint label)."""
    roll = make_rollup_fn(strict=False)
    toplevel = load_toplevel()  # top-level id -> name

    def toplevel_name(bagf: str) -> str:
        top_id = roll(bagf)
        return toplevel.get(top_id, UNLABELED_TOPLEVEL) if top_id else UNLABELED_TOPLEVEL

    return toplevel_name


def make_label_path(
    ba_lib: BenthicAttributeLibrary, gf_lib: GrowthFormLibrary
) -> Callable[[str], list[str]]:
    """BA_ID::GF_ID -> taxonomy name path (root..leaf, plus growth form leaf)."""

    def label_path(bagf: str) -> list[str]:
        ba_id, gf_id = split_ba_gf(bagf)
        path_ids = ba_lib.get_ancestor_ids(ba_id) + [ba_id]
        parts = [ba_lib.id_to_name(i) for i in path_ids]
        if gf_id:
            parts.append(gf_lib.id_to_name(gf_id))
        return parts

    return label_path


def make_path_to_bagf(
    ba_lib: BenthicAttributeLibrary, gf_lib: GrowthFormLibrary
) -> Callable[[list[str]], str]:
    """Inverse of label_path: a taxonomy name path -> BA_ID::GF_ID.

    A trailing element that matches a growth-form name is taken as the growth form;
    the element before it (or the last element otherwise) is the benthic-attribute leaf.
    """
    gf_name_to_id = {name: gid for gid, name in gf_lib.by_id.items()}

    def path_to_bagf(path: list[str]) -> str:
        parts = list(path)
        gf_id = ""
        if len(parts) > 1 and parts[-1] in gf_name_to_id:
            gf_id = gf_name_to_id[parts[-1]]
            parts = parts[:-1]
        ba_id = ba_lib.name_to_id(parts[-1]) if parts else ""
        return combine_ba_gf(ba_id, gf_id)

    return path_to_bagf


def make_v1_label_mapper(
    raw_mapper: Callable[[str], str | None],
    roll_up: Callable[[str | None], str | None],
    classes: set[str],
) -> Callable[[str], str | None]:
    """coralnet_id -> V1-label-set BA::GF: map to raw BA::GF, apply V1's training
    rollup, and keep only labels that are actually V1 classes (drop the rest, exactly
    as V1's training filter did)."""

    def v1_label(coralnet_id: str) -> str | None:
        raw = raw_mapper(coralnet_id)
        if raw is None:
            return None
        rolled = roll_up(raw)
        return rolled if rolled in classes else None

    return v1_label


def make_presigner(
    s3_client: Any, image_bucket: str, key_template: str
) -> Callable[[str, str], str]:
    def image_url(source_id: str, image_id: str) -> str:
        key = key_template.format(source_id=source_id, image_id=image_id)
        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": image_bucket, "Key": key},
            ExpiresIn=604800,  # 7 days
        )

    return image_url


def _image_sizer(
    s3_client: Any, image_bucket: str, key_template: str
) -> Callable[[str, str], tuple[int, int]]:
    import io

    from PIL import Image

    def image_size(source_id: str, image_id: str) -> tuple[int, int]:
        key = key_template.format(source_id=source_id, image_id=image_id)
        obj = s3_client.get_object(Bucket=image_bucket, Key=key)
        img = Image.open(io.BytesIO(obj["Body"].read()))
        return img.size  # (width, height)

    return image_size


def build_tasks_command(args: argparse.Namespace) -> None:
    import boto3

    from mermaid_classifier.pyspacer.annotation import resolve_classifier_artifact
    from mermaid_classifier.pyspacer.inference import load_predictor
    from mermaid_classifier.pyspacer.label_specs import LabelRollupSpec

    ba_lib = get_benthic_attribute_library()
    gf_lib = get_growth_form_library()
    label_path = make_label_path(ba_lib, gf_lib)

    # V1's label set = its class list; GT is rolled into it with V1's training rollup
    # and any label outside V1's classes is dropped (exactly as V1's training filter did).
    model_pt, model_json = resolve_classifier_artifact(args.classifier)
    predictor = load_predictor(model_pt, model_json)
    classes = set(predictor.classes)
    with open(args.v1_rollup_csv) as f:
        rollup = LabelRollupSpec(f)
    v1_mapper = make_v1_label_mapper(sample.default_coralnet_mapper(), rollup.roll_up, classes)

    points = sample.select_images_full_gt(
        val_csv=args.heldout_csv,
        manifest_uri=args.manifest_uri,
        coralnet_bucket=args.feature_bucket,
        n_images=args.n_images,
        seed=args.seed,
        min_points_per_image=args.min_points,
        map_coralnet=v1_mapper,
    )
    v1_preds = v1_infer.predict_points(points, args.classifier, predictor=predictor)

    s3 = boto3.client("s3")
    toplevel_name = make_toplevel_name()
    image_url = make_presigner(s3, args.image_bucket, args.image_key_template)
    image_size = _image_sizer(s3, args.image_bucket, args.image_key_template)

    tasks = ls_tasks.build_tasks(points, v1_preds, label_path, toplevel_name, image_url, image_size)
    with open(args.tasks_out, "w") as f:
        json.dump(tasks, f, indent=2)

    # Restrict the expert taxonomy to V1's label set (one path per V1 class).
    restrict_paths = sorted(label_path(c) for c in classes)
    config = ls_config.build_taxonomy_config(ba_lib, gf_lib, restrict_paths=restrict_paths)
    with open(args.config_out, "w") as f:
        f.write(config)
    print(f"Wrote {len(tasks)} tasks -> {args.tasks_out} and config -> {args.config_out}")


def synthesize_command(args: argparse.Namespace) -> None:
    with open(args.tasks) as f:
        tasks = json.load(f)
    with open(args.export) as f:
        export = json.load(f)

    ba_lib = get_benthic_attribute_library()
    gf_lib = get_growth_form_library()
    path_to_bagf = make_path_to_bagf(ba_lib, gf_lib)
    expert_labels, notes = ls_export.parse_export(export, path_to_bagf)
    # strict=False: UNLABELED/unmapped labels roll to None rather than
    # raising, matching synthesis's handling of unrollable points.
    roll = make_rollup_fn(strict=False)
    df = synthesis.build_point_table(tasks, expert_labels, roll)
    df.to_csv(args.points_out, index=False)

    summary = synthesis.agreement_summary(df, tasks, roll)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    with open(args.notes_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "expert", "note"])
        for n in notes:
            w.writerow([n.image_id, n.expert, n.note])
    print("Comparison summary:", summary)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="model-review")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-tasks")
    b.add_argument(
        "--heldout-csv", default="../reports/model_benchmark/data/v1_annotations_val.csv"
    )
    b.add_argument("--n-images", type=int, default=50)
    b.add_argument(
        "--min-points",
        type=int,
        default=15,
        help="only sample images with at least this many total ground-truth points",
    )
    b.add_argument("--seed", type=int, default=1)
    b.add_argument(
        "--manifest-uri",
        default=(
            "s3://dev-datamermaid-sm-sources/etl-outputs/coralnet/top108_full/"
            "coralnet_classifier_manifest_top108_full.parquet"
        ),
        help="CoralNet manifest parquet: the full ground-truth points per image",
    )
    b.add_argument(
        "--feature-bucket",
        default="2605-coralnet-public-sources",
        help="S3 bucket holding the per-image .featurevector files",
    )
    b.add_argument(
        "--classifier", required=True, help="MLflow model id, S3 dir, or local dir for V1"
    )
    b.add_argument(
        "--v1-rollup-csv",
        default="sagemaker/configs/coralnet_top108_full/rollups.csv",
        help="V1 training rollup (from_ba_id,from_gf_id,to_ba_id,to_gf_id) to map GT into V1's label set",
    )
    b.add_argument("--image-bucket", default="dev-datamermaid-sm-sources")
    b.add_argument(
        "--image-key-template",
        default="coralnet-public-images/s{source_id}/images/{image_id}.jpg",
    )
    b.add_argument("--tasks-out", default="review_tasks.json")
    b.add_argument("--config-out", default="review_config.xml")
    b.set_defaults(func=build_tasks_command)

    s = sub.add_parser("synthesize")
    s.add_argument("--tasks", required=True)
    s.add_argument("--export", required=True)
    s.add_argument("--points-out", default="review_points.csv")
    s.add_argument("--summary-out", default="review_summary.json")
    s.add_argument("--notes-out", default="review_notes.csv")
    s.set_defaults(func=synthesize_command)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
