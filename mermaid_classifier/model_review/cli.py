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
from mermaid_classifier.model_review.rollup import make_rollup_fn


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

    ba_lib = get_benthic_attribute_library()
    gf_lib = get_growth_form_library()

    points = sample.select_images(
        args.heldout_csv,
        n_images=args.n_images,
        seed=args.seed,
        min_points_per_image=args.min_points,
    )
    v1_preds = v1_infer.predict_points(points, classifier_location=args.classifier)

    s3 = boto3.client("s3")
    label_path = make_label_path(ba_lib, gf_lib)
    image_url = make_presigner(s3, args.image_bucket, args.image_key_template)
    image_size = _image_sizer(s3, args.image_bucket, args.image_key_template)

    tasks = ls_tasks.build_tasks(points, v1_preds, label_path, image_url, image_size)
    with open(args.tasks_out, "w") as f:
        json.dump(tasks, f, indent=2)

    config = ls_config.build_taxonomy_config(ba_lib, gf_lib)
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
    b.add_argument("--n-images", type=int, default=15)
    b.add_argument(
        "--min-points",
        type=int,
        default=15,
        help="only sample images with at least this many val points (grid, not 1-2)",
    )
    b.add_argument("--seed", type=int, default=1)
    b.add_argument(
        "--classifier", required=True, help="MLflow model id, S3 dir, or local dir for V1"
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
