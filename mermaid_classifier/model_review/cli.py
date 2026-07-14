"""CLI to build Label Studio tasks and synthesize exported results."""

import argparse
import csv
import json
import os
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
from mermaid_classifier.model_review.image_set import IMAGE_SET, ImageSet
from mermaid_classifier.model_review.ls_client import LabelStudioClient
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


def make_s3_uri(image_bucket: str, key_template: str) -> Callable[[str, str], str]:
    """(source_id, image_id) -> a durable ``s3://bucket/key`` URI.

    Tasks carry the plain S3 URI, not a presigned URL: Label Studio resolves it at
    view time via its S3 source storage (proxy mode) using the instance IAM role,
    so nothing in the task data expires. See HOSTING.md §3.
    """

    def image_url(source_id: str, image_id: str) -> str:
        key = key_template.format(source_id=source_id, image_id=image_id)
        return f"s3://{image_bucket}/{key}"

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


def image_set_from_args(args: argparse.Namespace) -> ImageSet:
    return ImageSet(
        name=args.name,
        classifier=args.classifier,
        heldout_csv=args.heldout_csv,
        manifest_uri=args.manifest_uri,
        feature_bucket=args.feature_bucket,
        image_bucket=args.image_bucket,
        image_prefix=args.image_prefix,
        image_bucket_region=args.image_bucket_region,
        image_key_template=args.image_key_template,
        v1_rollup_csv=args.v1_rollup_csv,
        n_images=args.n_images,
        min_points=args.min_points,
        seed=args.seed,
    )


def build_tasks_and_config(image_set: ImageSet) -> tuple[list[dict[str, Any]], str]:
    """Build LS tasks + labeling config from an ImageSet (reads S3/model/manifest)."""
    import boto3

    from mermaid_classifier.pyspacer.annotation import resolve_classifier_artifact
    from mermaid_classifier.pyspacer.inference import load_predictor
    from mermaid_classifier.pyspacer.label_specs import LabelRollupSpec

    ba_lib = get_benthic_attribute_library()
    gf_lib = get_growth_form_library()
    label_path = make_label_path(ba_lib, gf_lib)

    # V1's label set = its class list; GT is rolled into it with V1's training rollup
    # and any label outside V1's classes is dropped (exactly as V1's training filter did).
    model_pt, model_json = resolve_classifier_artifact(image_set.classifier)
    predictor = load_predictor(model_pt, model_json)
    classes = set(predictor.classes)
    with open(image_set.v1_rollup_csv) as f:
        rollup = LabelRollupSpec(f)
    v1_mapper = make_v1_label_mapper(sample.default_coralnet_mapper(), rollup.roll_up, classes)

    points = sample.select_images_full_gt(
        val_csv=image_set.heldout_csv,
        manifest_uri=image_set.manifest_uri,
        coralnet_bucket=image_set.feature_bucket,
        n_images=image_set.n_images,
        seed=image_set.seed,
        min_points_per_image=image_set.min_points,
        map_coralnet=v1_mapper,
    )
    v1_preds = v1_infer.predict_points(points, image_set.classifier, predictor=predictor)

    s3 = boto3.client("s3")
    toplevel_name = make_toplevel_name()
    # Tasks carry durable s3:// URIs (LS resolves them via S3 source storage); the
    # image dimensions are still read at build time from the objects themselves.
    image_url = make_s3_uri(image_set.image_bucket, image_set.image_key_template)
    image_size = _image_sizer(s3, image_set.image_bucket, image_set.image_key_template)

    tasks = ls_tasks.build_tasks(
        points,
        v1_preds,
        label_path,
        toplevel_name,
        image_url,
        image_size,
        image_set=image_set.name,
    )
    # Restrict the expert taxonomy to V1's label set (one path per V1 class).
    restrict_paths = sorted(label_path(c) for c in classes)
    config = ls_config.build_taxonomy_config(ba_lib, gf_lib, restrict_paths=restrict_paths)
    return tasks, config


def build_tasks_command(args: argparse.Namespace) -> None:
    tasks, config = build_tasks_and_config(image_set_from_args(args))
    with open(args.tasks_out, "w") as f:
        json.dump(tasks, f, indent=2)
    with open(args.config_out, "w") as f:
        f.write(config)
    print(f"Wrote {len(tasks)} tasks -> {args.tasks_out} and config -> {args.config_out}")


def orchestrate_create_project(
    client: Any,
    image_set: ImageSet,
    tasks: list[dict[str, Any]],
    config_xml: str,
) -> int:
    """Create a BRAND-NEW LS project for this image set and load it.

    Refuses (SystemExit) if a project with the same title already exists, so existing
    projects are never touched — to make another set, change `name` in image_set.py.
    """
    if image_set.name in client.project_titles():
        raise SystemExit(
            f"A Label Studio project titled {image_set.name!r} already exists. "
            f"Change `name` in image_set.py to create a new image set (existing "
            f"projects are never modified)."
        )
    project_id = client.create_project(image_set.name, config_xml)
    client.add_s3_presign_storage(
        project_id, image_set.image_bucket, image_set.image_prefix, image_set.image_bucket_region
    )
    client.import_tasks(project_id, tasks)
    # Reviewers start BLIND from the unlabelled starting layer, not from v1.
    client.set_model_version(project_id, ls_tasks.BLANK_MODEL_VERSION)
    return project_id


def create_project_command(args: argparse.Namespace) -> None:
    token = args.token or os.environ.get("LABEL_STUDIO_TOKEN")
    if not token:
        raise SystemExit("Provide --token or set LABEL_STUDIO_TOKEN (see HOSTING.md §2).")
    image_set = image_set_from_args(args)
    tasks, config_xml = build_tasks_and_config(image_set)
    client = LabelStudioClient(args.ls_url, token)
    project_id = orchestrate_create_project(client, image_set, tasks, config_xml)
    print(
        f"Created project {project_id!r} ({image_set.name!r}) with {len(tasks)} tasks: "
        f"{args.ls_url}/projects/{project_id}/data"
    )


def synthesize_command(args: argparse.Namespace) -> None:
    with open(args.tasks) as f:
        tasks = json.load(f)
    with open(args.export) as f:
        export = json.load(f)

    ba_lib = get_benthic_attribute_library()
    gf_lib = get_growth_form_library()
    path_to_bagf = make_path_to_bagf(ba_lib, gf_lib)
    # Resolve each annotation's author id -> email (attribution/filtering by reviewer).
    user_map: dict[str, str] | None = None
    if args.users_json:
        with open(args.users_json) as f:
            users = json.load(f)
        user_map = {str(u["id"]): u["email"] for u in users if u.get("email")}
    expert_labels, notes = ls_export.parse_export(export, path_to_bagf, user_map=user_map)
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


def _add_image_set_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by build-tasks and create-project; every default is from IMAGE_SET."""
    p.add_argument("--name", default=IMAGE_SET.name, help="image-set name / LS project title")
    p.add_argument("--heldout-csv", default=IMAGE_SET.heldout_csv)
    p.add_argument("--n-images", type=int, default=IMAGE_SET.n_images)
    p.add_argument(
        "--min-points",
        type=int,
        default=IMAGE_SET.min_points,
        help="only sample images with at least this many total ground-truth points",
    )
    p.add_argument("--seed", type=int, default=IMAGE_SET.seed)
    p.add_argument(
        "--manifest-uri",
        default=IMAGE_SET.manifest_uri,
        help="CoralNet manifest parquet: the full ground-truth points per image",
    )
    p.add_argument(
        "--feature-bucket",
        default=IMAGE_SET.feature_bucket,
        help="S3 bucket holding the per-image .featurevector files",
    )
    p.add_argument(
        "--classifier",
        default=IMAGE_SET.classifier,
        help="MLflow model id, S3 dir, or local dir for the reviewed model (auto-downloaded)",
    )
    p.add_argument(
        "--v1-rollup-csv",
        default=IMAGE_SET.v1_rollup_csv,
        help=(
            "training rollup (from_ba_id,from_gf_id,to_ba_id,to_gf_id) to map GT into the label set"
        ),
    )
    p.add_argument("--image-bucket", default=IMAGE_SET.image_bucket)
    p.add_argument("--image-prefix", default=IMAGE_SET.image_prefix)
    p.add_argument("--image-bucket-region", default=IMAGE_SET.image_bucket_region)
    p.add_argument("--image-key-template", default=IMAGE_SET.image_key_template)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-review")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-tasks")
    _add_image_set_args(b)
    b.add_argument("--tasks-out", default="review_tasks.json")
    b.add_argument("--config-out", default="review_config.xml")
    b.set_defaults(func=build_tasks_command)

    c = sub.add_parser("create-project")
    _add_image_set_args(c)
    c.add_argument("--ls-url", default="http://localhost:8080")
    c.add_argument("--token", help="LS API token (or set LABEL_STUDIO_TOKEN)")
    c.set_defaults(func=create_project_command)

    s = sub.add_parser("synthesize")
    s.add_argument("--tasks", required=True)
    s.add_argument("--export", required=True)
    s.add_argument(
        "--users-json",
        help="JSON dump of LS /api/users (id->email) to attribute annotations by reviewer email",
    )
    s.add_argument("--points-out", default="review_points.csv")
    s.add_argument("--summary-out", default="review_summary.json")
    s.add_argument("--notes-out", default="review_notes.csv")
    s.set_defaults(func=synthesize_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
