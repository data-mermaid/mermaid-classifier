"""CLI to build Label Studio tasks and synthesize exported results."""

import argparse
import csv
import json
import os
from collections.abc import Callable, Mapping
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
    beta_infer,
    features,
    ls_config,
    ls_export,
    ls_tasks,
    sample,
    synthesis,
    v1_infer,
)
from mermaid_classifier.model_review.image_set import IMAGE_SET, ImageSet, SiteSpec
from mermaid_classifier.model_review.ls_client import (
    PROJECT_TITLE_MAX_LENGTH,
    LabelStudioClient,
)
from mermaid_classifier.model_review.ls_config import UNLABELED_TOPLEVEL
from mermaid_classifier.model_review.rollup import load_toplevel, make_rollup_fn
from mermaid_classifier.model_review.sample import ReviewPoint


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
    """raw label key -> V1-label-set BA::GF: map to raw BA::GF, apply V1's training
    rollup, and keep only labels that are actually V1 classes (drop the rest, exactly
    as V1's training filter did).

    The key is a CoralNet label id for CoralNet points; MERMAID ground truth is already
    BA::GF, so it passes an identity `raw_mapper` and the key is the BA::GF itself.
    """

    def v1_label(label_key: str) -> str | None:
        raw = raw_mapper(label_key)
        if raw is None:
            return None
        rolled = roll_up(raw)
        return rolled if rolled in classes else None

    return v1_label


class ImageResolver:
    """Resolves a review point to its durable display image, per site.

    Tasks carry the plain ``s3://bucket/key`` URI, not a presigned URL: Label Studio
    resolves it at view time via that bucket's S3 source storage using the instance IAM
    role, so nothing in the task data expires. See HOSTING.md §3.

    A site whose ``image_key_extensions`` is non-empty has its key probed in that order
    (MERMAID stores ``.png``, and ~0.4% of images have no display object at all).
    Each image is probed once and the result cached, so ``can_display``, ``uri`` and
    ``size`` all agree on one key and cost at most one HEAD per image.
    """

    def __init__(self, s3_client: Any, specs: Mapping[str, SiteSpec]):
        self._s3 = s3_client
        self._specs = specs
        self._located: dict[tuple[str, str], tuple[str, str] | None] = {}

    def _exists(self, bucket: str, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._s3.head_object(Bucket=bucket, Key=key)
        except ClientError:
            return False
        return True

    def locate(self, point: ReviewPoint) -> tuple[str, str] | None:
        cache_key = (point.site, point.image_id)
        if cache_key in self._located:
            return self._located[cache_key]
        spec = self._specs[point.site]

        def key_for(ext: str) -> str:
            return spec.image_key_template.format(
                source_id=point.source_id, image_id=point.image_id, ext=ext
            )

        located: tuple[str, str] | None = None
        if not spec.image_key_extensions:
            located = (spec.image_bucket, key_for(""))
        else:
            for ext in spec.image_key_extensions:
                key = key_for(ext)
                if self._exists(spec.image_bucket, key):
                    located = (spec.image_bucket, key)
                    break
        self._located[cache_key] = located
        return located

    def can_display(self, point: ReviewPoint) -> bool:
        return self.locate(point) is not None

    def _require(self, point: ReviewPoint) -> tuple[str, str]:
        located = self.locate(point)
        if located is None:
            spec = self._specs[point.site]
            raise FileNotFoundError(
                f"no display image for {point.site} image {point.image_id} in "
                f"s3://{spec.image_bucket}/ (tried {', '.join(spec.image_key_extensions)})"
            )
        return located

    def uri(self, point: ReviewPoint) -> str:
        bucket, key = self._require(point)
        return f"s3://{bucket}/{key}"

    def size(self, point: ReviewPoint) -> tuple[int, int]:
        import io

        from PIL import Image

        bucket, key = self._require(point)
        obj = self._s3.get_object(Bucket=bucket, Key=key)
        return Image.open(io.BytesIO(obj["Body"].read())).size  # (width, height)


def image_set_from_args(args: argparse.Namespace) -> ImageSet:
    """Build an ImageSet from CLI flags.

    `sites` is not flag-configurable: half-specifying a site's S3 layout from the command
    line is worse than not offering it, so `image_set.py` stays the single place for it.
    """
    return ImageSet(
        name=args.name,
        classifier=args.classifier,
        beta_classifier=args.beta_classifier,
        heldout_csv=args.heldout_csv,
        v1_rollup_csv=args.v1_rollup_csv,
        n_images=args.n_images,
        min_points=args.min_points,
        seed=args.seed,
        sites=IMAGE_SET.sites,
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
    coralnet_mapper = make_v1_label_mapper(
        sample.default_coralnet_mapper(), rollup.roll_up, classes
    )
    # MERMAID ground truth is already BA::GF, so it skips the provider-id lookup and
    # goes straight into the same rollup + class filter.
    mermaid_mapper = make_v1_label_mapper(lambda bagf: bagf, rollup.roll_up, classes)

    s3 = boto3.client("s3")
    specs = {spec.site: spec for spec in image_set.sites}
    resolver = ImageResolver(s3, specs)
    builders = {sample.CORALNET: sample.coralnet_source, sample.MERMAID: sample.mermaid_source}
    mappers = {sample.CORALNET: coralnet_mapper, sample.MERMAID: mermaid_mapper}
    sources = [
        builders[spec.site](
            manifest_uri=spec.manifest_uri,
            feature_bucket=spec.feature_bucket,
            weight=spec.weight,
            map_label=mappers[spec.site],
            # An image with no display object cannot be reviewed, so displayability is
            # part of qualifying — the draw stays an exact SRS of a well-defined pool.
            keep_image=resolver.can_display,
        )
        for spec in image_set.sites
    ]

    points = sample.select_images_stratified(
        val_csv=image_set.heldout_csv,
        sources=sources,
        n_images=image_set.n_images,
        seed=image_set.seed,
        min_points_per_image=image_set.min_points,
    )
    # One S3 read per image feeds both reference models from the same rows.
    keys, feature_matrix = features.stack_features(points)
    v1_preds = v1_infer.predict_from_features(keys, feature_matrix, predictor)
    beta_preds = beta_infer.predict_points(keys, feature_matrix, image_set.beta_classifier)

    toplevel_name = make_toplevel_name()
    tasks = ls_tasks.build_tasks(
        points,
        v1_preds,
        beta_preds,
        label_path,
        toplevel_name,
        resolver.uri,
        resolver.size,
        image_set=image_set.name,
    )
    # Restrict the expert taxonomy to V1's label set plus the Beta labels this set
    # actually shows, so every reference label a reviewer can see has a selectable path.
    # Ground truth is already rolled into V1's classes, so it adds nothing.
    restrict_paths = sorted(label_path(c) for c in classes | set(beta_preds.values()))
    config = ls_config.build_taxonomy_config(ba_lib, gf_lib, restrict_paths=restrict_paths)
    return tasks, config


def site_mix(tasks: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """site -> (images, points) for a built task list.

    The quota is on images, so the realized point-level share differs from it: MERMAID
    grids are a fixed 25 points, CoralNet's average ~20.
    """
    mix: dict[str, tuple[int, int]] = {}
    for task in tasks:
        site = task["data"]["source"]
        images, points = mix.get(site, (0, 0))
        mix[site] = (images + 1, points + len(task["data"]["original_points"]))
    return dict(sorted(mix.items()))


def build_tasks_command(args: argparse.Namespace) -> None:
    tasks, config = build_tasks_and_config(image_set_from_args(args))
    with open(args.tasks_out, "w") as f:
        json.dump(tasks, f, indent=2)
    with open(args.config_out, "w") as f:
        f.write(config)
    print(f"Wrote {len(tasks)} tasks -> {args.tasks_out} and config -> {args.config_out}")
    mix = site_mix(tasks)
    total_points = sum(points for _, points in mix.values()) or 1
    for site, (images, points) in mix.items():
        print(f"  {site}: {images} images, {points} points ({100 * points / total_points:.1f}%)")


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
    # One presign storage per distinct bucket+prefix; LS picks the one matching each
    # task's s3:// URI. Config order keeps the call sequence deterministic.
    attached: set[tuple[str, str, str]] = set()
    for spec in image_set.sites:
        location = (spec.image_bucket, spec.image_prefix, spec.image_bucket_region)
        if location in attached:
            continue
        attached.add(location)
        client.add_s3_presign_storage(project_id, *location, title=f"{spec.site}-images")
    client.import_tasks(project_id, tasks)
    # Reviewers start BLIND from the unlabelled starting layer, not from v1.
    client.set_model_version(project_id, ls_tasks.BLANK_MODEL_VERSION)
    return project_id


def check_project_title(client: Any, name: str) -> None:
    """Reject a title Label Studio will not accept, or one already in use.

    Called before the tasks are built: the build downloads features and scores every
    model, so a title problem found afterwards costs that work for nothing.
    """
    if len(name) > PROJECT_TITLE_MAX_LENGTH:
        raise SystemExit(
            f"Label Studio project titles are limited to {PROJECT_TITLE_MAX_LENGTH} "
            f"characters; `name` in image_set.py is {len(name)}: {name!r}"
        )
    if name in client.project_titles():
        raise SystemExit(
            f"A Label Studio project titled {name!r} already exists. "
            f"Change `name` in image_set.py to create a new image set (existing "
            f"projects are never modified)."
        )


def create_project_command(args: argparse.Namespace) -> None:
    token = args.token or os.environ.get("LABEL_STUDIO_TOKEN")
    if not token:
        raise SystemExit("Provide --token or set LABEL_STUDIO_TOKEN (see HOSTING.md §2).")
    image_set = image_set_from_args(args)
    client = LabelStudioClient(args.ls_url, token)
    check_project_title(client, image_set.name)
    tasks, config_xml = build_tasks_and_config(image_set)
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

    summary = {
        "overall": synthesis.agreement_summary(df, tasks, roll),
        "by_site": synthesis.agreement_by_site(df, tasks, roll),
        "images_per_site": synthesis.image_counts_by_site(tasks),
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    with open(args.notes_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "expert", "note"])
        for n in notes:
            w.writerow([n.image_id, n.expert, n.note])
    print("Comparison summary:", summary)


def _add_image_set_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by build-tasks and create-project; every default is from IMAGE_SET.

    Per-site S3 layout (buckets, prefixes, key templates, manifests, weights) is
    deliberately not exposed here — it lives only in `image_set.py`.
    """
    p.add_argument("--name", default=IMAGE_SET.name, help="image-set name / LS project title")
    p.add_argument("--heldout-csv", default=IMAGE_SET.heldout_csv)
    p.add_argument(
        "--n-images",
        type=int,
        default=IMAGE_SET.n_images,
        help="total images, split across sites by their weights in image_set.py",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=IMAGE_SET.min_points,
        help="only sample images with at least this many ground-truth points",
    )
    p.add_argument("--seed", type=int, default=IMAGE_SET.seed)
    p.add_argument(
        "--classifier",
        default=IMAGE_SET.classifier,
        help="MLflow model id, S3 dir, or local dir for the reviewed model (auto-downloaded)",
    )
    p.add_argument(
        "--beta-classifier",
        default=IMAGE_SET.beta_classifier,
        help="Beta classifier pickle (scored in an isolated scikit-learn 1.1.3 subprocess)",
    )
    p.add_argument(
        "--v1-rollup-csv",
        default=IMAGE_SET.v1_rollup_csv,
        help=(
            "training rollup (from_ba_id,from_gf_id,to_ba_id,to_gf_id) to map GT into the label set"
        ),
    )


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
