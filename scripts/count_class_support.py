"""Count annotation points per (class, group) for a config, and write them to a CSV.

Reads the config's annotations (read, rollup, filter, dedupe; no feature
listing or download) and writes ``class_key, site, project_id, n``. A group is
a CoralNet source or a MERMAID project; a class key is ``<ba>::<gf>``, so an
empty growth form leaves a trailing ``::``. Prints points and distinct groups
per class. With ``--check-min-points`` / ``--check-min-groups`` it then exits
non-zero if any counted class is below either floor, listing each one.

Usage (from the repo root)::

    uv run python scripts/count_class_support.py \\
        --config-dir sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped \\
        --out class_support.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from classifier_train import (
    _resolve_local_aws_credentials,  # pyright: ignore[reportPrivateUsage]  # shared local-credential bootstrap
)
from select_holdout_groups import (
    REPO_ROOT,
    _resolve,  # pyright: ignore[reportPrivateUsage]  # shared path resolution
    load_config_then_pyspacer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--config-dir", required=True, help="Config dir, repo-root-relative or absolute."
    )
    parser.add_argument("--out", required=True, help="Output CSV, repo-root-relative or absolute.")
    parser.add_argument(
        "--check-min-points",
        type=int,
        default=0,
        help="After writing the CSV, fail if any class has fewer points.",
    )
    parser.add_argument(
        "--check-min-groups",
        type=int,
        default=0,
        help="After writing the CSV, fail if any class has fewer distinct groups.",
    )
    return parser.parse_args(argv)


def class_summary(counts: pd.DataFrame, extra_keys: Iterable[str] = ()) -> pd.DataFrame:
    """Per class: total `points` and distinct `groups` (site, project_id).

    `extra_keys` adds classes with no rows, at zero support. A key `X::` is
    judged on the union of its own rows and every `X::<gf>` row, matching
    the generator's Porites union; the GF rows stay as they are.
    """
    points: dict[str, int] = dict.fromkeys(extra_keys, 0)
    groups: dict[str, set[str]] = {key: set() for key in points}
    for key, site, project_id, n in zip(
        counts["class_key"], counts["site"], counts["project_id"], counts["n"], strict=True
    ):
        points[key] = points.get(key, 0) + int(n)
        groups.setdefault(key, set()).add(f"{site}/{project_id}")
    judged_points, judged_groups = dict(points), {k: set(v) for k, v in groups.items()}
    for key in points:
        ba_id, _, gf_id = key.partition("::")
        if gf_id:
            continue
        for other in points:
            other_ba, _, other_gf = other.partition("::")
            if other_ba == ba_id and other_gf:
                judged_points[key] += points[other]
                judged_groups[key] |= groups[other]
    index = pd.Index(list(points), name="class_key")
    return pd.DataFrame(
        {
            "points": [judged_points[k] for k in index],
            "groups": [len(judged_groups[k]) for k in index],
        },
        index=index,
    )


def included_class_keys(included_labels_csv: str) -> list[str]:
    """The `<ba>::<gf>` keys of an included-labels CSV, trailing `::` kept for an empty GF."""
    with Path(included_labels_csv).open(newline="") as f:
        return [f"{row['ba_id']}::{row.get('gf_id') or ''}" for row in csv.DictReader(f)]


def below_floors(summary: pd.DataFrame, min_points: int, min_groups: int) -> pd.DataFrame:
    """The rows of a `class_summary` frame below `min_points` or `min_groups`."""
    return pd.DataFrame(
        summary.loc[(summary["points"] < min_points) | (summary["groups"] < min_groups)]
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_dir = _resolve(args.config_dir, REPO_ROOT)
    out_path = _resolve(args.out, REPO_ROOT)

    _resolve_local_aws_credentials()

    from mermaid_classifier.training.class_support import class_group_counts

    config, training_dataset_cls = load_config_then_pyspacer(config_dir)
    dataset_options, _, _ = config.build_options(config_dir=config_dir)
    if dataset_options.include_mermaid and not dataset_options.mermaid_project_lookup_uri:
        raise SystemExit(
            "The config includes MERMAID but sets no mermaid_project_lookup_uri, so every"
            " MERMAID row would count as one project 'all'. Set the lookup, or set"
            " include_mermaid: false."
        )
    # Counts cover all data: an existing holdout CSV must not tag or drop rows.
    dataset_options = dataclasses.replace(
        dataset_options, holdout_groups_csv=None, holdout_fold="test"
    )
    dataset = training_dataset_cls.filtered_annotations_only(dataset_options)
    try:
        counts = class_group_counts(dataset)
    finally:
        dataset.cleanup()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_path, index=False)
    print(f"Wrote {len(counts)} class-group rows to {out_path}")
    checking = args.check_min_points > 0 or args.check_min_groups > 0
    keys: list[str] = []
    if checking and dataset_options.included_labels_csv:
        keys = included_class_keys(dataset_options.included_labels_csv)
    elif checking:
        print(
            "The config sets no included_labels_csv, so classes with zero annotations"
            " cannot be checked."
        )
    summary = class_summary(counts, keys)
    if not summary.empty:
        print("Per class (points, distinct groups):")
        print(summary.sort_values("points", ascending=False).to_string())
    failing = below_floors(summary, args.check_min_points, args.check_min_groups)
    if not failing.empty:
        print(
            f"{len(failing)} class(es) below the floor of {args.check_min_points} points"
            f" and {args.check_min_groups} groups:"
        )
        print(failing.sort_values("points").to_string())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
