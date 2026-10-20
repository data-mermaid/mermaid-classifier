"""Count annotation points per (class, group) for a config, and write them to a CSV.

Reads the config's annotations (read, rollup, filter, dedupe; no feature
listing or download) and writes ``class_key, site, project_id, n``. A group is
a CoralNet source or a MERMAID project; a class key is ``<ba>::<gf>``, so an
empty growth form leaves a trailing ``::``. Prints points and distinct groups
per class.

Usage (from the repo root)::

    uv run python scripts/count_class_support.py \\
        --config-dir sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped \\
        --out class_support.csv
"""

from __future__ import annotations

import argparse
import dataclasses

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
    return parser.parse_args(argv)


def class_summary(counts: pd.DataFrame) -> pd.DataFrame:
    """Per class: total `points` and the number of distinct `groups` (site, project_id)."""
    grouped = counts.assign(group=counts["site"] + "/" + counts["project_id"]).groupby("class_key")
    return pd.DataFrame({"points": grouped["n"].sum(), "groups": grouped["group"].nunique()})


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_dir = _resolve(args.config_dir, REPO_ROOT)
    out_path = _resolve(args.out, REPO_ROOT)

    _resolve_local_aws_credentials()

    from mermaid_classifier.training.class_support import class_group_counts

    config, training_dataset_cls = load_config_then_pyspacer(config_dir)
    dataset_options, _, _ = config.build_options(config_dir=config_dir)
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
    if counts.empty:
        return
    print("Per class (points, distinct groups):")
    print(class_summary(counts).sort_values("points", ascending=False).to_string())


if __name__ == "__main__":
    main()
