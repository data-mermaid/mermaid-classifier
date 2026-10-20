"""Select early-stopping val groups for a config, from the groups outside test.

Reads the test groups from the config's ``holdout_groups.csv`` (never modified),
counts points per (class, group) from the config's own post-rollup labels, and
picks about ``--val-frac`` of each site's non-test groups so every class keeps
training groups. ``--max-val-point-frac`` caps points, not groups: per site, val
groups hold at most that share of the site's points. At 0.10 only 43 of 107
CoralNet groups were reached on the min-support config, so the default is 0.15.
The script exits without writing when any site falls short of its target group
count, unless ``--allow-short-val`` is given. Writes ``site, group_id`` (the config's
``dataset.val_groups_csv``) and a per-class ``val_groups_coverage.csv`` beside
it. ``--counts-csv`` reads the counts from a ``count_class_support.py`` output
instead of the config's labels; those keys must be the config's post-rollup keys.

Usage (from the repo root)::

    uv run python scripts/select_val_groups.py \\
        --config-dir sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from classifier_train import (
    _resolve_local_aws_credentials,  # pyright: ignore[reportPrivateUsage]  # shared local-credential bootstrap
)
from select_holdout_groups import (
    CONFIG_FILENAME,
    MERMAID_SITE,
    REPO_ROOT,
    _resolve,  # pyright: ignore[reportPrivateUsage]  # shared path resolution
    all_data_options,
    load_config_then_pyspacer,
)

from mermaid_classifier.training.holdout_selection import (
    TEST_FOLD,
    VAL_FOLD,
    Group,
    SelectionParams,
    build_coverage_report,
    select_val_groups,
    site_test_shares,
    without_groups,
)

if TYPE_CHECKING:
    from mermaid_classifier.pyspacer.dataset import TrainingDataset
    from mermaid_classifier.sagemaker.config import TrainingRunConfig

DEFAULT_VAL_CSV = "val_groups.csv"
HOLDOUT_FILENAME = "holdout_groups.csv"
COVERAGE_NAME = "val_groups_coverage.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--config-dir", required=True, help="Config dir, repo-root-relative or absolute."
    )
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--min-val-points", type=int, default=30)
    parser.add_argument("--max-class-group-loss", type=float, default=0.20)
    parser.add_argument(
        "--max-val-point-frac",
        type=float,
        default=0.15,
        help="Caps points, not groups: per site, val groups hold at most this fraction of the"
        " site's points (0 disables). 0.10 reached only 43 of 107 CoralNet groups on the"
        " min-support config.",
    )
    parser.add_argument(
        "--allow-short-val",
        action="store_true",
        help="Write the CSV even when a site gets fewer val groups than its target.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--counts-csv",
        default=None,
        help="A count_class_support.py output (class_key, site, project_id, n) to use as counts.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV. Default: the config's dataset.val_groups_csv, else <config-dir>/val_groups.csv.",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.max_class_group_loss < 1:
        parser.error("--max-class-group-loss must be in [0, 1).")
    for name in ("val_frac", "max_val_point_frac"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1].")
    return args


def check_out_path(out_path: Path, holdout_path: Path) -> None:
    """Refuse an output that could overwrite a holdout (fold) file."""
    if out_path == holdout_path or out_path.name == HOLDOUT_FILENAME:
        raise SystemExit(f"--out {out_path} is a holdout groups file, which is never modified.")
    if out_path.is_file():
        with out_path.open(newline="") as f:
            header = next(csv.reader(f), [])
        if "fold" in header:
            raise SystemExit(f"--out {out_path} exists and has a 'fold' column; not overwriting.")


def check_counts_classes(counts: pd.DataFrame, included_labels_csv: str | None) -> None:
    """Refuse counts holding a class the config's included labels do not list."""
    from count_class_support import included_class_keys

    if included_labels_csv is None:
        return
    extra = set(counts["class_key"]) - set(included_class_keys(included_labels_csv))
    if extra:
        raise SystemExit(
            f"--counts-csv has {len(extra)} class key(s) outside the config's"
            f" included_labels_csv (e.g. {sorted(extra)[:3]}); coarser counts could strand"
            " classes. Regenerate it from this config's labels."
        )


def selection_params(args: argparse.Namespace) -> SelectionParams:
    return SelectionParams(
        test_frac=args.val_frac,
        folds=1,
        min_test_points=args.min_val_points,
        max_class_group_loss=args.max_class_group_loss,
        seed=args.seed,
        max_test_point_frac=args.max_val_point_frac,
    )


def read_test_groups(path: Path) -> set[Group]:
    from mermaid_classifier.pyspacer.label_specs import HoldoutGroups

    with path.open() as f:
        spec = HoldoutGroups(f)
    return {g for g, fold in spec.group_folds.items() if fold == TEST_FOLD}


def write_val_csv(path: Path, groups: set[Group]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["site", "group_id"])
        for site, group_id in sorted(groups):
            writer.writerow([site, group_id])


def _config_counts(
    config: TrainingRunConfig, training_dataset_cls: type[TrainingDataset], config_dir: Path
) -> pd.DataFrame:
    """Per-(class, group) counts from the config's post-rollup labels, over all data."""
    from mermaid_classifier.training.class_support import class_group_counts

    dataset_options, _, _ = config.build_options(config_dir=config_dir)
    dataset_options = all_data_options(dataset_options)
    dataset = training_dataset_cls.filtered_annotations_only(dataset_options)
    try:
        counts = class_group_counts(dataset)
    finally:
        dataset.cleanup()
    if dataset_options.include_mermaid and not dataset_options.mermaid_project_lookup_uri:
        print(
            "No mermaid_project_lookup_uri in the config: every MERMAID row is project 'all',"
            " so MERMAID val groups are skipped."
        )
        counts = pd.DataFrame(counts.loc[counts["site"] != MERMAID_SITE]).reset_index(drop=True)
    return counts


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_dir = _resolve(args.config_dir, REPO_ROOT)

    if args.counts_csv:
        from mermaid_classifier.sagemaker.config import TrainingRunConfig

        config = TrainingRunConfig.from_yaml_path(config_dir / CONFIG_FILENAME)
        training_dataset_cls = None
    else:
        _resolve_local_aws_credentials()
        config, training_dataset_cls = load_config_then_pyspacer(config_dir)

    holdout_path = config.dataset.holdout_groups_csv_path(config_dir)
    if holdout_path is None:
        raise SystemExit(
            "The config sets no dataset.holdout_groups_csv, so test groups are unknown."
        )
    holdout_path = holdout_path.resolve()
    out_path = (
        _resolve(args.out, REPO_ROOT)
        if args.out
        else (
            config.dataset.val_groups_csv_path(config_dir) or config_dir / DEFAULT_VAL_CSV
        ).resolve()
    )
    check_out_path(out_path, holdout_path)

    test_groups = read_test_groups(holdout_path)
    if training_dataset_cls is None:
        counts_path = _resolve(args.counts_csv, REPO_ROOT)
        counts = pd.read_csv(counts_path, dtype={"class_key": str, "site": str, "project_id": str})
        included = config.dataset.included_labels_csv_path(config_dir)
        check_counts_classes(counts, str(included) if included else None)
        print(f"Counts come from {counts_path}, not from the config's labels.")
    else:
        counts = _config_counts(config, training_dataset_cls, config_dir)
    counts_nt = without_groups(counts, test_groups)
    if counts_nt.empty:
        raise SystemExit("No non-test annotations to select val groups from.")

    params = selection_params(args)
    selection = select_val_groups(counts, test_groups, params)
    val_groups = set(selection.assignments)
    if val_groups & test_groups:
        raise SystemExit("Val groups overlap the test groups; refusing to write.")
    if selection.short_sites and not args.allow_short_val:
        for warning in selection.warnings:
            print(f"WARNING: {warning}")
        raise SystemExit(
            f"Site(s) {selection.short_sites} fell short of their val group target; nothing"
            " written. Raise --max-val-point-frac or --max-class-group-loss, or pass"
            " --allow-short-val."
        )
    write_val_csv(out_path, val_groups)

    # The coverage builder reads the `test` label; its test_* columns are val_* here.
    as_val = dict.fromkeys(val_groups, TEST_FOLD)
    report = build_coverage_report(counts_nt, as_val, folds=0)
    report = report.rename(columns=lambda c: c.replace("test_", "val_", 1) if "test_" in c else c)
    coverage_path = out_path.with_name(COVERAGE_NAME)
    report.to_csv(coverage_path, index=False)

    print(f"Wrote {len(val_groups)} {VAL_FOLD} groups to {out_path}")
    print(f"Wrote coverage report to {coverage_path}")
    print("Val share of non-test groups and points, per site:")
    for site, share in site_test_shares(counts_nt, as_val).items():
        print(
            f"  {site}: {share.test_groups}/{share.groups} groups"
            f" ({share.test_groups / max(share.groups, 1):.1%}),"
            f" {share.test_points}/{share.total_points} points"
            f" ({share.test_points / max(share.total_points, 1):.1%})"
        )
    for warning in selection.warnings:
        print(f"WARNING: {warning}")
    uncovered = sorted(report.loc[report["val_points"] == 0, "class_key"])
    print(f"{len(uncovered)} class(es) with 0 val points:")
    for class_key in uncovered:
        print(f"  {class_key}")


if __name__ == "__main__":
    main()
