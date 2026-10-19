"""Select source-grouped test groups and cross-validation folds for a config.

Reads the config's annotations (read, rollup, filter, dedupe; no feature
listing or download), then writes ``holdout_groups.csv`` (columns
``site, group_id, fold``) and a per-class coverage report beside it. A group
is a CoralNet source or a MERMAID project. CoralNet test candidates exclude
every source listed in ``--beta-sources``; MERMAID groups need the config's
``mermaid_project_lookup_uri``, otherwise MERMAID holdout is skipped.

Usage (from the repo root)::

    uv run python scripts/select_holdout_groups.py \\
        --config-dir sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import pandas as pd
from classifier_train import (
    _resolve_local_aws_credentials,  # pyright: ignore[reportPrivateUsage]  # shared local-credential bootstrap
)

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_BETA_SOURCES = WORKSPACE_ROOT / "sagemaker" / "sources" / "OldCoralNetSourcesToKeep.csv"
CONFIG_FILENAME = "training_config.yaml"
COVERAGE_SUFFIX = "_coverage.csv"
MERMAID_SITE = "mermaid"


def read_beta_sources(path: Path) -> set[str]:
    """The `id` column of the Beta source list, as strings."""
    with path.open(newline="") as f:
        return {row["id"].strip() for row in csv.DictReader(f) if row.get("id", "").strip()}


def write_holdout_csv(path: Path, assignments: dict[tuple[str, str], str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["site", "group_id", "fold"])
        for (site, group_id), fold in sorted(assignments.items()):
            writer.writerow([site, group_id, fold])


def _resolve(path: str, base: Path) -> Path:
    p = Path(path)
    return (p if p.is_absolute() else base / p).resolve()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--config-dir", required=True, help="Config dir, repo-root-relative or absolute."
    )
    parser.add_argument(
        "--beta-sources",
        default=str(DEFAULT_BETA_SOURCES),
        help="CSV with an `id` column of CoralNet sources barred from test (workspace-root default).",
    )
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-test-points", type=int, default=30)
    parser.add_argument("--max-class-group-loss", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", default=None, help="Output CSV. Default: <config-dir>/holdout_groups.csv."
    )
    args = parser.parse_args(argv)

    config_dir = _resolve(args.config_dir, REPO_ROOT)
    out_path = _resolve(args.out, REPO_ROOT) if args.out else config_dir / "holdout_groups.csv"
    beta_sources = read_beta_sources(_resolve(args.beta_sources, REPO_ROOT))

    _resolve_local_aws_credentials()

    from mermaid_classifier.pyspacer.dataset import TrainingDataset
    from mermaid_classifier.sagemaker.config import TrainingRunConfig
    from mermaid_classifier.training.holdout_selection import (
        SelectionParams,
        build_coverage_report,
        coverage_warnings,
        select_holdout,
    )

    config = TrainingRunConfig.from_yaml_path(config_dir / CONFIG_FILENAME)
    config.apply_env()
    dataset_options, _, _ = config.build_options(config_dir=config_dir)
    # Selection runs on all data: an existing holdout CSV must not tag or drop rows.
    dataset_options = dataclasses.replace(
        dataset_options, holdout_groups_csv=None, holdout_fold="test"
    )

    dataset = TrainingDataset.filtered_annotations_only(dataset_options)
    try:
        rows = dataset.duck_conn.execute(
            "SELECT benthic_attribute_id || '::' || growth_form_id AS class_key,"
            " site, project_id, COUNT(*) AS n"
            " FROM annotations GROUP BY 1, 2, 3"
        ).fetchall()
    finally:
        dataset.cleanup()
    if dataset_options.include_mermaid and not dataset_options.mermaid_project_lookup_uri:
        print(
            "No mermaid_project_lookup_uri in the config: every MERMAID row is project 'all',"
            " so MERMAID holdout is skipped."
        )
        rows = [r for r in rows if r[1] != MERMAID_SITE]
    if not rows:
        raise SystemExit("No annotations to select groups from.")
    counts = pd.DataFrame(
        {
            "class_key": [str(r[0]) for r in rows],
            "site": [str(r[1]) for r in rows],
            "project_id": [str(r[2]) for r in rows],
            "n": [int(r[3]) for r in rows],
        }
    )

    params = SelectionParams(
        test_frac=args.test_frac,
        folds=args.folds,
        min_test_points=args.min_test_points,
        max_class_group_loss=args.max_class_group_loss,
        seed=args.seed,
    )
    selection = select_holdout(counts, params, beta_sources)
    write_holdout_csv(out_path, selection.assignments)

    report = build_coverage_report(counts, selection.assignments, params.folds)
    coverage_path = out_path.with_name(out_path.stem + COVERAGE_SUFFIX)
    report.to_csv(coverage_path, index=False)

    print(f"Wrote {len(selection.assignments)} groups to {out_path}")
    print(f"Wrote coverage report to {coverage_path}")
    by_fold = pd.Series(selection.assignments).value_counts().sort_index()
    print("Groups per fold:")
    print(by_fold.to_string())
    for warning in [
        *selection.warnings,
        *coverage_warnings(report, params.min_test_points, params.folds),
    ]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
