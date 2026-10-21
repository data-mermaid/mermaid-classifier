"""Compare two training runs on the same source-grouped holdout, with paired CIs.

Reads each run's ``annotations_holdout.csv`` (from MLflow with ``--run-a/--run-b``,
or local files with ``--csv-a/--csv-b``), checks both runs scored the same
points with the same ground truth, and writes per-class and overall metrics for
each run plus the difference B - A with source-clustered paired bootstrap 95%
CIs, for all points and for CoralNet and MERMAID separately. With MLflow runs
it also checks both runs logged byte-identical ``holdout_groups.csv`` and the
same ``holdout_max_per_class`` param. The MLflow tracking URI comes from the
MLFLOW_TRACKING_SERVER env var / .env value unless ``--mlflow-tracking-uri`` is
given.

``--real-mix`` weights each point by available / (compared points of its fine
ground-truth class), undoing the per-class holdout cap; per-class recall is
unchanged by it. The outputs record the weighting (``capped`` or ``real_mix``). ``--check-groups`` checks run B's group split (train+ref, val
and holdout disjoint; val and holdout groups as listed in ``val_groups.csv``,
columns site,group_id, and the ``test`` fold of ``holdout_groups.csv``).

Writes compare_metrics.csv, compare_metrics.md and point_set_check.txt.

Usage (from the repo root)::

    uv run python scripts/compare_holdout_runs.py --run-a RUN_A --run-b RUN_B \\
        --mlflow-tracking-uri ARN --out-dir compare_report \\
        --top-level-rollup-csv ../sagemaker/labels/ba_rollup_top_level.csv --real-mix
    uv run python scripts/compare_holdout_runs.py --csv-a a.csv --csv-b b.csv --out-dir out
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import tempfile
from pathlib import Path
from typing import cast

import mlflow
import mlflow.artifacts
import pandas as pd
from mlflow.tracking import MlflowClient

from mermaid_classifier.pyspacer.utils import mlflow_connect
from mermaid_classifier.training.cv_aggregate import DEFAULT_REPS
from mermaid_classifier.training.run_compare import (
    align_holdout,
    check_group_disjointness,
    paired_summary,
    real_mix_weights,
    to_markdown_paired,
    top_level_map,
    with_class_names,
)

logger = logging.getLogger(__name__)

HOLDOUT_ARTIFACT = "annotations_holdout.csv"
GROUPS_ARTIFACT = "holdout_groups.csv"
COUNTS_ARTIFACT = "holdout/per_class_counts.csv"
PROJECT_STATS_ARTIFACT = "project_stats_train_data.csv"
VAL_GROUPS_ARTIFACT = "val_groups.csv"
CAP_PARAM = "holdout_max_per_class"
TEST_FOLD = "test"
COUNT_COLUMNS = ["benthic_attribute_id", "growth_form_id", "available", "realized"]


def _read_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _download(run_id: str, name: str, dst_dir: Path) -> Path:
    return Path(
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=name, dst_path=str(dst_dir / run_id)
        )
    )


def _groups(text: str, fold: str | None = None) -> set[tuple[str, str]]:
    """(site, group_id) pairs of a groups CSV, limited to `fold` when given."""
    rows = csv.DictReader(io.StringIO(text))
    return {(r["site"], r["group_id"]) for r in rows if fold is None or r.get("fold") == fold}


def _normalized_counts(df: pd.DataFrame) -> pd.DataFrame:
    subset = cast(pd.DataFrame, df[COUNT_COLUMNS])
    return subset.sort_values(by=COUNT_COLUMNS[:2]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--run-a", help="Baseline MLflow run id.")
    parser.add_argument("--run-b", help="Compared MLflow run id.")
    parser.add_argument("--csv-a", type=Path, help="Baseline annotations_holdout.csv (offline).")
    parser.add_argument("--csv-b", type=Path, help="Compared annotations_holdout.csv (offline).")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="Bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--top-level-rollup-csv", type=Path, help="Also compare at top level via this rollup."
    )
    parser.add_argument(
        "--top-level-classes-csv",
        type=Path,
        help="Top-level classes (id,name): each id maps to itself, and names label the output.",
    )
    parser.add_argument(
        "--real-mix", action="store_true", help="Weight points back to the uncapped class mix."
    )
    parser.add_argument(
        "--class-counts-csv", type=Path, help="Per-class holdout counts for --real-mix (offline)."
    )
    parser.add_argument(
        "--allow-point-mismatch",
        action="store_true",
        help="Compare on the agreeing intersection when the point sets differ.",
    )
    parser.add_argument(
        "--check-groups", action="store_true", help="Check run B's group split (MLflow only)."
    )
    parser.add_argument("--mlflow-tracking-uri", default=None)
    args = parser.parse_args(argv)

    run_a: str | None = args.run_a
    run_b: str | None = args.run_b
    csv_a: Path | None = args.csv_a
    csv_b: Path | None = args.csv_b
    class_counts_csv: Path | None = args.class_counts_csv
    use_mlflow = run_a is not None or run_b is not None
    use_csv = csv_a is not None or csv_b is not None
    if use_mlflow == use_csv:
        parser.error("Give either --run-a and --run-b, or --csv-a and --csv-b.")
    if use_csv and args.check_groups:
        parser.error("--check-groups needs MLflow runs.")
    if use_csv and args.real_mix and class_counts_csv is None:
        parser.error("--real-mix offline needs --class-counts-csv.")
    if class_counts_csv is not None and not args.real_mix:
        parser.error("--class-counts-csv needs --real-mix.")
    if use_mlflow and class_counts_csv is not None:
        parser.error("--class-counts-csv is for the offline path; runs supply their own.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    counts: pd.DataFrame | None = None
    group_problems: list[str] = []
    if run_a is not None or run_b is not None:
        if run_a is None or run_b is None:
            parser.error("--run-a and --run-b go together.")
        logger.info(
            "Connected to MLflow in %s", mlflow_connect(tracking_uri=args.mlflow_tracking_uri)
        )
        client = MlflowClient()
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp)
            holdout_a = _read_csv(_download(run_a, HOLDOUT_ARTIFACT, dst))
            holdout_b = _read_csv(_download(run_b, HOLDOUT_ARTIFACT, dst))
            groups_a = _download(run_a, GROUPS_ARTIFACT, dst).read_bytes()
            groups_b = _download(run_b, GROUPS_ARTIFACT, dst).read_bytes()
            if groups_a != groups_b:
                raise SystemExit("The runs' holdout_groups.csv artifacts differ.")
            caps = [client.get_run(r).data.params.get(CAP_PARAM) for r in (run_a, run_b)]
            if caps[0] != caps[1]:
                raise SystemExit(f"The runs' {CAP_PARAM} params differ: {caps[0]} vs {caps[1]}.")
            if args.real_mix:
                counts_a = _read_csv(_download(run_a, COUNTS_ARTIFACT, dst))
                counts_b = _read_csv(_download(run_b, COUNTS_ARTIFACT, dst))
                if not _normalized_counts(counts_a).equals(_normalized_counts(counts_b)):
                    raise SystemExit(f"The runs' {COUNTS_ARTIFACT} artifacts differ.")
                counts = counts_a
            if args.check_groups:
                stats = _read_csv(_download(run_b, PROJECT_STATS_ARTIFACT, dst))
                val = _groups(_download(run_b, VAL_GROUPS_ARTIFACT, dst).read_text())
                test = _groups(groups_b.decode(), fold=TEST_FOLD)
                group_problems = check_group_disjointness(stats, val, test)
    else:
        if csv_a is None or csv_b is None:
            parser.error("--csv-a and --csv-b go together.")
        holdout_a = _read_csv(csv_a)
        holdout_b = _read_csv(csv_b)
        if class_counts_csv is not None:
            counts = _read_csv(class_counts_csv)

    try:
        aligned, report = align_holdout(
            holdout_a, holdout_b, allow_mismatch=args.allow_point_mismatch
        )
    except ValueError as exc:
        raise SystemExit(f"Not comparable: {exc}") from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    check_path = args.out_dir / "point_set_check.txt"
    weighting = "real_mix" if args.real_mix else "capped"
    check_text = report.to_text() + f"\nweighting: {weighting}\n"
    if args.check_groups:
        check_text += "group problems: " + str(len(group_problems)) + "\n"
        check_text += "".join(f"  {p}\n" for p in group_problems)
    check_path.write_text(check_text)
    print(check_text, end="")
    if group_problems:
        raise SystemExit(
            f"Run B's group split has {len(group_problems)} problem(s); see {check_path}."
        )

    top_level_ids: list[str] = []
    class_names: dict[str, str] = {}
    if args.top_level_classes_csv is not None:
        top = _read_csv(args.top_level_classes_csv)
        top_level_ids = top["id"].tolist()
        # Top-level classes have no growth form, so the key is "<id>::".
        class_names = {
            f"{i}::": name for i, name in zip(top_level_ids, top["name"].tolist(), strict=True)
        }

    try:
        weights = (
            real_mix_weights(counts, aligned["gt_bagf"].tolist())
            if args.real_mix and counts is not None
            else None
        )
        summaries = [paired_summary(aligned, args.reps, args.seed, point_weights=weights)]
        if args.top_level_rollup_csv is not None:
            labels = set(aligned["gt_bagf"]) | set(aligned["pred_a"]) | set(aligned["pred_b"])
            mapping = top_level_map(
                sorted(labels), _read_csv(args.top_level_rollup_csv), top_level_ids
            )
            summaries.append(
                paired_summary(
                    aligned, args.reps, args.seed, class_map=mapping, point_weights=weights
                )
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    summary = with_class_names(pd.concat(summaries, ignore_index=True), class_names).assign(
        weighting=weighting
    )

    csv_path = args.out_dir / "compare_metrics.csv"
    md_path = args.out_dir / "compare_metrics.md"
    summary.to_csv(csv_path, index=False)
    md_path.write_text(f"Weighting: {weighting}\n\n" + to_markdown_paired(summary))
    print(f"Wrote {csv_path}, {md_path} and {check_path}")


if __name__ == "__main__":
    main()
