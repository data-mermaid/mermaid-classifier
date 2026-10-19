"""Aggregate the K cross-validation fold runs into one set of held-out metrics.

Downloads ``annotations_holdout.csv`` from each fold run, checks the folds are
disjoint, and writes per-class and overall metrics with source-clustered
bootstrap 95% CIs, for all points and for CoralNet and MERMAID separately.
The MLflow tracking URI comes from the MLFLOW_TRACKING_SERVER env var / .env
value unless ``--mlflow-tracking-uri`` is given.

Usage (from the repo root)::

    uv run python scripts/aggregate_cv.py --run-ids RUN0 RUN1 RUN2 RUN3 RUN4 --out-dir cv_report
    uv run python scripts/aggregate_cv.py --experiment my-experiment \\
        --run-name-contains grouped_fold --out-dir cv_report
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import mlflow
import mlflow.artifacts
import pandas as pd
from mlflow.tracking import MlflowClient

from mermaid_classifier.pyspacer.utils import mlflow_connect
from mermaid_classifier.training.cv_aggregate import (
    DEFAULT_REPS,
    concat_folds,
    summarize,
    to_markdown,
    unknown_class_counts,
)

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "annotations_holdout.csv"


def find_run_ids(client: MlflowClient, experiment: str, name_contains: str | None) -> list[str]:
    """Finished runs of `experiment` whose name contains `name_contains`, oldest first."""
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"No MLflow experiment named {experiment!r}.")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
        max_results=1000,
    )
    return [
        r.info.run_id
        for r in runs
        if name_contains is None or name_contains in (r.info.run_name or "")
    ]


def fetch_holdout(run_id: str, dst_dir: Path) -> pd.DataFrame:
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=ARTIFACT_NAME, dst_path=str(dst_dir / run_id)
    )
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-ids", nargs="+", help="The K fold run ids.")
    source.add_argument("--experiment", help="Experiment name; use with --run-name-contains.")
    parser.add_argument("--run-name-contains", default=None, help="Run-name substring filter.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="Bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Connected to MLflow in %s", mlflow_connect(tracking_uri=args.mlflow_tracking_uri))
    client = MlflowClient()

    run_ids: list[str] = args.run_ids or find_run_ids(
        client, args.experiment, args.run_name_contains
    )
    if not run_ids:
        raise SystemExit("No fold runs found.")
    logger.info("Aggregating %d run(s): %s", len(run_ids), ", ".join(run_ids))

    with tempfile.TemporaryDirectory() as tmp:
        frames = {run_id: fetch_holdout(run_id, Path(tmp)) for run_id in run_ids}
    combined = concat_folds(frames)

    summary = summarize(combined, reps=args.reps, seed=args.seed)
    unknowns = unknown_class_counts(combined)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "cv_metrics.csv"
    md_path = args.out_dir / "cv_metrics.md"
    summary.to_csv(csv_path, index=False)
    unknowns.to_csv(args.out_dir / "cv_unknown_class_counts.csv", index=False)
    md_path.write_text(to_markdown(summary, unknowns))
    print(unknowns.to_string(index=False))
    print(f"Wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
