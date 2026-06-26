"""
Generate a self-contained HTML report from an MLflow run's metrics and artifacts.

Usage:
    cd mermaid-classifier

    uv run python scripts/generate_report.py \
        --run-id 19e53cf4c2704d08adaaac3e74d925fe \
        --output report.html

    # Custom title:
    uv run python scripts/generate_report.py \
        --run-id 19e53cf4c2704d08adaaac3e74d925fe \
        --title "Training Run - 160 Sources" \
        --output report.html

    # Against a SageMaker Studio MLflow App (requires AWS_PROFILE active):
    uv run python scripts/generate_report.py \
        --run-id 70ad001e58c1442681e9221ed6a06a9d \
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-2OMU4VP53ZS2 \
        --output report.html
"""

import argparse
import base64
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import mlflow
import mlflow.artifacts
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader
from mlflow.tracking import MlflowClient

from mermaid_classifier.pyspacer.settings import set_env_vars_for_packages
from mermaid_classifier.pyspacer.utils import mlflow_connect

logger = logging.getLogger(__name__)

# -- Metric groupings for the executive summary --

EXECUTIVE_METRICS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("f1_macro", "F1 (Macro)"),
    ("precision_macro", "Precision (Macro)"),
    ("recall_macro", "Recall (Macro)"),
    ("mcc", "MCC"),
    ("ece", "ECE"),
    ("log_loss", "Log Loss"),
]

TOPK_METRICS = [
    ("top_1_accuracy", "Top-1"),
    ("top_3_accuracy", "Top-3"),
    ("top_5_accuracy", "Top-5"),
    ("top_10_accuracy", "Top-10"),
    ("mrr", "MRR"),
    ("hierarchical_top_5_mean_similarity", "Hierarchical Top-5 Similarity"),
]

COVER_METRICS = [
    ("cover_mean_abs_bias_pct", "Mean Abs Bias %"),
    ("cover_mean_rmse_pct", "Mean RMSE %"),
    ("cover_mean_mae_pct", "Mean MAE %"),
    ("cover_median_r_squared", "Median R-squared"),
]

TAXONOMIC_METRICS = [
    ("cross_branch_error_rate", "Cross-Branch Error Rate"),
    ("within_branch_error_rate", "Within-Branch Error Rate"),
    ("gf_accuracy_gf_relevant", "GF Accuracy (GF-relevant)"),
    ("within_ba_gf_accuracy", "Within-BA GF Accuracy"),
]

# -- Artifact manifest --
# (artifact_path, loader_key): loader_key is 'png', 'csv', or 'yaml'.
# Organized by section for building the template context.


class _SectionDefRequired(TypedDict):
    title: str
    artifacts: list[tuple[str, str]]


class _SectionDef(_SectionDefRequired, total=False):
    optional: bool


EVALUATION_SECTIONS: dict[str, _SectionDef] = {
    "confusion_matrix": {
        "title": "Confusion Matrices",
        "artifacts": [
            ("confusion_matrix/frequencies.png", "png"),
            ("confusion_matrix/percents.png", "png"),
            ("confusion_matrix/frequencies.csv", "csv"),
            ("confusion_matrix/percents.csv", "csv"),
        ],
    },
    "calibration": {
        "title": "Calibration",
        "artifacts": [
            ("calibration/reliability_diagram.png", "png"),
            ("calibration/per_bin_details.csv", "csv"),
            ("calibration/per_category_ece.csv", "csv"),
        ],
    },
    "cover": {
        "title": "Cover Analysis",
        "optional": True,
        "artifacts": [
            ("cover/per_class_bias.png", "png"),
            ("cover/per_class_cover_metrics.csv", "csv"),
        ],
    },
    "probability": {
        "title": "Probability / Log Loss",
        "optional": True,
        "artifacts": [
            ("probability/per_category_log_loss.png", "png"),
            ("probability/per_category_log_loss.csv", "csv"),
        ],
    },
    "ranking": {
        "title": "Ranking",
        "optional": True,
        "artifacts": [
            ("ranking/per_category_topk.png", "png"),
            ("ranking/per_category_topk.csv", "csv"),
            ("ranking/hierarchical_topk.csv", "csv"),
        ],
    },
    "taxonomic": {
        "title": "Taxonomic Error Analysis",
        "artifacts": [
            ("taxonomic/error_attribution.png", "png"),
            ("taxonomic/error_attribution.csv", "csv"),
            ("taxonomic/top_level_confusion.png", "png"),
            ("taxonomic/top_level_confusions.csv", "csv"),
            ("taxonomic/gf_confusion.png", "png"),
            ("taxonomic/gf_precision_recall_f1.csv", "csv"),
        ],
    },
    "per_source": {
        "title": "Per-Source Breakdown",
        "optional": True,
        "artifacts": [
            ("per_source/accuracy_by_source.png", "png"),
            ("per_source/metrics.csv", "csv"),
        ],
    },
}

ROOT_EVALUATION_ARTIFACTS = [
    ("metrics_per_label.csv", "csv"),
    ("metrics_overall.yaml", "yaml"),
]

TRAINING_ARTIFACTS = [
    ("system_specs.yaml", "yaml"),
    ("train_summary.yaml", "yaml"),
    ("other_options.yaml", "yaml"),
    ("epoch_ref_accuracies.yaml", "yaml"),
    ("coralnet_sources_included.csv", "csv"),
    ("labels_included.csv", "csv"),
    ("labels_excluded.csv", "csv"),
    ("rollup_spec.csv", "csv"),
    ("project_stats_raw.csv", "csv"),
    ("project_stats_train_data.csv", "csv"),
    ("ba_counts.csv", "csv"),
    ("bagf_counts.csv", "csv"),
    ("coralnet_label_mapping.csv", "csv"),
    ("unmapped_labels.csv", "csv"),
]


# -- Leaf loaders --


def encode_png_as_base64(png_path: Path) -> str:
    """Read a PNG file and return a data URI string."""
    data = png_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def load_csv_as_html_table(
    csv_path: Path,
    sort_by: str | None = None,
    ascending: bool = True,
) -> str:
    """Read a CSV with pandas and return an HTML table string."""
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        # Optional artifacts (e.g. excluded labels or a rollup spec) that
        # weren't specified in the training parameters are still logged as
        # empty (0-cell) CSVs, which pandas can't parse.
        return "<span>(Empty)</span>"
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)
    return df.to_html(
        index=False,
        classes="dataframe",
        float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}",
        na_rep="",
    )


def load_yaml_file(yaml_path: Path) -> dict[str, Any]:
    """Read a YAML file and return the parsed dict."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _load_artifact(
    artifact_dir: Path, artifact_path: str, loader_key: str
) -> str | dict[str, Any] | None:
    """Load a single artifact if it exists, return None otherwise."""
    full_path = artifact_dir / artifact_path
    if not full_path.exists():
        return None

    if loader_key == "png":
        return encode_png_as_base64(full_path)
    if loader_key == "csv":
        if full_path.stat().st_size == 0:
            return None
        try:
            if artifact_path == "metrics_per_label.csv":
                return load_csv_as_html_table(full_path, sort_by="f1_score", ascending=True)
            return load_csv_as_html_table(full_path)
        except pd.errors.EmptyDataError:
            return None
    elif loader_key == "yaml":
        return load_yaml_file(full_path)
    else:
        logger.warning(f"Unknown loader key: {loader_key}")
        return None


def _artifact_key(artifact_path: str) -> str:
    """Convert an artifact path like 'confusion_matrix/frequencies.png'
    to a template key like 'frequencies_png'."""
    name = Path(artifact_path).name
    return name.replace(".", "_")


# -- Data fetching --


def fetch_run_metadata(client: MlflowClient, run: Any) -> dict[str, Any]:
    """Extract run info, params, tags, and experiment name."""
    experiment = client.get_experiment(run.info.experiment_id)

    start_ms = run.info.start_time
    end_ms = run.info.end_time
    start_time = datetime.fromtimestamp(start_ms / 1000, tz=UTC) if start_ms else None
    end_time = datetime.fromtimestamp(end_ms / 1000, tz=UTC) if end_ms else None
    duration = None
    if start_time and end_time:
        delta = end_time - start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f"{hours}h {minutes}m {seconds}s"

    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name or run.info.run_id[:8],
        "experiment_name": experiment.name,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M UTC") if start_time else "N/A",
        "end_time": end_time.strftime("%Y-%m-%d %H:%M UTC") if end_time else "N/A",
        "duration": duration or "N/A",
        "status": run.info.status,
        "params": dict(run.data.params),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
    }


def fetch_scalar_metrics(run: Any) -> dict[str, Any]:
    """Organize run.data.metrics into named groups for the template.

    Each group is a list of (label, value) tuples, or None if no
    metrics in that group are present.
    """
    all_metrics = run.data.metrics

    def _build_group(spec: list[tuple[str, str]]) -> list[tuple[str, Any]] | None:
        items: list[tuple[str, Any]] = []
        for key, label in spec:
            if key in all_metrics:
                items.append((label, all_metrics[key]))
        return items if items else None

    return {
        "executive": _build_group(EXECUTIVE_METRICS),
        "topk": _build_group(TOPK_METRICS),
        "cover": _build_group(COVER_METRICS),
        "taxonomic": _build_group(TAXONOMIC_METRICS),
    }


def download_run_artifacts(run_id: str, dst_dir: Path) -> Path:
    """Download all non-model artifacts for a run to dst_dir.

    Uses list_artifacts() to enumerate top-level entries, then
    selectively downloads each one to avoid the large model pickle.
    Returns the directory where artifacts were downloaded.
    """
    client = MlflowClient()
    top_level = client.list_artifacts(run_id)

    for artifact in top_level:
        # Skip registered model artifacts (can be hundreds of MB).
        if artifact.path.startswith("model"):
            logger.info(f"Skipping model artifact: {artifact.path}")
            continue
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact.path,
            dst_path=str(dst_dir),
        )

    return dst_dir


def load_artifact_data(artifact_dir: Path) -> dict[str, Any]:
    """Load all known artifacts from the downloaded directory.

    Returns a nested dict suitable for the Jinja2 template context.
    """
    # Load evaluation sections (subdirectory-organized).
    sections = {}
    for section_id, section_def in EVALUATION_SECTIONS.items():
        section_data = {}
        has_any = False
        for artifact_path, loader_key in section_def["artifacts"]:
            key = _artifact_key(artifact_path)
            value = _load_artifact(artifact_dir, artifact_path, loader_key)
            section_data[key] = value
            if value is not None:
                has_any = True

        if has_any:
            section_data["title"] = section_def["title"]
            sections[section_id] = section_data
        elif not section_def.get("optional"):
            # Required section with no artifacts — include with empty data.
            section_data["title"] = section_def["title"]
            sections[section_id] = section_data

    # Load root-level evaluation artifacts.
    root_eval = {}
    for artifact_path, loader_key in ROOT_EVALUATION_ARTIFACTS:
        key = _artifact_key(artifact_path)
        root_eval[key] = _load_artifact(artifact_dir, artifact_path, loader_key)

    # Load training/config artifacts.
    training = {}
    has_training = False
    for artifact_path, loader_key in TRAINING_ARTIFACTS:
        key = _artifact_key(artifact_path)
        value = _load_artifact(artifact_dir, artifact_path, loader_key)
        training[key] = value
        if value is not None:
            has_training = True

    return {
        "sections": sections,
        "root_eval": root_eval,
        "training": training,
        "has_training": has_training,
    }


def build_template_context(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    title: str | None = None,
) -> dict[str, Any]:
    """Assemble everything into the context dict for Jinja2."""
    if title is None:
        title = f"Classifier Report - {metadata['experiment_name']} - {metadata['run_name']}"

    params = metadata.get("params") or {}
    n_classes = params.get("num_classes", "")
    n_predictions = params.get("num_predictions", "")

    return {
        "title": title,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "metadata": metadata,
        "metrics": metrics,
        "sections": artifacts["sections"],
        "root_eval": artifacts["root_eval"],
        "training": artifacts["training"],
        "has_training": artifacts["has_training"],
        "section_order": [
            "confusion_matrix",
            "calibration",
            "cover",
            "probability",
            "ranking",
            "taxonomic",
            "per_source",
        ],
        "n_classes": n_classes,
        "n_predictions": n_predictions,
    }


def render_report(context: dict[str, Any], output_path: Path) -> None:
    """Load the Jinja2 template, render with context, write to output_path."""
    template_dir = Path(__file__).parent
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
    )
    template = env.get_template("report_template.html.j2")
    html = template.render(**context)
    output_path.write_text(html)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Report written to {output_path} ({size_mb:.1f} MB)")


def main():
    set_env_vars_for_packages()
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML report from an MLflow run."
    )
    parser.add_argument("--run-id", required=True, help="MLflow run ID (32-char hex string)")
    parser.add_argument(
        "--output", default=None, help="Output HTML file path (default: report_<run_id>.html)"
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom report title (default: auto-generated from experiment/run)",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help=(
            "Override MLflow tracking URI (e.g. a SageMaker MLflow App ARN). "
            "Defaults to the MLFLOW_TRACKING_SERVER env var / .env value."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Connect to MLflow.
    connect_time = mlflow_connect(tracking_uri=args.mlflow_tracking_uri)
    logger.info(f"Connected to MLflow in {connect_time}")

    # Fetch run data.
    client = MlflowClient()
    run = client.get_run(args.run_id)
    logger.info(f"Fetched run {args.run_id} (status: {run.info.status})")

    metadata = fetch_run_metadata(client, run)
    metrics = fetch_scalar_metrics(run)

    # Download artifacts to a temp directory.
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = download_run_artifacts(args.run_id, Path(tmpdir))
        logger.info(f"Downloaded artifacts to {artifact_dir}")

        artifacts = load_artifact_data(artifact_dir)

    # Build context and render.
    context = build_template_context(metadata, metrics, artifacts, title=args.title)

    output_path = Path(args.output) if args.output else Path(f"report_{args.run_id[:8]}.html")
    render_report(context, output_path)


if __name__ == "__main__":
    main()
