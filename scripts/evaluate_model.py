"""
Evaluate a pre-trained model's validation results using the metrics system
and log everything to MLflow.

Usage:
    cd mermaid-classifier

    # With MERMAID BA::GF class IDs (valresult.json):
    uv run python scripts/evaluate_model.py \
        --valresult ../v1_beta_model/valresult.json \
        --experiment "v1-beta-evaluation" \
        --run-name "v1-beta" \
        --model ../v1_beta_model/classifier.pkl \
        --model-name "v1-beta"

    # With CoralNet label IDs (valresult_labels.json), remapped via label map:
    uv run python scripts/evaluate_model.py \
        --valresult ../v1_beta_model/valresult_labels.json \
        --label-map ../v1_beta_model/LabelMap.csv \
        --class-map ../v1_beta_model/CoralNetMermaidMatchedCoralFocusModel2Reassign_master*.csv \
        --experiment "v1-beta-labels-evaluation" \
        --run-name "v1-beta-labels" \
        --model ../v1_beta_model/classifier_labels.pkl \
        --model-name "v1-beta-labels"
"""
import argparse
import csv
import json
import logging

import duckdb
import mlflow
import sklearn.metrics
from spacer.data_classes import DataLocation, ValResults
from spacer.storage import load_classifier

from mermaid_classifier.common.benthic_attributes import (
    BAGF_SEP,
    BenthicAttributeLibrary,
    GrowthFormLibrary,
)
from mermaid_classifier.pyspacer.metrics import MetricsContext, MetricsCoordinator
from mermaid_classifier.pyspacer.settings import set_env_vars_for_packages
from mermaid_classifier.pyspacer.utils import mlflow_connect


def format_metric(value: float) -> float:
    return round(float(value), 3)


def remap_classes(val_data: dict, label_map_path: str, class_map_path: str,
                  logger: logging.Logger) -> dict:
    """Remap class IDs from CoralNet label UUIDs to MERMAID BA::GF IDs.

    Uses two CSVs to build the mapping chain:
      label_map (LabelMap.csv): label_id -> CoralFocus3Label
      class_map (spreadsheet):  CoralFocus3Label -> (CoralFocus3_BaID, CoralFocus3_GfID)
    """
    # label_id -> CoralFocus3Label
    label_to_cf3 = {}
    with open(label_map_path) as f:
        for row in csv.DictReader(f):
            label_to_cf3[row['label_id']] = row['CoralFocus3Label']

    # CoralFocus3Label -> MERMAID BA::GF
    cf3_to_bagf = {}
    with open(class_map_path) as f:
        for row in csv.DictReader(f):
            label = row.get('CoralFocus3Label', '').strip()
            ba_id = row.get('CoralFocus3_BaID', '').strip()
            gf_id = row.get('CoralFocus3_GfID', '').strip()
            if label and ba_id and ba_id != 'NA' and label not in cf3_to_bagf:
                gf = gf_id if gf_id and gf_id != 'NA' else ''
                cf3_to_bagf[label] = BAGF_SEP.join([ba_id, gf])

    # Remap each class, merging classes that map to the same BA::GF.
    old_classes = val_data['classes']
    new_class_list = []  # ordered unique MERMAID BA::GF IDs
    new_class_index = {}  # BA::GF -> index in new_class_list
    old_to_new = {}  # old index -> new index

    for old_idx, label_id in enumerate(old_classes):
        cf3 = label_to_cf3.get(label_id)
        if cf3 is None:
            raise ValueError(
                f"Label ID {label_id!r} not found in {label_map_path}")
        bagf = cf3_to_bagf.get(cf3)
        if bagf is None:
            raise ValueError(
                f"CoralFocus3Label {cf3!r} (from {label_id!r}) not found in "
                f"{class_map_path}")

        if bagf not in new_class_index:
            new_class_index[bagf] = len(new_class_list)
            new_class_list.append(bagf)
        old_to_new[old_idx] = new_class_index[bagf]

    val_data['classes'] = new_class_list
    val_data['gt'] = [old_to_new[i] for i in val_data['gt']]
    val_data['est'] = [old_to_new[i] for i in val_data['est']]

    logger.info(
        f"Remapped {len(old_classes)} label IDs to "
        f"{len(new_class_list)} MERMAID BA::GF classes")
    return val_data


def main():
    set_env_vars_for_packages()
    parser = argparse.ArgumentParser(
        description="Evaluate validation results and log metrics to MLflow.")
    parser.add_argument(
        "--valresult", required=True,
        help="Path to valresult.json file")
    parser.add_argument(
        "--experiment", default="v1-beta-evaluation",
        help="MLflow experiment name (default: v1-beta-evaluation)")
    parser.add_argument(
        "--run-name", default=None,
        help="MLflow run name (default: auto-generated)")
    parser.add_argument(
        "--model", default=None,
        help="Path to a classifier pickle file to register in MLflow")
    parser.add_argument(
        "--model-name", default=None,
        help="MLflow registered model name (required if --model is given)")
    parser.add_argument(
        "--label-map", default=None,
        help="Path to LabelMap.csv (columns: label_id, CoralFocus3Label). "
             "Remaps CoralNet label UUIDs to CoralFocus3Label names. "
             "Requires --class-map.")
    parser.add_argument(
        "--class-map", default=None,
        help="Path to class mapping CSV (columns: CoralFocus3Label, "
             "CoralFocus3_BaID, CoralFocus3_GfID). Maps CoralFocus3Label "
             "names to MERMAID BA::GF IDs. Requires --label-map.")
    args = parser.parse_args()

    if args.model and not args.model_name:
        parser.error("--model-name is required when --model is given")
    if bool(args.label_map) != bool(args.class_map):
        parser.error("--label-map and --class-map must be used together")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    # 1. Load validation results from local JSON.
    logger.info(f"Loading validation results from {args.valresult}")
    with open(args.valresult) as f:
        val_data = json.load(f)

    # Remap CoralNet label UUIDs to MERMAID BA::GF IDs if mapping files given.
    if args.label_map and args.class_map:
        val_data = remap_classes(val_data, args.label_map, args.class_map,
                                logger)
    # If classes are bare BA UUIDs (no :: separator), append :: to make
    # them valid BA::GF IDs with empty growth form.
    elif val_data['classes'] and BAGF_SEP not in val_data['classes'][0]:
        val_data['classes'] = [c + BAGF_SEP for c in val_data['classes']]
        logger.info("Appended '::' to class IDs (BA-only format detected)")

    val_results = ValResults(**val_data)
    logger.info(
        f"Loaded {len(val_results.gt)} predictions across "
        f"{len(val_results.classes)} classes")

    # 2. Fetch BA and GF libraries from MERMAID API.
    logger.info("Fetching BenthicAttributeLibrary from MERMAID API...")
    ba_library = BenthicAttributeLibrary()
    logger.info("Fetching GrowthFormLibrary from MERMAID API...")
    gf_library = GrowthFormLibrary()

    # 3. Build MetricsContext (no dataset, no clf).
    ctx = MetricsContext(
        val_results=val_results,
        ba_library=ba_library,
        gf_library=gf_library,
        format_func=format_metric,
    )

    # 4. Optionally load a pre-trained classifier for MLflow registration.
    clf = None
    if args.model:
        logger.info(f"Loading classifier from {args.model}")
        loc = DataLocation('filesystem', args.model)
        clf = load_classifier(loc)
        logger.info(f"Loaded classifier: {type(clf).__name__}")

    # 5. Connect to MLflow and create a run.
    logger.info("Connecting to MLflow...")
    mlflow_connect()
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run: {run_id}")

        mlflow.log_param("valresult_source", args.valresult)
        mlflow.log_param("num_predictions", len(val_results.gt))
        mlflow.log_param("num_classes", len(val_results.classes))

        # Overall accuracy (normally comes from pyspacer's return_msg).
        accuracy = sum(g == e for g, e in zip(val_results.gt, val_results.est)) / len(val_results.gt)
        mlflow.log_metric("accuracy", format_metric(accuracy))
        logger.info(f"Accuracy: {format_metric(accuracy)}")

        # 5. Compute and log all applicable metrics.
        # Always log basic sklearn metrics (no BA library needed).
        gt_labels = [val_results.classes[i] for i in val_results.gt]
        est_labels = [val_results.classes[i] for i in val_results.est]

        balanced_acc = sklearn.metrics.balanced_accuracy_score(
            gt_labels, est_labels)
        precision_macro = sklearn.metrics.precision_score(
            gt_labels, est_labels, average='macro', zero_division=0.0)
        recall_macro = sklearn.metrics.recall_score(
            gt_labels, est_labels, average='macro', zero_division=0.0)
        if (precision_macro + recall_macro) > 0:
            f1_macro = (2 * precision_macro * recall_macro
                        / (precision_macro + recall_macro))
        else:
            f1_macro = 0.0
        mcc = sklearn.metrics.matthews_corrcoef(gt_labels, est_labels)

        for name, value in [
            ('balanced_accuracy', balanced_acc),
            ('precision_macro', precision_macro),
            ('recall_macro', recall_macro),
            ('f1_macro', f1_macro),
            ('mcc', mcc),
        ]:
            mlflow.log_metric(name, format_metric(value))
            logger.info(f"{name}: {format_metric(value)}")

        # Run the full metrics pipeline (confusion matrices, taxonomic,
        # calibration, etc.) if BA library can resolve the class IDs.
        duck_conn = duckdb.connect()
        coordinator = MetricsCoordinator(ctx, duck_conn=duck_conn)
        coordinator.compute_and_log_all()
        duck_conn.close()

        # Register the pre-trained model (same pattern as training).
        if clf is not None:
            logger.info(f"Registering model as '{args.model_name}'...")
            model_info = mlflow.sklearn.log_model(
                sk_model=clf,
                registered_model_name=args.model_name,
            )
            logger.info(f"Model ID: {model_info.model_id}")

        logger.info("Metrics logged successfully.")


if __name__ == "__main__":
    main()
