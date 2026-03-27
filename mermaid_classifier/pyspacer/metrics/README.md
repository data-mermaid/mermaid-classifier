# metrics

## Architecture

`MetricsCoordinator` orchestrates all metric computation using a dispatcher pattern. It validates the `MetricsContext`, builds taxonomy caches, then conditionally calls specialized metric functions (calibration, classification, cover, probability, ranking, taxonomic). Each metric group returns a `MetricGroupResult` that can contain heterogeneous result types (scalars, figures, DataFrames, dicts).

## Design Decisions

**Coordinator as single entry point**: All metrics are computed through `MetricsCoordinator`, not by calling individual metric functions directly. This ensures consistent context validation and taxonomy cache setup.

**MLflow CSV workaround**: MLflow natively outputs JSON for tabular data. `_logging.py` works around this by using DuckDB's COPY to generate CSV, then logging via `mlflow.log_text()`. This provides better compatibility with external analysis tools.

**Hierarchical confusion matrix reordering**: `classification.py` reorders confusion matrix rows and columns by clustering normalized prediction profiles using cosine distance. This reveals block-diagonal structure where related classes cluster together, making the matrix more interpretable.

**LCA-based error attribution**: `taxonomic.py` maps each misclassification to its lowest common ancestor in the BA hierarchy, grouping confusions by how deep in the taxonomy the prediction diverges from ground truth. This distinguishes "close" errors (e.g., two coral species) from "far" errors (e.g., coral vs. algae).

## Invariants

- `MetricsContext` validates that class indices are in range and mappable to the BA library. Metrics functions assume this validation has already passed.
- `val_results.scores` contains max probabilities only (not full probability vectors). Metrics that need full probability distributions use `val_proba`, which is pre-computed by the coordinator.
