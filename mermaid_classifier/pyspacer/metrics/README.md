# metrics

## Architecture

`MetricsCoordinator` orchestrates all metric computation using a dispatcher pattern. It validates the `MetricsContext`, builds taxonomy caches, then conditionally calls specialized metric functions (calibration, classification, cover, per_source, probability, ranking, region, region_probe, taxonomic). Each metric group returns a `MetricGroupResult` that can contain heterogeneous result types (scalars, figures, DataFrames, dicts).

`region` computes out-of-region rates over the run's own validation split. It requires a dataset and returns an empty result when no validation image carries a region — the quiet no-op for configs that train on CoralNet alone, which is most of them.

`region_probe` scores a frozen, versioned probe set through `ctx.clf`, the predictor already built from the exported artifact, gated by a `requires_clf` flag on `MetricGroupSpec`. It is skipped silently when no probe is configured, but raises when one is configured and missing, because scoring zero points would read as a model with no incidents.

The two prefixes `region_val/*` and `region_probe/*` deliberately never merge: they measure different denominators over different populations, and identical names over incomparable denominators is how an evaluation program rots.

## Design Decisions

**Coordinator as single entry point**: All metrics are computed through `MetricsCoordinator`, not by calling individual metric functions directly. This ensures consistent context validation and taxonomy cache setup.

**MLflow CSV workaround**: MLflow natively outputs JSON for tabular data. `_logging.py` works around this by using DuckDB's COPY to generate CSV, then logging via `mlflow.log_text()`. This provides better compatibility with external analysis tools.

**Hierarchical confusion matrix reordering**: `classification.py` reorders confusion matrix rows and columns by clustering normalized prediction profiles using cosine distance. This reveals block-diagonal structure where related classes cluster together, making the matrix more interpretable.

**LCA-based error attribution**: `taxonomic.py` maps each misclassification to its lowest common ancestor in the BA hierarchy, grouping confusions by how deep in the taxonomy the prediction diverges from ground truth. This distinguishes "close" errors (e.g., two coral species) from "far" errors (e.g., coral vs. algae).

**Intervals resample images, not points**: Annotation points cluster roughly 25 to an image, so a point-level interval is several-fold too narrow — every confidence interval in `region.py` and `region_probe.py` resamples whole images instead. Design effects measured on a real run ranged 1.87–4.73.

**A metric group must not rely on raising**: `MetricsCoordinator` wraps every group in `except Exception: logger.warning(...)`, so an exception inside a group is swallowed and its metrics simply vanish behind a log line. `region.py` and `region_probe.py` filter unscorable points up front and emit the excluded count as a metric instead.

## Invariants

- `MetricsContext` validates that class indices are in range and mappable to the BA library. Metrics functions assume this validation has already passed.
- `val_results.scores` contains max probabilities only (not full probability vectors). Metrics that need full probability distributions use `val_proba`, which is pre-computed by the coordinator.
