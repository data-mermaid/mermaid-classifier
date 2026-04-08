# pyspacer

## Architecture

The training pipeline orchestrates two distinct data sources into a single training set:

- **CoralNet**: Per-source CSV files in S3, one CSV per source. Labels use CoralNet-specific IDs that must be mapped to MERMAID BA+GF taxonomy via `CoralNetMermaidMapping`.
- **MERMAID**: A single monolithic Parquet file in S3. Labels are already in BA+GF format.

Both sources are loaded into a shared DuckDB `annotations` table using SQL (not pandas) as the ETL engine. All filtering, rollups, and validation happen via DuckDB queries.

## Design Decisions

**Side-effect-on-import**: `__init__.py` calls `set_env_vars_for_packages()` immediately on import. This must run before any PySpacer or MLflow modules are imported. Code that uses pyspacer internals depends on this having executed first.

**Memory-efficient streaming**: `MermaidTrainer` in `trainer.py` uses a streaming design throughout. Training data is streamed from disk each epoch via `StreamingFeatureDataset(IterableDataset)`, yielding fixed-size minibatches without materializing the full dataset. Reference accuracy and calibration also stream features in batches, accumulating only scalar predictions. Memory usage is O(minibatch_size) instead of O(dataset_size). The `minibatch_size` parameter (default 512) controls both gradient update size and streaming chunk size.

**Settings case convention**: Setting names are lowercase in Python code but UPPERCASE in `.env` files. The `.env` file is loaded from the current working directory (via pydantic-settings), not from the package location. This is intentional for installed-package usage.

**MLflow run ID regex**: `annotation.py` uses a forgiving regex for MLflow model IDs, accepting 30-32 hex digits instead of exact length, to handle accidentally-truncated IDs in user input.

**Logging reset on each run**: `logging_config_for_script()` in `utils.py` clears previous logs (`mode='w'`) on each invocation, so log files start fresh per run.
