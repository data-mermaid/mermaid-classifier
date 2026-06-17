# pyspacer

## Architecture

The training pipeline orchestrates two distinct data sources into a single training set:

- **CoralNet**: Per-source CSV files in S3, one CSV per source. Labels use CoralNet-specific IDs that must be mapped to MERMAID BA+GF taxonomy via `CoralNetMermaidMapping`.
- **MERMAID**: A single monolithic Parquet file in S3. Labels are already in BA+GF format.

Both sources are loaded into a shared DuckDB `annotations` table using SQL (not pandas) as the ETL engine. All filtering, rollups, and validation happen via DuckDB queries.

## Design Decisions

**Side-effect-on-import**: `__init__.py` calls `set_env_vars_for_packages()` immediately on import. This must run before any PySpacer or MLflow modules are imported. Code that uses pyspacer internals depends on this having executed first.

**Memory-efficient streaming in trainer**: `MermaidTrainer` in `trainer.py` uses a streaming design -- reference accuracy and calibration both load features in batches from disk, accumulating only scalar predictions. Reference and training data are never held in memory simultaneously. This prevents OOM on large datasets.

**Automatic batch size**: `Settings.automatic_batch_size()` reserves 3GB of overhead and assumes peak memory is batch size + 60% for sklearn buffers. These constants are tuned specifically for EfficientNet feature vectors and may need adjustment for different extractors.

**Settings case convention**: Setting names are lowercase in Python code but UPPERCASE in `.env` files. The `.env` file is loaded from the current working directory (via pydantic-settings), not from the package location. This is intentional for installed-package usage.

**MLflow run ID regex**: `annotation.py` uses a forgiving regex for MLflow model IDs, accepting 30-32 hex digits instead of exact length, to handle accidentally-truncated IDs in user input.

**Logging reset on each run**: `logging_config_for_script()` in `utils.py` clears previous logs (`mode='w'`) on each invocation, so log files start fresh per run.

**Feature caching: ours vs. PySpacer's**: There are two distinct feature-caching mechanisms, and we deliberately use ours instead of PySpacer's.

- *Ours* (`settings.feature_cache_dir`, set up in `TrainingDataset.__init__`): we **eagerly bulk-download** every feature vector from S3 in parallel (`download_features_parallel`) *before* calling PySpacer, then hand PySpacer `filesystem` `DataLocation`s pointing at the local files. If `feature_cache_dir` is set, the downloads land in that directory with `exist_ok=True`, so they **persist and are reused across runs** (a true on-disk cache). If it's unset, we download into a `TemporaryDirectory` that is cleaned up when the run ends.
- *PySpacer's* (`TrainClassifierMsg.feature_cache_dir`): a **lazy** cache that only activates when `labels.has_remote_data` is true. It creates a temp dir (inside the given path, or OS-chosen for `AUTO`) and caches remotely-loaded vectors *during* training so they aren't re-fetched each epoch. It is **always ephemeral** — the temp dir is deleted when training finishes, so even pointing it at a fixed path does not persist across runs.

Because we pre-download and pass `filesystem` locations, `labels.has_remote_data` is false and PySpacer's cache would be a no-op anyway. We pass `feature_cache_dir=TrainClassifierMsg.FeatureCache.DISABLED` explicitly to make that intent clear. Net effect: our cache controls the cross-run, parallel-download behavior; PySpacer's would only matter if we let it load from S3 itself, which we don't.
