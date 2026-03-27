# common

## Design Decisions

**BA+GF separator contract**: The `::` separator between benthic attribute and growth form is a hard contract used across the entire codebase. Empty growth forms must still have the trailing `::` (e.g., `Hard coral::`), not just the BA name alone. This ensures round-trip integrity via `combine_ba_gf()` and `split_ba_gf()`.

**NULL growth forms stored as empty string**: CoralNet labels without a growth form must be stored as `''` (empty string) in DuckDB, never NULL. NULL breaks DuckDB JOINs silently, producing incorrect results. Tests specifically verify this invariant.

**CsvSpec template method pattern**: `LabelFilter`, `LabelRollupSpec`, and `CNSourceFilter` all inherit from `CsvSpec` in `csv_utils.py`. The base class validates CSV column structure upfront, supports flexible column naming (multiple aliases per column via `ColumnSpec`), and delegates per-row processing to subclass implementations via `per_item_init_action()`. New CSV-based config loaders should follow this pattern.

**Singleton API clients**: `BenthicAttributeLibrary` and `GrowthFormLibrary` fetch from the live MERMAID API on instantiation, making construction expensive. They are intended as singletons -- reuse a single cached instance rather than constructing multiple times.

**Blank CSV cell semantics**: CSV cells are kept as empty strings (not NaN) to avoid mathematical context confusion. Check truthiness rather than NaN.
