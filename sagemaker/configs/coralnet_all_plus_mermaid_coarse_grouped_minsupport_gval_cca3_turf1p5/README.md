# Training Config: coralnet_all_plus_mermaid_coarse_grouped_minsupport_gval_cca3_turf1p5

Follow-up to `coralnet_all_plus_mermaid_coarse_grouped_minsupport_gval_turfcca`
(MLflow run `7a46d513b2ac40ed99300b8423473e7f`) with one change: the Turf algae
subsample multiplier drops from 3.0 to 1.5 (target 352,500). Crustose
coralline algae stays at 3.0 (705,000). Early stopping is unchanged
(patience 3).

## Why

On the source-grouped holdout, run 7a46d513 raised Turf recall from 36% to
62% against the min-support run `31c89e1bdc8b4c699161be40ba701632`, but it
over-predicted Turf (per-image cover bias +6.4 points) and pulled points away
from macroalgae (recall 51% to 39%), bare substrate and cyanobacteria. CCA
gained recall (52% to 64%) with flat precision. This run tests whether a
smaller Turf boost keeps part of the Turf gain without that leakage.

## Provenance

Every file except `training_config.yaml`, `excluded_images.csv` and this
README is a byte-identical copy from
`coralnet_all_plus_mermaid_coarse_grouped_minsupport_gval_turfcca/`, including
`val_groups.csv` (112 groups) and `holdout_groups.csv`. The
`training_config.yaml` differs only in the Turf multiplier, the header comment
and `mlflow.model_name`. See that directory's README for how `val_groups.csv`
was generated.

`excluded_images.csv` adds 100 MERMAID images (151 in all) that reached the
MERMAID training data on 2026-10-20/21, after the 2026-10-19 project lookup, so
the lookup leaves them unmatched and the run refuses to start. Excluding them
keeps the training data the same as run 7a46d513's. They were listed with the
pipeline's own loader as the MERMAID images with `project_id = 'unknown'` after
filters.

## Comparing

Same holdout as both earlier runs. Against 7a46d513 the only difference is the
Turf multiplier; against 31c89e1b the differences are grouped val, CCA x3 and
Turf x1.5.

    uv run python scripts/compare_holdout_runs.py \
        --run-a 7a46d513b2ac40ed99300b8423473e7f --run-b <new run id> \
        --top-level-rollup-csv ../sagemaker/labels/ba_rollup_top_level.csv \
        --top-level-classes-csv ../sagemaker/labels/top_level_all.csv \
        --real-mix --check-groups --mlflow-tracking-uri <arn> --out-dir <dir>
