# Training Config: coralnet_all_plus_mermaid_coarse_grouped_minsupport_gval_cca3_turf1p5_accstop

Rerun of `coralnet_all_plus_mermaid_coarse_grouped_minsupport_gval_cca3_turf1p5`
(MLflow run `fa4b78434b554a989de77999d01598eb`) with one change: early stopping
selects on validation accuracy (`training.early_stopping_metric: val_accuracy`)
with patience 8 instead of validation loss with patience 3.

## Why

Val comes from 112 sources/projects never used in training. In run fa4b7843,
`epoch/val_loss` was lowest at epoch 3 (1.951) and then rose (2.056 at epoch 5,
2.048 at epoch 6), while `epoch/val_accuracy` kept rising (0.556 at epoch 3,
0.573 at epoch 6). The model grows more confident everywhere, which raises the
loss on unseen-source points it gets wrong even as more points come out right,
so loss-based selection kept restoring epoch 3. Run 7a46d513 showed the same
pattern.

## Provenance

Every file except `training_config.yaml` and this README is a byte-identical
copy from `coralnet_all_plus_mermaid_coarse_grouped_minsupport_gval_cca3_turf1p5/`
(CCA x3, Turf x1.5, the same 112 val groups, holdout and 151 excluded MERMAID
images). `training_config.yaml` differs only in the two early-stopping keys, the
header comment and `mlflow.model_name`. Compare against fa4b7843 with
`scripts/compare_holdout_runs.py` to isolate the stopping change.
