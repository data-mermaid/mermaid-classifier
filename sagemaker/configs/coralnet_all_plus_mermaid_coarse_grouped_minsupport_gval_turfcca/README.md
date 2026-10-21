# Training Config: coralnet_all_plus_mermaid_coarse_grouped_minsupport_gval_turfcca

One combined experiment on top of `coralnet_all_plus_mermaid_coarse_grouped_minsupport`.
Baseline for comparison: MLflow run `31c89e1bdc8b4c699161be40ba701632` (the
min-support run), same test fold.

## Provenance

`rollups.csv`, `included_labels.csv`, `excluded_images.csv` and
`holdout_groups.csv` are byte-identical copies from
`sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped_minsupport/`.
`support_counts.csv` is not copied. `training_config.yaml` is the min-support
config with these changes only:

1. Source-grouped validation (`dataset.val_groups_csv`, commit e1d5123).
   `val_max_per_class` is 5000, larger than the holdout's 3000, because val
   drives epoch selection. `ref_val_ratios` is `[0.1, 0.0]` because val comes
   from `val_groups.csv`.
2. Turf algae and Crustose coralline algae subsample targets x3
   (`subsample.class_target_multipliers`, commit 87bbe98). The balanced cap is
   18.8M // 80 = 235,000, so each class gets 705,000. Turf kept 5% and CCA kept
   18% of available points in run 31c89e1b.

## val_groups.csv

Generated with:

```
AWS_PROFILE=wcs-admin uv run --no-sync python scripts/select_val_groups.py --config-dir sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped_minsupport_gval_turfcca --min-val-points 100 --max-val-point-frac 0.20
```

`--max-val-point-frac 0.20` is needed because the default 0.15 reached only
94 of 107 CoralNet val groups, and the script wrote nothing. Other options are
defaults. `val_groups_coverage.csv` is the script's coverage report.

Printed result: 112 val groups, and 0 classes with 0 val points.

```
Val share of non-test groups and points, per site:
  coralnet: 107/1067 groups (10.0%), 2521510/15108893 points (16.7%)
  mermaid: 5/47 groups (10.6%), 46914/419114 points (11.2%)
```

The val groups are disjoint from `holdout_groups.csv`. The holdout is unchanged.

## Reading the results

Val is still the model-selection set (epoch selection, early stopping), so its
metrics are optimistic. The holdout is the number to report. Compare runs with
`scripts/compare_holdout_runs.py`.

This is one combined run, so the effects of the two changes are not separable.
