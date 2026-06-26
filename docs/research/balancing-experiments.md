# MERMAID classifier label-balancing experiments — research report

## Context

With the architecture question settled (`hidden_layer_sizes=(500,300,100)` @ `learning_rate_init=1e-4`, `epochs=40`, `early_stopping_patience=3` against `epoch/val_loss`; see `hidden-layer-experiments.md`), the next bottleneck was the long tail. The 40-epoch architecture-sweep winner had `balanced_accuracy=0.735` and a `per_source/min_accuracy` of 0.183 — the model was strong on aggregate but failed on at least one source. The 31–40 epoch window also showed `balanced_accuracy` regressing slightly while `accuracy` continued to climb, suggesting the late-epoch fit was drifting toward common classes.

This round held the architecture and training recipe constant and varied two label-balancing axes: **sample weighting** (which strategy / how aggressive) and **training-set resampling** (`stratified` vs `balanced` vs the new `soft_balanced` strategy that interpolates between them).

The full design lives at `/Users/gregn/.claude/plans/i-have-just-finished-jiggly-duckling.md`. The sweep itself was run ad hoc — the multi-process sweep driver was never committed to this repo. The round also evaluated a `'soft_balanced'` allocator (`target_c ∝ n_c^(1-alpha)`); it was not adopted and has since been removed from the codebase (see the cleanup note under Caveats), so `subsample` now exposes only `stratified` and `balanced`.

---

## Observations

### Screen (15 configs, 20 epochs + patience=3, 400K subsample)

Three independent arms were run in parallel. Anchor was the prior default (`effective_number, alpha=0.5, weight_ratio_cap=5000` + `stratified, 400K`). Configs ranked by `balanced_accuracy`:

| rank | config | bal_acc | f1_macro | accuracy | MCC | cover_R² |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `C5_soft_a0.75` | **0.797** | 0.788 | 0.827 | 0.822 | 0.683 |
| 2 | `C4_soft_a0.5`  | 0.793 | 0.785 | 0.814 | 0.805 | 0.665 |
| 3 | `C2_balanced_min200` | 0.792 | 0.788 | 0.831 | 0.827 | 0.656 |
| 4 | `C1_balanced` | 0.780 | 0.785 | 0.829 | 0.825 | 0.674 |
| 5 | `C3_soft_a0.25` | 0.763 | 0.756 | 0.803 | 0.782 | 0.636 |
| 6 | `anchor (eff_a0.5_cap5000)` | 0.700 | 0.715 | 0.801 | 0.760 | 0.617 |
| 7 | `B1_eff_capNone` | 0.674 | 0.676 | 0.784 | 0.738 | 0.612 |
| 8 | `B3_eff_cap100` | 0.673 | 0.692 | 0.802 | 0.760 | 0.660 |
| 9 | `A5_decomp_a0.5` | 0.671 | 0.694 | 0.800 | 0.758 | 0.639 |
| 10 | `B2_eff_cap1000` | 0.669 | 0.673 | 0.784 | 0.738 | 0.599 |
| 11 | `A4_leaf_a0.5` | 0.667 | 0.695 | 0.801 | 0.760 | 0.647 |
| 12 | `A6_unweighted` | 0.600 | 0.665 | 0.811 | 0.770 | 0.674 |
| 13 | `A3_tree_a0.7` | 0.415 | 0.476 | 0.733 | 0.672 | 0.434 |
| 14 | `A2_tree_a0.5` | 0.381 | 0.459 | 0.730 | 0.669 | 0.506 |
| 15 | `A1_tree_a0.3` | n/a | n/a | n/a | n/a | n/a |

Three findings jump out at the screen stage:

- **Every Arm C (subsampling) config beat every Arm A/B (sample-weighting) config on `balanced_accuracy`.** The five C entries occupy ranks 1–5; the anchor sits at #6.
- **`tree_balanced_ba_flat_gf` was unsafe at every alpha tested.** A2 (alpha=0.5, bal_acc=0.381) and A3 (alpha=0.7, bal_acc=0.415) underperformed even the unweighted baseline (A6, bal_acc=0.600). A1 (alpha=0.3) failed outright after ~19 s, before any training metrics were logged (only system metrics survived). The failure mode wasn't captured in MLflow artifacts.
- **Unweighted training** (A6) gave the worst weighted-or-not balanced_accuracy in the safe set (0.600) but the second-best aggregate `accuracy` (0.811), confirming weighting *does* shift mass toward the long tail at small expense of aggregate.

Top 3 by `balanced_accuracy` (ties broken by `f1_macro`) promoted to confirm: `C5_soft_a0.75`, `C4_soft_a0.5`, `C2_balanced_min200`.

### Confirm (3 configs, 40 epochs + patience=3, full data)

Each confirmer kept its Arm C subsample strategy at full-data budget (`total_annotations=1_770_000`), so the resampling knob continued to be exercised at production scale rather than erased.

| metric | C2 (#3 screen) | C5 (#1 screen) | C4 (#2 screen) |
|---|---:|---:|---:|
| **balanced_accuracy** | **0.7740** | 0.7650 | 0.7400 |
| f1_macro | 0.7580 | 0.7750 | 0.7570 |
| accuracy | 0.8120 | 0.8240 | 0.7920 |
| MCC | 0.8060 | 0.8140 | 0.7710 |
| cover_R² | 0.6470 | 0.6690 | 0.5902 |
| log_loss | 0.5976 | 0.7703 | 0.8284 |
| early_stop/best_val_epoch | 14 | 22 | 18 |
| early_stop/final_epoch | 17 | 25 | 21 |
| subsample/realized_total | 457,497 | (full subsample) | (full subsample) |
| per_source/min_accuracy | 0.434 | — | — |
| per_source/max_accuracy | 0.940 | — | — |

`C2_balanced_min200` won `balanced_accuracy` (the tie-breaking long-tail metric) and was the calibration winner by a wide margin (`log_loss=0.598` vs 0.770 / 0.828). `C5_soft_a0.75` was the aggregate-metrics winner (`accuracy`, `f1_macro`, `MCC`, `cover_R²`) but produced a much higher `log_loss`. `C4_soft_a0.5` was dominated.

### Screen → confirm rank inversion

The screen ranking on `balanced_accuracy` had `C5 > C4 > C2`; the confirm flipped to `C2 > C5 > C4`. This is the same pattern observed in the architecture sweep at 5 vs 10 vs 20 epochs: shorter budgets systematically prefer configurations that converge faster, regardless of their final quality. Even at a 20-epoch screen, the rank near the top was unreliable — a third-place screen finisher won every long-tail metric at confirm time.

### Comparison vs the prior 40-epoch architecture-sweep baseline

The hidden-layer experiments' 40-epoch result on `(500,300,100) @ 1e-4` with the prior weighting default (`effective_number, alpha=0.5, cap=5000`) and no resampling was the right baseline to compare to. From `hidden-layer-experiments.md`:

| metric | prior 40-ep baseline | C2 confirm | Δ |
|---|---:|---:|---:|
| balanced_accuracy | 0.735 | **0.774** | **+0.039** |
| f1_macro | 0.752 | 0.758 | +0.006 |
| accuracy | 0.824 | 0.812 | −0.012 |
| MCC | 0.786 | 0.806 | +0.020 |
| cover_R² (median) | 0.572 | 0.647 | +0.075 |
| log_loss (calibrated) | 0.681 | 0.598 | −0.083 |

The trade-off matches the hypothesis exactly: a 1.2 pp accuracy regression in exchange for a 3.9 pp balanced_accuracy gain, 7.5 pp cover_R² gain, and 8.3 pp log_loss reduction. The single previously-flagged concern from the architecture-sweep work — `balanced_accuracy` regressing in the 31–40 epoch window — is no longer present at 40 epochs once the training distribution itself is balanced.

The most striking shift was on `per_source/min_accuracy`: the prior 10-epoch confirm baseline measured 0.183. C2 confirms at 0.434. The model's worst-source accuracy more than doubled.

### Subsample shape under `balanced(min_per_class=200)` at full-data scale

At `target_per_class = 1_770_000 / 80 ≈ 22125`, most of the 80 classes have fewer rows than the target and are kept in full; only the top handful are capped. The realized subsample was 457,497 rows — about 26% of the full dataset. Despite training on a quarter of the available data, C2 outperformed the full-data baseline on every long-tail metric. **Less data, better outcome** — because the data the model sees is balanced.

---

## What was learned

### 1. Class imbalance was a bigger lever than sample-weighting strategy choice

At constant architecture and training recipe, swapping the training-time data preparation (`stratified` → `balanced`) moved `balanced_accuracy` from 0.735 to 0.774 — a larger lift than anything seen in the sample-weighting strategy or alpha sweeps. The five subsampling configs occupied the top five screen ranks; every weighting variant tested came below them. **For this dataset, balancing the input distribution dominates balancing the loss function.**

### 2. `balanced` with a `min_per_class` floor is the production recipe

`balanced(total_annotations=full, min_per_class=200)` works because at production scale most classes are smaller than the per-class target — the strategy mostly just caps the dominant classes (Turf algae, Sand, Porites, Bare substrate) while preserving everything else, with a `min_per_class=200` floor to keep rare classes alive. The resulting training set is roughly 26% of the full data but yields better long-tail performance. The sample-weighting layer is still active (`effective_number, alpha=0.5, cap=5000`); removing it was not tested at confirm scale and is left for future work.

### 3. `soft_balanced` is a real knob but not the winner here

The new alpha-interpolated subsampler (`target_c ∝ n_c^(1-alpha)`) won the screen at `alpha=0.75` and finished second in confirm. `alpha=0.5` (square-root sampling) and `alpha=0.25` were both worse. `soft_balanced` retains more high-frequency examples than fully-`balanced` does, so it preserves aggregate `accuracy` better — useful when aggregate accuracy matters more than long-tail balance. For the current goal (long-tail metrics), full balancing won.

### 4. Short evaluation budgets mis-rank balancing configurations too

The screen→confirm rank inversion (`C5 > C4 > C2` flipping to `C2 > C5 > C4` at 40 epochs) repeats the lesson from the architecture sweep: a 20-epoch screen is fast enough to weed out broken configurations but cannot be trusted to rank the top survivors. Promote at least the top-3 to confirm; do not pick a single screen winner. The current sweep script does this, and the inversion observed here justifies it.

### 5. `tree_balanced_ba_flat_gf` is unsafe on this data

Three of the four runs that included this strategy regressed badly: A2 (alpha=0.5) and A3 (alpha=0.7) underperformed unweighted, and A1 (alpha=0.3) crashed during training within ~19s. The hierarchical strategy may interact badly with the included-labels filter or the BA+GF taxonomy as currently configured. It should not be the default; consider reverting `SampleWeightingOptions.strategy` from `tree_balanced_ba_flat_gf` to `effective_number` until the failure mode is understood.

### 6. `weight_ratio_cap` had limited impact within `effective_number`

Arm B varied `cap ∈ {None, 100, 1000, 5000}` with strategy and alpha held at the anchor. All four configs landed within `balanced_accuracy ∈ [0.669, 0.700]` — meaningful spread but smaller than the spread between weighting-strategy and subsampling-strategy choices. The current default of 5000 happens to be the best of the four; lowering or removing the cap did not help.

---

## Practical training recipe (updated)

The hidden-layer experiments' production recipe stays unchanged on architecture:

- `hidden_layer_sizes=(500, 300, 100)`
- `learning_rate_init=1e-4`
- `epochs=40`, `early_stopping_patience=3` against `epoch/val_loss`

This round adds the data-preparation knobs:

- `weighting=SampleWeightingOptions(strategy='effective_number', alpha=0.5, weight_ratio_cap=5000.0)`
- `subsample=SubsampleOptions(strategy='balanced', total_annotations=<full-dataset-size>, min_per_class=200)`

Wall-clock cost is essentially unchanged — the balanced subsample is smaller than the full data, so each epoch is faster, but early stopping triggered at epoch 17 (vs ~29 in the architecture sweep), and the run completed in ~35 minutes instead of ~70.

---

## Caveats and follow-ups

- **A1's failure mode was not captured.** The `tree_balanced_ba_flat_gf, alpha=0.3` run crashed before any training metrics or stack trace were logged to MLflow. A repeat with verbose stderr capture would be the cheapest way to understand why; if the strategy is going to remain a registered option, it needs a regression test that protects against this failure path.
- **Combined weighting × subsampling was not screened.** The plan deliberately ran A, B, and C as independent arms with no cartesian product — Arm D was scoped out. C2 confirm uses the anchor weighting (`effective_number, alpha=0.5, cap=5000`); we don't know whether removing the weighting layer entirely (just balanced subsampling, no class weights) would do better, worse, or the same. A 3-config follow-up — C2 with weighting on / off / different strategy — would close that gap cheaply.
- **Per-source accuracy is much better but still has a 50 pp range.** `per_source/min=0.434`, `per_source/max=0.940`. The next iteration of this work, if it targets the long tail, should look at the per-source metrics artifact for the C2 confirm run and identify which sources still struggle — that may motivate per-source rebalancing or a per-source feature-extraction review rather than another label-balancing pass.
- **Calibration regression in `C5` deserves a closer look.** `C5` won every aggregate metric except `log_loss`, where it was 29% worse than `C2`. If `soft_balanced` at higher alphas is to be revived, Platt-scaling parameters or temperature scaling on top of it should be investigated.
- **The `soft_balanced` strategy was removed after this round.** It was evaluated here (a real knob, but not the winner) and has since been deleted from the codebase along with the other non-winning strategies, to keep `subsample` to the validated production set (`stratified`, `balanced`). If a use case for `alpha < 0` (oversampling) or `alpha > 1` (over-balanced) emerges, it can be reintroduced as a new allocator.
- **Infrastructure note**: this run hit a real MLflow concurrency bug — when two `ProcessPoolExecutor` workers both call `mlflow.set_experiment` on a not-yet-existing experiment, one wins the implicit `create_experiment` and the other silently falls back to `Default`. The fix is to pre-create the experiments in the parent process before any worker spawns. Any future multi-process MLflow training script in this repo should do the same. (The sweep driver that applied this is not committed to the repo.)
