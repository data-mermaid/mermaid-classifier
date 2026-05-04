# MERMAID classifier hidden-layer experiments — research report

## Context

We ran a multi-stage architectural search to determine the optimal MLP head configuration for the PySpacer-based classifier on CoralNet-derived training data, focusing on `hidden_layer_sizes` and `learning_rate_init`. The search expanded from a small benchmark (10 sources, 51 classes, ~336K annotations) to the production-scale label set (20 sources, 80 classes, ~1.77M annotations) and then probed training-budget effects (5, 10, 20, 40 epochs) once the architectural ordering was established.

This report summarizes what was observed and what those observations mean for future training.

---

## Observations

### Small-scale sweep (10 sources, 51 classes, 10 epochs)

A 4-architecture × 2-learning-rate grid was run on the original benchmark dataset. Findings:

- **`(500, 300, 100) @ lr=1e-4`** was the aggregate-metric winner (accuracy 0.828, MCC 0.783, cover_R² 0.680).
- **`(200, 100) @ lr=1e-4`** was the long-tail winner (balanced_accuracy 0.703, recall_macro 0.703).
- **`lr=1e-3` was unstable for deeper architectures**: one outright training failure (`(500,300,100)`), one final-epoch ref-accuracy collapse (`(300,200,100)`), one mid-training loss spike (`(200,100)`). Only the shallowest MLP (`(100,)`) trained stably at the higher learning rate.
- A counterintuitive **invariance** appeared: top_K accuracy, MRR, log_loss, and hierarchical similarity were identical across all 7 trained heads to four decimal places. Investigation traced this to high-volume classes (Turf algae, Sand, Porites, Bare substrate) constituting ~75% of the validation samples and saturating those metrics regardless of architecture.

### First scale-up to 20 sources / 80 classes — confounded

A two-stage screen-then-confirm pipeline was set up to retest the small-sweep findings on the larger dataset. The first attempt produced uncomparable results:

- The three screen runs nominally received the same `annotation_limit=400000`, yet they trained on **different data**: 5, 6, and 12 sources respectively, and 73, 75, and 77 BA+GF classes. The largest single-source contribution varied threefold between runs.
- Per-source totals showed completely disjoint distributions per run. One run got 156K rows from a single source; another got 0 from that same source.
- The ranking on every primary metric inverted relative to the small sweep, but the inversion couldn't be trusted because the runs weren't comparing the same problem.

The root cause was traced to `LIMIT` without `ORDER BY` in the data-prep SQL, which is non-deterministic under DuckDB parallel execution. **Replacing it with deterministic per-class stratified subsampling** restored cross-run identity: re-run screens produced byte-identical 399,998-row, 81-class, 35,761-image subsamples across all three configs.

### Large-scale screen (5 epochs, 400K stratified subsample, deterministic)

After the determinism fix, `(100,) @ lr=1e-3` won every primary metric:

| metric | (200,100)@1e-4 | (500,300,100)@1e-4 | (100,)@1e-3 |
|---|---:|---:|---:|
| accuracy | 0.752 | 0.768 | **0.772** |
| balanced_accuracy | 0.618 | 0.644 | **0.689** |
| cover_R² | 0.510 | 0.517 | **0.581** |

Per-epoch loss curves at 5 epochs told a different story: `(500,300,100)`'s loss was still descending steeply (0.93 → 0.73 over the last two epochs), whereas `(100,)`'s loss had largely flattened. The shallow architecture's apparent "win" coincided with it being the only one that had finished its useful descent.

### Large-scale confirm (10 epochs, full data, all 3 configs)

When all three configs ran for the full epoch budget on the full dataset, **the ordering inverted on every primary metric**:

| metric | (200,100)@1e-4 | (500,300,100)@1e-4 | (100,)@1e-3 |
|---|---:|---:|---:|
| accuracy | 0.773 | **0.796** | 0.770 |
| balanced_accuracy | 0.664 | **0.706** | 0.681 |
| f1_macro | 0.671 | **0.722** | 0.668 |
| MCC | 0.723 | **0.751** | 0.719 |
| cover_R² | 0.462 | **0.486** | 0.454 |
| per_source/min | 0.067 | **0.183** | 0.100 |
| per_source/max | 0.851 | **0.876** | 0.850 |

`(500,300,100) @ 1e-4` became the decisive winner on every primary metric except calibration error (ECE), winning both the aggregate and the long-tail axes that had been split between two configs at small scale. Per-epoch curves for the deep net continued to descend at epoch 10, while the shallow net's reference accuracy was flat over the last 3 epochs.

### 20-epoch follow-up on `(500,300,100) @ 1e-4`

Suspecting the deep net had been under-evaluated at 10 epochs, we ran the same configuration to 20 epochs. Every primary metric improved:

| metric | 10-ep | 20-ep | Δ |
|---|---:|---:|---:|
| accuracy | 0.796 | 0.824 | +0.028 |
| balanced_accuracy | 0.706 | 0.735 | +0.029 |
| f1_macro | 0.722 | 0.752 | +0.030 |
| MCC | 0.751 | 0.786 | +0.035 |
| cover_R² | 0.486 | 0.514 | +0.028 |
| log_loss (calibrated) | 0.760 | 0.681 | −0.079 |

13 of 13 primary metrics improved or stayed flat. The **20-epoch large-data result matched or beat the small-sweep accuracy** (0.824 vs 0.828) on classification metrics, and beat it on balanced_accuracy / f1_macro / MCC — i.e., the apparent "scaling penalty" at 10 epochs disappeared.

The training-loss and reference-accuracy curves were still descending at epoch 20, indicating the model was not yet fully converged.

### 40-epoch overfitting test (with new per-epoch val_loss logging)

To find the actual plateau and detect overfitting, the same configuration was extended to 40 epochs with `epoch/val_loss` and `epoch/val_accuracy` logged every epoch (independent held-out set, distinct from the reference set already logged).

The canonical overfitting signature appeared, but in a soft form:

| epoch window | Δ training_loss | Δ val_loss | Δ ref_accuracy | Δ val_accuracy |
|---|---:|---:|---:|---:|
| 1–10  | −0.448 | −0.181 | +0.046 | +0.045 |
| 11–20 | −0.139 | −0.050 | +0.025 | +0.024 |
| 21–30 | −0.065 | −0.014 | +0.020 | +0.019 |
| **31–40** | **−0.057** | **+0.006** | **+0.014** | **+0.014** |

- `val_loss` reached its minimum at **epoch 29 (0.5373)** and rose to 0.5537 by epoch 40.
- `training_loss` continued to decrease through epoch 40 — the divergence between training and validation loss is the textbook overfitting fingerprint.
- However, `val_accuracy` and `ref_accuracy` kept climbing through epoch 40 — the model became overconfident on borderline samples (worse calibration / log_loss) but its argmax decisions remained correct or improved.
- Final calibrated `log_loss` at 40 epochs was still better than at 20 epochs because Platt scaling absorbed most of the raw-probability degradation.
- **Only `balanced_accuracy` regressed** between 20 and 40 epochs (0.735 → 0.731), suggesting the late-epoch fit was biased toward common classes at slight expense of the long tail.
- `cover_R²` continued to improve from 20 → 40 epochs (0.514 → 0.572), but the cover_R² gap to the small-sweep value (0.680) closed only partially.

---

## What was learned

### 1. Architecture choice has a real, scale-dependent effect

The deep `(500,300,100)` MLP was the strongest configuration on the small dataset and, after sufficient training, became more decisively dominant on the larger dataset. At the 80-class / 1.77M-annotation scale, the deep MLP wins not only on aggregate accuracy but also on long-tail metrics that the medium MLP (`(200,100)`) had owned at smaller scale. Per-source robustness scales with capacity in the same direction: the deep MLP has both the highest worst-source and best-source accuracy. The shallow `(100,)` head saturates around `ref_accuracy ≈ 0.766` regardless of how much data or how many epochs it sees — its reference-accuracy curve goes flat by epoch 5–6 at every scale tested. **Capacity is the load-bearing variable for this problem.**

### 2. The optimal learning rate is architecture-dependent

The small-sweep instability of `lr=1e-3` for deeper MLPs (one outright failure, one collapse, one spike) demonstrated that the higher learning rate is unsafe for any depth ≥ 2. The shallow `(100,)` head tolerates `1e-3` because its loss landscape is simpler, but this advantage doesn't extend to other architectures. **`lr=1e-4` is the safe default for all non-trivial MLP heads on this data.**

### 3. Short evaluation budgets systematically penalize larger architectures

The 5-epoch screen ranked `(100,) @ 1e-3` first on every primary metric; the 10-epoch confirm ranked it last. The reason — visible directly in the loss curves — is that deeper architectures need more epochs to demonstrate their advantage. At 5 epochs the deep MLP was still in its steep initial descent; at 10 it had passed the shallow net but its loss was still falling; at 20 it was still descending; at 30+ it began to plateau. **Any screening or hyperparameter-search procedure that uses a small fraction of the eventual epoch budget will systematically prefer architectures that converge faster, regardless of their final quality.** Either run screens at the full epoch budget, or use early stopping with sufficient patience so each architecture is allowed to converge.

### 4. Cross-scale findings replicate, but only with adequate training

The small-sweep verdict (`(500,300,100) @ 1e-4` is the aggregate winner) replicates at the larger scale, both qualitatively (architecture ordering preserved) and quantitatively (absolute accuracy returns to small-sweep levels at 20+ epochs). The apparent "scaling penalty" observed at 10 epochs of the large dataset was a training-budget artifact, not an intrinsic property of the harder problem. **Hyperparameter conclusions from a small benchmark are likely to transfer to a larger dataset, provided the larger evaluation also gets enough training time.**

### 5. The classification problem is genuinely harder at the larger scale

Per-source minimum accuracies dropped from ~0.70 (small sweep) to as low as 0.067 (some sources at large scale); cover_R² dropped from ~0.68 to ~0.51 even at 40 epochs. These shifts are uniform across architectures and reflect the increase from 51 to 80 classes and 10 to 20 source distributions. The per-source minimum recovers with more capacity (deep MLP > shallow), but cover regression appears to plateau at a level meaningfully below the small-sweep result regardless of architecture or epoch count. **If cover prediction is the downstream metric of record, additional epochs and capacity buy decreasing returns; the next round of work should target loss formulation or feature representation rather than head architecture.**

### 6. Overfitting is mild, late, and largely calibration-recoverable

`val_loss` did begin to rise in the 31–40 epoch window for `(500,300,100) @ 1e-4` while `training_loss` continued falling — strict overfitting onset at epoch 29. But `val_accuracy` continued to climb, and Platt-scaling calibration at the end of training mostly repaired the resulting overconfidence. The only macro-averaged metric that regressed between 20 and 40 epochs was `balanced_accuracy` (−0.4 pp), suggesting the late-epoch fit modestly favors common classes. **There is no hard generalization regression within 40 epochs at this scale**, but training past ~30 epochs gives diminishing returns on accuracy at small but real cost to long-tail performance.

### 7. Practical training recipe — actionable defaults

The combination of observations supports a single recommendation for production training on this data:

- Architecture: `hidden_layer_sizes=(500, 300, 100)`
- Learning rate: `learning_rate_init=1e-4`
- Epoch budget: `epochs=40` (generous upper bound)
- Early stopping: `early_stopping_patience=3` against `epoch/val_loss`

Under this configuration, the runner trains each model to its individual val_loss minimum without manual tuning of the epoch count. On the data and architecture explored here the val_loss minimum lands around epoch 29; on smaller heads or different dataset sizes the trainer adapts automatically. The full-budget cost (~70 min/run on the larger dataset) becomes a soft cap that's only paid by configurations that genuinely benefit from it.

### 8. Infrastructure caveats surfaced during the experiment

Two methodological problems uncovered during this work are worth documenting:

- **The original `annotation_limit` knob was non-deterministic under parallel execution** because of `LIMIT` without `ORDER BY` in DuckDB. This silently produced cross-run dataset divergence in the first scale-up attempt. Resolved by replacing the knob with deterministic per-class stratified subsampling. Future analyses that subsample data should use `SubsampleOptions(strategy='stratified', total_annotations=N)`.

- **Per-epoch validation metrics did not exist in the trainer**, which blocked principled overfitting detection. Resolved by adding `epoch/val_loss` and `epoch/val_accuracy` to the per-epoch MLflow logging. Every future training run now produces the canonical overfitting plot for free in MLflow's UI.

Both fixes are in production and should be the default approach for any subsequent architecture or data experiments on this stack.
