# Autoresearch: Mermaid Coral Reef Classifier

## Objective

Maximize `balanced_accuracy` on the validation set.
Current baseline: {BASELINE_METRIC} (from initial run)

## Frozen Constraints (DO NOT modify)

- The `FROZEN_DATA` dict and `MLFLOW_OPTIONS` in `train_experiment.py` — these define what data enters training, how labels are mapped/filtered, and how data is split into train/ref/val sets.
- The `mermaid_classifier` package — all imports from this package are read-only. You cannot modify the package source code. SHA256 hashes are verified before every run; any change aborts the experiment.
- The evaluation metric (`balanced_accuracy`) and how it's computed — this is done by the MetricsCoordinator in the frozen package.
- Do not install new packages or add dependencies.

## Modifiable Files

You may modify these files in the `experiment/` directory:

- **train_experiment.py** — Hyperparameters (`TRAINING_OPTIONS`), subsampling config (`SUBSAMPLE`), weighting config (`WEIGHTING`), and how the `ExperimentRunner` wires components together.
- **classifier.py** — The MLP architecture (`ExperimentMLPClassifier` and `_MLPModule`). Change layer sizes, add dropout, batch normalization, skip connections, different activation functions (GELU, SiLU, etc.), weight initialization schemes, or replace the MLP entirely.
- **trainer.py** — The training loop (`ExperimentTrainer`). Change the optimizer (SGD+momentum, AdamW, etc.), add learning rate schedules (cosine, warmup, cyclical), gradient clipping, modify the epoch loop, etc.
- **strategies.py** — Define new sample weighting or subsampling strategies. Import existing strategies from the registry or create novel ones inline.

## How Components Connect

The pipeline works as follows:
1. `train_experiment.py` creates `DatasetOptions` (frozen data fields + modifiable subsample/weighting) and `TrainingOptions`
2. `ExperimentRunner` (subclass of `MLflowTrainingRunner`) handles dataset preparation, MLflow logging, and metrics — all from the frozen package
3. `ExperimentRunner._create_trainer()` constructs your `ExperimentTrainer` from `trainer.py`
4. `ExperimentTrainer.__call__()` constructs the `ExperimentMLPClassifier` from `classifier.py`, runs the training loop, calibrates, and evaluates
5. After training, the frozen `MetricsCoordinator` computes all metrics including `balanced_accuracy`

To inject a custom component (e.g., a new weighting strategy from `strategies.py`), import it in `train_experiment.py` and wire it into the runner or trainer.

## Research Directions (suggestions, not exhaustive)

### Architecture (classifier.py)
- Wider or deeper networks (e.g., (800, 400, 200) or (500, 300, 200, 100))
- Shallower networks (e.g., (300, 100) or even (500,))
- Add dropout between layers (0.1-0.5)
- Add batch normalization
- Skip/residual connections
- Different activation functions: GELU, SiLU/Swish, Mish, LeakyReLU
- Different weight initialization: Kaiming, orthogonal
- Layer normalization instead of batch normalization

### Training Loop (trainer.py)
- Different optimizers: AdamW, SGD+momentum+Nesterov, RAdam, LAMB
- Learning rate schedules: cosine annealing, warmup + cosine, step decay, one-cycle
- Gradient clipping (by norm or value)
- Label smoothing in the cross-entropy loss
- Mixup or CutMix data augmentation on feature vectors
- Stochastic Weight Averaging (SWA)
- Gradient accumulation for effective larger batch sizes

### Hyperparameters (train_experiment.py)
- Learning rate: try 5e-5, 3e-4, 5e-4, 1e-3
- Early stopping patience: try 2, 5, 7
- More epochs (60, 80) with lower patience
- Different random seeds to assess variance

### Sample Weighting (train_experiment.py + strategies.py)
- Different existing strategies: leaf_inverse, decomposed
- Different alpha values: 0.0, 0.25, 0.75, 1.0
- Different weight_ratio_cap values: 100, 1000, 10000, None
- Novel weighting strategies (define in strategies.py)
- Disable weighting entirely (set weighting=None) to compare

### Subsampling (train_experiment.py)
- Different strategies: stratified, soft_balanced
- Different total_annotations: 500K, 1M, 2M
- Different min_per_class: 50, 100, 500, 1000
- soft_balanced with balance_alpha: 0.3, 0.5, 0.7
- No subsampling (set subsample=None) to compare

## Rules

1. **One hypothesis per experiment.** State your hypothesis clearly. Change one thing at a time so results are interpretable. Exception: if combining two previously-successful changes, state that explicitly.
2. **Prefer simple changes over complex ones.** A single-line hyperparameter change that improves the metric is better than a 50-line architecture rewrite that improves it by the same amount. With only ~14 experiments per overnight run, prioritize high-impact changes.
3. **Training must complete within 75 minutes.** A typical run takes ~45 minutes. If your change makes training significantly slower, reconsider. You have ~14 experiment slots per overnight run — spend them wisely.
4. **If an experiment crashes, analyze the error and try something different.** Don't retry the same thing. Read the error output provided in the next prompt.
5. **Review results.tsv.** Don't repeat failed approaches. Build on successful ones. Look for patterns in what works.
6. **Build on success.** The current code reflects all kept improvements. Your next change should build on this improved baseline.
7. **The FROZEN section must not be modified.** This is enforced by SHA256 verification.
8. **All experiment files must remain valid Python.** Syntax errors waste an experiment slot.
9. **Don't make the code unnecessarily complex.** If you're adding 100 lines to try something that could be done in 10, simplify.
