# Autoresearch Harness for Mermaid-Classifier

## Context

Training coral reef classifiers involves a large hyperparameter and architecture search space: MLP layer sizes, learning rates, sample weighting strategies, subsampling approaches, regularization techniques, and training dynamics. Manual experimentation is slow (10-30 min per run) and tedious. Inspired by Karpathy's autoresearch (March 2026), this design enables an AI agent to run autonomous overnight experiments, systematically improving the classifier while keeping data integrity and evaluation methodology frozen.

The goal: wake up to 20-50 validated experiments, a clear results log, and full MLflow tracking for every run.

## Architecture Overview

Four components following Karpathy's three-file contract, adapted to this pipeline's multi-file training structure:

```
autoresearch/
├── autoresearch.py          # Harness driver (IMMUTABLE)
├── program.md               # Research instructions (IMMUTABLE)
├── results.tsv              # Experiment log (harness-managed, append-only)
├── baseline_hashes.json     # SHA256 of frozen files (generated at init)
└── experiment/
    ├── train_experiment.py  # Entry point — wires components, sets frozen DatasetOptions
    ├── classifier.py        # MLP architecture — seeded from torch_classifier.py
    ├── trainer.py           # Training loop — seeded from trainer.py
    └── strategies.py        # Custom weighting/subsampling strategies — starts near-empty
```

**Ownership rules:**
- `autoresearch.py`, `program.md`, `baseline_hashes.json`: human-owned, never modified by the agent
- `experiment/*.py`: agent-owned, freely modifiable
- `results.tsv`: harness-owned, append-only
- Everything in `mermaid_classifier/`: frozen, SHA256-verified before every run

## Frozen vs Modifiable Boundary

### Frozen (agent CANNOT modify)

| Parameter / Component | Location | Why frozen |
|---|---|---|
| `include_mermaid` | DatasetOptions | Defines what data sources enter training |
| `coralnet_sources_csv` | DatasetOptions | Selects which CoralNet sources to use |
| `drop_growthforms` | DatasetOptions | Structural label decision |
| `label_rollup_spec_csv` | DatasetOptions | Defines BA+GF taxonomy mapping |
| `included_labels_csv` / `excluded_labels_csv` | DatasetOptions | Defines label set |
| `ref_val_ratios` | DatasetOptions | Train/ref/val split proportions |
| Data loading pipeline | `mermaid_classifier/pyspacer/train.py` | Data integrity |
| Label mapping (CoralNetMermaidMapping) | `mermaid_classifier/pyspacer/train.py` | Taxonomy consistency |
| MetricsCoordinator + all metric modules | `mermaid_classifier/pyspacer/metrics/` | Evaluation integrity |
| MLflow logging infrastructure | `mermaid_classifier/pyspacer/train.py` | Tracking consistency |
| DuckDB helpers | `mermaid_classifier/common/duckdb_utils.py` | Data pipeline integrity |
| Settings / env vars | `mermaid_classifier/pyspacer/settings.py` | Infrastructure config |
| Config CSVs | `../sagemaker/configs/` | Data definition |

### Modifiable (agent CAN change)

| Parameter / Component | File | What the agent can do |
|---|---|---|
| MLP architecture | `experiment/classifier.py` | Change layer sizes, add dropout, batch norm, skip connections, different activations, weight init, replace MLP entirely |
| Training loop | `experiment/trainer.py` | Change optimizer (SGD, AdamW, etc.), add LR schedules, gradient clipping, warmup, modify epoch logic |
| Hyperparameters | `experiment/train_experiment.py` | epochs, learning_rate_init, early_stopping_patience, random_state, hidden_layer_sizes |
| Sample weighting strategy | `experiment/strategies.py` + `experiment/train_experiment.py` | Use existing strategies (effective_number, tree_balanced, leaf_inverse, decomposed) or define new ones inline |
| Subsampling strategy/params | `experiment/train_experiment.py` | Change strategy (stratified, balanced, soft_balanced), total_annotations, min_per_class, balance_alpha |
| Custom components | `experiment/strategies.py` | Define novel loss functions, custom weighting schemes, new subsampling allocators |

## Harness Loop (`autoresearch.py`)

### Initialization

```python
def init():
    1. Compute SHA256 hashes of all frozen files → baseline_hashes.json
    2. Seed experiment/ files from current production code (if not already seeded)
    3. Run baseline experiment (current experiment/ state)
    4. Record baseline balanced_accuracy in results.tsv
    5. Create initial commit on autoresearch branch: "baseline: balanced_accuracy=X.XXX"
```

### Main Loop

```python
def loop():
    while not should_stop():  # time budget or manual interrupt
        
        # 1. Build Claude prompt
        prompt = build_prompt(
            program_md=read("program.md"),
            experiment_files=read_all("experiment/"),
            results_history=read("results.tsv"),
            recent_diffs=git_log_diffs(last_n=5),
        )
        
        # 2. Call Claude API
        response = call_claude(prompt)
        # Response contains: file changes + hypothesis description
        
        # 3. Apply changes
        apply_file_changes(response.changes)
        git_commit(f"experiment {n}: {response.hypothesis}")
        
        # 4. Verify frozen files
        if not verify_hashes("baseline_hashes.json"):
            git_reset_hard()
            log_result(experiment_id=n, status="ABORT: frozen file modified")
            continue
        
        # 5. Run training with timeout
        result = subprocess.run(
            ["uv", "run", "python", "autoresearch/experiment/train_experiment.py"],
            timeout=1800,  # 30 min default
            capture_output=True,
        )
        
        # 6. Handle crash
        if result.returncode != 0:
            git_reset_hard()
            log_result(experiment_id=n, status="CRASH", error=result.stderr)
            continue
        
        # 7. Extract metric from MLflow
        balanced_accuracy = query_mlflow_latest_run("balanced_accuracy")
        
        # 8. Keep or revert
        if balanced_accuracy > best_so_far:
            best_so_far = balanced_accuracy
            log_result(experiment_id=n, status="KEPT", metric=balanced_accuracy)
        else:
            git_reset_hard()
            log_result(experiment_id=n, status="REVERTED", metric=balanced_accuracy)
```

### Stopping Conditions

- `--max-hours N` (default: 12): total wall-clock time budget
- `--max-experiments N` (default: 100): experiment count limit
- Manual interrupt (Ctrl+C): graceful shutdown, current experiment completes or is reverted
- Consecutive failure limit (default: 10): if 10 experiments in a row crash or are reverted, stop and alert

## Component Integration

The key integration challenge: `TrainingRunner.run()` (line 1649 of `train.py`) directly constructs `MermaidTrainer`, which in turn constructs `TorchMLPClassifier` (line 124 of `trainer.py`). For the agent to swap in custom components, the experiment uses **subclassing**:

1. `experiment/classifier.py` defines a custom classifier that can be imported by the trainer
2. `experiment/trainer.py` subclasses `MermaidTrainer` and overrides `__call__` to use the custom classifier
3. `experiment/train_experiment.py` subclasses `MLflowTrainingRunner` and overrides the trainer construction in `run()` (line 1649) to use the custom trainer

This preserves all frozen infrastructure (dataset prep, MLflow logging, metrics, calibration) while letting the agent fully control the model architecture and training loop.

## Experiment File Seeding

### `experiment/train_experiment.py`

Adapted from `scripts/classifier_train.py`. Subclasses `MLflowTrainingRunner` to inject the custom trainer. Frozen DatasetOptions are hardcoded with subsampling/weighting clearly separated as modifiable:

```python
"""Autoresearch experiment entry point.

FROZEN: DatasetOptions data fields (sources, rollups, labels, splits),
        MLflowOptions, AWS credential setup.
MODIFIABLE: Everything else — TrainingOptions, subsample/weighting
            config, runner subclass, component wiring.
"""
import os
import boto3

# AWS credential resolution (same as production)
os.environ['AWS_PROFILE'] = 'wcs'
session = boto3.Session()
# ... (credential setup)

from mermaid_classifier.pyspacer.train import (
    DatasetOptions, MLflowOptions, MLflowTrainingRunner, TrainingOptions)
from mermaid_classifier.training.sample_weighting import SampleWeightingOptions
from mermaid_classifier.training.subsample import SubsampleOptions

# Import custom components from experiment directory
from trainer import ExperimentTrainer

# ── FROZEN: data identity ──────────────────────────────────────────
FROZEN_DATA = dict(
    include_mermaid=False,
    coralnet_sources_csv='../sagemaker/configs/tiela77_top100_min1k/sources.csv',
    label_rollup_spec_csv='../sagemaker/configs/tiela77_top100_min1k/rollups.csv',
    included_labels_csv='../sagemaker/configs/tiela77_top100_min1k/included_labels.csv',
    drop_growthforms=False,
    ref_val_ratios=(0.1, 0.1),
)
MLFLOW_OPTIONS = MLflowOptions(
    experiment_name="autoresearch",
    model_name='AutoResearch',
)
# ── END FROZEN ──────────────────────────────────────────────────────

# ── MODIFIABLE: subsampling, weighting, training ────────────────────
SUBSAMPLE = SubsampleOptions(
    strategy='balanced',
    total_annotations=1_770_000,
    min_per_class=200,
)
WEIGHTING = SampleWeightingOptions(
    strategy='effective_number',
    alpha=0.5,
    weight_ratio_cap=5000.0,
)
TRAINING_OPTIONS = TrainingOptions(
    hidden_layer_sizes=(500, 300, 100),
    learning_rate_init=1e-4,
    epochs=40,
    early_stopping_patience=3,
)


class ExperimentRunner(MLflowTrainingRunner):
    """Overrides trainer construction to use experiment's custom trainer."""

    def _create_trainer(self, batch_size, class_weight):
        return ExperimentTrainer(
            batch_size=batch_size,
            on_epoch_end=self._on_epoch_end,
            class_weight=class_weight,
            hidden_layer_sizes=self.training_options.hidden_layer_sizes,
            learning_rate_init=self.training_options.learning_rate_init,
            early_stopping_patience=self.training_options.early_stopping_patience,
            random_state=self.training_options.random_state,
        )


if __name__ == "__main__":
    dataset_options = DatasetOptions(
        **FROZEN_DATA,
        subsample=SUBSAMPLE,
        weighting=WEIGHTING,
    )
    runner = ExperimentRunner(
        dataset_options=dataset_options,
        training_options=TRAINING_OPTIONS,
        mlflow_options=MLFLOW_OPTIONS,
    )
    runner.run()
```

**Note**: This requires a small refactor in `TrainingRunner.run()` to extract trainer construction into a `_create_trainer()` method (currently inline at lines 1649-1658). This is the only change to the frozen codebase needed before the harness can run.

### `experiment/classifier.py`

Seeded from `mermaid_classifier/pyspacer/torch_classifier.py`. Contains `_MLPModule` and `TorchMLPClassifier`. The agent can modify the architecture, add layers, change activations, etc.

### `experiment/trainer.py`

Seeded from `mermaid_classifier/pyspacer/trainer.py`. Contains `ExperimentTrainer` (initially identical to `MermaidTrainer`). The agent can modify the training loop, optimizer, learning rate schedule, and swap in the custom classifier from `classifier.py`.

### `experiment/strategies.py`

Starts near-empty with imports from existing registries:

```python
"""Custom strategies for autoresearch experiments.

Import existing strategies or define new ones here.
"""
from mermaid_classifier.training.sample_weighting.registry import (
    compute_class_weights,
)
from mermaid_classifier.training.subsample.registry import (
    compute_per_class_targets,
)

# Agent can define new strategies here, e.g.:
# class MyCustomWeighting(Strategy):
#     ...
```

## `program.md` Structure

```markdown
# Autoresearch: Mermaid Coral Reef Classifier

## Objective
Maximize `balanced_accuracy` on the validation set.
Current baseline: {BASELINE_METRIC} (from initial run)

## Frozen Constraints (DO NOT modify)
- Data sources, label rollups, included_labels, ref_val_ratios in train_experiment.py
- The mermaid_classifier package (verified by SHA256)
- The evaluation metric and how it's computed
- Do not install new packages

## Modifiable Files
- experiment/train_experiment.py — hyperparameters, component wiring, 
  subsampling/weighting config
- experiment/classifier.py — MLP architecture
- experiment/trainer.py — training loop, optimizer, LR schedule
- experiment/strategies.py — custom weighting/subsampling strategies

## Research Directions (suggestions)
- Architecture: wider/deeper/shallower networks, residual connections, 
  dropout, batch norm, different activations (GELU, SiLU)
- Optimizer: SGD+momentum, AdamW, different LR schedules (cosine, warmup)
- Regularization: weight decay, label smoothing, mixup
- Sample weighting: novel strategies, different alpha/cap values
- Subsampling: different balance points, soft_balanced with various alphas
- Training dynamics: warmup, cosine annealing, cycle learning rates

## Rules
1. Each experiment tests ONE hypothesis. State it clearly.
2. Prefer simple changes over complex ones.
3. Training must complete within 30 minutes.
4. If an experiment crashes, analyze the error and try something different.
5. Review results.tsv — don't repeat failed approaches.
6. Build on successful changes — the current code reflects all kept improvements.
7. The FROZEN section of train_experiment.py must not be modified.
```

## `results.tsv` Format

```
id	timestamp	hypothesis	balanced_accuracy	best_so_far	status	duration_s	commit_sha	error
1	2026-05-09T22:00:00	baseline	0.4523	0.4523	KEPT	1200	abc1234	
2	2026-05-09T22:25:00	add dropout 0.3 between layers	0.4601	0.4601	KEPT	1150	def5678	
3	2026-05-09T22:50:00	try GELU activation	0.4498	0.4601	REVERTED	1180	ghi9012	
4	2026-05-09T23:15:00	increase width to (800,500,200)	0.0000	0.4601	CRASH	30	jkl3456	OOM...
```

## Safety Mechanisms

### 1. SHA256 Hash Verification

At init, compute SHA256 for every file in:
- `mermaid_classifier/` (recursive)
- `autoresearch/autoresearch.py`
- `autoresearch/program.md`
- Config CSVs referenced in DatasetOptions

Store in `baseline_hashes.json`. Before every training run, re-hash and compare. Abort on any mismatch.

### 2. Training Timeout

`subprocess.run()` with `timeout=4500` (75 min, configurable via `--timeout`). Training takes ~45 min with the 7-source dataset. On timeout, `SIGKILL` the process, revert the experiment, log as `TIMEOUT`.

### 3. Git Reset on Failure

Failed/reverted experiments use `git reset --hard HEAD~1` to discard the experiment commit entirely. This keeps git history clean — only successful experiments appear in the log. The full audit trail lives in `results.tsv`, which records every experiment (kept, reverted, crashed) with its hypothesis and metrics.

### 4. No Package Installation

The harness runs training in a subprocess using the existing `uv` environment. No mechanism is provided for `pip install` or `uv add`. Claude is instructed in `program.md` not to install packages.

### 5. Crash Recovery

Non-zero exit codes → revert + include stderr in next Claude prompt. The agent learns from crashes.

### 6. Total Time Budget

`--max-hours` flag (default: 12). Checked before starting each new experiment.

### 7. Consecutive Failure Limit

If N consecutive experiments crash or are reverted (default: 10), the harness stops and prints a summary. This prevents infinite loops of bad ideas.

## Claude API Integration

The harness calls the Claude API directly (not Claude Code). Key parameters:

- **Model**: claude-sonnet-4-6 (fast, good enough for code modifications; opus for complex reasoning if needed)
- **System prompt**: Contents of `program.md`
- **User message**: Current experiment files + results.tsv + recent diffs + any error output from last run
- **Response format**: Structured output with `hypothesis` (string) and `file_changes` (list of {filename, content})
- **Max tokens**: ~8000 (enough for file rewrites)
- **Temperature**: 0.7-1.0 (encourage exploration)

### Prompt Construction

```python
def build_prompt(program_md, experiment_files, results_tsv, recent_diffs, last_error=None):
    parts = [
        "## Current Experiment Files\n",
        *[f"### {name}\n```python\n{content}\n```" for name, content in experiment_files],
        f"\n## Experiment History\n```\n{results_tsv}\n```",
        f"\n## Recent Diffs (last 5 kept experiments)\n{recent_diffs}",
    ]
    if last_error:
        parts.append(f"\n## Last Experiment Error\n```\n{last_error}\n```")
    return "\n".join(parts)
```

## MLflow Integration

The harness reads metrics from MLflow after each training run:

```python
def query_mlflow_latest_run(metric_name, experiment_name="autoresearch"):
    """Query the most recent MLflow run for a specific metric."""
    import mlflow
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs.iloc[0][f"metrics.{metric_name}"]
```

All existing MLflow logging (per-epoch metrics, confusion matrices, calibration plots, per-source breakdowns) continues to work unchanged. The harness only reads `balanced_accuracy` for the keep/revert decision.

## Verification Plan

### Unit Tests
- Hash verification: test that modifying a frozen file triggers abort
- Results.tsv parsing and appending
- Git commit/revert logic
- Timeout enforcement
- Prompt construction

### Integration Test
1. Seed experiment files from production code
2. Run baseline — verify balanced_accuracy is logged to MLflow and results.tsv
3. Make a known-good change (e.g., add dropout 0.1) — verify it's kept
4. Make a known-bad change (e.g., LR=100) — verify it crashes and is reverted
5. Modify a frozen file — verify hash check aborts the run

### End-to-End Test
1. Run the full harness with `--max-experiments 3` against a small dataset (first 10 sources)
2. Verify: 3 experiments logged, git history is clean, results.tsv has 3+1 rows (including baseline)
3. Review MLflow UI for all runs under "autoresearch" experiment

### Overnight Run
1. Start with `--max-hours 8`
2. Check in the morning: review results.tsv, MLflow dashboard, git log
3. Verify the working tree reflects only kept improvements
