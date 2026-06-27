# Training a PySpacer model

Training is driven by a **committed config directory** — a `training_config.yaml`
plus the CSVs it references — under `sagemaker/configs/<name>/`. The *same* config
drives both local and SageMaker runs, so there's a single source of truth and the
two paths can't drift:

- **Local:** `uv run python scripts/classifier_train.py --config-dir sagemaker/configs/<name>`
  (defaults to `coralnet_top108_best`). Requires AWS SSO and an
  `MLFLOW_TRACKING_SERVER` to log to — a local SQLite DB / store
  (e.g. `sqlite:///mlflow.db`, no server to run) or the SageMaker MLflow App ARN.
- **SageMaker:** `scripts/launch_training.py` uploads the config dir and runs it in a
  TrainingJob — see [training_at_scale.md](../training_at_scale.md).

Generate a fresh config dir (CoralNet→MERMAID label mapping + rollups + included
labels) with `scripts/generate_training_config.py`; see
[../workflow.md](../workflow.md) for where this sits in the overall pipeline.

For programmatic / library use (driving training from Python without a YAML file),
see [Programmatic use](#programmatic--library-use) at the end.


## The config file

A `training_config.yaml` has four blocks — `dataset`, `training`, `mlflow`, and
`env`. CSV paths are **bare filenames resolved as siblings of the YAML file**.
An illustrative config (the committed `sagemaker/configs/example/training_config.yaml`
is a runnable 2-source fixture; the values below are illustrative, not a verbatim
copy):

```yaml
dataset:
  include_mermaid: false          # false = train on CoralNet sources only
  coralnet_sources_csv: sources.csv
  drop_growthforms: false
  label_rollup_spec_csv: rollups.csv
  included_labels_csv: included_labels.csv
  ref_val_ratios: [0.1, 0.1]      # 10% ref, 10% val, 80% train
  subsample:                       # omit the block to disable subsampling
    strategy: balanced             # 'stratified' or 'balanced'
    total_annotations: 1000
    min_per_class: 10
  weighting:                       # omit (or enabled: false) to disable
    enabled: true
    weight_ratio_cap: 5000.0

training:
  epochs: 1                        # upper bound; early stopping may end sooner
  early_stopping_patience: 3
  # The MLP architecture and learning rate are fixed at MermaidTrainer's
  # production values, so only epochs / early stopping are configurable.

mlflow:
  experiment_name: example-smoke-test
  model_name: ExampleModel
  # annotations_to_log: all        # log all annotations as an artifact (not just the val split)

env:                               # applied before pyspacer is imported (so Settings() picks them up)
  MLFLOW_TRACKING_SERVER: file:./mlruns
  WEIGHTS_LOCATION: s3://mermaid-config/classifier/v1/efficientnet_weights.pt
  CORALNET_TRAIN_DATA_BUCKET: 2605-coralnet-public-sources
  MERMAID_TRAIN_DATA_BUCKET: coral-reef-training
```

The full schema (every accepted key + validation) lives in
`mermaid_classifier/sagemaker/config.py` (`TrainingRunConfig`).


## Choosing data sources

Set `dataset.include_mermaid` to include/exclude the MERMAID annotation set, and
point `dataset.coralnet_sources_csv` at a CSV listing the public CoralNet source
IDs to pull training data from:

```
id
23
1579
3064
```

With `include_mermaid: true` and that `sources.csv`, training runs on all MERMAID
data **plus** those three CoralNet sources.


## Rolling up annotations of certain labels

`dataset.label_rollup_spec_csv` points at a CSV mapping MERMAID benthic-attribute +
growth-form (BA+GF) combos onto other combos. Only the IDs (MERMAID UUIDs) are
parsed; the `*_name` columns are purely for human readability:

```
from_ba_id,from_ba_name,from_gf_id,from_gf_name,to_ba_id,to_ba_name,to_gf_id,to_gf_name
050c0689-0b01-4a6c-afbf-a5d3ca39c310,Acropora,cf2deca6-53b8-4096-916f-32c2c71d14bf,Branching,350e9eb4-5e6b-48f5-aeb8-0bfdf023bf1c,Hard coral,cf2deca6-53b8-4096-916f-32c2c71d14bf,Branching
2b55697f-f26e-433e-9070-f1fe9748bf7b,Porites,888609b5-b58a-4d57-addc-a6935bba284b,Massive,2b55697f-f26e-433e-9070-f1fe9748bf7b,Porites,,
```

Here, annotations labeled `Acropora::Branching` are fed into training as
`Hard coral::Branching`, and `Porites::Massive` becomes `Porites` (no growth form).


## Including or excluding certain labels

Point `dataset.included_labels_csv` **or** `dataset.excluded_labels_csv` (at most
one) at a CSV with one BA+GF combo per row. `included_labels_csv` keeps only the
listed combos; `excluded_labels_csv` keeps everything *except* them. Only the IDs
are read; the `*_name` columns are for readability:

```
ba_id,ba_name,gf_id,gf_name
050c0689-0b01-4a6c-afbf-a5d3ca39c310,Acropora,cf2deca6-53b8-4096-916f-32c2c71d14bf,Branching
31c5af16-30d7-4966-97f4-01889a5cf973,Other,,
```


## Subsampling and class weighting

- **`dataset.subsample`** caps the training set for long-tailed taxonomies.
  `strategy: stratified` keeps the original class distribution; `strategy: balanced`
  equalizes toward `total_annotations / num_classes`, with `min_per_class` flooring
  rare classes. (This replaced the old non-deterministic `annotation_limit` knob.)
- **`dataset.weighting`** applies effective-number-of-samples class weights;
  `weight_ratio_cap` bounds the max:min weight ratio.

See [../research/balancing-experiments.md](../research/balancing-experiments.md)
for the experiments behind the production defaults.


## Programmatic / library use

The config path is the recommended workflow, but you can also drive training from
Python — this is exactly the machinery the config path uses under the hood
(`TrainingRunConfig.build_options()` turns a YAML config into the option dataclasses
below). Two runner classes are available:

- `TrainingRunner` — runs training but doesn't log anything to MLflow (no tracking
  server or experiment needed; handy for tests). The `mlflow` package is still a
  dependency of the training install either way.
- `MLflowTrainingRunner` — logs the model, metrics, and artifacts to MLflow. Set
  `MLFLOW_TRACKING_SERVER` to where it should log — a local SQLite DB/store or the
  SageMaker MLflow App ARN (no server to run; see [../mlflow.md](../mlflow.md)).

With no arguments, either runner trains on all MERMAID annotations with no
filtering or rollup:

```python
from mermaid_classifier.pyspacer.runner import MLflowTrainingRunner

MLflowTrainingRunner().run()
```

To configure it, construct the option dataclasses directly (note the modules:
runners live in `pyspacer.runner`, the option dataclasses in `pyspacer.options`,
and `SubsampleOptions` in `training.subsample`):

```python
from mermaid_classifier.pyspacer.runner import MLflowTrainingRunner
from mermaid_classifier.pyspacer.options import (
    DatasetOptions,
    MLflowOptions,
    TrainingOptions,
)
from mermaid_classifier.training.subsample import SubsampleOptions

runner = MLflowTrainingRunner(
    dataset_options=DatasetOptions(
        include_mermaid=False,            # CoralNet sources only
        coralnet_sources_csv="sources/sample.csv",
        drop_growthforms=True,
        label_rollup_spec_csv="labels/rollups.csv",
        included_labels_csv="labels/inclusions.csv",
        # Specify at most one of included / excluded, not both.
        # excluded_labels_csv="labels/exclusions.csv",
        ref_val_ratios=(0.075, 0.125),    # 7.5% ref, 12.5% val, 80% train
        # Subsample for a quick test (replaces the old annotation_limit knob).
        subsample=SubsampleOptions(
            strategy="balanced", total_annotations=5000, min_per_class=10
        ),
    ),
    training_options=TrainingOptions(epochs=5, early_stopping_patience=3),
    mlflow_options=MLflowOptions(
        experiment_name="My Experiment 1",
        model_name="my-model",            # date/time is appended
        extra_annotations_to_log="all",   # YAML key for this is `annotations_to_log`
    ),
)
runner.run()
```

For the complete set of fields, see the `DatasetOptions`, `TrainingOptions`, and
`MLflowOptions` dataclasses in
[`mermaid_classifier/pyspacer/options.py`](../../mermaid_classifier/pyspacer/options.py).
