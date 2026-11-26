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
labels + `groups.csv`) with `scripts/generate_training_config.py`; see
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
  datasets:                        # pinned dataset versions in mermaid-ic-datasets
    - name: coralnet-raw
      version: 20261124-5ab21b3
      manifest_sha256: b74e2a9fa32b340e117c0e6f2e0c6fcc64dc1d2df5011778ba0bde76e2568e8b
      features:
        family: efficientnet-v1
        manifest_sha256: "c557feffdeb3f89ec82270563b2942d0c4bbe7c976a3568cb92bb91dcfe9935e"
      labels: coralnet            # or: mermaid
      groups_csv: groups.csv      # optional; omit to train on every group
  # taxonomy:                      # required when any dataset has `labels: mermaid`
  #   version: 20261124-e6abfad
  #   manifest_sha256: fc12b99c119b0151e6fb02ea7bb9870f423bd220764d5f47b57d2305fc3f6b43
  feature_extractor_weights: s3://mermaid-config/classifier/v1/efficientnet_weights.pt
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
```

The full schema (every accepted key + validation) lives in
`mermaid_classifier/sagemaker/config.py` (`TrainingRunConfig`).

`dataset.feature_extractor_weights` is required, and is the run's statement of
which EfficientNet produced the feature vectors it trains on. Each feature set's
manifest records its extractor, and each must agree with this field.
`TrainingDataset` hashes the weights object and seals the result into
`model.json` under `feature_extraction`, which the release gate then verifies
the shipped `efficientnet.pt` against.


## Choosing data sources

Training reads points and feature vectors only from pinned dataset versions in
the `mermaid-ic-datasets` bucket (`segdata-spec`). Each `dataset.datasets` entry
takes:

| Key | Meaning |
|---|---|
| `name`, `version`, `manifest_sha256` | The dataset version and the sha256 of its published manifest. Names are unique within a config. |
| `features.family`, `features.manifest_sha256` | The feature set read for that version. |
| `labels` | `coralnet`: label IDs go through `CoralNetMermaidMapping`; only for coralnet-scrape datasets. `mermaid`: labels carry a MERMAID BA id and a growth-form name, mapped to a growth-form id through the taxonomy. |
| `groups_csv` | Optional. A file of dataset groups, one per line, relative to the YAML. Omitted, every group of the version trains. A group absent from the version stops the run. |

`dataset.taxonomy` (`version`, `manifest_sha256`) pins the `mermaid-benthic`
taxonomy. It is required when any entry has `labels: mermaid`.

For a CoralNet dataset the groups are `s<id>`, one per source:

```
s23
s1579
s3064
```

Feature vectors are fetched with `segdata-spec`'s `sync_feature_set` into the
run's feature cache (`FEATURE_CACHE_DIR`, a temporary directory when unset),
using `DOWNLOAD_MAX_WORKERS` threads. The bucket is not listed. Images that a
feature set does not index are dropped from the annotations and count against
`TRAINING_INPUTS_PERCENT_MISSING_ALLOWED`; above that percentage the run stops.

Training on CoralNet sources plus MERMAID takes two entries (see
`sagemaker/configs/coralnet_all_plus_mermaid/training_config.yaml`): a
`coralnet-raw` entry with `labels: coralnet` and a `mermaid` entry with
`labels: mermaid`, plus the `taxonomy` pin.

`scripts/generate_training_config.py` writes the `groups.csv` for the CoralNet
sources it was given.

### Run lineage

Each run records the pins it trained on:

- MLflow params `dataset/<name>/{version,manifest_sha256,labels,groups_sha256,groups_n}`,
  `dataset/<name>/features/{family,manifest_sha256}`, and
  `taxonomy/<name>/{version,manifest_sha256}`. `groups_sha256` and `groups_n` are
  `all` when no `groups_csv` is set.
- A `groups/<name>.csv` artifact for each dataset with a group selection.
- A `training_data` block in `model.json`.


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

A run needs `DatasetOptions.datasets` (and `feature_extractor_weights`), so the
option dataclasses are constructed directly (note the modules: runners live in
`pyspacer.runner`, the option dataclasses in `pyspacer.options`, and
`SubsampleOptions` in `training.subsample`). The pins are `segdata_spec.manifest`
`Pin` and `FeatureSetPin` values:

```python
from mermaid_classifier.pyspacer.runner import MLflowTrainingRunner
from mermaid_classifier.pyspacer.options import (
    DatasetOptions,
    MLflowOptions,
    PinnedDataset,
    TrainingOptions,
)
from segdata_spec.manifest import FeatureSetPin, Pin
from mermaid_classifier.training.subsample import SubsampleOptions

dataset = Pin(
    name="coralnet-raw",
    version="20261124-5ab21b3",
    manifest_sha256="b74e2a9fa32b340e117c0e6f2e0c6fcc64dc1d2df5011778ba0bde76e2568e8b",
)

runner = MLflowTrainingRunner(
    dataset_options=DatasetOptions(
        datasets=[
            PinnedDataset(
                pin=dataset,
                features=FeatureSetPin(
                    dataset=dataset,
                    family="efficientnet-v1",
                    manifest_sha256="c557feffdeb3f89ec82270563b2942d0c4bbe7c976a3568cb92bb91dcfe9935e",
                ),
                labels="coralnet",
                groups_csv="labels/groups.csv",   # None trains on every group
            )
        ],
        feature_extractor_weights="s3://mermaid-config/classifier/v1/efficientnet_weights.pt",
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
