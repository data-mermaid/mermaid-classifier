# Training a PySpacer model


## Basic training run with or without MLflow

There are two training runner classes available:

- `TrainingRunner`, which runs training but doesn't save any results.
- `MLflowTrainingRunner`, which logs the model and associated artifacts to MLflow. Requires an MLflow installation and an MLflow tracking server to be running.

By default, either runner gets all MERMAID annotations and passes them directly into training, without excluding or rolling up any of the labels.

```python
from mermaid_classifier.pyspacer.train import TrainingRunner

runner = TrainingRunner()
return_msg, model_loc = runner.run()
```

```python
from mermaid_classifier.pyspacer.train import MLflowTrainingRunner

runner = MLflowTrainingRunner()
# This returns a message and model location too, but the runner already
# logs most of the message's contents and the model.
runner.run()
```


## Choosing data sources

You can create a CSV file which lists IDs of public CoralNet sources to get training data (annotations of confirmed images) from:

```
id
23
1579
3064
```

Supposing that the above CSV content is entered into a file located at `sources/sample.csv`, the following code would run training on all MERMAID data plus the data from these three CoralNet sources:

```python
from mermaid_classifier.pyspacer.train import (
    DatasetOptions, MLflowTrainingRunner)

runner = MLflowTrainingRunner(
    dataset_options=DatasetOptions(
        coralnet_sources_csv='sources/sample.csv',
    ),
)
runner.run()
```


## Rolling up annotations of certain labels

You can create a CSV file which says what MERMAID benthic attribute + growth form combinations to roll up to what other combinations:

```
from_ba_id,from_ba_name,from_gf_id,from_gf_name,to_ba_id,to_ba_name,to_gf_id,to_gf_name
050c0689-0b01-4a6c-afbf-a5d3ca39c310,Acropora,cf2deca6-53b8-4096-916f-32c2c71d14bf,Branching,350e9eb4-5e6b-48f5-aeb8-0bfdf023bf1c,Hard coral,cf2deca6-53b8-4096-916f-32c2c71d14bf,Branching
2b55697f-f26e-433e-9070-f1fe9748bf7b,Porites,888609b5-b58a-4d57-addc-a6935bba284b,Massive,2b55697f-f26e-433e-9070-f1fe9748bf7b,Porites,,
```

Only the IDs (MERMAID UUIDs) are actually parsed; the names are purely for human readability in this example.

In this example, any annotations labeled as Acropora::Branching will be fed into training as Hard coral::Branching. Similarly, any annotations of Porites::Massive will be fed into training as Porites (without growth form specified).

Supposing that the above CSV content is entered into a file located at `labels/coral_rollups.csv`, running training would go like this:

```python
from mermaid_classifier.pyspacer.train import (
    DatasetOptions, MLflowTrainingRunner)

runner = MLflowTrainingRunner(
    dataset_options=DatasetOptions(
        label_rollup_spec_csv='labels/coral_rollups.csv',
    ),
)
runner.run()
```


## Excluding annotations of certain labels

You can create a CSV file specifying either the MERMAID benthic attribute + growth form combos to accept into the training data (excluding all others), or the ones to leave out from the training data (including all others). Either way, one BA + GF combo would be specified per row:

```
ba_id,ba_name,gf_id,gf_name
050c0689-0b01-4a6c-afbf-a5d3ca39c310,Acropora,cf2deca6-53b8-4096-916f-32c2c71d14bf,Branching
31c5af16-30d7-4966-97f4-01889a5cf973,Other,,
```

Only the IDs are read in; the names are purely for human readability in this example.

Supposing that the above CSV content is entered into a file located at `labels/exclusions.csv`, and we want to exclude these BA+GF combos while including all others, running training would go like this:

```python
from mermaid_classifier.pyspacer.train import (
    DatasetOptions, MLflowTrainingRunner)

runner = MLflowTrainingRunner(
    dataset_options=DatasetOptions(
        excluded_labels_csv='labels/exclusions.csv',
    ),
)
runner.run()
```

As a result, if any annotations are found using either of the above BA+GF combos, those annotations will not go into training.


## More available parameters

Here are examples demonstrating other available parameters. For more details, browse [train.py](../../mermaid_classifier/pyspacer/train.py) for the `DatasetOptions`, `MLflowOptions`, and `TrainingOptions` classes.

```python
from mermaid_classifier.pyspacer.train import (
    DatasetOptions, MLflowOptions, MLflowTrainingRunner, TrainingOptions)

runner = MLflowTrainingRunner(
    mlflow_options=MLflowOptions(
        model_name='CustomModelNameHere',
    ),
)
runner.run()

runner = MLflowTrainingRunner(
    # These options can be useful for quick tests.
    dataset_options=DatasetOptions(
        annotation_limit=2000,
    ),
    training_options=TrainingOptions(
        epochs=2,
    ),
)
runner.run()

# The rest of the parameters available.
runner = MLflowTrainingRunner(
    dataset_options=DatasetOptions(
        # Specifying False here means you're only training on CoralNet sources.
        include_mermaid=False,
        coralnet_sources_csv='sources/sample.csv',
        drop_growthforms=True,
        label_rollup_spec_csv='labels/rollups.csv',
        included_labels_csv='labels/inclusions.csv',
        # Specify at most one of included and excluded, not both.
        # excluded_labels_csv='labels/exclusions.csv',
        # 7.5% ref, 12.5% val, 80% train.
        ref_val_ratios=(0.075, 0.125),
        annotation_limit=5000,
    ),
    training_options=TrainingOptions(
        epochs=5,
    ),
    mlflow_options=MLflowOptions(
        # This basically helps to group models.
        experiment_name="My Experiment 1",
        # This gets date/time appended to the end of it.
        model_name='my-model',
        # Logs all input annotations to MLflow. Also possible to just log a subset.
        annotations_to_log='all',
    ),
)
runner.run()
```
