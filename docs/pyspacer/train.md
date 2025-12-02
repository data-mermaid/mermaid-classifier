# Training a PySpacer model


## Unfiltered training run

By default, `run_training()` gets all MERMAID annotations and passes them directly into training, without excluding or rolling up any of the labels.

```python
from mermaid_classifier.pyspacer.train import run_training

run_training()
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
from mermaid_classifier.pyspacer.train import run_training

run_training(
    coralnet_sources_csv='sources/sample.csv',
)
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
from mermaid_classifier.pyspacer.train import run_training

run_training(
    label_rollup_spec_csv='labels/coral_rollups.csv',
)
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
from mermaid_classifier.pyspacer.train import run_training

run_training(
    excluded_labels_csv='labels/exclusions.csv',
)
```

As a result, if any annotations are found using either of the above BA+GF combos, those annotations will not go into training.


## More available parameters

Here are examples demonstrating other available parameters. For more details, browse [train.py](../../mermaid_classifier/pyspacer/train.py) for the start of the `run_training()` function, where the parameters are listed and explained in comments.

```python
from mermaid_classifier.pyspacer.train import run_training

run_training(
    model_name='CustomModelNameHere',
)

# These parameters can be useful for quick tests.
run_training(
    epochs=2,
    disable_mlflow=True,
    annotation_limit=2000,
)

# The rest of the parameters available.
run_training(
    # Specifying False here means you're only training on CoralNet sources. 
    include_mermaid=False,
    coralnet_sources_csv='sources/sample.csv',
    label_rollup_spec_csv='labels/rollups.csv',
    included_labels_csv='labels/inclusions.csv',
    # Specify at most one of included and excluded, not both.
    # excluded_labels_csv='labels/exclusions.csv',
    drop_growthforms=True,
    epochs=5,
    # This basically helps to group models.
    experiment_name="My Experiment 1",
    # This gets date/time appended to the end of it.
    model_name='my-model',
    # This is already False by default, but it doesn't make sense to
    # specify experiment and model names when MLflow usage is disabled.
    disable_mlflow=False,
    annotation_limit=5000,
)
```
