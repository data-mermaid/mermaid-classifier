# Training a PySpacer model


## Unfiltered training run

By default, `run_training()` gets all MERMAID annotations and passes them directly into training, without excluding or rolling up any of the labels.

```python
from mermaid_classifier.pyspacer.train import run_training

run_training()
```


## Excluding annotations of certain labels

You can create a CSV file which enumerates certain MERMAID benthic attributes to exclude from training:

```
id,name
31c5af16-30d7-4966-97f4-01889a5cf973,Other
7e426637-0aac-4945-913d-6ab23cdcd8c2,Obscured
b72e02c1-eab3-4781-9eb4-d191bb84e5c7,Unknown
3dc38d25-ed80-4049-af16-e85f04f97cc6,Other invertebrates
```

You need either an `id` column, corresponding to MERMAID benthic attribute UUIDs, or a `name` column, corresponding to MERMAID benthic attribute names (not case sensitive). If you prefer to rely on IDs but still want names present for human readability, you can specify both, as above.

Supposing that the above CSV content is entered into a file located at `labels/inspecific.csv`, running training would go like this:

```python
from mermaid_classifier.pyspacer.train import run_training

run_training(
    excluded_benthicattrs_csv='labels/inspecific.csv',
)
```

As a result, if any annotations are found using any of the above benthic attributes, those annotations will not go into training.

The presence or absence of MERMAID growth forms has no effect on this option.


## Rolling up annotations of certain labels

You can create a CSV file which enumerates target MERMAID benthic attributes to roll up to:

```
id,name
f4df7abd-3d51-42fb-8cab-5102b95fad8e,Crustose coralline algae
350e9eb4-5e6b-48f5-aeb8-0bfdf023bf1c,Hard coral
09226989-50e7-4c40-bd36-5bcef32ee7a1,Macroalgae
30a987e9-b420-4db6-a83a-a1f7cabd14fb,Soft coral
20090bf4-868e-431b-974c-ab9be5bbdb5f,Turf algae
```

The CSV format is the same as for label exclusions.

If these are the rollup targets, then for example, any descendant benthic attributes of Hard coral will get rolled up to Hard coral. In other words, any annotations of the genus Acropora, the species Porites lobata, etc. will be fed into training as Hard coral instead of as the original benthic attribute. And similarly for descendants of Crustose coralline algae, Macroalgae, Soft coral, and Turf algae. 

Supposing that the above CSV content is entered into a file located at `labels/top_level_coral_algae.csv`, running training would go like this:

```python
from mermaid_classifier.pyspacer.train import run_training

run_training(
    benthicattr_rollup_targets_csv='labels/top_level_coral_algae.csv',
)
```

The presence or absence of MERMAID growth forms has no effect on this option.


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
    included_benthicattrs_csv='labels/included.csv',
    # Specify at most one of included and excluded, not both.
    # excluded_benthicattrs_csv='labels/excluded.csv',
    benthicattr_rollup_targets_csv='labels/rollups.csv',
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
