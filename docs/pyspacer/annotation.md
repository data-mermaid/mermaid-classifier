# Image classification and viewing point annotations


## Machine-classifying points and viewing the results

For this, you need to specify an image, point locations in the image, and a machine classifier.

Point locations are expected in CSV format. Say we have `0032dba6_points.csv` with contents as follows:

```
row,column
812,928
812,1856
812,2784
812,3712
812,4640
<continues to define a total of 25 point positions>
```

The classifier location can be specified as any of the following strings:

- An MLflow registered Model ID; starts with `m-`. Resolves the logged model's `model.pt` + `model.json` artifact. Requires the applicable MLflow tracking server to be running.
- An S3 URI to the **directory** containing `model.pt` + `model.json`; starts with `s3://` (e.g. `s3://bucket/models/run-abc/`). Both files are downloaded to a temp dir.
- A local **directory** path containing `model.pt` + `model.json`.

  > Note: as of issue #60, the classifier is the portable TorchScript artifact (`model.pt` + `model.json`), not a single `.pkl`. The S3-URI and local-path forms now name the *directory* holding both files, not a pickle file.

Putting it together, we set up an `AnnotationRun` and show its result:

```python
from mermaid_classifier.pyspacer.annotation import AnnotationRun

annotation_run = AnnotationRun(
    image='s3://coral-reef-training/mermaid/0032dba6-8357-42e2-bace-988f99032286.png',
    points_csv='0032dba6_points.csv',
    classifier='m-48f52362eb404e09bfd1587974e066a8',
)
annotation_run.show()
```

When it's done, it should display the labeled points overlaid on the image, using matplotlib.

To view the matplotlib result interactively in a JupyterLab notebook:

1. Add `%matplotlib widget` at the beginning of the above code snippet, and use the snippet content as your notebook cell
2. Ensure the pip package `ipympl` is installed
3. Ensure the browser tab running JupyterLab has been [hard-refreshed](https://www.howtogeek.com/672607/how-to-hard-refresh-your-web-browser-to-bypass-your-cache/) since `ipympl` was installed
4. Run the notebook cell

This also applies to other annotation-viewing code snippets below.


## Viewing CSV-defined point annotations

Alternatively, you can include a `label` column in your `points_csv` to assign a label to each point, and show the results with an `AnnotationRun`.

Suppose we have `0032dba6_annotations.csv` here:

```
row,column,label
812,928,ed2332ed-0762-45fb-87a3-d315e218faf1
812,1856,f4df7abd-3d51-42fb-8cab-5102b95fad8e
812,2784,5f1f7956-bc21-4bfd-a409-9740e614b2ac
812,3712,f4df7abd-3d51-42fb-8cab-5102b95fad8e
812,4640,b76bca12-884b-4404-bb9f-97d505b0fe58
<continues to define a total of 25 annotated points>
```

Then this will display the annotations:

```python
from mermaid_classifier.pyspacer.annotation import AnnotationRun

annotation_run = AnnotationRun(
    image='s3://coral-reef-training/mermaid/0032dba6-8357-42e2-bace-988f99032286.png',
    points_csv='0032dba6_annotations.csv',
)
annotation_run.show()
```


## More available parameters

For more details, browse [annotation.py](../../mermaid_classifier/pyspacer/annotation.py) for the start of the `__init__()` method of the `AnnotationRun` class, where the parameters are listed and explained in comments.

```python
from mermaid_classifier.pyspacer.annotation import AnnotationRun

annotation_run = AnnotationRun(
    # Can be a CoralNet public image ID.
    image='12903',
    points_csv='cn_12903_points.csv',
    classifier='s3://my-bucket/classifier.pkl',
    # Can also be specified in a .env file.
    weights='s3://my-bucket/weights.pt',
    # Custom mapping of label IDs to display names.
    labelset_csv='labelset.csv',
    # Save machine predictions back to points CSV.
    num_predictions_to_save=5,
    # Downloads cache when specifying images as CoralNet IDs.
    coralnet_cache_dir='/var/tmp/coralnet_cache',
    plot_title="Bocas del Toro classification example",
)
annotation_run.show()
```
