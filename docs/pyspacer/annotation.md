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

- An S3 URI to the model's .pkl file; starts with `s3://`. MLflow-logged models can be specified like this, if you can find where MLflow logged it in S3.
- A MLflow registered Model ID; starts with `m-`. This option requires the applicable MLflow tracking server to be running.
- A local filepath.

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
