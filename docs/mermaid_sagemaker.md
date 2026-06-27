# MERMAID SageMaker notes

These notes are for MERMAID team's data scientists and developers who are working with SageMaker.


## How this project uses SageMaker

There are two distinct ways you interact with SageMaker, and they happen in
different places — don't conflate them:

1. **Submit jobs from your *local* machine** (not from inside Studio). With AWS
   SSO configured, the launcher scripts create and run SageMaker jobs; the heavy
   compute runs in SageMaker while you just kick it off from your terminal:
   - `scripts/launch_training.py` → a **TrainingJob** that trains the classifier
     (CPU). See [training_at_scale.md](training_at_scale.md).
   - `scripts/launch_processing.py` → **ProcessingJob(s)** that do feature-vector
     extraction (GPU; optionally sharded across parallel jobs). See
     [feature_extraction_at_scale.md](feature_extraction_at_scale.md).

2. **Use SageMaker Studio (in the browser)** to *look at* things, not to launch
   jobs:
   - Open the **MLflow App** to browse logged runs and models (see
     [mlflow.md](mlflow.md)).
   - Monitor the **Training / Processing Jobs** you submitted (status, logs,
     metrics) under SageMaker AI → Training / Processing.
   - Optionally, work inside SageMaker via the in-browser IDEs (below).

So: you do **not** open Studio to run training or feature extraction — submit
those from your local machine. Studio is for MLflow and for watching the jobs
run.


## Accessing SageMaker Studio

- Sign into AWS through MERMAID's SSO setup, using the awsapps link for that.
- Click the appropriate AWS account for your work, and click the SageMaker role.
- On the header bar, ensure your AWS region is "United States (N. Virginia)" (code us-east-1), since that is where MERMAID's SageMaker resources are set up.
- Navigate to Amazon SageMaker AI > Studio. Domain = dev-SG-Project, user profile = your user profile, and click Open Studio.
- Under Spaces, next to the space mermaid-ic-pipeline, click Launch Shared Studio.


## SageMaker Studio apps

There are three in-browser IDEs offered by SageMaker: JupyterLab, Code Editor (VSCode), and RStudio.

- MERMAID has only used the first two; the RStudio option requires a professional license for Posit Workbench.
- Each IDE offering can be used by starting and then opening a Space. There are shared spaces and private spaces; shared spaces enable real-time collaboration. Each space has its own file storage.

MERMAID runs an always-on SageMaker MLflow App for PySpacer classifiers. To log to it, point `MLFLOW_TRACKING_SERVER` at the App's ARN (see [mlflow.md](mlflow.md)) — there's no server to start.

MERMAID isn't using Canvas, which is a "no-code" offering and likely not as flexible.


## JupyterLab spaces

When you're working with a Jupyter notebook (.ipynb file), it'll default to the "Glue PySpark" kernel, but using this kernel requires additional AWS permissions. You can instead select a different kernel such as "Python 3 (ipykernel)".
