# MERMAID SageMaker notes

These notes are for MERMAID team's data scientists and developers who are working with SageMaker.


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

Under the MLflow app, MERMAID has an MLflow tracking server set up for PySpacer classifiers.

MERMAID isn't using Canvas, which is a "no-code" offering and likely not as flexible.


## JupyterLab spaces

When you're working with a Jupyter notebook (.ipynb file), it'll default to the "Glue PySpark" kernel, but using this kernel requires additional AWS permissions. You can instead select a different kernel such as "Python 3 (ipykernel)".
