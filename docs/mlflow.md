# MLflow notes


## MLflow tracking server

To log models to MLflow or look up info on logged models, you need to have an MLflow tracking server running.

- On a local development machine, this can be done with `mlflow ui --port 8080` in a terminal with the mermaid-classifier Python environment activated. Then the server can be visited using a web browser at a URL like `http://localhost:8080`. Artifacts and runs get saved to the current directory by default, but this can be [configured](https://mlflow.org/docs/latest/ml/tracking/server/#tracking-server-artifact-store).

- On SageMaker, this means navigating to SageMaker Studio > MLflow, and starting the MLflow tracking server. You'll probably need to wait around 20 minutes for it to start up. Then the server can be visited by clicking the ... next to the server, and choosing Open MLflow. Artifacts and runs get saved to the location specified in the SageMaker-defined server options.