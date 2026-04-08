import os
import boto3

# Resolve AWS SSO credentials before importing mermaid_classifier,
# whose Settings() object is created at import time and reads env vars then.
os.environ['AWS_PROFILE'] = 'wcs'
session = boto3.Session()
credentials = session.get_credentials()
if credentials:
    creds = credentials.get_frozen_credentials()
    os.environ['SPACER_AWS_ACCESS_KEY_ID'] = creds.access_key
    os.environ['SPACER_AWS_SECRET_ACCESS_KEY'] = creds.secret_key
    if creds.token:
        os.environ['SPACER_AWS_SESSION_TOKEN'] = creds.token

from mermaid_classifier.pyspacer.config import (
    DatasetOptions, MLflowOptions, TrainingOptions)
from mermaid_classifier.pyspacer.train import MLflowTrainingRunner


if __name__ == "__main__":
    runner = MLflowTrainingRunner(
        dataset_options=DatasetOptions(
            # Specifying False here means you're only training on CoralNet sources.
            include_mermaid=False,
            coralnet_sources_csv='../sagemaker/sources/OldCoralNetSourcesToKeep.csv',
            # label_rollup_spec_csv='../sagemaker/labels/ba_rollup_top_level.csv',
            excluded_labels_csv='../sagemaker/labels/inspecific-top-level.csv',
            drop_growthforms=False,
            #annotation_limit=200000,
        ),
        mlflow_options=MLflowOptions(
            experiment_name="pyspacer-beta-test",
            model_name='GregTest',
            # Logs all input annotations to MLflow. Also possible to just log a subset.
            #annotations_to_log='all',
        ),
        training_options=TrainingOptions(
            class_balancing=True,
            epochs=10,
            device='auto',
            # minibatch_size=512,
            # optimizer='adamw',
            # learning_rate=1e-4,
            # weight_decay=1e-4,
            # hidden_layer_sizes=(200, 100),
        ),
    )
    runner.run()