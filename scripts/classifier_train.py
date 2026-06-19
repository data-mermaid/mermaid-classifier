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

from mermaid_classifier.pyspacer.train import (
        DatasetOptions, MLflowOptions, MLflowTrainingRunner)


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
            # Logs all input annotations to MLflow (in addition to the
            # always-logged validation split). Also possible to just log a subset.
            #extra_annotations_to_log='all',
        ),
    )
    runner.run()