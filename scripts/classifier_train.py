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
from mermaid_classifier.training.label_transforms import (
        LabelTransformsOptions, TransformSpec)
from mermaid_classifier.training.sample_weighting import (
        SampleWeightingOptions)


if __name__ == "__main__":
    runner = MLflowTrainingRunner(
        dataset_options=DatasetOptions(
            # Specifying False here means you're only training on CoralNet sources.
            include_mermaid=False,
            coralnet_sources_csv='../sagemaker/sources/CoralNetSourcesFirst10.csv',
            # label_rollup_spec_csv='../sagemaker/labels/ba_rollup_top_level.csv',
            excluded_labels_csv='../sagemaker/labels/inspecific-top-level.csv',
            drop_growthforms=False,
            #annotation_limit=200000,
            # Optional sample weighting. Default tree_balanced_ba_flat_gf
            # uses the BA hierarchy for sibling balancing and a flat
            # per-GF inverse-frequency factor. Other registered
            # strategies: leaf_inverse, decomposed, effective_number.
            # Pass None (or omit) to disable weighting entirely.
            weighting=SampleWeightingOptions(
                strategy='effective_number',
                alpha=0.1,
                weight_ratio_cap=500.0,  # bound max:min ratio of weights
            ),
            # Optional label-transforms pipeline. Drops/merges rare
            # classes from train/ref/val so all three splits and the
            # trained classifier share one consistent label space.
            # Available transforms (see
            # mermaid_classifier.training.label_transforms): drop_rare,
            # merge_rare. Use a list of TransformSpec to compose stages.
            label_transforms=LabelTransformsOptions(
                enabled=True,
                pipeline=[
                    TransformSpec(name='drop_rare', params={'min_count': 50}),
                ],
            ),
        ),
        mlflow_options=MLflowOptions(
            experiment_name="pyspacer-beta-test",
            model_name='GregTest',
            # Logs all input annotations to MLflow. Also possible to just log a subset.
            #annotations_to_log='all',
        ),
    )
    runner.run()
