import os
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings to be read from environment variables or a .env file.

    The below names are lower_case, but in env vars or .env they
    should be UPPER_CASE.
    (Note: that matters for Linux/Mac, but might not for Windows,
    which is case-insensitive about that)

    Though none of the settings are 100% strictly required, some are
    needed in common cases.
    """
    # ML inputs

    coralnet_train_data_bucket: str = 'coral-reef-training'
    mermaid_train_data_bucket: str = 'coral-reef-training'
    # Annotation file paths/patterns require settings to be overrideable
    # for unit tests, because these files are read in with DuckDB,
    # and DuckDB is not easy (if even possible) to use with Python
    # mock, being C++ under the hood.
    coralnet_annotations_csv_pattern: str = (
        # The placeholders get filled in with a format() call during use.
        # Custom patterns don't HAVE to include the placeholders.
        # They just get used if they're present.
        's3://{coralnet_train_data_bucket}'
        '/s{source_id}/annotations.csv'
    )
    mermaid_annotations_parquet_pattern: str = (
        's3://{mermaid_train_data_bucket}'
        '/mermaid/mermaid_confirmed_annotations.parquet'
    )
    weights_location: str | None = None
    aws_region: str = 'us-east-1'
    aws_anonymous: Literal['False', 'True'] = 'False'
    aws_key_id: str | None = None
    aws_secret: str | None = None
    aws_session_token: str | None = None

    # Other

    mlflow_tracking_server: str | None = None
    training_inputs_percent_missing_allowed: int = 0
    spacer_extractors_cache_dir: str | None = None
    # Yes, this is str. After it gets through the settings machinery,
    # pyspacer will convert to int.
    spacer_ref_set_max_size: str | None = None
    mlflow_http_request_max_retries: str | None = None
    mlflow_default_experiment_name: str | None = None

    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8')


settings = Settings()


def set_env_vars_for_packages():
    """
    This allows certain spacer variables and MLflow variables to be
    configured with .env, rather than having to be manually set as
    env vars.
    """
    setting_to_env_name = dict(
        # Filesystem directory to use for caching extractor weights that
        # were downloaded from S3 or from a URL.
        # This is required if loading weights from such a source.
        spacer_extractors_cache_dir='SPACER_EXTRACTORS_CACHE_DIR',
        # This pyspacer training setting is:
        # - A cap on the reference set size, as a number of point-features
        # - The size of batches used during training
        # Raising this can better accommodate rare classes in large training
        # runs, and can improve the trainer's ability to calibrate between
        # epochs.
        # However, this setting is also tied to training's memory usage,
        # so monitor memory usage when increasing this setting.
        spacer_ref_set_max_size='SPACER_TRAINING_BATCH_LABEL_COUNT',
        aws_region='SPACER_AWS_REGION',
        # If True, AWS is accessed without any credentials, which can
        # simplify setup while still allowing access to public S3 files.
        aws_anonymous='SPACER_AWS_ANONYMOUS',
        # Accessing private AWS data is most easily done by using the
        # instance metadata service within AWS, but here are other
        # ways. For example, log into AWS with SSO, choose a role to view
        # a temporary key+secret+token for, and plug those 3 values in.
        aws_key_id='SPACER_AWS_ACCESS_KEY_ID',
        aws_secret='SPACER_AWS_SECRET_ACCESS_KEY',
        aws_session_token='SPACER_AWS_SESSION_TOKEN',
        # When retrieving or saving models, this is the number of
        # exponential-backoff retries that MLflow will attempt when it's
        # having trouble connecting.
        # Default 7, but that takes forever; 2 is recommended.
        mlflow_http_request_max_retries='MLFLOW_HTTP_REQUEST_MAX_RETRIES',
    )
    for setting_name, env_var_name in setting_to_env_name.items():
        var_value = getattr(settings, setting_name)
        if var_value:
            # Ensure the value's set in the OS environment
            # so that spacer or MLflow sees it.
            os.environ[env_var_name] = var_value
