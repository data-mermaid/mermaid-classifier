import os
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings to be read from environment variables or a .env file.

    The below names are lower_case, but in env vars or .env they
    should be UPPER_CASE.
    (Note: that matters for Linux/Mac, but might not for Windows,
    which is case-insensitive about that)
    """

    # Required settings
    mlflow_tracking_server: str = Field()
    weights_location: str = Field()

    # Optional settings (may need to set in specific cases)
    training_inputs_percent_missing_allowed: int = 0
    spacer_extractors_cache_dir: str | None = None
    spacer_aws_anonymous: Literal['False', 'True'] = 'False'
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
    var_names = [
        # Filesystem directory to use for caching extractor weights that
        # were downloaded from S3 or from a URL.
        'spacer_extractors_cache_dir',
        # If True, AWS is accessed without any credentials, which can
        # simplify setup while still allowing access to public S3 files.
        'spacer_aws_anonymous',
        # When retrieving or saving models, this is the number of
        # exponential-backoff retries that MLflow will attempt when it's
        # having trouble connecting.
        # Default 7, but that takes forever; 2 is recommended.
        'mlflow_http_request_max_retries',
    ]
    for var_name in var_names:
        var_value = getattr(settings, var_name)
        if var_value:
            # Ensure the value's set in the OS environment
            # so that spacer sees it.
            caps_var_name = var_name.upper()
            os.environ[caps_var_name] = var_value
