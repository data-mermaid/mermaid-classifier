import os
from typing import Literal

import psutil
from pydantic_settings import BaseSettings, SettingsConfigDict

# EfficientNet feature vector dimensionality (4096 floats).
_FEATURE_DIM = 4096
# Bytes per float64 element (numpy / sklearn default).
_BYTES_PER_FLOAT = 8
_FEATURE_BYTES = _FEATURE_DIM * _BYTES_PER_FLOAT

# Minimum batch size — don't go lower than pyspacer's default.
_MIN_BATCH_SIZE = 5000


def training_batch_size(
    num_classes: int = 300,
) -> tuple[int, float]:
    """
    Calculate batch size based on *currently available* memory.

    Call after data-prep is complete so that
    psutil.virtual_memory().available reflects what the OS actually
    has free — including DuckDB tables, ImageLabels structures, and
    everything else already allocated.

    Accounts for sklearn copy overhead and MLP activation buffers.

    Returns (batch_size, available_gb) so callers can log the memory
    snapshot that was actually used in the calculation.
    """
    available_bytes = psutil.virtual_memory().available
    available_gb = available_bytes / 1e9

    # Per-point peak memory during partial_fit:
    #   1. Feature vector loaded from disk:  4096 × 8 bytes
    #   2. sklearn copies to C-contiguous float64: 4096 × 8 bytes
    #   3. MLP forward/backward activation buffers per layer
    sklearn_copy_bytes = _FEATURE_BYTES  # worst-case full copy

    # MLP hidden_layer_sizes is fixed at the production (500, 300, 100)
    # architecture (see MermaidTrainer / docs/hidden-layer-experiments.md);
    # num_classes is the output layer. sklearn's backprop holds both
    # forward activations and backprop deltas (plus gradient buffers) per
    # layer, so per-sample activation memory is roughly double the
    # forward-pass size.
    activation_units = 500 + 300 + 100 + num_classes
    activation_bytes = 2 * activation_units * _BYTES_PER_FLOAT

    bytes_per_point = _FEATURE_BYTES + sklearn_copy_bytes + activation_bytes

    # Reserve 20% headroom for heap fragmentation, Python GC peaks, and
    # any other transient allocations.
    usable_bytes = available_bytes * 0.80

    batch_size = int(usable_bytes / bytes_per_point)
    return max(batch_size, _MIN_BATCH_SIZE), available_gb


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

    coralnet_train_data_bucket: str = "coral-reef-training"
    mermaid_train_data_bucket: str = "coral-reef-training"
    # Annotation file paths/patterns require settings to be overrideable
    # for unit tests, because these files are read in with DuckDB,
    # and DuckDB is not easy (if even possible) to use with Python
    # mock, being C++ under the hood.
    coralnet_annotations_csv_pattern: str = (
        # The placeholders get filled in with a format() call during use.
        # Custom patterns don't HAVE to include the placeholders.
        # They just get used if they're present.
        "s3://{coralnet_train_data_bucket}/s{source_id}/annotations.csv"
    )
    mermaid_annotations_parquet_pattern: str = (
        "s3://{mermaid_train_data_bucket}/mermaid/mermaid_confirmed_annotations.parquet"
    )
    weights_location: str | None = None
    aws_region: str = "us-east-1"
    aws_anonymous: Literal["False", "True"] = "False"
    aws_key_id: str | None = None
    aws_secret: str | None = None
    aws_session_token: str | None = None

    # Other

    mlflow_tracking_server: str | None = None
    training_inputs_percent_missing_allowed: int = 0
    spacer_extractors_cache_dir: str | None = None
    # Override for training batch size. If None (default),
    # training_batch_size() auto-calculates at runtime. Typed as int so
    # Pydantic validates the env var at startup rather than failing
    # mid-run, after the (expensive) data-prep step.
    spacer_batch_size: int | None = None
    feature_cache_dir: str | None = None
    download_max_workers: int = 50
    mlflow_http_request_max_retries: str | None = None
    mlflow_default_experiment_name: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()


def set_env_vars_for_packages():
    """
    This allows certain spacer variables and MLflow variables to be
    configured with .env, rather than having to be manually set as
    env vars.
    """
    setting_to_env_name = {
        # Filesystem directory to use for caching extractor weights that
        # were downloaded from S3 or from a URL.
        # This is required if loading weights from such a source.
        "spacer_extractors_cache_dir": "SPACER_EXTRACTORS_CACHE_DIR",
        "aws_region": "SPACER_AWS_REGION",
        # If True, AWS is accessed without any credentials, which can
        # simplify setup while still allowing access to public S3 files.
        "aws_anonymous": "SPACER_AWS_ANONYMOUS",
        # Accessing private AWS data is most easily done by using the
        # instance metadata service within AWS, but here are other
        # ways. For example, log into AWS with SSO, choose a role to view
        # a temporary key+secret+token for, and plug those 3 values in.
        "aws_key_id": "SPACER_AWS_ACCESS_KEY_ID",
        "aws_secret": "SPACER_AWS_SECRET_ACCESS_KEY",
        "aws_session_token": "SPACER_AWS_SESSION_TOKEN",
        # When retrieving or saving models, this is the number of
        # exponential-backoff retries that MLflow will attempt when it's
        # having trouble connecting.
        # Default 7, but that takes forever; 2 is recommended.
        "mlflow_http_request_max_retries": "MLFLOW_HTTP_REQUEST_MAX_RETRIES",
    }
    for setting_name, env_var_name in setting_to_env_name.items():
        var_value = getattr(settings, setting_name)
        if var_value:
            # Ensure the value's set in the OS environment
            # so that spacer or MLflow sees it.
            os.environ[env_var_name] = var_value

    # Effectively disable PySpacer's ref set size cap.
    # PySpacer's split_labels() caps the ref set via
    # config.TRAINING_BATCH_LABEL_COUNT. With batch calibration in
    # trainer.py, the ref set no longer needs to fit in memory at once,
    # so we set this to a very large value to let ref_val_ratios alone
    # determine the ref set size.
    os.environ["SPACER_TRAINING_BATCH_LABEL_COUNT"] = str(2**31 - 1)
