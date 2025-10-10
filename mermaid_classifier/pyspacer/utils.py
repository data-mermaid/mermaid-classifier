from datetime import datetime, timedelta
import logging

import mlflow

from mermaid_classifier.pyspacer.settings import settings


def logging_config_for_script(name):
    """
    Call this to set up a logging config that prints info messages,
    and file-logs info and debug messages.
    """
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            }
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': f'{name}.log',
                # Clear logs of the previous run
                'mode': 'w',
            },
        },
        'loggers': {
            name: {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
            }
        },
    })
    return logging.getLogger(name)


def mlflow_connect() -> timedelta:
    mlflow.set_tracking_uri(uri=settings.mlflow_tracking_server)

    try:
        # Do something to test the server connection.
        time_before_connect = datetime.now()
        mlflow.search_experiments(max_results=1)
    except mlflow.exceptions.MlflowException as e:
        # Note that this may take a long time to reach
        # unless you set MLFLOW_HTTP_REQUEST_MAX_RETRIES to
        # a low number.
        if "Max retries exceeded" in str(e):
            raise RuntimeError(
                "Could not connect to the MLflow tracking server."
                " Is the tracking server up and running?"
            )
        # If it's some other kind of MlflowException, just re-raise
        # for debugging purposes.
        raise e

    time_after_connect = datetime.now()
    # Return the time taken to connect.
    return time_after_connect - time_before_connect
