import logging

# python-decouple makes configuration more explicit compared to python-dotenv.
from decouple import config
import mlflow


class Settings:
    def __init__(self):
        # This is coded up a bit oddly and boilerplate-ly because it was
        # first an attempt to use pydantic-settings, but something about it
        # wasn't working. So, minimal changes were made to use python-decouple
        # instead of pydantic.
        self.SPACER_EXTRACTORS_CACHE_DIR = config(
            'SPACER_EXTRACTORS_CACHE_DIR')

        self.MLFLOW_TRACKING_SERVER_ARN = config('MLFLOW_TRACKING_SERVER_ARN')

        self.MLFLOW_DEFAULT_EXPERIMENT_NAME = config(
            'MLFLOW_DEFAULT_EXPERIMENT_NAME')

        self.WEIGHTS_LOCATION = config('WEIGHTS_LOCATION')


def logging_config_for_script(name):

    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            }
        },
        'handlers': {
            # Print info messages; file-log info and debug messages
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


def mlflow_connect():
    # Navigate to your MLflow tracking server in SageMaker Studio, click the
    # Copy icon to get the tracking server's ARN, and paste that in as this
    # env value.
    mlflow.set_tracking_uri(uri=Settings().MLFLOW_TRACKING_SERVER_ARN)

    try:
        # Do something to test the server connection.
        mlflow.search_experiments(max_results=1)
    except mlflow.exceptions.MlflowException as e:
        # TODO: It takes far too long to give up and enter this block.
        #  Ideally there'd be a way to set the max retries, or force it
        #  to give up after N seconds.
        if "Max retries exceeded" in str(e):
            raise RuntimeError(
                "Could not connect to the MLflow tracking server."
                " Is the tracking server up and running?"
            )
