"""
Train and classify from multisource (or project) using featurevector files provided on S3.

"""
import json
import logging
import os
import pickle
import threading
from datetime import datetime

import boto3
import duckdb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import pyarrow as pa
import s3fs
from botocore.exceptions import BotoCoreError, ClientError

from pyarrow import fs
from sklearn.model_selection import train_test_split

from spacer.data_classes import ImageLabels
from spacer.messages import DataLocation, TrainClassifierMsg, TrainingTaskLabels
from spacer.task_utils import preprocess_labels
from spacer.tasks import (
    train_classifier,
)
from spacer.task_utils import preprocess_labels


# Set up logging 
logging.basicConfig(
    filename='mod_res.log',
     filemode='w',
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
def log_memory_usage(message):
    memory_usage = psutil.virtual_memory()
    logger.info(f"{message} - Memory usage: {memory_usage.percent}%")

logger = logging.getLogger(__name__)  # Create a logger for your script

logger.info('Setting up connections')
log_memory_usage('Initial memory usage')
# Set up connections
# Set up Sources
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
bucketname = "coralnet-mermaid-share"
prefix = "coralnet_public_features/"

with open('secrets.json') as f:
    secrets = json.load(f)
# Use pyarrow to read the parquet file from S3
fs = fs.S3FileSystem(
    region=secrets["AWS_REGION"],
    access_key=secrets["AWS_ACCESS_KEY_ID"],
    secret_key=secrets["AWS_SECRET_ACCESS_KEY"]
)

# TODO check this is the most efficient way to load in parquet
# Load Parquet file from S3 bucket using s3_client
parquet_file = "pyspacer-test/allsource/selected_subsample_ModRes_reduced.parquet"
logger.info('Reading parquet file from S3 bucket')
log_memory_usage('Memory usage after reading parquet file')


#selected_sources = pq.read_table(parquet_file, filesystem=fs)
selected_sources = pq.read_table(parquet_file, filesystem=fs)
# Connect to DuckDB
logger.info('Connecting to DuckDB')
conn = duckdb.connect()

selected_sources = conn.execute("SELECT * FROM selected_sources").fetchdf()
log_memory_usage('Memory usage after wrangling in DuckDB')
#
# Remove label counts lower than 1
count_labels = selected_sources["Label ID"].value_counts()
selected_sources = selected_sources[selected_sources["Label ID"].isin(count_labels[count_labels > 1].index)]

# Create a dictionary of the labels to go into existing pyspacer code
logger.info('Restructure as Tuples for ImageLabels')

rng = np.random.RandomState(0)
train_labels, val_test_labels = train_test_split(selected_sources,
                                                test_size=0.20,
                                                random_state=rng,
                                                shuffle=True,
                                                stratify=selected_sources["Label ID"])
# Split the validation and test sets

val_labels, test_labels = train_test_split(val_test_labels, test_size=0.5, random_state=rng)

# Create a dictionary of the labels to go into existing pyspacer code
train_labels = {
    f"{key}": [tuple(x) for x in group[["Row", "Column", "Label ID"]].values]
    for key, group in train_labels.groupby("key")
}

val_labels = {
    f"{key}": [tuple(x) for x in group[["Row", "Column", "Label ID"]].values]
    for key, group in val_labels.groupby("key")
}

test_labels = {
    f"{key}": [tuple(x) for x in group[["Row", "Column", "Label ID"]].values]
    for key, group in test_labels.groupby("key")
}
log_memory_usage('Memory usage after creating train_labels_data and val_labels_data')
logger.info('Create TrainClassifierMsg')

train_msg = TrainClassifierMsg(
    job_token="mulitest",
    trainer_name="minibatch",
    nbr_epochs=10,
    clf_type="MLP",
    labels = TrainingTaskLabels(
    train = ImageLabels(train_labels),
    val = ImageLabels(val_labels),
    ref = ImageLabels(test_labels)
    ),
    features_loc=DataLocation("s3", bucket_name=bucketname, key=""),
    previous_model_locs=[],
    model_loc=DataLocation(
        "s3", bucket_name="pyspacer-test", key="allsource" + f"/classifier_ba_{current_time}.pkl"
    ),
    valresult_loc=DataLocation(
        "s3", bucket_name="pyspacer-test", key="allsource" + f"/valresult_ba_{current_time}.json"
    ),
    feature_cache_dir=TrainClassifierMsg.FeatureCache.AUTO,
)
logger.info('Train Classifier')
log_memory_usage('Memory usage before training')
print('Training Classifier')
return_msg = train_classifier(train_msg)

logger.info(f'Train time: {return_msg.runtime:.1f} s')
log_memory_usage('Memory usage after training')
logger.info('Write return_msg to S3')


print(f"Train time: {return_msg.runtime:.1f} s")

path = f'allsource/return_msg_ba_{current_time}.pkl'

# Use pickle to serialize the object
return_msg_bytes = pickle.dumps(return_msg)
# Use s3fs to write the bytes to the file
with s3.open(path, 'wb') as f:
    f.write(return_msg_bytes)

ref_accs_str = ", ".join([f"{100*acc:.1f}" for acc in return_msg.ref_accs])

print("------------------------------")

print(f"New model's accuracy: {100*return_msg.acc:.1f}%")

logger.info(ref_accs_str)
logger.info(f'New model accuracy: {100*return_msg.acc:.1f}%')
logger.info(f' New model accuracy progression: {ref_accs_str}')
logger.info('Done --------- Closing DuckDB connection')
duckdb.close(conn)


