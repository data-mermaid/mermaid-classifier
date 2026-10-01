import csv
import json
from operator import itemgetter
import urllib.request

from spacer.messages import ClassifyReturnMsg


ENDPOINT_NAME = 'pyspacer-inference'

# https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html
# Endpoints are scoped to an individual account, and are not public.
# The URL does not contain the account ID, but Amazon SageMaker AI
# determines the account ID from the authentication token that is
# supplied by the caller.
ENDPOINT_URL = f'/endpoints/{ENDPOINT_NAME}/invocations'

IMAGE_URI = 's3://coral-reef-training/mermaid/0032dba6-8357-42e2-bace-988f99032286.png'

ROWCOLS_PATH = '0032dba6_points.csv'


if __name__ == '__main__':

    rowcols = []
    with open(ROWCOLS_PATH) as rowcols_csv:
        reader = csv.DictReader(rowcols_csv)
        for csv_row in reader:
            row = int(csv_row['row'])
            column = int(csv_row['column'])
            rowcols.append((row, column))

    request_data = dict(
        image_uri=IMAGE_URI,
        rowcols=rowcols,
    )

    request = urllib.request.Request(
        ENDPOINT_URL,
        data=json.dumps(request_data).encode(),
        method='POST',
    )
    response = urllib.request.urlopen(request)

    serialized_return_msg = response.data
    return_message = ClassifyReturnMsg.deserialize(
        json.loads(serialized_return_msg))

    # Print results in a simple format since this is just a
    # proof of concept.
    labels = return_message.classes
    for row, column, msg_scores in return_message.scores:
        predictions_top_first = sorted(
            zip(labels, msg_scores), key=itemgetter(1), reverse=True)
        top_label, top_score = predictions_top_first[0]
        print(f"ID {top_label} - {top_score} confidence")
