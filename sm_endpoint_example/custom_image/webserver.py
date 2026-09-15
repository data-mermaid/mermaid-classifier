# Example we're using some parts from:
# https://github.com/niklas-palm/sagemaker-endpoint-custom-container/blob/bf8a374cf85cd7a0af72725a8cda27330ea3045b/custom_image/app.py

import os
from pathlib import Path
from urllib.parse import urlparse

import flask
from spacer.extractors import EfficientNetExtractor
from spacer.messages import DataLocation, ClassifyImageMsg
from spacer.tasks import classify_image


app = flask.Flask(__name__)

# Use ProxyFix middleware for running behind a proxy
# TODO: Is this needed?
# app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)


@app.route("/ping", methods=["GET"])
def ping():
    """
    Healthcheck function.
    """
    return "pong"


@app.route('/invocations', methods=["POST"])
def invocations(request):

    image_s3_uri = urlparse(request.data['image_uri'])
    image_stem = Path(image_s3_uri.path).stem
    image_loc = DataLocation(
        's3',
        bucket_name=image_s3_uri.netloc,
        # url.path probably begins with a slash, which isn't
        # what we want when ultimately passing this to boto's
        # Object().
        key=image_s3_uri.path.strip('/'),
    )

    model_path = os.environ.get('SM_MODEL_DIR')
    classifier_loc = DataLocation(
        'filesystem',
        key=f'{model_path}/classifier.pkl',
    )
    weights_loc = DataLocation(
        'filesystem',
        key=f'{model_path}/weights.pt',
    )

    message = ClassifyImageMsg(
        job_token=f'{image_stem}_classify',
        image_loc=image_loc,
        extractor=EfficientNetExtractor(
            data_locations=dict(weights=weights_loc),
        ),
        # (row, column) tuples specifying pixel locations in the image.
        # Note that row is y, column is x.
        rowcols=request.data['rowcols'],
        classifier_loc=classifier_loc,
    )
    return_message_as_dict = classify_image(message).serialize()
    return flask.Response(
        return_message_as_dict, mimetype='application/json')


if __name__ == "__main__":
    app.run(debug=True)
