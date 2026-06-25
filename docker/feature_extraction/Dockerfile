# Container for running scripts/build_feature_bucket.py as a SageMaker
# Processing Job worker. One container per worker; the launcher
# (scripts/launch_feature_extraction_sagemaker.py) submits N of these
# in parallel against a sharded subset of source IDs.
#
# Base: official PyTorch image from Docker Hub. Ships CUDA, cuDNN,
# Python 3.10, and torch -- everything we need for SageMaker
# Processing Jobs to pull and run on a g5/g4dn instance. We deliberately
# avoid the AWS SageMaker Deep Learning Containers (private ECR, requires
# `aws ecr get-login-password` at build time, occasionally flaky to
# reach from non-AWS networks). SageMaker Processing Jobs only require
# the image to expose an executable entrypoint -- the DLC's
# `sagemaker-training` framework is not used here.

ARG BASE_IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

FROM ${BASE_IMAGE}

# Some pyspacer deps need build toolchains and image libs. The
# pytorch/pytorch image is minimal so we install them explicitly.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Install project + pyspacer extras. We use pip (not uv) inside the
# container; the base image ships pip but not uv, and pulling uv just
# to install once isn't worth the complexity.
COPY pyproject.toml /opt/ml/code/pyproject.toml
COPY README.md /opt/ml/code/README.md
COPY mermaid_classifier /opt/ml/code/mermaid_classifier
COPY scripts /opt/ml/code/scripts

# Editable install so scripts/ stays importable as a top-level module.
# Pulls in pyspacer 0.14.0 (pinned in pyproject.toml) and its transitive
# deps. pip will reconcile torch with whatever pyspacer needs.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[pyspacer]"

COPY docker/feature_extraction/entrypoint.sh /opt/ml/code/entrypoint.sh
RUN chmod +x /opt/ml/code/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AWS_DEFAULT_REGION=us-east-1 \
    SPACER_EXTRACTORS_CACHE_DIR=/opt/ml/cache/spacer-extractors

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
