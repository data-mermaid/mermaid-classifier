# Container for running scripts/build_feature_set.py as a SageMaker
# Processing Job worker on a GPU instance. Launched by
# scripts/launch_processing.py with `--shard i/n` fan-out. Pushes to the
# mermaid-classifier-jobs ECR repo under the `features-*` tag prefix.
#
# Build from the mermaid-classifier/ directory:
#   docker buildx build --platform linux/amd64 \
#       -t <ECR_URI>:features-<tag> -f docker/jobs/features.Dockerfile .
#
# See docs/feature_extraction_at_scale.md for the full build/push recipe.

# CUDA 12.8 matches the cu128 torch wheels the `cuda` extra locks.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# git fetches the segdata-spec git dependency; libgl1/libglib2.0-0 back the image libs.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates git build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /usr/local/bin/

# Ubuntu 22.04 ships Python 3.10, so uv installs a managed 3.12 under /opt/python.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /opt/ml/code

RUN uv python install 3.12

# Layer 1: dependencies only, cached while pyproject.toml + uv.lock are unchanged.
# --extra cuda brings cu128 torch; it conflicts with the default cpu group.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --python 3.12 --no-install-project \
        --extra training --extra cuda --no-group cpu

# Layer 2: the project itself (non-editable).
COPY mermaid_classifier/ ./mermaid_classifier/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --python 3.12 --no-editable \
        --extra training --extra cuda --no-group cpu

# Layer 3: scripts + entrypoint. The entrypoint runs the script by path.
COPY scripts ./scripts
COPY docker/jobs/features-entrypoint.sh /opt/ml/code/entrypoint.sh
RUN chmod +x /opt/ml/code/entrypoint.sh

ENV AWS_DEFAULT_REGION=us-east-1 \
    SPACER_EXTRACTORS_CACHE_DIR=/opt/ml/cache/spacer-extractors

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
