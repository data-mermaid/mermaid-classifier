# CPU-only image for the mermaid-classifier SageMaker TrainingJob.
# Build from the mermaid-classifier/ directory:
#   docker buildx build --platform linux/amd64 \
#       -t <ECR_URI>:training-<tag> -f docker/jobs/training.Dockerfile .
#
# See docker/jobs/CLAUDE.md for the full build/push recipe.

FROM python:3.12-slim

# Build deps for pyarrow, duckdb, and the rest of the pyspacer chain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# uv: fast, lockfile-driven installs. Pinned by digest tag for reproducibility.
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /usr/local/bin/

# Don't write .pyc; don't buffer stdout (SageMaker tails it to CloudWatch).
# UV_LINK_MODE=copy avoids hardlink warnings when the cache is on a different fs
# (e.g. a BuildKit cache mount). UV_COMPILE_BYTECODE=1 precompiles .pyc once
# at install time so the first import inside SageMaker isn't paying that cost.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /opt/ml/code

# Layer 1: resolve and install dependencies only. This layer is cached as long
# as pyproject.toml + uv.lock are unchanged, so code edits don't reinstall torch.
# torch + torchvision come from the pytorch-cpu index (see [tool.uv.sources] in
# pyproject.toml) so no CUDA wheels are downloaded.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra training

# Layer 2: install the project itself (non-editable).
COPY mermaid_classifier/ ./mermaid_classifier/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --extra training

# Layer 3: entrypoint + SageMaker scripts.
COPY scripts/sagemaker_train_entrypoint.py ./scripts/
COPY docker/jobs/training-entrypoint.sh ./docker/jobs/
RUN chmod +x ./docker/jobs/training-entrypoint.sh

ENTRYPOINT ["/opt/ml/code/docker/jobs/training-entrypoint.sh"]
