# CPU-only image for the mermaid-classifier SageMaker TrainingJob.
# Build from the mermaid-classifier/ directory:
#   docker buildx build --platform linux/amd64 \
#       -t <ECR_URI>:<tag> -f docker/training/Dockerfile .
#
# See docs/training_at_scale.md for the full build/push runbook.

FROM python:3.10-slim

# Build deps for pyarrow, duckdb, and the rest of the pyspacer chain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Don't write .pyc; don't buffer stdout (SageMaker tails it to CloudWatch).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/ml/code

# 1. Install CPU-only PyTorch FIRST. Without this, pyspacer's transitive
#    resolution pulls the default ~2GB CUDA wheel even though we never
#    use GPU at training time.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# 2. Copy the project metadata and install the package with training
#    extras. torch is already satisfied so this is fast.
COPY pyproject.toml ./
COPY mermaid_classifier/ ./mermaid_classifier/
RUN pip install -e .[pyspacer]

# 3. Copy in the entrypoint and SageMaker scripts.
COPY scripts/sagemaker_train_entrypoint.py ./scripts/
COPY docker/training/entrypoint.sh ./docker/training/
RUN chmod +x ./docker/training/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/docker/training/entrypoint.sh"]
