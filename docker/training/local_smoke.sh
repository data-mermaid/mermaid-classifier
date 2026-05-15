#!/usr/bin/env bash
# Local Docker smoke test for the mermaid-classifier training image.
#
# Goal: prove that the image starts, reads the example config, applies
# env vars, and gets as far as constructing MLflowTrainingRunner.
# Without real AWS credentials and a real MLflow server the run cannot
# complete; we treat "made it past build_options stage" as success.
#
# Usage (from mermaid-classifier/):
#   bash docker/training/local_smoke.sh
#
# Pass --with-aws to mount your local ~/.aws and attempt a fuller run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE_TAG=mermaid-classifier-training:smoke
EXAMPLE_DIR="${REPO_ROOT}/sagemaker/configs/example"

WITH_AWS=0
for arg in "$@"; do
    case "$arg" in
        --with-aws) WITH_AWS=1 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [[ ! -d "$EXAMPLE_DIR" ]]; then
    echo "Missing example config at $EXAMPLE_DIR" >&2
    exit 1
fi

echo "==> Building image $IMAGE_TAG"
docker buildx build --platform linux/amd64 \
    -t "$IMAGE_TAG" \
    -f "${REPO_ROOT}/docker/training/Dockerfile" \
    "$REPO_ROOT"

echo "==> Running container with example config mounted"
DOCKER_ARGS=(
    --rm
    --platform linux/amd64
    -v "${EXAMPLE_DIR}:/opt/ml/input/data/config:ro"
)
if [[ "$WITH_AWS" -eq 1 ]]; then
    DOCKER_ARGS+=( -v "${HOME}/.aws:/root/.aws:ro" )
fi

LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT

# Capture both stdout and stderr.
set +e
docker run "${DOCKER_ARGS[@]}" "$IMAGE_TAG" 2>&1 | tee "$LOG_FILE"
RC=$?
set -e

echo
echo "==> Smoke verification"
if grep -q '\[stage:build_options\] ENTER' "$LOG_FILE"; then
    echo "OK: reached build_options stage"
else
    echo "FAIL: build_options stage marker missing in container log"
    exit 1
fi

if grep -q '\[stage:apply_env\] EXIT' "$LOG_FILE"; then
    echo "OK: applied env vars"
else
    echo "FAIL: apply_env did not complete"
    exit 1
fi

echo "==> Smoke test passed (container exit code was $RC; non-zero is"
echo "    expected without real AWS creds and an MLflow server)."
