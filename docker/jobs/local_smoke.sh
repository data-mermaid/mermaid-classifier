#!/usr/bin/env bash
# Local smoke test for the mermaid-classifier jobs images.
#
# This is a *packaging* smoke, not a behavioural test: it builds the
# image and checks that the entrypoint's Python modules import cleanly
# inside the container. It deliberately does NOT exercise the training
# or feature-extraction pipelines — those run in CloudWatch under
# `scripts/launch_training.py` / `launch_processing.py`.
#
# Usage: bash docker/jobs/local_smoke.sh [training|features]
# Default: training.

set -euo pipefail

KIND="${1:-training}"
case "$KIND" in
    training|features) ;;
    *) echo "Usage: $0 [training|features]" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/../.."

DOCKERFILE="docker/jobs/${KIND}.Dockerfile"
IMAGE="mermaid-classifier-jobs:${KIND}-smoke-local"

echo "[smoke] Building ${IMAGE} from ${DOCKERFILE}..."
docker buildx build \
    --platform linux/amd64 \
    --load \
    -t "${IMAGE}" \
    -f "${DOCKERFILE}" \
    .

if [ "$KIND" = "training" ]; then
    echo "[smoke] Verifying entrypoint module imports..."
    docker run --rm --entrypoint python "${IMAGE}" -c "
from mermaid_classifier.sagemaker.config import TrainingRunConfig
from mermaid_classifier.sagemaker.launcher_config import RunConfig, parse_run_config
# Verify the in-container entrypoint script parses and imports.
import importlib.util
spec = importlib.util.spec_from_file_location(
    'sagemaker_train_entrypoint',
    '/opt/ml/code/scripts/sagemaker_train_entrypoint.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('imports OK')
"
else
    # The features image's `--help` exec imports torch+CUDA at top of
    # build_feature_bucket.py, which segfaults under qemu emulation on
    # ARM hosts. Use a lighter check that runs only the Python
    # interpreter and verifies the worker script is present.
    echo "[smoke] Verifying worker script is reachable in the image..."
    docker run --rm --entrypoint python "${IMAGE}" -c "
import os
worker = '/opt/ml/code/scripts/build_feature_bucket.py'
assert os.path.exists(worker), f'worker script missing: {worker}'
# Verify Python can compile (catches syntax errors) without executing
# the top-level torch import.
with open(worker) as f:
    compile(f.read(), worker, 'exec')
print('worker script reachable + compiles')
"
fi

echo "[smoke] OK"
