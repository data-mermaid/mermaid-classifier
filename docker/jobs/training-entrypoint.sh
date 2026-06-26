#!/usr/bin/env bash
# SageMaker training-container entrypoint shim. Dispatches to the
# script named by CONTAINER_ENTRYPOINT_SCRIPT (set by the launcher
# from the YAML's job.entrypoint). Falls back to the historic default
# for backward compat with one-off `docker run`s that don't set it.
set -euo pipefail
cd /opt/ml/code

# SageMaker invokes the container with `train` as the first positional
# arg. Drop it so it doesn't leak into the script's argv.
if [ "${1:-}" = "train" ]; then
    shift
fi

SCRIPT="${CONTAINER_ENTRYPOINT_SCRIPT:-scripts/sagemaker_train_entrypoint.py}"
exec python -u "${SCRIPT}" "$@"
