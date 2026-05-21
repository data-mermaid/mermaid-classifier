#!/usr/bin/env bash
# SageMaker passes arbitrary args through to ENTRYPOINT. Use `exec` so
# signals (SageMaker sends SIGTERM on stop) reach Python directly.
set -euo pipefail
cd /opt/ml/code
# SageMaker invokes the container with `train` as the first positional arg
# by convention (containers may also implement `serve`). We only train, so
# drop it before forwarding to Python.
if [ "${1:-}" = "train" ]; then
    shift
fi
exec python -u scripts/sagemaker_train_entrypoint.py "$@"
