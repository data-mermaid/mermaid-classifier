#!/usr/bin/env bash
# SageMaker passes arbitrary args through to ENTRYPOINT. Use `exec` so
# signals (SageMaker sends SIGTERM on stop) reach Python directly.
set -euo pipefail
cd /opt/ml/code
exec python -u scripts/sagemaker_train_entrypoint.py "$@"
