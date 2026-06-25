#!/usr/bin/env bash
# SageMaker Processing Job container entrypoint.
#
# The launcher (scripts/launch_feature_extraction_sagemaker.py) passes
# build_feature_bucket.py's CLI args directly via SageMaker's
# AppSpecification.ContainerArguments, which arrive here as "$@". We
# just exec the script. The script's --aws-profile flag is passed as ""
# by the launcher, telling its SSO bootstrap to fall through to the
# default credential chain (i.e. the SageMaker task role).

set -euo pipefail

# Pyspacer downloads the EfficientNet weights from S3 on first use and
# caches them under this directory. Ensure it exists before the worker
# starts -- pyspacer assumes the path is writable but doesn't create it.
mkdir -p "${SPACER_EXTRACTORS_CACHE_DIR:-/opt/ml/cache/spacer-extractors}"

exec python /opt/ml/code/scripts/build_feature_bucket.py "$@"
