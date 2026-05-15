# This test package is named 'sagemaker' to mirror the AWS SageMaker SDK
# package structure. The installed sagemaker SDK submodules (estimator,
# inputs, session, etc.) must remain importable as 'sagemaker.<sub>' even
# though this package shadows the SDK at the top level.
#
# Solution: extend __path__ with the installed SDK's path from site-packages
# so 'from sagemaker.estimator import Estimator' resolves to the real SDK.
import site
from pathlib import Path

for _site in site.getsitepackages():
    _candidate = Path(_site) / "sagemaker"
    if _candidate.is_dir() and (_candidate / "estimator.py").exists():
        _sdk_path = str(_candidate)
        if _sdk_path not in __path__:
            __path__.append(_sdk_path)
        break
