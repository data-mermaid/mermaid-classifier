"""Fixtures and helpers shared by more than one test package.

A fixture used inside a single package stays in that package (see
pyspacer/metrics_test_helpers.py); this is where the cross-package ones live,
so no test module has to import from another test module.

The suite runs with tests/ on sys.path, so this package's name must not match
an installed distribution -- see the shadowing note in CLAUDE.md.
"""

import os

# MLflow phones home to mlflow-telemetry.io the first time it is imported.
# Set before any test imports it; this package is imported by everything that
# reaches the MLflow lane, and the suite has no conftest to hang it on.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
