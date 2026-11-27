"""The holdout_groups.csv fold vocabulary and site names.

Shared by the training run, the SageMaker config and the selection/CV scripts.
"""

from __future__ import annotations

HOLDOUT_TEST_FOLD = "test"
CORALNET_SITE = "coralnet"
MERMAID_SITE = "mermaid"
SITES = (CORALNET_SITE, MERMAID_SITE)


def is_cv_fold(value: str) -> bool:
    """True for a cross-validation fold label: a non-empty string of ASCII digits."""
    return value.isascii() and value.isdigit()
