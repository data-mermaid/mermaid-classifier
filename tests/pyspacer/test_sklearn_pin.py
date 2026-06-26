"""CI guard: the installed scikit-learn must match the version the portable
artifact's parity gate was proven against. A silent sklearn bump can change
CalibratedClassifierCV / _SigmoidCalibration semantics, so this test fails the
build until parity is re-proven and the pin + constant are updated together."""

import unittest
from importlib.metadata import version

from mermaid_classifier.pyspacer.inference import PARITY_PROVEN_SKLEARN


class SklearnPinTest(unittest.TestCase):
    def test_installed_sklearn_matches_parity_proven(self):
        installed = version("scikit-learn")
        self.assertEqual(
            installed,
            PARITY_PROVEN_SKLEARN,
            msg=(
                f"scikit-learn {installed} != parity-proven"
                f" {PARITY_PROVEN_SKLEARN}. To bump: (1) update the"
                " scikit-learn pin in pyproject.toml and run `uv lock`,"
                " (2) set PARITY_PROVEN_SKLEARN to the new version,"
                " (3) re-run the live-feature parity gate"
                " (PORTABLE_ARTIFACT_LIVE_MODEL + PORTABLE_ARTIFACT_LIVE_FEATURES)"
                " and confirm max|Δ| < 1e-6 on REAL EfficientNet features."
            ),
        )


if __name__ == "__main__":
    unittest.main()
