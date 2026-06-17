"""Tests for the portable classifier artifact (model.pt + model.json)."""
import subprocess
import sys
import unittest


class ScaffoldTest(unittest.TestCase):
    def test_constants_and_exceptions_exported(self):
        from mermaid_classifier.pyspacer.inference import (
            SCHEMA_VERSION, TASK_NAME, ParityError, ManifestError,
        )
        self.assertEqual(SCHEMA_VERSION, 1)
        self.assertEqual(TASK_NAME, "pyspacer_mlp_classifier")
        self.assertTrue(issubclass(ParityError, Exception))
        self.assertTrue(issubclass(ManifestError, Exception))

    def test_importing_inference_does_not_import_settings(self):
        # The [inference] extra must not pull training-only settings deps.
        child = (
            "import sys\n"
            "import mermaid_classifier.pyspacer.inference  # noqa: F401\n"
            "mod = 'mermaid_classifier.pyspacer.settings'\n"
            "raise SystemExit(1 if mod in sys.modules else 0)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
