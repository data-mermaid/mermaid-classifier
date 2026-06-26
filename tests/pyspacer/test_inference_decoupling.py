"""Guards the inference/training dependency split.

Loading a trained classifier for inference (mermaid-classifier[inference]) must
not require the training-only settings deps (pydantic-settings, psutil). That
holds only if importing the ``mermaid_classifier.pyspacer`` subpackage does NOT
import ``mermaid_classifier.pyspacer.settings`` (whose import pulls those deps
and used to run as a side effect of the subpackage ``__init__``).

We assert this in a fresh subprocess so an already-imported settings module in
the test session can't mask a regression.
"""

import subprocess
import sys
import unittest


class InferenceDecouplingTest(unittest.TestCase):
    CHILD = """
import sys
import mermaid_classifier.pyspacer  # noqa: F401
mod = "mermaid_classifier.pyspacer.settings"
if mod in sys.modules:
    raise SystemExit(
        "importing mermaid_classifier.pyspacer must not import settings "
        "(breaks the [inference] dependency split)"
    )
print("ok")
"""

    def test_importing_pyspacer_subpackage_does_not_import_settings(self):
        result = subprocess.run(
            [sys.executable, "-c", self.CHILD],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
