"""Guards the inference/training dependency split.

Loading a trained classifier for inference (mermaid-classifier[inference]) must
not require the training-only settings deps (pydantic-settings, psutil). That
holds only if importing the inference-facing packages does NOT import
``mermaid_classifier.pyspacer.settings`` (whose import pulls those deps and used
to run as a side effect of the subpackage ``__init__``).

Both serving-relevant entry points are guarded: the ``mermaid_classifier.pyspacer``
subpackage and the ``mermaid_classifier.pyspacer.inference`` module that serving
code imports directly. Each import is checked in a fresh subprocess so an
already-imported settings module in the test session can't mask a regression.
"""

import subprocess
import sys
import unittest

_SETTINGS_MODULE = "mermaid_classifier.pyspacer.settings"

# Entry points the [inference] extra must be able to import without pulling in
# the training-only settings layer.
_INFERENCE_IMPORTS = [
    "mermaid_classifier.pyspacer",
    "mermaid_classifier.pyspacer.inference",
]


def _child_script(import_target: str) -> str:
    return (
        "import sys\n"
        f"import {import_target}  # noqa: F401\n"
        f"if {_SETTINGS_MODULE!r} in sys.modules:\n"
        f"    sys.stderr.write('settings was imported via {import_target}')\n"
        "    raise SystemExit(1)\n"
        "print('ok')\n"
    )


class InferenceDecouplingTest(unittest.TestCase):
    def test_inference_imports_do_not_import_settings(self):
        for import_target in _INFERENCE_IMPORTS:
            with self.subTest(import_target=import_target):
                result = subprocess.run(
                    [sys.executable, "-c", _child_script(import_target)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
