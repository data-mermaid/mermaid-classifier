"""Guards the inference/training dependency split.

Loading a trained classifier for inference (mermaid-classifier[inference])
must not require anything outside that extra's closure. The extra resolves to
pyspacer + scikit-learn and their transitives -- torch, torchvision, numpy,
Pillow, boto3 -- so an import that reaches mlflow, duckdb, pandas, matplotlib
or the settings layer has broken the split, and the serving image would fail
to build rather than merely grow.

Checking a list of modules rather than one name is the point: guarding only
``mermaid_classifier.pyspacer.settings`` would pass for a module that had
grown an ``import duckdb``. Each import runs in a fresh subprocess so an
already-imported module in the test session cannot mask a regression.
"""

import subprocess
import sys
import unittest

# Installed in the dev environment but absent from the [inference] closure,
# so their presence in sys.modules means the import pulled them in.
_TRAINING_ONLY_MODULES = (
    "mermaid_classifier.pyspacer.settings",
    "pydantic_settings",
    "psutil",
    "mlflow",
    "duckdb",
    "pandas",
    "matplotlib",
    "s3fs",
    "pyarrow",
    "bs4",
    "yaml",
)

# Entry points the [inference] extra must be able to import. torch_classifier
# is here because CLAUDE.md names it as part of the serving lane's contract;
# nothing else reaches it, so without an entry of its own it goes unchecked.
_INFERENCE_IMPORTS = [
    "mermaid_classifier.pyspacer",
    "mermaid_classifier.pyspacer.inference",
    "mermaid_classifier.pyspacer.inference.extractor_spec",
    "mermaid_classifier.pyspacer.torch_classifier",
]


def _child_script(import_target: str) -> str:
    return (
        "import sys\n"
        f"import {import_target}  # noqa: F401\n"
        f"leaked = [m for m in {_TRAINING_ONLY_MODULES!r} if m in sys.modules]\n"
        "if leaked:\n"
        f"    sys.stderr.write('{import_target} pulled in: ' + ','.join(leaked))\n"
        "    raise SystemExit(1)\n"
        "print('ok')\n"
    )


class InferenceDecouplingTest(unittest.TestCase):
    def test_inference_imports_stay_inside_the_extra(self):
        for import_target in _INFERENCE_IMPORTS:
            with self.subTest(import_target=import_target):
                result = subprocess.run(
                    [sys.executable, "-c", _child_script(import_target)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("ok", result.stdout)

    def test_the_guard_would_catch_a_leak(self):
        # A test that can only pass is not a guard. Importing a training-only
        # module directly has to trip the same script.
        result = subprocess.run(
            [sys.executable, "-c", _child_script("duckdb")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("duckdb", result.stderr)


if __name__ == "__main__":
    unittest.main()
