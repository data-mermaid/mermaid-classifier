"""Guards that importing region_eval.features stays inside the evaluation lane.

The evaluation CLI reaches this module with no training run in progress, so
importing it must not run the training scripts' logging setup -- which opens
`train.log` in truncate mode in the current working directory -- or pull
mlflow and pydantic-settings into sys.modules. Both arrive only through
``mermaid_classifier.pyspacer.utils.logging_config_for_script``, called at
module scope by whichever module holds the shared S3 downloader; duckdb and
boto3 are not checked here because region_eval already depends on them
directly (probe_set.py and the S3 downloader itself), so their presence is not
a regression this test could distinguish from normal operation.

Each import runs in a fresh subprocess with its own working directory, so an
already-imported module in the test session, or a stray train.log in this
repo, cannot mask a regression.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TRAINING_ONLY_MODULES = ("mlflow", "pydantic_settings")

_SCRIPT = (
    "import sys\n"
    "import mermaid_classifier.region_eval.features  # noqa: F401\n"
    f"leaked = [m for m in {_TRAINING_ONLY_MODULES!r} if m in sys.modules]\n"
    "if leaked:\n"
    "    sys.stderr.write('training-only module(s) imported: ' + ','.join(leaked))\n"
    "    raise SystemExit(1)\n"
    "print('ok')\n"
)


class RegionEvalImportDecouplingTest(unittest.TestCase):
    def _import_in_subprocess(self, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", _SCRIPT],
            cwd=cwd,
            capture_output=True,
            text=True,
        )

    def test_importing_features_does_not_truncate_train_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_log = Path(tmp) / "train.log"
            train_log.write_text("training log content")

            result = self._import_in_subprocess(Path(tmp))

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(train_log.read_text(), "training log content")

    def test_importing_features_does_not_pull_in_mlflow_or_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._import_in_subprocess(Path(tmp))

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
