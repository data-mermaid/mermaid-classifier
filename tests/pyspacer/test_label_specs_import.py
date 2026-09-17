"""Guards against a filesystem side effect from importing label_specs.

Merely importing a CSV-spec module must not touch the filesystem. Configuring
logging at import time (`logging.config.dictConfig`, which a file handler
truncates on open) would create a log file in whatever directory the process
happens to be running from and reconfigure every logging handler globally, for
a module doing nothing more than defining the exclusion/rollup/filter specs.

The import runs in a subprocess with a freshly created, empty directory as its
cwd, so a module already imported elsewhere in the test session -- or a
leftover file from another test -- can't mask a regression.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LabelSpecsImportSideEffectTest(unittest.TestCase):
    def test_importing_label_specs_creates_no_file_in_the_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [sys.executable, "-c", "import mermaid_classifier.pyspacer.label_specs"],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(
                list(Path(tmp_dir).iterdir()),
                [],
                msg="importing label_specs must not write files into the cwd",
            )


if __name__ == "__main__":
    unittest.main()
