"""Guards that importing s3_utils stays stdlib-only.

`parse_s3_uri` and `is_s3_uri` are the lightweight surface the module's
docstring advertises; only `download_features_parallel`'s nested `_download`
needs `spacer.aws.get_s3_resource`, and that import is deferred into
`download_features_parallel` rather than sitting at module scope. The import
runs in a fresh subprocess, so an already-imported spacer in the test
session can't mask a regression.
"""

import subprocess
import sys
import unittest

_SCRIPT = (
    "import sys\n"
    "import mermaid_classifier.common.s3_utils  # noqa: F401\n"
    "if 'spacer' in sys.modules:\n"
    "    sys.stderr.write('spacer was imported via s3_utils')\n"
    "    raise SystemExit(1)\n"
    "print('ok')\n"
)


class S3UtilsImportTest(unittest.TestCase):
    def test_importing_s3_utils_does_not_import_spacer(self):
        result = subprocess.run(
            [sys.executable, "-c", _SCRIPT],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
