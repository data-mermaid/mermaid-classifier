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


import numpy as np
import torch

from pyspacer._calibrated_model_fixture import make_calibrated_model


class HeadParityTest(unittest.TestCase):
    def test_head_matches_source_predict_proba(self):
        from mermaid_classifier.pyspacer.inference.head import (
            build_calibrated_head,
        )
        model, X = make_calibrated_model()
        head = build_calibrated_head(model)
        head.eval()
        with torch.no_grad():
            got = head(torch.from_numpy(X.astype(np.float32))).numpy()
        expected = model.predict_proba(X)
        self.assertEqual(got.shape, expected.shape)
        self.assertLess(float(np.max(np.abs(got - expected))), 1e-6)

    def test_rows_sum_to_one(self):
        from mermaid_classifier.pyspacer.inference.head import (
            build_calibrated_head,
        )
        model, X = make_calibrated_model()
        head = build_calibrated_head(model)
        head.eval()
        with torch.no_grad():
            got = head(torch.from_numpy(X.astype(np.float32))).numpy()
        np.testing.assert_allclose(got.sum(axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
