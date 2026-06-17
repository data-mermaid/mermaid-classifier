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


import json
import tempfile
from pathlib import Path


class ExportTest(unittest.TestCase):
    def test_export_writes_pt_and_manifest_and_passes_parity(self):
        from mermaid_classifier.pyspacer.inference import export_artifact
        model, X = make_calibrated_model()
        with tempfile.TemporaryDirectory() as d:
            model_pt, manifest, max_diff = export_artifact(model, d, X)
            self.assertTrue(Path(model_pt).is_file())
            self.assertTrue((Path(d) / "model.json").is_file())
            self.assertLess(max_diff, 1e-6)

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["task"], "pyspacer_mlp_classifier")
            self.assertEqual(manifest["classes"], model.classes_.tolist())
            self.assertEqual(manifest["input_dim"], X.shape[1])
            self.assertEqual(manifest["config"], {"patch_size": 224})
            self.assertIn("torch", manifest["trained_with"])
            self.assertIn("sklearn", manifest["trained_with"])

            on_disk = json.loads((Path(d) / "model.json").read_text())
            self.assertEqual(on_disk, manifest)

    def test_frozen_graph_reloads_and_matches_source(self):
        import torch
        from mermaid_classifier.pyspacer.inference import export_artifact
        model, X = make_calibrated_model()
        with tempfile.TemporaryDirectory() as d:
            model_pt, _, _ = export_artifact(model, d, X)
            graph = torch.jit.load(str(model_pt))
            graph.eval()
            with torch.no_grad():
                got = graph(torch.from_numpy(X.astype(np.float32))).numpy()
        self.assertLess(float(np.max(np.abs(got - model.predict_proba(X)))), 1e-6)

    def test_parity_gate_raises_when_graph_diverges(self):
        from mermaid_classifier.pyspacer.inference import (
            export_artifact, ParityError,
        )
        model, X = make_calibrated_model()
        # Corrupt a calibrator so the source model no longer matches a head
        # rebuilt from the (now-tampered) parameters via an impossible tol.
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ParityError):
                export_artifact(model, d, X, tol=-1.0)


class LoadValidationTest(unittest.TestCase):
    def _export(self, d):
        from mermaid_classifier.pyspacer.inference import export_artifact
        model, X = make_calibrated_model()
        model_pt, manifest, _ = export_artifact(model, d, X)
        return model_pt, Path(d) / "model.json", model, X

    def test_load_predictor_round_trip_matches_source(self):
        from mermaid_classifier.pyspacer.inference import load_predictor
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, model, X = self._export(d)
            predictor = load_predictor(model_pt, model_json)
            self.assertEqual(predictor.classes, model.classes_.tolist())
            self.assertEqual(predictor.input_dim, X.shape[1])
            got = predictor.predict_proba(X)
            self.assertLess(
                float(np.max(np.abs(got - model.predict_proba(X)))), 1e-6)

    def test_schema_version_mismatch_raises(self):
        from mermaid_classifier.pyspacer.inference import (
            load_predictor, ManifestError,
        )
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, _, _ = self._export(d)
            manifest = json.loads(Path(model_json).read_text())
            manifest["schema_version"] = 999
            Path(model_json).write_text(json.dumps(manifest))
            with self.assertRaises(ManifestError):
                load_predictor(model_pt, model_json)

    def test_class_count_mismatch_raises(self):
        from mermaid_classifier.pyspacer.inference import (
            load_predictor, ManifestError,
        )
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, _, _ = self._export(d)
            manifest = json.loads(Path(model_json).read_text())
            manifest["classes"] = manifest["classes"][:-1]  # drop one class
            Path(model_json).write_text(json.dumps(manifest))
            with self.assertRaises(ManifestError):
                load_predictor(model_pt, model_json)

    def test_input_dim_mismatch_raises(self):
        from mermaid_classifier.pyspacer.inference import (
            load_predictor, ManifestError,
        )
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, _, _ = self._export(d)
            manifest = json.loads(Path(model_json).read_text())
            manifest["input_dim"] = manifest["input_dim"] + 7  # wrong dim
            Path(model_json).write_text(json.dumps(manifest))
            with self.assertRaises(ManifestError):
                load_predictor(model_pt, model_json)


import os


class LiveModelParityTest(unittest.TestCase):
    def _load_live_model(self):
        from spacer.storage import storage_factory
        from spacer.data_classes import DataLocation
        from urllib.parse import urlparse
        import pickle

        loc = os.environ["PORTABLE_ARTIFACT_LIVE_MODEL"]
        uri = urlparse(loc)
        if uri.scheme == "s3":
            data_loc = DataLocation(
                "s3", bucket_name=uri.netloc, key=uri.path.strip("/"))
        else:
            data_loc = DataLocation("filesystem", key=loc)
        storage = storage_factory(data_loc.storage_type, data_loc.bucket_name)
        with storage.load(data_loc.key) as stream:
            return pickle.load(stream)

    @unittest.skipUnless(
        os.environ.get("PORTABLE_ARTIFACT_LIVE_MODEL"),
        "set PORTABLE_ARTIFACT_LIVE_MODEL to run live-model parity",
    )
    def test_live_model_export_round_trip_within_tolerance(self):
        from mermaid_classifier.pyspacer.inference import (
            export_artifact, load_predictor,
        )
        model = self._load_live_model()
        input_dim = int(model.calibrated_classifiers_[0].estimator.n_features_in_)

        feats_path = os.environ.get("PORTABLE_ARTIFACT_LIVE_FEATURES")
        if feats_path:
            X = np.load(feats_path).astype(np.float32)
        else:
            rng = np.random.default_rng(0)
            X = rng.normal(0, 1, size=(256, input_dim)).astype(np.float32)

        with tempfile.TemporaryDirectory() as d:
            model_pt, manifest, max_diff = export_artifact(model, d, X)
            self.assertLess(max_diff, 1e-6)
            self.assertEqual(manifest["input_dim"], input_dim)
            self.assertEqual(manifest["classes"], model.classes_.tolist())
            predictor = load_predictor(model_pt, Path(d) / "model.json")
            got = predictor.predict_proba(X)
        self.assertLess(
            float(np.max(np.abs(got - model.predict_proba(X)))), 1e-6)


if __name__ == "__main__":
    unittest.main()
