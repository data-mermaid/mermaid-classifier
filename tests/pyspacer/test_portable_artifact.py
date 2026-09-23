"""Tests for the portable classifier artifact (model.pt + model.json)."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from support.calibrated_model import make_calibrated_model
from support.extractor import make_extractor_spec

from mermaid_classifier.pyspacer.inference import (
    ManifestError,
    ParityError,
    SklearnPinError,
    export_artifact,
    load_predictor,
)


class ExportTest(unittest.TestCase):
    def test_export_writes_pt_and_manifest_and_passes_parity(self):
        model, X = make_calibrated_model()
        with tempfile.TemporaryDirectory() as d:
            model_pt, manifest, max_diff = export_artifact(
                model, d, X, extractor=make_extractor_spec(X.shape[1])
            )
            self.assertTrue(Path(model_pt).is_file())
            self.assertTrue((Path(d) / "model.json").is_file())
            self.assertLess(max_diff, 1e-6)

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["task"], "pyspacer_mlp_classifier")
            self.assertEqual(manifest["classes"], model.classes_.tolist())
            self.assertEqual(manifest["input_dim"], X.shape[1])
            # patch_size is the extractor's crop, not a literal: asserting it
            # against the spec is what would catch the two drifting apart.
            spec = make_extractor_spec(X.shape[1])
            self.assertEqual(manifest["config"], {"patch_size": spec.crop_size})
            self.assertEqual(manifest["feature_extraction"], spec.to_dict())
            self.assertIn("torch", manifest["trained_with"])
            self.assertIn("sklearn", manifest["trained_with"])
            # trained_with must record pyspacer so the serving runtime can verify
            # feature-extraction compatibility.
            from importlib.metadata import version

            self.assertEqual(manifest["trained_with"]["pyspacer"], version("pyspacer"))

            on_disk = json.loads((Path(d) / "model.json").read_text())
            self.assertEqual(on_disk, manifest)

    def test_parity_gate_raises_when_graph_diverges(self):
        model, X = make_calibrated_model()
        # Force the gate to fire with an impossible tolerance: any non-negative max diff > -1.0 always raises.
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ParityError):
            export_artifact(model, d, X, extractor=make_extractor_spec(X.shape[1]), tol=-1.0)

    def test_manifest_records_model_pt_sha256(self):
        model, X = make_calibrated_model()
        with tempfile.TemporaryDirectory() as d:
            model_pt, manifest, _ = export_artifact(
                model, d, X, extractor=make_extractor_spec(X.shape[1])
            )
            expected = hashlib.sha256(Path(model_pt).read_bytes()).hexdigest()
            self.assertEqual(manifest["model_pt_sha256"], expected)

    def test_export_raises_when_sklearn_unpinned(self):
        model, X = make_calibrated_model()
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch(
                "mermaid_classifier.pyspacer.inference.export.PARITY_PROVEN_SKLEARN",
                "0.0.0-never",
            ),
            self.assertRaises(SklearnPinError),
        ):
            # Patch the proven version to something the runner can't have, so
            # the installed sklearn is guaranteed to differ.
            export_artifact(model, d, X, extractor=make_extractor_spec(X.shape[1]))


class LoadValidationTest(unittest.TestCase):
    def _export(self, d):
        model, X = make_calibrated_model()
        model_pt, manifest, _ = export_artifact(
            model, d, X, extractor=make_extractor_spec(X.shape[1])
        )
        return model_pt, Path(d) / "model.json", model, X

    def test_load_predictor_round_trip_matches_source(self):
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, model, X = self._export(d)
            predictor = load_predictor(model_pt, model_json)
            self.assertEqual(predictor.classes, model.classes_.tolist())
            self.assertEqual(predictor.input_dim, X.shape[1])
            got = predictor.predict_proba(X)
            self.assertLess(float(np.max(np.abs(got - model.predict_proba(X)))), 1e-6)

    def test_schema_version_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, _, _ = self._export(d)
            manifest = json.loads(Path(model_json).read_text())
            manifest["schema_version"] = 999
            Path(model_json).write_text(json.dumps(manifest))
            with self.assertRaises(ManifestError):
                load_predictor(model_pt, model_json)

    def test_class_count_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, _, _ = self._export(d)
            manifest = json.loads(Path(model_json).read_text())
            manifest["classes"] = manifest["classes"][:-1]  # drop one class
            Path(model_json).write_text(json.dumps(manifest))
            with self.assertRaises(ManifestError):
                load_predictor(model_pt, model_json)

    def test_input_dim_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, _, _ = self._export(d)
            manifest = json.loads(Path(model_json).read_text())
            manifest["input_dim"] = manifest["input_dim"] + 7  # wrong dim
            Path(model_json).write_text(json.dumps(manifest))
            with self.assertRaises(ManifestError):
                load_predictor(model_pt, model_json)

    def test_model_pt_rewritten_after_export_raises(self):
        # Digest check runs before torch.jit.load, so corrupt bytes never
        # reach the graph loader -- this exercises the digest compare, not a
        # jit failure.
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, _, _ = self._export(d)
            Path(model_pt).write_bytes(b"not the exported graph")
            with self.assertRaises(ManifestError):
                load_predictor(model_pt, model_json)

    def test_missing_model_pt_sha256_warns_and_still_loads(self):
        with tempfile.TemporaryDirectory() as d:
            model_pt, model_json, model, X = self._export(d)
            manifest = json.loads(Path(model_json).read_text())
            del manifest["model_pt_sha256"]
            Path(model_json).write_text(json.dumps(manifest))

            with self.assertLogs(
                "mermaid_classifier.pyspacer.inference.loader", level="WARNING"
            ) as ctx:
                predictor = load_predictor(model_pt, model_json)

            self.assertTrue(any("[loader.unverified_graph]" in msg for msg in ctx.output))
            self.assertEqual(predictor.classes, model.classes_.tolist())


class LiveModelParityTest(unittest.TestCase):
    def _load_live_model(self):
        # spacer.storage pulls heavy S3 machinery only this env-gated test
        # needs, so it stays a function-local import (deferred on purpose).
        import pickle
        from urllib.parse import urlparse

        from spacer.data_classes import DataLocation
        from spacer.storage import storage_factory

        loc = os.environ["PORTABLE_ARTIFACT_LIVE_MODEL"]
        uri = urlparse(loc)
        if uri.scheme == "s3":
            data_loc = DataLocation("s3", bucket_name=uri.netloc, key=uri.path.strip("/"))
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
        model = self._load_live_model()
        input_dim = int(model.calibrated_classifiers_[0].estimator.n_features_in_)

        # Parity must be proven on REAL EfficientNet features. Random vectors
        # sit in flat softmax regions and under-exercise the per-class
        # calibration tails, where the frozen graph diverges most — so we
        # refuse to "prove" parity on them. Build the .npy with
        # scripts/extract_reference_features.py.
        feats_path = os.environ.get("PORTABLE_ARTIFACT_LIVE_FEATURES")
        if not feats_path:
            self.fail(
                "PORTABLE_ARTIFACT_LIVE_MODEL is set but"
                " PORTABLE_ARTIFACT_LIVE_FEATURES is not. The live parity gate"
                " requires REAL EfficientNet features (no random fallback)."
                " Generate them with"
                " `python scripts/extract_reference_features.py --out X.npy"
                " <.featurevector files>` and set PORTABLE_ARTIFACT_LIVE_FEATURES=X.npy."
            )
        X = np.load(feats_path).astype(np.float32)
        if X.ndim != 2 or X.shape[1] != input_dim:
            self.fail(
                f"real features must be (N, {input_dim}) to match the live model; got {X.shape}."
            )

        with tempfile.TemporaryDirectory() as d:
            model_pt, manifest, max_diff = export_artifact(
                model, d, X, extractor=make_extractor_spec(X.shape[1])
            )
            self.assertLess(max_diff, 1e-6)
            self.assertEqual(manifest["input_dim"], input_dim)
            self.assertEqual(manifest["classes"], model.classes_.tolist())
            predictor = load_predictor(model_pt, Path(d) / "model.json")
            got = predictor.predict_proba(X)
        self.assertLess(float(np.max(np.abs(got - model.predict_proba(X)))), 1e-6)


if __name__ == "__main__":
    unittest.main()
