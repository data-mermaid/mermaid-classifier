"""Tests for scripts/release_artifact.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from botocore.exceptions import ClientError
from support.extractor import make_extractor_spec
from support.paths import add_scripts_to_path

add_scripts_to_path()

import release_artifact as ra  # noqa: E402


def _not_found_error():
    return ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")


class VersionValidationTest(unittest.TestCase):
    def test_accepts_vN(self):
        ra.validate_version("v3")  # no raise
        ra.validate_version("v0")

    def test_rejects_bad_versions(self):
        for bad in ("3", "v", "latest", "v1.0", "V3", "v-1", ""):
            with self.assertRaises(ValueError, msg=bad):
                ra.validate_version(bad)


class ParseS3UriTest(unittest.TestCase):
    def test_parses_bucket_and_key(self):
        bucket, key = ra.parse_s3_uri("s3://mermaid-config/classifier/efficientnet.pt")
        self.assertEqual(bucket, "mermaid-config")
        self.assertEqual(key, "classifier/efficientnet.pt")

    def test_rejects_non_s3(self):
        for bad in ("https://x/y", "/local/path", "s3://", "file.pt"):
            with self.assertRaises(ValueError, msg=bad):
                ra.parse_s3_uri(bad)


class ValidateArtifactTest(unittest.TestCase):
    def _export(self, tmp):
        """Export a real small artifact; return (model_pt, model_json)."""
        from support.calibrated_model import make_calibrated_model

        from mermaid_classifier.pyspacer.inference import export_artifact

        model, X = make_calibrated_model()
        model_pt, _manifest, _ = export_artifact(
            model, tmp, X, extractor=make_extractor_spec(X.shape[1])
        )
        return Path(model_pt), Path(tmp) / "model.json"

    def test_valid_artifact_returns_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_pt, model_json = self._export(tmp)
            manifest = ra.validate_artifact(model_pt, model_json)
            self.assertEqual(manifest["task"], "pyspacer_mlp_classifier")
            self.assertTrue(manifest["classes"])

    def test_rejects_wrong_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_pt, model_json = self._export(tmp)
            m = json.loads(model_json.read_text())
            m["task"] = "something_else"
            model_json.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                ra.validate_artifact(model_pt, model_json)

    def test_rejects_missing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_pt, model_json = self._export(tmp)
            m = json.loads(model_json.read_text())
            del m["trained_with"]
            model_json.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                ra.validate_artifact(model_pt, model_json)

    def test_rejects_incomplete_provenance(self):
        # The v2 regression (issue #91): trained_with is present but missing the
        # `pyspacer` key. The serving compat gate requires torch/sklearn/pyspacer,
        # so the release gate must reject it rather than defer the failure to
        # the Lambda's first invocation.
        with tempfile.TemporaryDirectory() as tmp:
            model_pt, model_json = self._export(tmp)
            m = json.loads(model_json.read_text())
            del m["trained_with"]["pyspacer"]
            model_json.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                ra.validate_artifact(model_pt, model_json)

    def test_rejects_bad_class_count(self):
        # load_predictor probes the graph: a manifest claiming the wrong class
        # count must raise (ManifestError is a subclass-agnostic failure here).
        from mermaid_classifier.pyspacer.inference import ManifestError

        with tempfile.TemporaryDirectory() as tmp:
            model_pt, model_json = self._export(tmp)
            m = json.loads(model_json.read_text())
            m["classes"] = m["classes"][:-1]  # drop one -> count mismatch
            model_json.write_text(json.dumps(m))
            with self.assertRaises(ManifestError):
                ra.validate_artifact(model_pt, model_json)


class S3ExistsTest(unittest.TestCase):
    def test_reraises_other_clienterror(self):
        client = mock.Mock()
        client.head_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "HeadObject"
        )
        with self.assertRaises(ClientError):
            ra.s3_object_exists(client, "b", "k")


class AssembleLayoutTest(unittest.TestCase):
    def test_uploads_pair_and_copies_weights(self):
        client = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            mp = Path(tmp) / "model.pt"
            mp.write_bytes(b"pt")
            mj = Path(tmp) / "model.json"
            mj.write_text("{}")
            uris = ra.assemble_s3_layout(
                client,
                dest_bucket="mermaid-config",
                dest_prefix="classifier",
                version="v7",
                model_pt=mp,
                model_json=mj,
                weights_uri="s3://mermaid-config/classifier/efficientnet.pt",
            )

        self.assertEqual(
            uris,
            {
                "model.pt": "s3://mermaid-config/classifier/v7/model.pt",
                "model.json": "s3://mermaid-config/classifier/v7/model.json",
                "efficientnet.pt": "s3://mermaid-config/classifier/v7/efficientnet.pt",
            },
        )
        # model.pt + model.json uploaded by file.
        uploaded_keys = {
            c.kwargs.get("Key") or c.args[2] for c in client.upload_file.call_args_list
        }
        self.assertEqual(uploaded_keys, {"classifier/v7/model.pt", "classifier/v7/model.json"})
        # efficientnet.pt is a server-side copy from the weights source.
        client.copy_object.assert_called_once_with(
            Bucket="mermaid-config",
            Key="classifier/v7/efficientnet.pt",
            CopySource={"Bucket": "mermaid-config", "Key": "classifier/efficientnet.pt"},
        )

    def test_cleans_up_already_written_objects_on_failure(self):
        # model.pt + model.json upload, then the weights copy fails: the two
        # uploaded objects must be deleted so the partial prefix doesn't brick
        # reruns of this version, and the original error must propagate.
        client = mock.Mock()
        client.copy_object.side_effect = RuntimeError("copy boom")
        with tempfile.TemporaryDirectory() as tmp:
            mp = Path(tmp) / "model.pt"
            mp.write_bytes(b"pt")
            mj = Path(tmp) / "model.json"
            mj.write_text("{}")
            with self.assertRaises(RuntimeError):
                ra.assemble_s3_layout(
                    client,
                    dest_bucket="mermaid-config",
                    dest_prefix="classifier",
                    version="v7",
                    model_pt=mp,
                    model_json=mj,
                    weights_uri="s3://mermaid-config/classifier/efficientnet.pt",
                )

        deleted_keys = {c.kwargs["Key"] for c in client.delete_object.call_args_list}
        self.assertEqual(deleted_keys, {"classifier/v7/model.pt", "classifier/v7/model.json"})


class MainTest(unittest.TestCase):
    def setUp(self):
        # A real exported pair the fetch seam will "return".
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        from support.calibrated_model import make_calibrated_model

        from mermaid_classifier.pyspacer.inference import export_artifact

        model, X = make_calibrated_model()
        self._spec = make_extractor_spec(X.shape[1])
        model_pt, _m, _ = export_artifact(model, tmp, X, extractor=self._spec)
        self._pair = (Path(model_pt), tmp / "model.json")
        self.addCleanup(self._tmp.cleanup)

    def _run(self, client, cwd, weights_sha256=None):
        """Run main() with the MLflow fetch and the weights hash stubbed.

        `weights_sha256` defaults to what the manifest records, i.e. the
        weights the model was actually trained through.
        """
        argv = ["--mlflow-model-id", "m-" + "a" * 30, "--version", "v9"]
        with (
            mock.patch.object(ra.boto3, "client", return_value=client),
            mock.patch.object(ra, "resolve_classifier_artifact", return_value=self._pair),
            mock.patch.object(
                ra,
                "sha256_of_uri",
                return_value=weights_sha256 or self._spec.weights_sha256,
            ),
            mock.patch.object(ra.Path, "cwd", return_value=cwd),
        ):
            return ra.main(argv)

    def test_happy_path_uploads_and_emits(self):
        client = mock.Mock()
        # destination model.pt absent (404), then weights source exists.
        client.head_object.side_effect = [_not_found_error(), {}]
        with tempfile.TemporaryDirectory() as cwd:
            rc = self._run(client, Path(cwd))
            self.assertEqual(rc, 0)
            self.assertEqual(client.upload_file.call_count, 2)
            # The extractor copied is the one the manifest names, not a default.
            client.copy_object.assert_called_once_with(
                Bucket="mermaid-config",
                Key="classifier/v9/efficientnet.pt",
                CopySource={"Bucket": "test-bucket", "Key": "efficientnet.pt"},
            )
            # Artifacts copied to CWD for the workflow to attach.
            self.assertTrue((Path(cwd) / "model.pt").is_file())
            self.assertTrue((Path(cwd) / "model.json").is_file())

    def test_existing_version_fails_before_any_write(self):
        client = mock.Mock()
        client.head_object.side_effect = [{}]  # destination model.pt exists
        with tempfile.TemporaryDirectory() as cwd, self.assertRaises(SystemExit):
            self._run(client, Path(cwd))
        client.upload_file.assert_not_called()
        client.copy_object.assert_not_called()

    def test_missing_weights_source_fails_before_any_write(self):
        client = mock.Mock()
        # destination absent, then the weights the manifest names are absent.
        client.head_object.side_effect = [_not_found_error(), _not_found_error()]
        with tempfile.TemporaryDirectory() as cwd, self.assertRaises(SystemExit):
            self._run(client, Path(cwd))
        client.upload_file.assert_not_called()
        client.copy_object.assert_not_called()

    def test_weights_hash_mismatch_fails_before_any_write(self):
        # The object exists at the URI the manifest names, but is not the file
        # the head was trained through. Shipping the two together would
        # mis-score every image while every other gate passed.
        client = mock.Mock()
        client.head_object.side_effect = [_not_found_error(), {}]
        with tempfile.TemporaryDirectory() as cwd, self.assertRaises(SystemExit) as ctx:
            self._run(client, Path(cwd), weights_sha256="b" * 64)
        self.assertIn("b" * 64, str(ctx.exception))
        self.assertIn(self._spec.weights_sha256, str(ctx.exception))
        client.upload_file.assert_not_called()
        client.copy_object.assert_not_called()

    def test_override_uri_still_has_to_hash_to_the_manifest(self):
        client = mock.Mock()
        client.head_object.side_effect = [_not_found_error(), {}]
        argv = [
            "--mlflow-model-id",
            "m-" + "a" * 30,
            "--version",
            "v9",
            "--extractor-weights-uri",
            "s3://elsewhere/other.pt",
        ]
        with (
            tempfile.TemporaryDirectory() as cwd,
            mock.patch.object(ra.boto3, "client", return_value=client),
            mock.patch.object(ra, "resolve_classifier_artifact", return_value=self._pair),
            mock.patch.object(ra, "sha256_of_uri", return_value="c" * 64),
            mock.patch.object(ra.Path, "cwd", return_value=Path(cwd)),
            self.assertRaises(SystemExit),
        ):
            ra.main(argv)
        client.upload_file.assert_not_called()
        client.copy_object.assert_not_called()


class ManifestWithoutExtractorTest(unittest.TestCase):
    """An artifact that cannot say which extractor produced its training
    features cannot be released: the release is what pairs a head with a
    backbone, and there would be nothing to pair it against."""

    def test_validate_artifact_refuses(self):
        from support.calibrated_model import make_calibrated_model

        from mermaid_classifier.pyspacer.inference import ManifestError, export_artifact

        with tempfile.TemporaryDirectory() as tmp:
            model, X = make_calibrated_model()
            model_pt, _m, _ = export_artifact(
                model, tmp, X, extractor=make_extractor_spec(X.shape[1])
            )
            model_json = Path(tmp) / "model.json"
            manifest = json.loads(model_json.read_text())
            del manifest["feature_extraction"]
            model_json.write_text(json.dumps(manifest))

            with self.assertRaises(ManifestError):
                ra.validate_artifact(Path(model_pt), model_json)


if __name__ == "__main__":
    unittest.main()
