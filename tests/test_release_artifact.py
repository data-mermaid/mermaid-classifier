"""Tests for scripts/release_artifact.py."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from botocore.exceptions import ClientError

# Allow importing scripts/release_artifact.py (mirrors test_generate_training_config).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import release_artifact as ra  # noqa: E402


def _not_found_error():
    return ClientError({'Error': {'Code': '404', 'Message': 'Not Found'}},
                       'HeadObject')


class VersionValidationTest(unittest.TestCase):
    def test_accepts_vN(self):
        ra.validate_version('v3')  # no raise
        ra.validate_version('v0')

    def test_rejects_bad_versions(self):
        for bad in ('3', 'v', 'latest', 'v1.0', 'V3', 'v-1', ''):
            with self.assertRaises(ValueError, msg=bad):
                ra.validate_version(bad)


class ParseS3UriTest(unittest.TestCase):
    def test_parses_bucket_and_key(self):
        bucket, key = ra.parse_s3_uri('s3://mermaid-config/classifier/v1/efficientnet_weights.pt')
        self.assertEqual(bucket, 'mermaid-config')
        self.assertEqual(key, 'classifier/v1/efficientnet_weights.pt')

    def test_rejects_non_s3(self):
        for bad in ('https://x/y', '/local/path', 's3://', 'file.pt'):
            with self.assertRaises(ValueError, msg=bad):
                ra.parse_s3_uri(bad)


class ValidateArtifactTest(unittest.TestCase):
    def _export(self, tmp):
        """Export a real small artifact; return (model_pt, model_json)."""
        from mermaid_classifier.pyspacer.inference import export_artifact
        from pyspacer._calibrated_model_fixture import make_calibrated_model
        model, X = make_calibrated_model()
        model_pt, _manifest, _ = export_artifact(model, tmp, X)
        return Path(model_pt), Path(tmp) / 'model.json'

    def test_valid_artifact_returns_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_pt, model_json = self._export(tmp)
            manifest = ra.validate_artifact(model_pt, model_json)
            self.assertEqual(manifest['task'], 'pyspacer_mlp_classifier')
            self.assertTrue(manifest['classes'])

    def test_rejects_wrong_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_pt, model_json = self._export(tmp)
            m = json.loads(model_json.read_text())
            m['task'] = 'something_else'
            model_json.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                ra.validate_artifact(model_pt, model_json)

    def test_rejects_missing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_pt, model_json = self._export(tmp)
            m = json.loads(model_json.read_text())
            del m['trained_with']
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
            m['classes'] = m['classes'][:-1]  # drop one -> count mismatch
            model_json.write_text(json.dumps(m))
            with self.assertRaises(ManifestError):
                ra.validate_artifact(model_pt, model_json)


class S3ExistsTest(unittest.TestCase):
    def test_true_when_head_succeeds(self):
        client = mock.Mock()
        client.head_object.return_value = {}
        self.assertTrue(ra.s3_object_exists(client, 'b', 'k'))

    def test_false_on_404(self):
        client = mock.Mock()
        client.head_object.side_effect = _not_found_error()
        self.assertFalse(ra.s3_object_exists(client, 'b', 'k'))

    def test_reraises_other_clienterror(self):
        client = mock.Mock()
        client.head_object.side_effect = ClientError(
            {'Error': {'Code': 'AccessDenied'}}, 'HeadObject')
        with self.assertRaises(ClientError):
            ra.s3_object_exists(client, 'b', 'k')


class AssembleLayoutTest(unittest.TestCase):
    def test_uploads_pair_and_copies_weights(self):
        client = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            mp = Path(tmp) / 'model.pt'
            mp.write_bytes(b'pt')
            mj = Path(tmp) / 'model.json'
            mj.write_text('{}')
            uris = ra.assemble_s3_layout(
                client,
                dest_bucket='mermaid-config',
                dest_prefix='classifier',
                version='v7',
                model_pt=mp,
                model_json=mj,
                weights_uri='s3://mermaid-config/classifier/v1/efficientnet_weights.pt',
            )

        self.assertEqual(uris, {
            'model.pt': 's3://mermaid-config/classifier/v7/model.pt',
            'model.json': 's3://mermaid-config/classifier/v7/model.json',
            'efficientnet.pt': 's3://mermaid-config/classifier/v7/efficientnet.pt',
        })
        # model.pt + model.json uploaded by file.
        uploaded_keys = {c.kwargs.get('Key') or c.args[2]
                         for c in client.upload_file.call_args_list}
        self.assertEqual(uploaded_keys,
                         {'classifier/v7/model.pt', 'classifier/v7/model.json'})
        # efficientnet.pt is a server-side copy from the weights source.
        client.copy_object.assert_called_once_with(
            Bucket='mermaid-config',
            Key='classifier/v7/efficientnet.pt',
            CopySource={'Bucket': 'mermaid-config',
                        'Key': 'classifier/v1/efficientnet_weights.pt'},
        )


import shutil


class MainTest(unittest.TestCase):
    def setUp(self):
        # A real exported pair the fetch seam will "return".
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        from mermaid_classifier.pyspacer.inference import export_artifact
        from pyspacer._calibrated_model_fixture import make_calibrated_model
        model, X = make_calibrated_model()
        model_pt, _m, _ = export_artifact(model, tmp, X)
        self._pair = (Path(model_pt), tmp / 'model.json')
        self.addCleanup(self._tmp.cleanup)

    def _run(self, client, cwd):
        argv = ['--mlflow-model-id', 'm-' + 'a' * 30, '--version', 'v9']
        with mock.patch.object(ra.boto3, 'client', return_value=client), \
             mock.patch.object(ra, 'resolve_classifier_artifact',
                               return_value=self._pair), \
             mock.patch.object(ra.Path, 'cwd', return_value=cwd):
            return ra.main(argv)

    def test_happy_path_uploads_and_emits(self):
        client = mock.Mock()
        # weights source exists (True), destination model.pt absent (404).
        client.head_object.side_effect = [{}, _not_found_error()]
        with tempfile.TemporaryDirectory() as cwd:
            rc = self._run(client, Path(cwd))
            self.assertEqual(rc, 0)
            self.assertEqual(client.upload_file.call_count, 2)
            client.copy_object.assert_called_once()
            # Artifacts copied to CWD for the workflow to attach.
            self.assertTrue((Path(cwd) / 'model.pt').is_file())
            self.assertTrue((Path(cwd) / 'model.json').is_file())

    def test_existing_version_fails_before_any_write(self):
        client = mock.Mock()
        # weights source exists (True), destination model.pt ALSO exists (True).
        client.head_object.side_effect = [{}, {}]
        with tempfile.TemporaryDirectory() as cwd:
            with self.assertRaises(SystemExit):
                self._run(client, Path(cwd))
        client.upload_file.assert_not_called()
        client.copy_object.assert_not_called()

    def test_missing_weights_source_fails_before_any_write(self):
        client = mock.Mock()
        client.head_object.side_effect = [_not_found_error()]  # weights absent
        with tempfile.TemporaryDirectory() as cwd:
            with self.assertRaises(SystemExit):
                self._run(client, Path(cwd))
        client.upload_file.assert_not_called()
