import json
import unittest

from mermaid_classifier.model_review.ls_client import LabelStudioClient


class _FakeResp:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _client(payloads):
    """payloads: list of dicts returned in order; records the sent Requests."""
    recorded = []
    seq = iter(payloads)

    def opener(req, timeout=None):
        recorded.append(req)
        return _FakeResp(json.dumps(next(seq)).encode())

    return LabelStudioClient("http://ls/", "TOK", opener=opener), recorded


class LsClientTest(unittest.TestCase):
    def test_create_project_posts_title_and_config_with_auth(self):
        client, recorded = _client([{"id": 42}])
        pid = client.create_project("My Set", "<View/>")
        self.assertEqual(pid, 42)
        req = recorded[-1]
        self.assertEqual(req.full_url, "http://ls/api/projects")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("Authorization"), "Token TOK")
        body = json.loads(req.data)
        self.assertEqual(body["title"], "My Set")
        self.assertEqual(body["label_config"], "<View/>")

    def test_project_titles_parses_results_envelope(self):
        client, _ = _client([{"results": [{"title": "A"}, {"title": "B"}]}])
        self.assertEqual(client.project_titles(), {"A", "B"})

    def test_project_titles_parses_bare_list(self):
        client, _ = _client([[{"title": "A"}]])
        self.assertEqual(client.project_titles(), {"A"})

    def test_add_s3_presign_storage_sets_presign_and_no_blob_urls(self):
        client, recorded = _client([{"id": 1}])
        client.add_s3_presign_storage(5, "bkt", "pfx/", "us-east-1")
        req = recorded[-1]
        self.assertEqual(req.full_url, "http://ls/api/storages/s3")
        body = json.loads(req.data)
        self.assertEqual(body["project"], 5)
        self.assertEqual(body["bucket"], "bkt")
        self.assertEqual(body["prefix"], "pfx/")
        self.assertEqual(body["region_name"], "us-east-1")
        self.assertTrue(body["presign"])
        self.assertFalse(body["use_blob_urls"])

    def test_import_tasks_posts_list_to_import_endpoint(self):
        client, recorded = _client([{"task_count": 2}])
        client.import_tasks(9, [{"data": {}}, {"data": {}}])
        req = recorded[-1]
        self.assertEqual(req.full_url, "http://ls/api/projects/9/import")
        self.assertEqual(json.loads(req.data), [{"data": {}}, {"data": {}}])

    def test_set_model_version_patches_project(self):
        client, recorded = _client([{"id": 3}])
        client.set_model_version(3, "Unlabelled Starting Set")
        req = recorded[-1]
        self.assertEqual(req.full_url, "http://ls/api/projects/3")
        self.assertEqual(req.get_method(), "PATCH")
        self.assertEqual(json.loads(req.data)["model_version"], "Unlabelled Starting Set")
