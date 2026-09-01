"""Thin Label Studio HTTP client (stdlib urllib only).

Wraps the exact REST calls used to create a review project, matching the proven
manual flow in HOSTING.md §5. `opener` is injectable so request construction is
unit-testable without a live server.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

# Label Studio's Project.title column width. Exceeding it fails the create call with a
# validation error, so the title is checked before a build rather than after.
PROJECT_TITLE_MAX_LENGTH = 50


class LabelStudioClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        opener: Callable[..., Any] = urllib.request.urlopen,
        timeout: float = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._opener = opener
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Any = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Token {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with self._opener(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as err:
            # urllib's message is just the status line; Label Studio puts the reason
            # (validation_errors, and a request id) in the body.
            detail = err.read().decode(errors="replace")[:1000]
            raise RuntimeError(f"{method} {path} -> HTTP {err.code}: {detail}") from err
        return json.loads(raw) if raw else None

    def project_titles(self) -> set[str]:
        data = self._request("GET", "/api/projects")
        items = data["results"] if isinstance(data, dict) else data
        return {p["title"] for p in items}

    def create_project(self, title: str, label_config: str) -> int:
        resp = self._request(
            "POST", "/api/projects", {"title": title, "label_config": label_config}
        )
        return int(resp["id"])

    def add_s3_presign_storage(
        self,
        project_id: int,
        bucket: str,
        prefix: str,
        region_name: str,
        title: str = "images",
    ) -> None:
        # PRESIGN mode: LS mints a presigned S3 URL per view and redirects the browser to
        # S3 (LS stays light; needs the image-bucket CORS rule). Do NOT sync — it only
        # resolves the s3:// links already in the tasks. A project carries one storage per
        # image bucket; LS matches each task's s3:// URI to the storage that covers it.
        self._request(
            "POST",
            "/api/storages/s3",
            {
                "project": project_id,
                "bucket": bucket,
                "prefix": prefix,
                "region_name": region_name,
                "use_blob_urls": False,
                "presign": True,
                "title": title,
            },
        )

    def import_tasks(self, project_id: int, tasks: list[Any]) -> None:
        self._request("POST", f"/api/projects/{project_id}/import", tasks)

    def set_model_version(self, project_id: int, model_version: str) -> None:
        self._request("PATCH", f"/api/projects/{project_id}", {"model_version": model_version})
