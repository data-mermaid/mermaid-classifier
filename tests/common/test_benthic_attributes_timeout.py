"""Unit tests asserting every MERMAID API call in common/benthic_attributes.py
bounds its wait, so a black-holed endpoint degrades instead of hanging
forever.

PR #103's drift diagnostic is documented to degrade -- not hang -- when the
MERMAID API is unreachable. A refused connection already raises quickly and
is covered elsewhere; this guards the case that review caught: a connection
that accepts and then never replies, which only a socket timeout escapes. The
server below is a raw loopback listener that completes the TCP handshake and
then never writes or closes, standing in for a firewalled/hung endpoint.

Each test bounds its own wait externally, in a background thread joined with
a 5 s timeout: a missing socket timeout then surfaces as this external
bound's own failure well before that ceiling, rather than as a hang that
blocks the test runner for as long as it is allowed to run. Every call site
`_HTTP_TIMEOUT_SECONDS` guards is exercised: the BenthicAttributeLibrary and
GrowthFormLibrary constructors, the mapping endpoint's first request, and the
mapping endpoint's pagination follow-up. RegionLibrary needs no case of its
own -- it and GrowthFormLibrary reach the network through the one
ChoiceLibrary fetch the growth-form test already bounds.
"""

import concurrent.futures
import http.server
import json
import socket
import threading
import time
import unittest
import urllib.request
from unittest import mock

from mermaid_classifier.common.benthic_attributes import (
    BenthicAttributeLibrary,
    CoralNetMermaidMapping,
    GrowthFormLibrary,
)

# Real elapsed time stays well under this on a correctly-guarded call (bounded
# by the mocked _HTTP_TIMEOUT_SECONDS below); a call missing its socket
# timeout instead blocks until the external 5 s bound in _run_bounded, which
# this margin distinguishes from a fast, real failure.
_ELAPSED_MARGIN_SECONDS = 2.0


class _BlackHoleServer:
    """A loopback listener that accepts a connection and then holds it open,
    sending and closing nothing, so a client is left waiting on a read."""

    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.port: int = self._listener.getsockname()[1]
        self._held_conn: socket.socket | None = None
        self._thread = threading.Thread(target=self._accept_and_hold, daemon=True)
        self._thread.start()

    def _accept_and_hold(self) -> None:
        try:
            conn, _addr = self._listener.accept()
        except OSError:
            return
        self._held_conn = conn

    def close(self) -> None:
        if self._held_conn is not None:
            self._held_conn.close()
        self._listener.close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"


class _JsonPageServer:
    """A local HTTP server returning one JSON page whose `next` points
    wherever the caller asks, so a pagination follow-up can be redirected at
    a black hole without a real endpoint's cooperation."""

    def __init__(self, next_url: str) -> None:
        page = json.dumps({"results": [], "next": next_url}).encode()

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

            def log_message(self, format_: str, *args: object) -> None:
                pass  # keep test output free of per-request access logs

        self._server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        self.port: int = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"


def _redirect_to(url: str):
    """A `urllib.request.urlopen` stand-in that opens `url` regardless of
    what the caller asked for, forwarding every other argument (including a
    missing or present `timeout=`) to the real `urlopen` unchanged.

    `BenthicAttributeLibrary`, `GrowthFormLibrary` and `RegionLibrary` each
    call a hardcoded `https://api.datamermaid.org/...` URL with no way to
    inject a test endpoint, so redirecting the shared `urlopen` function is
    the only way to point their real call sites at a local server.
    """
    real_urlopen = urllib.request.urlopen

    def _urlopen(_url, *args, **kwargs):
        return real_urlopen(url, *args, **kwargs)

    return _urlopen


class BenthicAttributesTimeoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _BlackHoleServer()
        self.addCleanup(self.server.close)

    def _run_bounded(self, call) -> float:
        """Run `call` off-thread and return the elapsed time to a
        `TimeoutError` -- from the real socket timeout when one fires, or
        from this 5 s external bound when nothing ever does.

        Not a `with ThreadPoolExecutor() as executor:` block: that form's
        exit calls `shutdown(wait=True)`, which would itself block forever on
        a `call` a missing socket timeout never lets finish.
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.addCleanup(executor.shutdown, wait=False, cancel_futures=True)
        future = executor.submit(call)

        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            future.result(timeout=5)
        return time.monotonic() - start

    def test_a_hung_connection_raises_within_a_bounded_wait(self):
        # A short timeout keeps the test fast; production picks its own
        # default via the same constant.
        with mock.patch("mermaid_classifier.common.benthic_attributes._HTTP_TIMEOUT_SECONDS", 0.2):
            mapping = CoralNetMermaidMapping(mapping_endpoint=self.server.url)
            elapsed = self._run_bounded(lambda: mapping.mapping)

        self.assertLess(elapsed, _ELAPSED_MARGIN_SECONDS)

    def test_the_paginated_next_call_bounds_its_wait(self):
        page_server = _JsonPageServer(next_url=self.server.url)
        self.addCleanup(page_server.close)

        with mock.patch("mermaid_classifier.common.benthic_attributes._HTTP_TIMEOUT_SECONDS", 0.2):
            mapping = CoralNetMermaidMapping(mapping_endpoint=page_server.url)
            elapsed = self._run_bounded(lambda: mapping.mapping)

        self.assertLess(elapsed, _ELAPSED_MARGIN_SECONDS)

    def test_benthic_attribute_library_bounds_its_wait(self):
        with (
            mock.patch("mermaid_classifier.common.benthic_attributes._HTTP_TIMEOUT_SECONDS", 0.2),
            mock.patch("urllib.request.urlopen", _redirect_to(self.server.url)),
        ):
            elapsed = self._run_bounded(BenthicAttributeLibrary)

        self.assertLess(elapsed, _ELAPSED_MARGIN_SECONDS)

    def test_growth_form_library_bounds_its_wait(self):
        with (
            mock.patch("mermaid_classifier.common.benthic_attributes._HTTP_TIMEOUT_SECONDS", 0.2),
            mock.patch("urllib.request.urlopen", _redirect_to(self.server.url)),
        ):
            elapsed = self._run_bounded(GrowthFormLibrary)

        self.assertLess(elapsed, _ELAPSED_MARGIN_SECONDS)


if __name__ == "__main__":
    unittest.main()
