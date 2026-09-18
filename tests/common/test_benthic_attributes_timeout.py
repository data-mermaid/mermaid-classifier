"""Unit test asserting MERMAID API calls in common/benthic_attributes.py bound
their wait, so a black-holed endpoint degrades instead of hanging forever.

PR #103's drift diagnostic is documented to degrade -- not hang -- when the
MERMAID API is unreachable. A refused connection already raises quickly and
is covered elsewhere; this guards the case that review caught: a connection
that accepts and then never replies, which only a socket timeout escapes. The
server below is a raw loopback listener that completes the TCP handshake and
then never writes or closes, standing in for a firewalled/hung endpoint --
without a bounded `timeout=`, the call would hang for as long as the test
runner allows it to.
"""

import socket
import threading
import time
import unittest
from unittest import mock

from mermaid_classifier.common.benthic_attributes import CoralNetMermaidMapping


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


class BenthicAttributesTimeoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _BlackHoleServer()
        self.addCleanup(self.server.close)

    def test_a_hung_connection_raises_within_a_bounded_wait(self):
        # A short timeout keeps the test fast; production picks its own
        # default via the same constant.
        with mock.patch("mermaid_classifier.common.benthic_attributes._HTTP_TIMEOUT_SECONDS", 0.2):
            mapping = CoralNetMermaidMapping(mapping_endpoint=self.server.url)

            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                mapping.mapping  # noqa: B018 -- property access triggers the download
            elapsed = time.monotonic() - start

        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
