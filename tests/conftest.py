"""Local HTTP server used by the integration tests (no internet access needed)."""

import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

PAYLOAD_SIZE = 1_000_000
PAYLOAD = b"x" * PAYLOAD_SIZE
SLOW_DELAY_S = 0.5


class RecordingServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that records the path of every GET request."""

    request_paths: list[str]

    def handle_error(self, request: object, client_address: object) -> None:
        """Stay quiet when the client has already disconnected (e.g. after a timeout)."""
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    """Serves /payload, /redirect, /truncated and /slow; anything else is 404."""

    # HTTP/1.1 enables keep-alive, so every response must send Content-Length.
    protocol_version = "HTTP/1.1"
    server: RecordingServer

    def do_GET(self) -> None:
        self.server.request_paths.append(self.path)
        if self.path == "/payload":
            self.send_body(200, PAYLOAD)
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/payload")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/truncated":
            # Promise the full payload, send half of it, then close the connection.
            self.send_response(200)
            self.send_header("Content-Length", str(PAYLOAD_SIZE))
            self.end_headers()
            self.wfile.write(PAYLOAD[: PAYLOAD_SIZE // 2])
            self.close_connection = True
        elif self.path == "/slow":
            time.sleep(SLOW_DELAY_S)
            self.send_body(200, b"too late")
        else:
            self.send_body(404, b"not found")

    def send_body(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Silence the default per-request logging to stderr."""


@pytest.fixture
def server() -> Iterator[RecordingServer]:
    """Start the server on a free port in a background thread."""
    httpd = RecordingServer(("127.0.0.1", 0), Handler)
    httpd.request_paths = []
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@pytest.fixture
def base_url(server: RecordingServer) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


@pytest.fixture
def request_paths(server: RecordingServer) -> list[str]:
    """Paths of all GET requests the server has received, in order."""
    return server.request_paths


@pytest.fixture
def payload_size() -> int:
    """Size in bytes of the body served at /payload."""
    return PAYLOAD_SIZE
