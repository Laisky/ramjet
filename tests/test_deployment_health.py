"""Exercise the health probe against synthetic real HTTP listeners."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from scripts.check_container_health import healthy, listening_ports
from tests.conftest import _test_ports


@contextmanager
def endpoint(status, body):
    """Run a local HTTP endpoint with a controlled health response."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    _test_ports.add(server.server_port)
    try:
        yield server.server_port
    finally:
        _test_ports.discard(server.server_port)
        server.shutdown()
        server.server_close()
        worker.join()


def test_health_accepts_only_existing_contract():
    with endpoint(200, b"hello, world") as port:
        assert healthy([port])


def test_health_rejects_wrong_body_and_error_status():
    for status, body in [(200, b"wrong application"), (503, b"hello, world")]:
        with endpoint(status, body) as port:
            assert not healthy([port])


def test_listener_discovery_handles_both_families_and_ignores_connections():
    table = "header\n 0: 00000000:1F90 00000000:0000 0A\n 1: 0100007F:1111 00000000:0000 01\n"
    ipv6 = "header\n 0: 00000000000000000000000000000000:1F90 00000000:0000 0A\n"
    assert listening_ports([table, ipv6]) == [8080]
