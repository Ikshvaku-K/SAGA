"""Typed places -> coordinates (Nominatim), against a local mock server."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ag.geocode as g


def test_geocode_us_only_cached_and_safe(monkeypatch):
    seen = []

    class Mock(BaseHTTPRequestHandler):
        def do_GET(self):
            q = parse_qs(urlparse(self.path).query)
            seen.append((q["q"][0], q["countrycodes"][0], self.headers["User-Agent"]))
            body = [{"lat": "33.7531", "lon": "-84.3853"}] if "state" in q["q"][0].lower() else []
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(body).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("AG_GEOCODE_URL", f"http://127.0.0.1:{srv.server_port}/search")
    g._cache.clear()
    assert g.geocode("Georgia  State university") == (33.7531, -84.3853)
    assert g.geocode("georgia state university") == (33.7531, -84.3853)  # cached: no second request
    assert len(seen) == 1 and seen[0][1] == "us" and "SAGA" in seen[0][2]
    assert g.geocode("nowhere at all") is None
    assert g.geocode("ab") is None                                        # too short: not sent
    monkeypatch.setenv("AG_GEOCODE_URL", "http://127.0.0.1:9/search")  # unreachable: no exception
    g._cache.clear()
    assert g.geocode("Georgia State") is None
    srv.shutdown()
