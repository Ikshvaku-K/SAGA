"""Spoken summaries: server-written text, cache keys, and the ElevenLabs request."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ag.voice import DEFAULT_MODEL, DEFAULT_VOICE, Voice, speech_text

F = lambda o=False, p=False, n=False: {"sidewalk_obstruction": o, "pedestrian_in_roadway": p, "no_visible_sidewalk": n}


def test_text_for_findings_and_residents():
    t = speech_text(65, F(o=True, p=True), [{"class": "car"}, {"class": "bicycle"}], [{}])
    assert t == ("Caution. A bicycle and a car are blocking the sidewalk. Someone is walking in the street. "
                 "Repair priority 65 out of 100, medium.")
    assert speech_text(25, F(o=True), [{"class": "car"}]).startswith("Caution. A car is blocking the sidewalk.")
    assert speech_text(40, F(n=True)).endswith("Repair priority 40 out of 100, medium.")
    assert speech_text(85, F(p=True, n=True), pedestrians=[{}, {}]).startswith("Caution. People are walking")
    assert speech_text(0, F()) == "No sidewalk barrier was found here."
    assert speech_text(40, F(n=True), verdict="confirmed", resident=True).endswith("Reported by a resident and verified.")
    assert "model missed" in speech_text(0, F(), verdict="missed", resident=True)


def test_disabled_without_key_and_cache_tag_tracks_voice():
    assert not Voice({}).enabled
    a, b = Voice({"AG_ELEVENLABS_KEY": "k"}), Voice({"AG_ELEVENLABS_KEY": "k", "AG_ELEVENLABS_VOICE": "other"})
    assert a.voice == DEFAULT_VOICE and a.model == DEFAULT_MODEL and a.enabled
    assert a.cache_tag("hi") == a.cache_tag("hi") != a.cache_tag("hello")
    assert a.cache_tag("hi") != b.cache_tag("hi")  # changing the voice re-renders


def test_request_shape_against_a_mock_server():
    seen = {}

    class Mock(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"], seen["key"] = self.path, self.headers.get("xi-api-key")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            code = 200 if seen["key"] == "good" else 401
            self.send_response(code); self.end_headers()
            self.wfile.write(b"ID3fake-mp3" if code == 200 else b'{"detail":"invalid key"}')

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1/text-to-speech/{{voice}}?output_format=mp3_44100_128"
    v = Voice({"AG_ELEVENLABS_KEY": "good", "AG_ELEVENLABS_URL": url})
    assert v.synthesize("Caution.") == b"ID3fake-mp3"
    assert seen["path"].startswith(f"/v1/text-to-speech/{DEFAULT_VOICE}?output_format=mp3")
    assert seen["body"] == {"text": "Caution.", "model_id": DEFAULT_MODEL}
    with pytest.raises(RuntimeError, match="ElevenLabs 401"):
        Voice({"AG_ELEVENLABS_KEY": "bad", "AG_ELEVENLABS_URL": url}).synthesize("x")
    srv.shutdown()
