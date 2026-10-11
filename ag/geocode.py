"""Turn a typed place ("Georgia State University", "Atlantic Ave & 4th Ave, Brooklyn") into
coordinates so place-only reports can appear on the map.

Uses OpenStreetMap's Nominatim (free; usage policy: identify the app, at most 1 request per
second, cache results). Only the typed place text is sent, never photos or user data.
Results are limited to the US, matching the platform map.
"""
import json
import os
import threading
import time
import urllib.parse
import urllib.request

URL = "https://nominatim.openstreetmap.org/search"
AGENT = "SAGA-accessibility-audit/1.0 (https://github.com/Ikshvaku-K/SAGA)"
_lock = threading.Lock()
_last = [0.0]
_cache = {}


def geocode(place, timeout=8):
    """-> (lat, lon) or None. Never raises."""
    q = " ".join(str(place or "").split())[:200]
    if len(q) < 3:
        return None
    if q.lower() in _cache:
        return _cache[q.lower()]
    url = (os.environ.get("AG_GEOCODE_URL") or URL) + "?" + urllib.parse.urlencode(
        {"q": q, "format": "jsonv2", "limit": 1, "countrycodes": "us"})
    with _lock:  # at most one request per second, per Nominatim's policy
        wait = 1.0 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept-Language": "en"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                hits = json.loads(r.read())
        except Exception:
            return None
    result = (float(hits[0]["lat"]), float(hits[0]["lon"])) if hits else None
    _cache[q.lower()] = result
    return result
