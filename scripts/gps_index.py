"""Extract one GPS point per official-test image from bdd100k_info.zip (read in place).

The info zip is ~60 GB uncompressed, so it is never extracted. The platform map
covers US locations only. Output: data/gps_test.json  {image_id: [lat, lon]}
"""
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ag.data import DATA, IMG_DIR

z = zipfile.ZipFile(DATA / "bdd100k_info.zip")
names = {os.path.basename(n)[:-5]: n for n in z.namelist() if n.endswith(".json")}
out = {}
for p in sorted((IMG_DIR / "test").glob("*.jpg")):
    if p.stem not in names:
        continue
    d = json.loads(z.read(names[p.stem]))
    locs = d.get("locations") or d.get("gps") or []
    if not locs:
        continue
    # 100K keyframes are taken at the 10th second; use the fix closest to that
    t = d.get("startTime", locs[0]["timestamp"]) + 10_000
    loc = min(locs, key=lambda l: abs(l["timestamp"] - t))
    if not (24 <= loc["latitude"] <= 50 and -125 <= loc["longitude"] <= -66):  # US only
        continue
    out[p.stem] = [round(loc["latitude"], 6), round(loc["longitude"], 6)]

(DATA / "gps_test.json").write_text(json.dumps(out))
print(f"{len(out)} test images with GPS")
