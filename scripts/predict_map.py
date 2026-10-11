"""Run the audit on every GPS-tagged official-test image (unlabelled, never trained on).

Writes results/map/<id>.jpg (photo) and <id>_o.jpg (overlay), plus
results/map_findings.json for the platform's map and repair-priority list.

  python scripts/predict_map.py --model checkpoints/b2_bdd/best --tta
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ag.audit import audit
from ag.data import DATA, IMG_DIR
from ag.model import BASELINE, load, predict
from ag.viz import overlay

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=BASELINE)
ap.add_argument("--tta", action="store_true")
a = ap.parse_args()

if __name__ == "__main__":
    gps = json.loads((DATA / "gps_test.json").read_text())
    out_dir = ROOT / "results" / "map"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load(a.model)
    feats, t = [], time.time()
    for k, (iid, (lat, lon)) in enumerate(sorted(gps.items())):
        img = Image.open(IMG_DIR / "test" / f"{iid}.jpg").convert("RGB")
        if img.size != (1280, 720):
            img = img.resize((1280, 720), Image.BILINEAR)
        lbl = predict(model, img, tta=a.tta)
        r = audit(lbl)
        img.resize((640, 360), Image.BILINEAR).save(out_dir / f"{iid}.jpg", quality=82)
        overlay(img, lbl, r, alpha=0.45, width=640).save(out_dir / f"{iid}_o.jpg", quality=82)
        feats.append({"id": iid, "lat": lat, "lon": lon, "score": r["score"], "flags": r["flags"],
                      "obstructions": r["obstructions"], "pedestrians_in_road": r["pedestrians_in_road"],
                      "coverage": r["coverage"]})
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(gps)}  {time.time() - t:.0f}s", flush=True)
    (ROOT / "results" / "map_findings.json").write_text(json.dumps(
        {"model": str(a.model), "tta": a.tta, "features": feats}))
    print(f"done: {len(feats)} images, {sum(f['score'] > 0 for f in feats)} with findings")
