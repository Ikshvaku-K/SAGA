"""Render correct / false-alarm / missed examples per audit flag for the report page.

  python scripts/make_gallery.py --name finetuned
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ag.audit import audit
from ag.data import IMG_DIR
from ag.viz import overlay

_model = None


def model():
    global _model
    if _model is None:
        from ag.model import load
        _model = load(a.model)
    return _model


def predict(m, img):
    from ag.model import predict as _p
    return _p(m, img)

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="finetuned")
ap.add_argument("--per", type=int, default=6)
ap.add_argument("--model", help="predict missing masks with this checkpoint instead of reading saved ones")
a = ap.parse_args()

res = json.loads((ROOT / "results" / f"audit_{a.name}_test.json").read_text())
# image ids kept off the public platform after visual review (one id per line)
skip_file = ROOT / "data" / "gallery_exclude.txt"
SKIP = set(skip_file.read_text().split()) if skip_file.exists() else set()
pred_dir = ROOT / "results" / f"pred_{a.name}_test"
out = ROOT / "results" / "gallery"
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("*.jpg"):
    old.unlink()
index = {}
for flag, m in res["flags"].items():
    index[flag] = {}
    for kind in ("tp", "fp", "fn"):
        names = []
        for iid in [i for i in m["examples"][kind] if i not in SKIP][:a.per]:
            img = Image.open(IMG_DIR / "val" / f"{iid}.jpg").convert("RGB")
            saved = pred_dir / f"{iid}.png"
            if saved.exists():
                lbl = np.asarray(Image.open(saved))
                if img.size != (lbl.shape[1], lbl.shape[0]):
                    img = img.transpose(Image.ROTATE_90)
            else:
                if img.size != (1280, 720):
                    img = img.transpose(Image.ROTATE_90)
                lbl = predict(model(), img)
            n = f"{iid}_{flag}_{kind}.jpg"
            overlay(img, lbl, audit(lbl), alpha=0.45, width=640).save(out / n, quality=82)
            names.append(n)
        index[flag][kind] = names
(out / "index.json").write_text(json.dumps(index))
print({f: {k: len(v) for k, v in d.items()} for f, d in index.items()})
