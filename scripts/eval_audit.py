"""Score audit flags from predicted masks against flags from human-labelled masks.

This measures how faithfully the model reproduces the audit that the ground-truth
segmentation would give; it does not validate the rules themselves (see README).

  python scripts/eval_audit.py --name baseline     # uses results/pred_baseline_test/
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
from ag.data import load_split, paths

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--split", default="test")
a = ap.parse_args()

pred_dir = ROOT / "results" / f"pred_{a.name}_{a.split}"
items = load_split(a.split)
rows, flags = [], None
for it in items:
    gt = audit(np.asarray(Image.open(paths(it)[1])))
    pr = audit(np.asarray(Image.open(pred_dir / f"{it['id']}.png")))
    flags = list(gt["flags"])
    rows.append({"id": it["id"], "gt": gt["flags"], "pred": pr["flags"],
                 "gt_score": gt["score"], "pred_score": pr["score"]})

out = {"name": a.name, "split": a.split, "images": len(rows), "flags": {}}
for f in flags:
    tp = [r["id"] for r in rows if r["gt"][f] and r["pred"][f]]
    fp = [r["id"] for r in rows if not r["gt"][f] and r["pred"][f]]
    fn = [r["id"] for r in rows if r["gt"][f] and not r["pred"][f]]
    p = len(tp) / max(1, len(tp) + len(fp))
    rc = len(tp) / max(1, len(tp) + len(fn))
    out["flags"][f] = {"support": len(tp) + len(fn), "predicted": len(tp) + len(fp),
                       "precision": round(p, 4), "recall": round(rc, 4),
                       "f1": round(2 * p * rc / max(1e-9, p + rc), 4),
                       "examples": {"tp": tp[:30], "fp": fp[:30], "fn": fn[:30]}}
g = np.array([r["gt_score"] for r in rows], float)
s = np.array([r["pred_score"] for r in rows], float)
out["score_mae"] = round(float(np.abs(g - s).mean()), 2)
rank = lambda v: np.argsort(np.argsort(v))
out["score_spearman"] = round(float(np.corrcoef(rank(g), rank(s))[0, 1]), 4)

(ROOT / "results" / f"audit_{a.name}_{a.split}.json").write_text(json.dumps(out, indent=1))
print(f"{a.name}: score MAE {out['score_mae']}  spearman {out['score_spearman']}")
for f, m in out["flags"].items():
    print(f"  {f:22s} support {m['support']:4d}  P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}")
