"""Evaluate a SegFormer checkpoint on a split; optionally save predicted label maps.

  python scripts/eval_seg.py --model nvidia/segformer-b2-finetuned-cityscapes-1024-1024 --name baseline
  python scripts/eval_seg.py --model checkpoints/b2_bdd/best --name finetuned --tta --save-preds
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ag.model import BASELINE, evaluate, load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=BASELINE)
ap.add_argument("--name", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--tta", action="store_true")
ap.add_argument("--save-preds", action="store_true")
ap.add_argument("--limit", type=int)
a = ap.parse_args()

if __name__ == "__main__":
    save = None
    if a.save_preds:
        save = ROOT / "results" / f"pred_{a.name}_{a.split}"
        save.mkdir(parents=True, exist_ok=True)
    t = time.time()
    r = evaluate(load(a.model), a.split, tta=a.tta, save_dir=save, limit=a.limit)
    r.update(model=str(a.model), name=a.name, seconds=round(time.time() - t, 1))
    out = ROOT / "results" / f"seg_{a.name}_{a.split}.json"
    out.write_text(json.dumps(r, indent=1))
    print(f"{a.name} {a.split}: mIoU {r['mIoU']:.4f}  acc {r['pixel_acc']:.4f}  ({r['seconds']}s)")
    for c, v in r["iou"].items():
        print(f"  {c:14s} {v:.3f}")
