"""Fine-tune SegFormer (Cityscapes init) on BDD100K 10K semantic segmentation.

Selects the checkpoint with the best holdout mIoU; the test split is never touched here.
Resumable: re-running with the same --out continues from <out>/last.

  python scripts/train.py --iters 8000 --out checkpoints/b2_bdd
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ag.data import BDDSeg
from ag.model import BASELINE, device, evaluate, load

ap = argparse.ArgumentParser()
ap.add_argument("--init", default=BASELINE)
ap.add_argument("--out", default="checkpoints/b2_bdd")
ap.add_argument("--iters", type=int, default=8000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--crop", type=int, default=512)
ap.add_argument("--lr", type=float, default=6e-5)
ap.add_argument("--head-lr-mult", type=float, default=10.0)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--eval-every", type=int, default=1000)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--amp", action="store_true", help="mixed precision (bf16 on CUDA, fp16 on MPS)")
a = ap.parse_args()


def lr_factor(it):
    if it < a.warmup:
        return 1e-2 + (1 - 1e-2) * it / a.warmup
    return (1 - (it - a.warmup) / (a.iters - a.warmup)) ** 1.0


def log(rec, path):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def main():
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "log.jsonl"
    dev = device()
    amp_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float16
    state = {"iter": 0, "best": -1.0}
    if (out / "last" / "state.pt").exists():
        model = load(out / "last", dev)
        state = torch.load(out / "last" / "state.pt", weights_only=False, map_location=dev)
        print(f"resuming from iter {state['iter']}")
    else:
        model = load(a.init, dev)
    model.train()

    head = [p for n, p in model.named_parameters() if n.startswith("decode_head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("decode_head")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr},
                             {"params": head, "lr": a.lr * a.head_lr_mult}], weight_decay=0.01)
    base_lrs = [a.lr, a.lr * a.head_lr_mult]
    if "opt" in state:
        opt.load_state_dict(state["opt"])

    dl = torch.utils.data.DataLoader(
        BDDSeg("train", train=True, crop=a.crop), batch_size=a.bs, shuffle=True,
        num_workers=a.workers, drop_last=True, persistent_workers=True, prefetch_factor=4)
    batches = iter(dl)
    it, t0, run_loss = state["iter"], time.time(), 0.0
    log({"event": "start", "iter": it, "args": vars(a)}, logf)

    while it < a.iters:
        try:
            x, y = next(batches)
        except StopIteration:
            batches = iter(dl)
            x, y = next(batches)
        f = lr_factor(it)
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=a.amp):
            loss = model(pixel_values=x.to(dev), labels=y.to(dev)).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run_loss += loss.item()
        it += 1

        if it % 50 == 0:
            el = time.time() - t0
            log({"iter": it, "loss": round(run_loss / 50, 4), "lr": opt.param_groups[0]["lr"],
                 "s_per_it": round(el / 50, 3),
                 "eta_min": round(el / 50 * (a.iters - it) / 60, 1)}, logf)
            run_loss, t0 = 0.0, time.time()

        if it % a.eval_every == 0 or it == a.iters:
            model.eval()
            te = time.time()
            # workers=0: spawning extra loader processes mid-training hung the
            # M4 Pro run right after the first holdout check (step 1000)
            r = evaluate(model, "holdout", workers=0)
            model.train()
            better = r["mIoU"] > state["best"]
            if better:
                state["best"] = r["mIoU"]
                model.save_pretrained(out / "best")
            state.update(iter=it, opt=opt.state_dict())
            model.save_pretrained(out / "last")
            torch.save(state, out / "last" / "state.pt")
            log({"event": "eval", "iter": it, "holdout_mIoU": round(r["mIoU"], 4),
                 "sidewalk": round(r["iou"]["sidewalk"], 4), "person": round(r["iou"]["person"], 4),
                 "best": better, "eval_s": round(time.time() - te, 1)}, logf)
            t0 = time.time()

    log({"event": "done", "best_holdout_mIoU": state["best"]}, logf)


if __name__ == "__main__":
    main()
