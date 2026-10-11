"""SegFormer loading, prediction, and mIoU evaluation."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from .classes import CLASSES, IGNORE
from .data import normalize

BASELINE = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load(path=BASELINE, dev=None):
    m = SegformerForSemanticSegmentation.from_pretrained(str(path))
    return m.to(dev or device()).eval()


@torch.no_grad()
def logits(model, x, tta=False):
    """x: Bx3xHxW normalised -> Bx19xHxW logits at input resolution."""
    out = model(pixel_values=x).logits
    if tta:
        out = out + model(pixel_values=x.flip(-1)).logits.flip(-1)
    return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


def predict(model, img: Image.Image, tta=False):
    """PIL image -> HxW uint8 label map (train ids)."""
    x = normalize(np.asarray(img.convert("RGB")))[None].to(next(model.parameters()).device)
    return logits(model, x, tta)[0].argmax(0).byte().cpu().numpy()


@torch.no_grad()
def evaluate(model, split, tta=False, save_dir=None, batch=4, workers=4, limit=None):
    from .data import BDDSeg
    ds = BDDSeg(split)
    if limit:
        ds.items = ds.items[:limit]
    dev = next(model.parameters()).device
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers,
                                     pin_memory=dev.type == "cuda")
    n = len(CLASSES)
    conf = torch.zeros(n * n, dtype=torch.int64)
    k = 0
    for x, y in dl:
        # bincount is ~300x slower on MPS than CPU, so the confusion matrix stays on CPU
        pred = logits(model, x.to(dev), tta).argmax(1).cpu()
        valid = y != IGNORE
        conf += torch.bincount(y[valid] * n + pred[valid], minlength=n * n)
        if save_dir:
            for p in pred.byte().cpu().numpy():
                Image.fromarray(p).save(save_dir / f"{ds.items[k]['id']}.png")
                k += 1
    conf = conf.view(n, n).double()
    tp = conf.diag()
    iou = tp / (conf.sum(0) + conf.sum(1) - tp).clamp(min=1)
    present = conf.sum(1) > 0
    return {
        "split": split, "images": len(ds), "tta": tta,
        "mIoU": float(iou[present].mean()),
        "pixel_acc": float(tp.sum() / conf.sum()),
        "iou": {c: float(iou[i]) for i, c in enumerate(CLASSES)},
        "confusion": conf.long().tolist(),
    }
