"""BDD100K 10K semantic-segmentation dataset and the project's fixed splits.

Splits (see scripts/make_splits.py):
  train   - 6500 of the official train images
  holdout - 500 of the official train images, used for checkpoint selection
  test    - the 1000 official val images, never used for training or selection
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMG_DIR = DATA / "10k"
LBL_DIR = DATA / "bdd100k_seg_maps" / "labels"
SPLITS = DATA / "splits.json"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_split(name):
    return json.loads(SPLITS.read_text())[name]


def paths(item):
    """item = {"id": ..., "src": "train"|"val"} -> (image path, label path)."""
    return (IMG_DIR / item["src"] / f"{item['id']}.jpg",
            LBL_DIR / item["src"] / f"{item['id']}_train_id.png")


def normalize(img_uint8):
    """HxWx3 uint8 -> 3xHxW float tensor, ImageNet-normalised (SegFormer convention)."""
    x = (img_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class BDDSeg(torch.utils.data.Dataset):
    def __init__(self, split, train=False, crop=512, scale=(0.5, 2.0)):
        self.items = load_split(split)
        self.train, self.crop, self.scale = train, crop, scale

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, lp = paths(self.items[i])
        img, lbl = Image.open(ip).convert("RGB"), Image.open(lp)
        if img.size != lbl.size:
            # 6 of the 8000 labelled images are stored portrait (no EXIF tag) against a
            # landscape mask; sky/road brightness checks show a 90deg CCW rotation realigns them
            img = img.transpose(Image.ROTATE_90)
        if self.train:
            img, lbl = self._augment(img, lbl)
        img, lbl = np.asarray(img), np.asarray(lbl)
        return normalize(img), torch.from_numpy(lbl.astype(np.int64))

    def _augment(self, img, lbl):
        s = random.uniform(*self.scale)
        w, h = int(img.width * s), int(img.height * s)
        img = img.resize((w, h), Image.BILINEAR)
        lbl = lbl.resize((w, h), Image.NEAREST)
        c = self.crop
        if w < c or h < c:  # pad with ignore so a full crop always fits
            pw, ph = max(w, c), max(h, c)
            pimg = Image.new("RGB", (pw, ph)); pimg.paste(img)
            plbl = Image.new("L", (pw, ph), 255); plbl.paste(lbl)
            img, lbl, w, h = pimg, plbl, pw, ph
        # prefer crops that are not mostly ignore/sky (cat_max_ratio-style retry)
        for _ in range(10):
            x, y = random.randint(0, w - c), random.randint(0, h - c)
            box = (x, y, x + c, y + c)
            cl = np.asarray(lbl.crop(box))
            vals, cnt = np.unique(cl[cl != 255], return_counts=True)
            if len(cnt) > 1 and cnt.max() / cnt.sum() < 0.75:
                break
        img, lbl = img.crop(box), lbl.crop(box)
        if random.random() < 0.5:
            img, lbl = img.transpose(Image.FLIP_LEFT_RIGHT), lbl.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.8:  # photometric jitter
            a = np.asarray(img).astype(np.float32)
            a = a * random.uniform(0.75, 1.25) + random.uniform(-25, 25)
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        return img, lbl
