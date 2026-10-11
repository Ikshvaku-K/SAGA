"""Write data/splits.json: train/holdout from official train, test = official val."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ag.data import IMG_DIR, LBL_DIR, SPLITS


def ids(src):
    out = []
    for p in sorted((IMG_DIR / src).glob("*.jpg")):
        if (LBL_DIR / src / f"{p.stem}_train_id.png").exists():
            out.append({"id": p.stem, "src": src})
    return out


train = ids("train")
random.Random(0).shuffle(train)
splits = {"train": train[500:], "holdout": train[:500], "test": ids("val")}
SPLITS.write_text(json.dumps(splits))
print({k: len(v) for k, v in splits.items()})
