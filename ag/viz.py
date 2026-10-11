"""Overlay rendering shared by the scripts and the platform."""
import numpy as np
from PIL import Image, ImageDraw

from .classes import PALETTE

_LUT = np.zeros((256, 3), dtype=np.uint8)
_LUT[:len(PALETTE)] = PALETTE


def colorize(label_map):
    return Image.fromarray(_LUT[np.asarray(label_map)])


def overlay(img, label_map, report=None, alpha=0.5, width=None):
    """Blend the class colours over the photo and box any audit findings."""
    img = img.convert("RGB")
    col = colorize(label_map).resize(img.size, Image.NEAREST)
    out = Image.blend(img, col, alpha)
    if report:
        d = ImageDraw.Draw(out)
        k = img.width / 1280  # audit boxes are in a width-1280 frame, aspect preserved
        for kind, color in (("obstructions", (255, 200, 0)), ("pedestrians_in_road", (255, 40, 40))):
            for h in report[kind]:
                x0, y0, x1, y1 = (v * k for v in h["box"])
                d.rectangle([x0, y0, x1, y1], outline=color, width=max(2, int(4 * k)))
    if width:
        out = out.resize((width, int(out.height * width / out.width)), Image.BILINEAR)
    return out
