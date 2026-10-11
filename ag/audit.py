"""Accessibility audit rules over a 19-class label map.

Each rule looks at what an object is *standing on*: the strip of pixels directly
below its lowest edge ("contact band"). A car whose contact band is sidewalk is
parked on the sidewalk; a person whose contact band is road is walking in the street.

Findings:
  sidewalk_obstruction  vehicle / motorcycle / bicycle standing on sidewalk
  pedestrian_in_roadway person standing on road (includes crossing; no crosswalk class)
  no_visible_sidewalk   built-up street scene with road but no visible sidewalk
                        (a single frame cannot tell "absent" from "hidden by parked cars")

A second check guards against perspective errors: an object must also have its
target surface beside it at ground level (left/right of its lowest rows). Without
this, cars on a cross street behind a corner read as "on the sidewalk" because the
near-side sidewalk sits between them and the camera.
"""
import numpy as np
from scipy import ndimage

from .classes import (BICYCLE, BUS, CAR, CLASSES, IGNORE, MOTORCYCLE, PERSON,
                      ROAD, SIDEWALK, TRUCK)

WORK_W = 640                # rules run at half of BDD's native 1280 width
BAND = 6                    # contact band height in pixels (at WORK_W)
MIN_CONTACT = 8             # need this many informative contact pixels to decide
SHARE = 0.6                 # sidewalk/road share of the contact band to fire
SIDE_W = 6                  # side-check strip width in pixels
SIDE_SHARE = 0.4            # target-surface share required beside the object
MIN_SIDE = 4                # informative side pixels needed (else: do not fire)
MIN_AREA = {CAR: 600, TRUCK: 600, BUS: 600, MOTORCYCLE: 150, BICYCLE: 150, PERSON: 60}
# Minimum object height (px at WORK_W) as a distance limit: ground contact is only
# trustworthy for nearby objects (roughly within 20-25 m for BDD's dashcam optics).
MIN_H = {CAR: 35, TRUCK: 35, BUS: 35, MOTORCYCLE: 25, BICYCLE: 25, PERSON: 35}
NO_SIDEWALK = {"building_min": 0.05, "road_min": 0.10, "sidewalk_max": 0.005,
               # A sidewalk can only be called absent if the curb zones are actually visible:
               "curb_vehicle_max": 0.25,   # vehicles in left/right curb zone (else: hidden)
               "barrier_max": 0.03}        # wall+fence share; highways read as built-up

OBSTRUCTORS = (CAR, TRUCK, BUS, MOTORCYCLE, BICYCLE)


def _downsample(lbl):
    h, w = lbl.shape
    if w == WORK_W:
        return lbl
    s = w / WORK_W
    ys = (np.arange(int(round(h / s))) * s).astype(int)
    xs = (np.arange(WORK_W) * s).astype(int)
    return lbl[np.ix_(ys, xs)]


def _informative(s):
    return s[(s != IGNORE) & (s != PERSON) & ~np.isin(s, OBSTRUCTORS)]


def _contact(lbl, comp, sl):
    """Surface pixels just below one component, and beside its lowest rows."""
    H, W = lbl.shape
    ys, xs = sl
    sub = comp[ys, xs]
    cols = np.flatnonzero(sub.any(0))
    last = sub.shape[0] - 1 - np.argmax(sub[::-1, cols], axis=0)
    y0 = ys.start + last + 1
    x = xs.start + cols
    keep = y0 + BAND <= H              # objects cut off by the image bottom: unknown
    y0, x = y0[keep], x[keep]
    if len(x) == 0:
        return np.array([], dtype=lbl.dtype), np.array([], dtype=lbl.dtype)
    yy = (y0[None, :] + np.arange(BAND)[:, None]).ravel()
    below = _informative(lbl[yy, np.tile(x, BAND)])
    rows = slice(max(ys.start, ys.stop - max(2, (ys.stop - ys.start) // 5)), ys.stop)
    left = lbl[rows, max(0, xs.start - SIDE_W):xs.start]
    right = lbl[rows, xs.stop:min(W, xs.stop + SIDE_W)]
    beside = _informative(np.concatenate([left.ravel(), right.ravel()]))
    return below, beside


def _objects(lbl, cls, fire_on):
    """Components of `cls` whose contact band is mostly `fire_on`."""
    comps, n = ndimage.label(lbl == cls)
    hits = []
    for i, sl in enumerate(ndimage.find_objects(comps), start=1):
        comp = comps == i
        area = int(comp[sl].sum())
        ys, xs = sl
        if area < MIN_AREA[cls] or ys.stop - ys.start < MIN_H[cls]:
            continue
        if xs.start == 0 or xs.stop == lbl.shape[1]:  # cut off by the frame: footprint unknown
            continue
        below, beside = _contact(lbl, comp, sl)
        if len(below) < MIN_CONTACT or len(beside) < MIN_SIDE:
            continue
        share = float((below == fire_on).mean())
        side_share = float((beside == fire_on).mean())
        if share >= SHARE and side_share >= SIDE_SHARE:
            s = 1280 / lbl.shape[1]  # boxes in a width-1280 frame (BDD native), aspect kept
            hits.append({"class": CLASSES[cls], "share": round(share, 2),
                         "side_share": round(side_share, 2), "area": area,
                         "box": [int(xs.start * s), int(ys.start * s),
                                 int(xs.stop * s), int(ys.stop * s)]})
    return hits


def _curb_vehicle_share(lbl):
    """Largest vehicle share among the left/right curb zones (outer 30%, ground band)."""
    H, W = lbl.shape
    band = lbl[int(H * 0.45):int(H * 0.8)]
    zones = (band[:, :int(W * 0.3)], band[:, int(W * 0.7):])
    return max(float(np.isin(z, OBSTRUCTORS).mean()) for z in zones)


def audit(label_map):
    lbl = _downsample(np.asarray(label_map))
    valid = lbl != IGNORE
    nvalid = max(int(valid.sum()), 1)
    frac = {c: float((lbl == i).sum() / nvalid) for i, c in enumerate(CLASSES)}

    obstruct = [h for c in OBSTRUCTORS for h in _objects(lbl, c, SIDEWALK)]
    in_road = _objects(lbl, PERSON, ROAD)
    no_side = (frac["building"] >= NO_SIDEWALK["building_min"]
               and frac["road"] >= NO_SIDEWALK["road_min"]
               and frac["sidewalk"] <= NO_SIDEWALK["sidewalk_max"]
               and frac["wall"] + frac["fence"] <= NO_SIDEWALK["barrier_max"]
               and _curb_vehicle_share(lbl) <= NO_SIDEWALK["curb_vehicle_max"])

    # Heuristic priority score (0-100) for ranking repairs; not a validated measure.
    score = 40 * no_side + min(50, 25 * len(obstruct)) + min(30, 15 * len(in_road))
    if no_side and in_road:
        score += 15  # pedestrians with nowhere else to walk
    return {
        "flags": {
            "sidewalk_obstruction": len(obstruct) > 0,
            "pedestrian_in_roadway": len(in_road) > 0,
            "no_visible_sidewalk": bool(no_side),
        },
        "obstructions": obstruct,
        "pedestrians_in_road": in_road,
        "score": int(min(100, score)),
        "coverage": {k: round(frac[k], 4) for k in ("road", "sidewalk", "building", "person", "car")},
    }
