"""Audit rules on synthetic 360x640 label maps with known answers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ag.audit import audit
from ag.classes import BICYCLE, BUILDING, CAR, FENCE, PERSON, ROAD, SIDEWALK, SKY

H, W = 360, 640


def street(sidewalk=True):
    """Sky / building band / road, with a sidewalk strip on the right if requested."""
    m = np.full((H, W), SKY, np.uint8)
    m[60:170] = BUILDING
    m[170:] = ROAD
    if sidewalk:
        m[170:, 450:] = SIDEWALK
    return m


def test_car_on_road_is_clean():
    m = street()
    m[220:300, 200:320] = CAR
    r = audit(m)
    assert not r["flags"]["sidewalk_obstruction"]
    assert r["score"] == 0


def test_car_on_sidewalk_is_obstruction():
    m = street()
    m[220:300, 500:600] = CAR
    r = audit(m)
    assert r["flags"]["sidewalk_obstruction"]
    assert r["obstructions"][0]["class"] == "car"


def test_bicycle_on_sidewalk_is_obstruction():
    m = street()
    m[250:300, 520:560] = BICYCLE
    assert audit(m)["flags"]["sidewalk_obstruction"]


def test_car_beyond_sidewalk_is_not_obstruction():
    # Sidewalk strip lies between camera and a car on a cross street: below = sidewalk,
    # but beside the car's wheels is road, so the side check must reject it.
    m = street(sidewalk=False)
    m[240:260, :] = SIDEWALK
    m[200:240, 280:360] = CAR
    assert not audit(m)["flags"]["sidewalk_obstruction"]


def test_distant_small_car_is_ignored():
    m = street()
    m[176:190, 500:520] = CAR  # 14 px tall: beyond the distance limit
    assert not audit(m)["flags"]["sidewalk_obstruction"]


def test_car_cut_by_frame_edge_is_ignored():
    m = street()
    m[220:300, 560:640] = CAR
    assert not audit(m)["flags"]["sidewalk_obstruction"]


def test_person_on_road_and_on_sidewalk():
    m = street()
    m[220:300, 200:225] = PERSON
    r = audit(m)
    assert r["flags"]["pedestrian_in_roadway"] and len(r["pedestrians_in_road"]) == 1
    m = street()
    m[220:300, 520:545] = PERSON
    assert not audit(m)["flags"]["pedestrian_in_roadway"]


def test_no_visible_sidewalk_requires_visible_curb():
    m = street(sidewalk=False)
    assert audit(m)["flags"]["no_visible_sidewalk"]
    # Parked cars filling the right curb zone: the sidewalk may just be hidden.
    m[180:300, 470:640] = CAR
    assert not audit(m)["flags"]["no_visible_sidewalk"]


def test_highway_barrier_suppresses_no_sidewalk():
    m = street(sidewalk=False)
    m[160:200, :] = FENCE
    assert not audit(m)["flags"]["no_visible_sidewalk"]


def test_full_resolution_input_matches_half():
    m = street()
    m[220:300, 500:600] = CAR
    full = np.repeat(np.repeat(m, 2, 0), 2, 1)
    assert audit(full)["flags"] == audit(m)["flags"]
