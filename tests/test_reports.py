"""Report store, anonymisation and EXIF GPS for resident reports."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ag.classes import CAR, PERSON, ROAD
from ag.reports import ReportStore, anonymize, describe, exif_latlon, in_us


def noisy(w=320, h=180, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def report(flags=(False, False, False), obstructions=(), peds=()):
    keys = ("sidewalk_obstruction", "pedestrian_in_roadway", "no_visible_sidewalk")
    return {"flags": dict(zip(keys, flags)), "obstructions": list(obstructions),
            "pedestrians_in_road": list(peds), "score": 0, "coverage": {}}


def test_anonymize_blurs_people_and_plate_band_only():
    img = noisy()
    lbl = np.full((180, 320), ROAD, np.uint8)
    lbl[40:140, 20:60] = PERSON
    lbl[60:160, 180:300] = CAR
    out = np.asarray(anonymize(img, lbl)).astype(float)
    src = np.asarray(img).astype(float)
    std = lambda a: a.std()
    assert std(out[50:130, 25:55]) < 0.5 * std(src[50:130, 25:55])        # person blurred
    assert std(out[125:158, 190:290]) < 0.5 * std(src[125:158, 190:290])  # plate band blurred
    assert np.array_equal(out[62:90, 200:280], src[62:90, 200:280])       # car roof kept
    assert np.array_equal(out[0:20, 100:160], src[0:20, 100:160])         # background kept


def test_anonymize_without_people_is_identity():
    img = noisy()
    assert np.array_equal(np.asarray(anonymize(img, np.zeros((180, 320), np.uint8))), np.asarray(img))


def test_exif_gps_round_trip():
    img = noisy()
    exif = Image.Exif()
    gps = exif.get_ifd(0x8825)
    gps.update({1: "N", 2: (40.0, 41.0, 6.0), 3: "W", 4: (73.0, 59.0, 24.0)})
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    lat, lon = exif_latlon(Image.open(io.BytesIO(buf.getvalue())))
    assert lat == pytest.approx(40.685, abs=1e-3) and lon == pytest.approx(-73.99, abs=1e-3)
    assert exif_latlon(noisy()) is None


def test_us_filter():
    assert in_us(40.7, -74.0) and in_us(37.8, -122.4)
    assert not in_us(None, None) and not in_us(51.5, -0.1)


def test_store_feedback_and_field_precision(tmp_path):
    st = ReportStore(tmp_path)
    img = noisy()
    a = st.create(img, img, report((True, False, False)), "+15550001", "app", "m")
    b = st.create(img, img, report((True, False, False)), "+15550002", "app", "m")
    c = st.create(img, img, report(), "+15550001", "app", "m")
    assert (tmp_path / f"{a['id']}.jpg").exists() and a["lat"] is None
    st.set_location(a["id"], 40.7, -73.9, source="pin")
    st.set_feedback(a["id"], "confirmed")
    st.set_feedback(b["id"], "rejected", note="it's a driveway")
    st.set_feedback(c["id"], "missed")
    s = st.stats()
    assert s["reports"] == 3 and s["reporters"] == 2 and s["with_location"] == 1
    assert s["flags"]["sidewalk_obstruction"]["precision"] == 0.5
    assert s["miss_rate"] == 1.0
    assert st.get(b["id"])["note"] == "it's a driveway"
    with pytest.raises(ValueError):
        st.set_feedback(a["id"], "maybe")
    assert st.delete(b["id"]) and st.get(b["id"]) is None
    assert not (tmp_path / f"{b['id']}.jpg").exists() and not st.delete(b["id"])


def test_edit_key_is_required_and_hashed(tmp_path):
    st = ReportStore(tmp_path)
    r = st.create(noisy(), noisy(), report(), "app:dev1", "app", "m", edit_key="s3cret-key")
    assert st.check_key(r["id"], "s3cret-key")
    assert not st.check_key(r["id"], "wrong") and not st.check_key(r["id"], "") and not st.check_key("nope", "s3cret-key")
    assert "key_hash" not in r and b"s3cret-key" not in (tmp_path / "reports.db").read_bytes()
    keyless = st.create(noisy(), noisy(), report(), "app:x", "app", "m")
    assert not st.check_key(keyless["id"], "anything")  # a report created without a key can't be edited


def test_demo_reports_are_private_and_temporary(tmp_path):
    st = ReportStore(tmp_path)
    real = st.create(noisy(), noisy(), report((True, False, False)), "app:me", "app", "m", lat=40.7, lon=-74.0)
    st.set_feedback(real["id"], "confirmed")
    demo = st.create(noisy(), noisy(), report((True, False, False)), "app:d", "demo", "m", lat=40.7, lon=-74.0,
                     demo_session="sessA", expires=1_000)
    st.set_feedback(demo["id"], "rejected")
    ids = lambda rows: {r["id"] for r in rows}
    assert ids(st.all()) == {real["id"]}                              # real users never see demo data
    assert ids(st.all("sessA")) == {real["id"], demo["id"]}           # the demo user sees their own
    assert ids(st.all("sessB")) == {real["id"]}                       # ...but not other demos'
    s = st.stats()
    assert s["reports"] == 1 and s["reporters"] == 1                  # metrics ignore demo data
    assert s["flags"]["sidewalk_obstruction"]["precision"] == 1.0
    assert "demo_session" not in demo
    assert st.purge_demo(now=999) == 0                                # still within its session
    assert st.purge_demo(now=1_000) == 1                              # session over: gone
    assert st.get(demo["id"]) is None and not (tmp_path / f"{demo['id']}.jpg").exists()
    assert st.get(real["id"]) is not None                             # real data untouched


def test_reporters_are_hashed(tmp_path):
    st = ReportStore(tmp_path)
    st.create(noisy(), noisy(), report(), "+15550001", "app", "m")
    raw = (tmp_path / "reports.db").read_bytes()
    assert b"+15550001" not in raw


def test_describe_counts_objects():
    r = report((True, True, True), [{"class": "car"}, {"class": "car"}, {"class": "bicycle"}], [{}])
    text = " ".join(d["text"] for d in describe(r))
    assert "a bicycle and 2 cars are standing on the sidewalk" in text
    assert "1 person is walking in the street" in text and "No visible sidewalk" in text
