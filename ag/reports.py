"""Resident reports sent from the SAGA mobile app.

A report is one photo a resident texted in. The raw photo is never written to disk:
people (and the plate band of each vehicle) are blurred first, using the model's
own segmentation. Reporters are stored as salted hashes, never phone numbers.

Feedback (the resident's 👍 / 👎 on the app's answer) turns each report into a
field label:
  confirmed  the flags the model raised are right        (true positive)
  rejected   the model raised flags that are wrong        (false positive)
  missed     the model found nothing, the resident says there is a barrier (false negative)
"""
import hashlib
import json
import os
import secrets
import sqlite3
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from .classes import PERSON, RIDER, VEHICLES

ROOT = Path(__file__).resolve().parent.parent
FLAGS = ("sidewalk_obstruction", "pedestrian_in_roadway", "no_visible_sidewalk")
VERDICTS = ("confirmed", "rejected", "missed")
US_BOX = (24.0, 50.0, -125.0, -66.0)  # the platform map covers US locations only


def anonymize(img, lbl, plate_band=0.4):
    """Blur every person/rider pixel and the lower band of every vehicle."""
    lbl = np.asarray(lbl)
    if lbl.shape != (img.height, img.width):
        lbl = np.asarray(Image.fromarray(lbl).resize(img.size, Image.NEAREST))
    mask = np.isin(lbl, (PERSON, RIDER))
    comps, _ = ndimage.label(np.isin(lbl, VEHICLES))
    for sl in ndimage.find_objects(comps):
        ys, xs = sl
        top = ys.stop - max(1, int((ys.stop - ys.start) * plate_band))
        mask[top:ys.stop, xs.start:xs.stop] = True
    if not mask.any():
        return img.copy()
    mask = ndimage.binary_dilation(mask, iterations=max(2, img.width // 160))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=max(8, img.width // 60)))
    return Image.composite(blurred, img, Image.fromarray((mask * 255).astype(np.uint8)))


def exif_latlon(img):
    """GPS from the photo's EXIF block, if the phone kept it."""
    try:
        gps = img.getexif().get_ifd(0x8825)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def deg(v, ref):
        d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -d if ref in ("S", "W") else d

    try:
        return deg(gps[2], gps.get(1, "N")), deg(gps[4], gps.get(3, "E"))
    except Exception:
        return None


def in_us(lat, lon):
    return lat is not None and lon is not None and \
        US_BOX[0] <= lat <= US_BOX[1] and US_BOX[2] <= lon <= US_BOX[3]


class ReportStore:
    def __init__(self, root=None):
        self.root = Path(root or ROOT / "results" / "reports")
        self.root.mkdir(parents=True, exist_ok=True)
        salt_file = self.root / ".salt"
        if not salt_file.exists():
            salt_file.write_text(secrets.token_hex(16))
        self.salt = os.environ.get("AG_REPORT_SALT") or salt_file.read_text().strip()
        self.db = sqlite3.connect(self.root / "reports.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY, created REAL, reporter TEXT, channel TEXT,
            lat REAL, lon REAL, place TEXT, location_source TEXT,
            score INTEGER, flags TEXT, findings TEXT, model TEXT,
            verdict TEXT, note TEXT, feedback_at REAL)""")
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(reports)")}
        if "key_hash" not in cols:  # per-report secret for the public app endpoints
            self.db.execute("ALTER TABLE reports ADD COLUMN key_hash TEXT")
        if "demo_session" not in cols:  # set only for demo reports: temporary, private to that session
            self.db.execute("ALTER TABLE reports ADD COLUMN demo_session TEXT")
            self.db.execute("ALTER TABLE reports ADD COLUMN expires REAL")
        self.db.commit()

    def reporter_hash(self, sender_id):
        return hashlib.sha256(f"{self.salt}:{sender_id}".encode()).hexdigest()[:16]

    def create(self, photo, overlay_img, report, sender_id, channel, model,
               lat=None, lon=None, place=None, location_source=None, edit_key=None,
               demo_session=None, expires=None):
        """edit_key, when given, is required later to change this report via the public app API.
        demo_session + expires mark a temporary demo report (see purge_demo)."""
        rid = uuid.uuid4().hex[:12]
        photo.save(self.root / f"{rid}.jpg", quality=85)
        overlay_img.save(self.root / f"{rid}_o.jpg", quality=85)
        findings = {"obstructions": report["obstructions"],
                    "pedestrians_in_road": report["pedestrians_in_road"],
                    "coverage": report["coverage"]}
        self.db.execute(
            "INSERT INTO reports (id, created, reporter, channel, lat, lon, place, location_source,"
            " score, flags, findings, model, key_hash, demo_session, expires)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rid, time.time(), self.reporter_hash(sender_id), channel, lat, lon, place,
             location_source, report["score"], json.dumps(report["flags"]), json.dumps(findings), model,
             _sha(edit_key) if edit_key else None, demo_session, expires))
        self.db.commit()
        return self.get(rid)

    def get(self, rid):
        row = self.db.execute("SELECT * FROM reports WHERE id = ?", (rid,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["flags"] = json.loads(d["flags"])
        d["findings"] = json.loads(d["findings"])
        d.pop("reporter")
        d.pop("key_hash", None)
        d.pop("demo_session", None)
        return d

    def check_key(self, rid, key):
        row = self.db.execute("SELECT key_hash FROM reports WHERE id = ?", (rid,)).fetchone()
        return bool(row and row["key_hash"] and key and secrets.compare_digest(row["key_hash"], _sha(key)))

    def set_location(self, rid, lat=None, lon=None, place=None, source=None):
        self.db.execute("UPDATE reports SET lat = COALESCE(?, lat), lon = COALESCE(?, lon),"
                        " place = COALESCE(?, place), location_source = COALESCE(?, location_source)"
                        " WHERE id = ?", (lat, lon, place, source, rid))
        self.db.commit()
        return self.get(rid)

    def set_feedback(self, rid, verdict=None, note=None):
        if verdict is not None and verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        self.db.execute("UPDATE reports SET verdict = COALESCE(?, verdict), note = COALESCE(?, note),"
                        " feedback_at = ? WHERE id = ?", (verdict, note, time.time(), rid))
        self.db.commit()
        return self.get(rid)

    def delete(self, rid):
        """Remove a report and its images (moderation, or test clean-up)."""
        for suffix in ("", "_o"):
            (self.root / f"{rid}{suffix}.jpg").unlink(missing_ok=True)
        for clip in self.root.glob(f"{rid}_speech_*.mp3"):  # cached voice summaries
            clip.unlink(missing_ok=True)
        n = self.db.execute("DELETE FROM reports WHERE id = ?", (rid,)).rowcount
        self.db.commit()
        return n > 0

    def all(self, demo_session=None):
        """Real reports, plus the caller's own demo reports when demo_session is given."""
        rows = self.db.execute("SELECT id FROM reports WHERE demo_session IS NULL OR demo_session = ?"
                               " ORDER BY created DESC", (demo_session or "\x00",)).fetchall()
        return [self.get(r["id"]) for r in rows]

    def purge_demo(self, now=None):
        """Delete every demo report (row + photos) whose session has expired. Returns the count."""
        rows = self.db.execute("SELECT id FROM reports WHERE demo_session IS NOT NULL AND expires <= ?",
                               (now or time.time(),)).fetchall()
        for r in rows:
            self.delete(r["id"])
        return len(rows)

    def stats(self):
        """Field metrics from resident feedback (real reports only, never demo)."""
        reps = self.all()
        judged = [r for r in reps if r["verdict"]]
        per_flag = {}
        for f in FLAGS:
            raised = [r for r in judged if r["flags"][f] and r["verdict"] in ("confirmed", "rejected")]
            ok = sum(r["verdict"] == "confirmed" for r in raised)
            per_flag[f] = {"raised_and_judged": len(raised), "confirmed": ok,
                           "precision": round(ok / len(raised), 4) if raised else None}
        clean = [r for r in judged if not any(r["flags"].values())]
        return {
            "reports": len(reps),
            "reporters": len({r for (r,) in self.db.execute("SELECT reporter FROM reports WHERE demo_session IS NULL")}),
            "with_location": sum(r["lat"] is not None for r in reps),
            "judged": len(judged),
            "verdicts": {v: sum(r["verdict"] == v for r in judged) for v in VERDICTS},
            "flags": per_flag,
            # of photos the model called clean and the resident judged, how many hid a barrier
            "miss_rate": round(sum(r["verdict"] == "missed" for r in clean) / len(clean), 4) if clean else None,
        }


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def describe(report):
    """Plain-language findings, for screen readers and spoken summaries."""
    out = []
    obs = report["obstructions"]
    if obs:
        counts = Counter(o["class"] for o in obs)
        things = [f"a {k}" if n == 1 else f"{n} {k}{'es' if k.endswith('s') else 's'}"
                  for k, n in sorted(counts.items())]
        verb = "is" if len(obs) == 1 else "are"
        out.append({"flag": "sidewalk_obstruction",
                    "text": f"Sidewalk obstruction: {_join(things)} {verb} standing on the sidewalk."})
    ped = report["pedestrians_in_road"]
    if ped:
        n = len(ped)
        out.append({"flag": "pedestrian_in_roadway",
                    "text": f"Pedestrian in roadway: {n} {'person is' if n == 1 else 'people are'} "
                            "walking in the street."})
    if report["flags"]["no_visible_sidewalk"]:
        out.append({"flag": "no_visible_sidewalk",
                    "text": "No visible sidewalk on this built-up street."})
    return out


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
