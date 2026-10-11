"""SAGA platform: photo audit, city map, model report, and the mobile app's report API.

  .venv/bin/python -m uvicorn app.server:app --port 8010
Model: $AG_MODEL, else checkpoints/b2_bdd/best if present, else the Cityscapes baseline.
Moderation (DELETE /api/reports/<id>) requires the X-Admin-Token header matching
$AG_ADMIN_TOKEN (older installs: $AG_AGENT_TOKEN).
"""
import base64
import io
import json
import os
import re
import secrets
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
# local development secrets (production uses /etc/saga/env via systemd)
if (ROOT / ".env.local").exists():
    for _line in (ROOT / ".env.local").read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            os.environ.setdefault(*_line.split("=", 1))
from ag.audit import audit
from ag.auth import COOKIE, DEMO_QUESTIONS, GATE_QUESTIONS, Gate
from ag.classes import CLASSES, PALETTE
from ag.model import BASELINE, load, predict
from ag.reports import ReportStore, anonymize, describe, exif_latlon, in_us
from ag.viz import overlay
from ag.voice import Voice, speech_text

register_heif_opener()  # iPhones send photos as HEIC

RESULTS = ROOT / "results"
FINETUNED = ROOT / "checkpoints" / "b2_bdd" / "best"
# Switch to the fine-tuned model only once its final test evaluation exists, so the
# photo tab and the report page always describe the same model.
READY = FINETUNED.exists() and (RESULTS / "seg_finetuned_test.json").exists()
MODEL_PATH = os.environ.get("AG_MODEL") or (str(FINETUNED) if READY else BASELINE)
MODEL = load(MODEL_PATH)
# training log for the report page's curves: next to the checkpoint (runs/<run>/best -> runs/<run>/log.jsonl)
TRAIN_LOG = Path(MODEL_PATH).parent / "log.jsonl" if MODEL_PATH != BASELINE else None
MAX_UPLOAD = 20 * 1024 * 1024
TTA = os.environ.get("AG_TTA", "1") != "0"  # flip-averaging: +0.4 mIoU, 2x CPU time
APP_LIMIT = (int(os.environ.get("AG_APP_LIMIT", 30)), 600)  # app uploads per IP per 10 minutes
_hits = defaultdict(deque)
VOICE = Voice()
SPEECH_DIR = RESULTS / "speech"
_voice_hits = defaultdict(deque)
_map_cache = {"mtime": None, "by_id": {}}

app = FastAPI(title="SAGA")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
# the mobile web app (PWA) lives under /m/ so its service worker scope stays /m/
app.mount("/m", StaticFiles(directory=ROOT / "app" / "static" / "m", html=True), name="mobile")
(RESULTS / "map").mkdir(parents=True, exist_ok=True)
(RESULTS / "gallery").mkdir(parents=True, exist_ok=True)
app.mount("/media/map", StaticFiles(directory=RESULTS / "map"), name="map")
app.mount("/media/gallery", StaticFiles(directory=RESULTS / "gallery"), name="gallery")
STORE = ReportStore()
app.mount("/media/reports", StaticFiles(directory=STORE.root), name="reports")
ADMIN_TOKEN = os.environ.get("AG_ADMIN_TOKEN") or os.environ.get("AG_AGENT_TOKEN")
GATE = Gate()
if not GATE.configured and os.environ.get("AG_AUTH") != "off":
    raise SystemExit("Login not configured: run scripts/set_password.py (or set AG_AUTH=off for tests).")

# Open without a session: the gate itself, its login API, and what iOS fetches for the
# home-screen icon before any cookie exists. Admin (moderation) calls carry X-Admin-Token.
PUBLIC = ("/gate", "/api/auth/", "/m/manifest.webmanifest", "/m/icons/", "/favicon.ico")
_fails = defaultdict(deque)
_last_purge = [0.0]
STORE.purge_demo()  # demo data never survives a restart past its session


@app.middleware("http")
async def require_login(request: Request, call_next):
    if time.time() - _last_purge[0] > 60:  # sweep expired demo reports at most once a minute
        _last_purge[0] = time.time()
        STORE.purge_demo()
    path = request.url.path
    request.state.session = GATE.session(request.cookies.get(COOKIE))
    if (os.environ.get("AG_AUTH") == "off" or path.startswith(PUBLIC)
            or request.state.session
            or _is_admin(request.headers.get("x-admin-token"))):
        return await call_next(request)
    if path.startswith("/api/") or path.startswith("/media/"):
        return JSONResponse({"detail": "Login required."}, status_code=401)
    return RedirectResponse(f"/gate?next={path}", status_code=303)


class LoginIn(BaseModel):
    mode: str = "user"
    answers: list[str]
    password: str | None = None
    name: str | None = None


def _demo(request):
    """Demo session id if this request comes from a demo session, else None."""
    sess = getattr(request.state, "session", None)
    return sess["s"] if sess and sess.get("r") == "demo" else None


@app.get("/gate")
def gate_page():
    return FileResponse(ROOT / "app" / "static" / "gate.html", headers={"Cache-Control": "no-store"})


@app.get("/api/auth/questions")
def gate_questions(mode: str = "user"):
    return {"questions": DEMO_QUESTIONS if mode == "demo" else GATE_QUESTIONS}


@app.get("/api/auth/me")
def whoami(request: Request):
    sess = request.state.session
    if not sess:
        raise HTTPException(401, "Login required.")
    return {"role": sess["r"], "name": sess.get("n"), "expires": sess["e"]}


@app.post("/api/auth/login")
def login(body: LoginIn, request: Request):
    ip = _client_ip(request)
    q, now = _fails[ip], time.time()
    while q and now - q[0] > 900:
        q.popleft()
    if len(q) >= 5:
        raise HTTPException(429, "Too many attempts. Try again in 15 minutes.")
    demo = body.mode == "demo"
    ok = GATE.verify_demo(body.answers, body.name) if demo else GATE.verify(body.answers, body.password)
    if not ok:
        q.append(now)
        raise HTTPException(401, "Please try again.")
    q.clear()
    token, max_age = GATE.issue("demo", body.name.strip()) if demo else GATE.issue()
    res = JSONResponse({"ok": True})
    res.set_cookie(COOKIE, token, max_age=max_age, httponly=True, samesite="lax",
                   secure=GATE.secure_cookie, path="/")
    return res


@app.post("/api/auth/logout")
def logout():
    res = JSONResponse({"ok": True})
    res.delete_cookie(COOKIE, path="/")
    return res


def _b64jpeg(img, quality=85):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _jpeg_data_url(img, quality=85):
    return "data:image/jpeg;base64," + _b64jpeg(img, quality)


def _read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


# Phones always get the mobile app; only laptops/desktops see the web version (and can switch).
PHONE_UA = re.compile(r"iPhone|iPod|Android.+Mobile|Windows Phone|Mobile Safari", re.I)


def is_phone(request: Request):
    ua = request.headers.get("user-agent", "")
    return bool(PHONE_UA.search(ua)) and "iPad" not in ua


@app.get("/")
def index(request: Request):
    if is_phone(request):
        return RedirectResponse("/m/", status_code=302)
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/info")
def info():
    name = "Cityscapes baseline" if MODEL_PATH == BASELINE else "fine-tuned on BDD100K"
    return {"model": MODEL_PATH, "model_label": f"SegFormer-B2 · {name}",
            "classes": CLASSES, "palette": PALETTE, "voice": VOICE.enabled}


# ---------- spoken summaries (ElevenLabs) ----------

def _map_feature(fid):
    path = RESULTS / "map_findings.json"
    if not path.exists():
        return None
    if _map_cache["mtime"] != path.stat().st_mtime:
        _map_cache["by_id"] = {f["id"]: f for f in json.loads(path.read_text())["features"]}
        _map_cache["mtime"] = path.stat().st_mtime
    return _map_cache["by_id"].get(fid)


@app.get("/api/speech/{kind}/{fid}.mp3")
def speech(kind: str, fid: str, request: Request):
    """Spoken summary of one map point. kind = map (dashcam audit) | report (resident report)."""
    if kind == "map":
        f = _map_feature(fid)
        if not f:
            raise HTTPException(404, "No such map point.")
        text = speech_text(f["score"], f["flags"], f.get("obstructions", []), f.get("pedestrians_in_road", []))
        folder, name = SPEECH_DIR, f"map_{fid}"
    elif kind == "report":
        r = STORE.get(fid)
        visible = {x["id"] for x in STORE.all(_demo(request))} if r else set()
        if not r or fid not in visible:  # another demo session's report counts as missing
            raise HTTPException(404, "No such report.")
        text = speech_text(r["score"], r["flags"], r["findings"]["obstructions"],
                           r["findings"]["pedestrians_in_road"], r["verdict"], resident=True)
        folder, name = STORE.root, f"{fid}_speech"
    else:
        raise HTTPException(404, "Unknown kind.")
    clip = folder / f"{name}_{VOICE.cache_tag(text)}.mp3"
    if not clip.exists():
        if not VOICE.enabled:
            raise HTTPException(503, "Voice is not configured on this server.")
        q, now = _voice_hits[_client_ip(request)], time.time()
        while q and now - q[0] > 600:
            q.popleft()
        if len(q) >= 40:
            raise HTTPException(429, "Too many new voice clips. Please wait a few minutes.")
        q.append(now)
        try:
            audio = VOICE.synthesize(text)
        except RuntimeError as e:
            raise HTTPException(502, str(e))
        folder.mkdir(parents=True, exist_ok=True)
        tmp = clip.with_suffix(".part")
        tmp.write_bytes(audio)
        tmp.replace(clip)
    return FileResponse(clip, media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=86400",
                                                                 "X-Speech-Text": text.encode("ascii", "ignore").decode()})


def _open_photo(raw):
    """Bytes -> (upright RGB image at model scale, EXIF lat/lon or None)."""
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Photo is larger than 20 MB.")
    try:
        src = Image.open(io.BytesIO(raw))
        gps = exif_latlon(src)
        img = ImageOps.exif_transpose(src).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read that file as an image.")
    # The model was trained on 1280x720 dashcam frames; bring other photos to that scale.
    s = 1280 / max(img.width, img.height)
    return img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR), gps


def _audit(img, tta):
    t = time.time()
    lbl = predict(MODEL, img, tta=tta)
    return lbl, audit(lbl), int((time.time() - t) * 1000)


def _is_admin(token):
    return bool(ADMIN_TOKEN and token and secrets.compare_digest(token, ADMIN_TOKEN))


@app.post("/api/analyze")
async def analyze(request: Request, file: UploadFile = File(...), tta: bool = TTA):
    _rate_limit(_client_ip(request))
    img, _ = _open_photo(await file.read())
    lbl, r, ms = _audit(img, tta)
    return {"report": r, "ms": ms, "size": [img.width, img.height],
            "photo": _jpeg_data_url(img.resize((960, int(960 * img.height / img.width)))),
            "overlay": _jpeg_data_url(overlay(img, lbl, r, alpha=0.45, width=960))}


# ---------- resident reports ----------

def _make_report(raw, sender, channel, lat=None, lon=None, place=None, edit_key=None, demo=None):
    img, gps = _open_photo(raw)
    lbl, r, ms = _audit(img, tta=TTA)
    safe = anonymize(img, lbl)  # the raw photo is never stored
    src = "message" if lat is not None else ("exif" if gps else None)
    if lat is None and gps:
        lat, lon = gps
    ov = overlay(safe, lbl, r, alpha=0.45, width=960)
    rec = STORE.create(safe, ov, r, sender, channel, MODEL_PATH, lat=lat, lon=lon, place=place,
                       demo_session=demo["s"] if demo else None, expires=demo["e"] if demo else None,
                       location_source=src, edit_key=edit_key)
    return {"id": rec["id"], "score": r["score"], "flags": r["flags"],
            "findings": describe(r), "has_location": rec["lat"] is not None or bool(rec["place"]),
            "location_source": src, "ms": ms, "overlay_jpeg_b64": _b64jpeg(ov)}, safe


@app.delete("/api/reports/{rid}")
def delete_report(rid: str, x_admin_token: str | None = Header(None)):
    """Moderation: remove a report and its photos. Always requires the admin token."""
    if not _is_admin(x_admin_token):
        raise HTTPException(401, "Missing or wrong X-Admin-Token.")
    if not STORE.delete(rid):
        raise HTTPException(404, "No such report.")
    return {"deleted": rid}


# ---------- mobile app (PWA) ----------
# The app cannot hold a server secret, so each report gets its own random edit key;
# only the phone that created a report can add its location or feedback.

class AppEditIn(BaseModel):
    key: str
    verdict: str | None = None
    note: str | None = None
    lat: float | None = None
    lon: float | None = None
    place: str | None = None


def _client_ip(request: Request):
    # behind the tunnel the server binds to 127.0.0.1, so this header comes from Cloudflare
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


def _rate_limit(ip):
    n, window = APP_LIMIT
    q, now = _hits[ip], time.time()
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= n:
        raise HTTPException(429, "Too many photos from this network. Please wait a few minutes.")
    q.append(now)


@app.post("/api/app/reports")
async def app_create_report(request: Request, file: UploadFile = File(...), device: str = Form(...),
                            lat: float | None = Form(None), lon: float | None = Form(None)):
    _rate_limit(_client_ip(request))
    key = secrets.token_urlsafe(18)
    sess = request.state.session
    demo = sess if sess and sess.get("r") == "demo" else None  # temporary: deleted when the session ends
    out, safe = _make_report(await file.read(), f"app:{device[:64]}", "demo" if demo else "app", lat, lon,
                             edit_key=key, demo=demo)
    photo = safe.resize((960, round(960 * safe.height / safe.width)))
    return {**out, "key": key, "photo_jpeg_b64": _b64jpeg(photo, 82)}


@app.post("/api/app/reports/{rid}")
def app_edit_report(rid: str, body: AppEditIn):
    if not STORE.check_key(rid, body.key):
        raise HTTPException(403, "This report can only be changed from the phone that sent it.")
    if body.lat is not None or body.place:
        STORE.set_location(rid, body.lat, body.lon, body.place[:200] if body.place else None,
                           "app-gps" if body.lat is not None else "text")
    if body.verdict or body.note:
        try:
            STORE.set_feedback(rid, body.verdict, body.note[:500] if body.note else None)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return STORE.get(rid)


@app.get("/api/reports")
def list_reports(request: Request):
    """Public list for the map: US-located reports only, no reporter ids."""
    return [r for r in STORE.all(_demo(request)) if in_us(r["lat"], r["lon"])]


@app.get("/api/reports/stats")
def report_stats():
    return STORE.stats()


@app.get("/api/map")
def map_data():
    d = _read_json(RESULTS / "map_findings.json")
    if not d:
        raise HTTPException(404, "Run scripts/predict_map.py first.")
    return d


@app.get("/api/report")
def report():
    seg = {n: _read_json(RESULTS / f"seg_{n}_test.json") for n in ("baseline", "finetuned")}
    for v in seg.values():
        if v:
            v.pop("confusion", None)
    aud = {n: _read_json(RESULTS / f"audit_{n}_test.json") for n in ("baseline", "finetuned")}
    log = TRAIN_LOG
    curve, loss = [], []
    if log and log.exists():
        for line in log.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("event") == "eval":
                curve.append([rec["iter"], rec["holdout_mIoU"]])
            elif "loss" in rec:
                loss.append([rec["iter"], rec["loss"]])
    gallery = _read_json(RESULTS / "gallery" / "index.json") or {}
    review = _read_json(RESULTS / "rule_review.json")
    ref = _read_json(RESULTS / "seg_baseline_holdout.json")
    return JSONResponse({"seg": seg, "audit": aud, "curve": curve, "loss": loss,
                         "curve_ref": ref["mIoU"] if ref else None,
                         "gallery": gallery, "rule_review": review})


@app.get("/api/samples")
def samples():
    d = ROOT / "app" / "static" / "samples"
    return sorted(p.name for p in d.glob("*.jpg")) if d.exists() else []
