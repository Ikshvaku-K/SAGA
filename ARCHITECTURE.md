# SAGA: Street Accessibility and Guidance Audit
## Architecture, pipeline, workflow and expected outcomes

*Status as of 10 Oct 2026. Companion documents: `PROJECT_DETAILS.md` (metrics in depth), `deploy/README.md` (operations).*

---

## 1. What SAGA is, in one paragraph

SAGA looks at a street photo, labels every pixel (road, sidewalk, person, car, bicycle and 14 other classes), and then checks **what each object is standing on**. From that it raises three accessibility flags (a vehicle blocking the sidewalk, a pedestrian forced into the roadway, a street with no visible sidewalk) and turns them into a **repair priority score** and a **city map**. Photos come from two places: a dataset of dashcam footage (BDD100K), and residents who photograph problems with the SAGA mobile app. Residents' 👍 / 👎 on each answer becomes a real-world label that measures, and later improves, the model.

**Design principle:** SAGA audits *places*, never people. Its output is a list of locations for a city to inspect and fix, not a ticket for a driver. Faces and licence plates are blurred before any photo is stored.

---

## 2. System architecture

```
                                   ┌──────────────────────────────────────────────┐
  iPhone (SAGA web app, /m/) ─┐    │  Vultr server  (Ubuntu)                       │
  Laptop browser (/)        ──┼──► │                                              │
                              │    │  Caddy  :443  HTTPS for sagatech.us          │
   https://sagatech.us        │    │    │   (automatic Let's Encrypt certificate) │
                              │    │    ▼                                         │
                              │    │  saga-api  (FastAPI/uvicorn, 127.0.0.1:8010) │
                              │    │    ├─ Login gate (4 questions + password)    │
                              │    │    ├─ SegFormer-B2 model (CPU)               │
                              │    │    ├─ Audit rules + priority score           │
                              │    │    ├─ Anonymiser (blur people & plates)      │
                              │    │    ├─ Report store (SQLite + JPEGs)          │
                              │    │    └─ Platform pages, map, report, gallery   │
                              │    └──────────────────────────────────────────────┘
                              │
  Training (offline, rented by the hour)
  RTX PRO 6000 GPU box ── trains model ── checkpoint copied to the server
```

### Components

| Component | Technology | Runs on | Role |
|---|---|---|---|
| **Web platform** | HTML/JS, Leaflet map | Browser | Photo audit, city map, model report, method page |
| **iPhone app** | Progressive web app at `/m/` | iPhone Safari / home screen | Camera → audit → 👍/👎 → location |
| **Login gate** | `ag/auth.py`, `gate.html` | Server | 4 Yes/No questions + shared password; 30-day signed cookie |
| **API server** | FastAPI + uvicorn | Server, port 8010, local only | All requests; loads the model once at startup |
| **Model** | SegFormer-B2 (27 M parameters, 104 MB) | Server CPU | Labels every pixel into 19 classes |
| **Audit engine** | `ag/audit.py` | Server | Turns the label map into flags and a 0–100 priority score |
| **Anonymiser** | `ag/reports.py` | Server | Blurs people and the plate band of vehicles before saving |
| **Report store** | SQLite + JPEG files | Server disk | Resident reports, locations, verdicts, notes |
| **HTTPS** | Caddy | Server, ports 80/443 | Certificate for sagatech.us, forwards to the API |
| **Training** | PyTorch on an NVIDIA GPU | Rented GPU box (off when idle) | Fine-tunes the model; produces the checkpoint |

### Security layers

1. **Network:** firewall allows only ports 22, 80 and 443. The API listens on `127.0.0.1` and is reachable only through Caddy.
2. **Login:** every page, photo and API call needs a session.
   - The gate answers and password are stored only as salted scrypt hashes.
   - 5 failed attempts lock an IP for 15 minutes.
   - The session cookie is HttpOnly and Secure.
3. **Per-report edit keys:** only the phone that sent a report can add its location or feedback.
4. **Abuse limits:** 30 uploads per IP per 10 minutes; 20 MB per photo.
5. **Privacy:**
   - Raw photos are never written to disk; only the blurred version is kept.
   - Senders are stored as salted hashes, never phone numbers.
6. **Secrets:** `/etc/saga/env` (root-only) holds the hashes, the session signing key and the admin (moderation) token.

---

## 3. The machine-learning pipeline

### 3.1 Data

| Data | Source | Used for |
|---|---|---|
| 7,000 labelled training photos | BDD100K "10K" set (official train split) | 6,500 for training, 500 for choosing the best checkpoint ("holdout") |
| 1,000 labelled photos | BDD100K official validation split | **The test set:** every reported score. Never used for training or selection |
| 747 unlabelled photos with US GPS | BDD100K official test split | The city map (New York, SF Bay Area, other US) |
| Resident photos | SAGA mobile app | Live audits and field validation (👍/👎) |

The labels give every pixel one of 19 classes (the Cityscapes scheme).

### 3.2 Model and training

1. **Starting point:** SegFormer-B2 pre-trained on Cityscapes (European streets), which uses the same 19 classes. Run unchanged, this is the **baseline**.
2. **Fine-tuning on BDD100K:** 8,000 steps, batch 8, random 512×512 crops, random scaling, flips, colour jitter, bf16 mixed precision.
3. **Hardware:**
   - Training ran on an RTX PRO 6000 Blackwell and took about 8 minutes.
   - The same job was estimated at about 3.5 hours on the Mac, which also stalled after the first holdout check (a bug, since fixed).
4. **Checkpoint selection:** the holdout set is checked every 1,000 steps, and the best one (step 6,000) is kept.
5. **Result** (test set, single pass):

| | Baseline | SAGA model |
|---|---|---|
| mIoU, all 19 classes | 50.4% | **63.4%** |
| Sidewalk IoU | 51.4% | **65.0%** |
| Person IoU | 65.0% | 74.5% |

### 3.3 From pixels to an audit

```
photo ──► resize to 1280 px ──► SegFormer ──► 19-class label map
                                                    │
                       ┌────────────────────────────┼─────────────────────────────┐
                       ▼                            ▼                             ▼
              each vehicle/bike:            each person:                  whole scene:
              what is directly below        what is below their feet?     building ≥5%, road ≥10%,
              it AND beside it?             road + road beside?           sidewalk ≤0.5%, curb zones
              sidewalk ≥60% / ≥40%          → PEDESTRIAN IN ROADWAY       visible, no highway barriers
              → SIDEWALK OBSTRUCTION                                      → NO VISIBLE SIDEWALK
                       └────────────────────────────┬─────────────────────────────┘
                                                    ▼
                        priority score 0–100  (40 no sidewalk, +25 per obstruction,
                        +15 per pedestrian in road, +15 if both pedestrians and no sidewalk)
```

**Safeguards**, each added after reviewing real mistakes:
- **Side check:** stops cars on a cross street, seen behind a near-side sidewalk, from being called "on the sidewalk".
- **Distance limit:** ignores tiny far-away objects whose contact with the ground can't be read reliably.
- **Frame-edge skip:** ignores objects cut off by the edge of the photo.
- **Curb-visibility check:** stops NYC streets lined with parked cars from being called "no sidewalk".

### 3.4 Accuracy of the audit

Each test photo is audited twice: from the model's labels and from the human annotators' labels. Agreement between the two:

| Flag | F1 (baseline → SAGA) | SAGA precision |
|---|---|---|
| Pedestrian in roadway | 0.69 → **0.82** | 0.88 |
| Sidewalk obstruction | 0.35 → **0.37** | 0.38 (only 14 real cases, so uncertain) |
| No visible sidewalk | 0.23 → **0.34** | 0.33 |
| Priority ranking (Spearman) | 0.70 → **0.85** | |

---

## 4. Process workflows

### 4.1 A resident reports a barrier with the iPhone app

```
 1. Open SAGA from the home screen ──► (first time) answer 4 questions + password ──► session for 30 days
 2. Tap "Take photo" ──► iPhone camera
 3. Phone shrinks the photo to 1600 px (faster upload) and asks for location in parallel
 4. Upload ──► server: model ──► audit ──► blur people/plates ──► save report ──► reply
 5. Screen shows: findings in plain text (VoiceOver-friendly), repair priority, overlay image
 6. "Did I get this right?"  👍 = confirmed · 👎 = wrong (optional note) · 👎 on "nothing found" = missed barrier
 7. Location: GPS from the phone, or a typed cross street
 8. The report appears on the city map's "Resident reports" layer
```

### 4.2 City staff or researchers on the web platform

| Tab | What they do |
|---|---|
| **Audit a photo** | Upload any street photo, or try a held-out sample |
| **City map** | Browse 747 audited US locations plus resident reports; filter by flag and city; click a point for the photo and findings; use the **repair priority list** |
| **Model report** | Test-set accuracy per class, audit agreement, training curves, field precision from residents, gallery of correct / false alarm / missed |
| **Method & limits** | Rules, thresholds, privacy, known limits |

### 4.3 Deployment workflow (what is happening right now)

| Step | Command | Status |
|---|---|---|
| Server check | `deploy.sh check` | ✅ Ubuntu 26.04, 2 vCPU, 3.3 GB RAM, 47 GB disk |
| Upload code, model, scores, images | `deploy.sh push` | ✅ 317 MB |
| Install environment, Caddy, firewall, Node | `deploy.sh setup` | ✅ |
| Build gallery + city map on the server | `deploy.sh rebuild` | ⏳ Gallery nearly done; map next (~2 h at ~10 s/photo on this plan) |
| Set gate answers + password (hashes only) | `deploy.sh login` | Next |
| Start services, get the HTTPS certificate | `deploy.sh start` | Next |
| Review the gallery by eye, exclude unsuitable images | manual | Next |

**Open decision:**
- **Option A:** let the server build the map itself (about 2 hours).
- **Option B:** build it once on the Mac (about 5 minutes) and upload it.
- **Option C:** upgrade to a 4-vCPU dedicated plan.

### 4.4 Improvement loop (how SAGA gets better over time)

```
 residents' 👍 / 👎  ──►  field precision per flag (live on the Model report page)
        │
        ├─► 👎 "wrong" and "missed" photos = the most useful ones to annotate
        ▼
 annotate a batch ──► retrain on a rented GPU (~10 min) ──► evaluate on the fixed test set
        ──► if better: copy the new checkpoint to the server (deploy.sh push + restart)
```

---

## 5. What we expect from SAGA

### 5.1 Expected behaviour today

| Situation | Expected result |
|---|---|
| Clear daytime street photo, landscape, from the curb | Reliable road/sidewalk/person/vehicle labels. Pedestrian-in-roadway flag is the most trustworthy (88% precision against human labels) |
| Car or bike parked on the sidewalk | Often detected, but this flag is the least certain (only 14 test cases; 5 caught). Expect misses |
| Street with parked cars hiding the sidewalk | Correctly *not* flagged as "no sidewalk" (by design) |
| Night, rain, snow | Lower accuracy; snow-covered sidewalks are sometimes flagged, which is useful |
| Portrait phone photo taken standing on the sidewalk | Works, but less accurate than dashcam-style views (the model learned from dashcams) |
| Response time | ~10 s per photo on the current 2-vCPU plan; ~3 s on 4 dedicated vCPU; under 1 s on a GPU |

### 5.2 Expected outcomes (what success looks like)

1. **For residents (especially wheelchair, walker and stroller users):** a 20-second way to report a barrier from a phone, with no account, no app store, and a text answer that works with VoiceOver.
2. **For a city or advocacy group:** a ranked repair list and a map that grows with every report, instead of waiting for complaints or manual surveys.
3. **For the model:** every 👍/👎 is a free, real-world label. Within a few hundred judged reports, SAGA can measure its own precision on residents' photos and target its weakest flag (sidewalk obstruction) with the right training examples.
4. **For the research claim:** a measurable chain from pixel accuracy (mIoU) to audit accuracy (flag F1) to ranking quality (Spearman 0.85) to field precision (residents' verdicts), with each layer reported honestly.

### 5.3 Known limits (what not to expect yet)

- **Not proof of a legal violation.** Flags are checked against human labels and residents' feedback, not field inspections.
- **No curb-ramp or crosswalk detection.** The training labels don't include them; people crossing at a crosswalk count as "in roadway".
- **One photo is not a street.** Aggregating several reports per block would make the map more reliable.
- **Shared password.** Anyone who has it gets in; individual accounts would be the next step if access needs to be revoked per person.
- **Speed** depends on the server plan (see 5.1).

---

## 6. Where everything lives

| Path | What |
|---|---|
| `saga/ag/` | Core Python: data, model, audit rules, reports, login |
| `saga/app/` | Server (`server.py`), platform (`static/index.html`), iPhone app (`static/m/`), gate (`static/gate.html`) |
| `saga/scripts/` | Training, evaluation, map, gallery, password tools |
| `saga/deploy/` | Server deployment kit |
| `saga/remote/` | Isolated GPU training kit |
| `saga/remote_runs/b2_8k/` | The deployed model and its metrics |
| Server `/opt/saga` | Deployed copy; `/etc/saga/env` holds secrets |
