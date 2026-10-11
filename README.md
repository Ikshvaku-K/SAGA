# SAGA: Street Accessibility and Guidance Audit

**SAGA turns a street photo into an accessibility audit.** A segmentation model labels every pixel; an audit engine then checks what each car, bike and person is *standing on* and flags three problems that block wheelchair, walker and stroller users:

| Flag | What it means |
|---|---|
| **Sidewalk obstruction** | A car, truck, bus, motorcycle or bicycle standing on the sidewalk |
| **Pedestrian in roadway** | A person forced to walk in the street |
| **No visible sidewalk** | A built-up street with a visible curb but no sidewalk |

Findings get a 0–100 repair priority, go on a city map, and can be read aloud. Residents report barriers from their phone and confirm or dispute each answer with 👍 / 👎, which become real-world labels. **SAGA audits places, never people:** faces and licence plates are blurred before anything is stored.

## Results

On the 1,000 BDD100K validation images (never used in training):

| | Cityscapes baseline | SAGA (fine-tuned) |
|---|---|---|
| mIoU, 19 classes | 50.4% | **63.4%** (63.8% with flip averaging) |
| Sidewalk IoU | 51.4% | **65.0%** |
| Pedestrian-in-roadway flag F1 | 0.69 | **0.82** (precision 0.88) |
| Repair-ranking agreement (Spearman) | 0.70 | **0.85** |

Full metrics and their meaning: [PROJECT_DETAILS.md](PROJECT_DETAILS.md). System design: [ARCHITECTURE.md](ARCHITECTURE.md).

## Features

- **Web platform** (`/`): photo audit, city map with repair priority list, model report, method and limits
- **Mobile app** (`/m/`, installable web app): camera → audit → 👍 / 👎 → GPS pin, plus an in-app map with New York / SF / Near me. Phones always get the mobile app; laptops can switch between the two.
- **Voice** (optional, ElevenLabs): 🔊 Listen on any report, and voice guidance that reads out reports within 60 m
- **Login gate**: *User* (yes/no questions + shared password, 30 days) or *Demo* (15 minutes; demo data is private to the session and deleted when it ends)
- **Privacy**: people and plates blurred server-side, raw photos never stored, no accounts or phone numbers

## Quick start (local)

Requires Python 3.12.

```bash
git clone <this repo> saga && cd saga
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Configuration: copy the template, then generate the login hashes into it
cp .env.example .env.local
.venv/bin/python scripts/set_password.py --questions "Q1?|Q2?|Q3?|Q4?" --answers yes,no,yes,yes --out .env.local
#    (asks for the shared password; only hashes are written)
echo "AG_ADMIN_TOKEN=$(openssl rand -hex 24)" >> .env.local

# 2. Run
.venv/bin/python -m uvicorn app.server:app --port 8010
# open http://localhost:8010
```

Without a fine-tuned model, the server uses the public Cityscapes SegFormer from Hugging Face. To use your own, set `AG_MODEL` in `.env.local` to a checkpoint folder (see Training). Every setting is documented in [.env.example](.env.example).

## Data

The BDD100K dataset is **not** included (size and licence). Download from https://bdd-data.berkeley.edu/ into `data/`:

| Package | Goes to |
|---|---|
| 10K Images | `data/10k/{train,val,test}` |
| Semantic Segmentation labels | `data/bdd100k_seg_maps/labels/{train,val}` |
| Info (GPS), only for building the map | `data/bdd100k_info.zip` |

## Training and evaluation

```bash
.venv/bin/python scripts/make_splits.py                       # fixed train / holdout / test split
.venv/bin/python scripts/eval_seg.py --name baseline --save-preds
.venv/bin/python scripts/train.py --iters 8000 --out checkpoints/b2_bdd --amp
.venv/bin/python scripts/gps_index.py                         # US GPS points for the map
./run_final.sh                                                # test metrics, audit scores, gallery, map
.venv/bin/python -m pytest -q tests                           # 27 tests
```

8,000 steps take about 8 minutes on an RTX PRO 6000 or about 3.5 hours on an Apple M4 Pro. `remote/gpu.sh` runs training on a rented GPU in an isolated folder (`AG_GPU_HOST=user@host remote/gpu.sh check|push|setup|train|eval|pull`). Model weights are not in the repo (over GitHub's 100 MB limit); train your own or host them on Hugging Face.

## Deployment

`deploy/deploy.sh` installs SAGA on an Ubuntu server with systemd, Caddy (automatic HTTPS) and a firewall. See [deploy/README.md](deploy/README.md):

```bash
export AG_HOST=root@your-server AG_DOMAIN=your-domain.example
deploy/deploy.sh check && deploy/deploy.sh push && deploy/deploy.sh setup && deploy/deploy.sh rebuild
AG_GATE_QUESTIONS='Q1?|Q2?|Q3?|Q4?' AG_GATE_ANSWERS=yes,no,yes,yes deploy/deploy.sh login
deploy/deploy.sh start
deploy/deploy.sh voice     # optional: ElevenLabs key, typed at a hidden prompt
```

## Security model

- **No secrets in the repo.** All credentials live in `.env.local` (local) or `/etc/saga/env` (server, mode 600); both are git-ignored. Gate answers and the password are stored only as salted scrypt hashes.
- **The API key stays on the server.** ElevenLabs is called server-side with text the server writes itself, and each clip is cached.
- **Limits:** 5 failed logins per IP lock it for 15 minutes; 30 uploads per IP per 10 minutes; 20 MB per photo; per-report edit keys; admin token for moderation.
- **Personal data is never committed:** `results/reports/` (resident photos and database) is git-ignored.

## Project layout

| Path | Contents |
|---|---|
| `ag/` | Core Python: data, model, audit rules, reports, login, voice |
| `app/` | FastAPI server, web platform, mobile app, login gate |
| `scripts/` | Training, evaluation, map, gallery, password tools |
| `tests/` | Unit tests (audit rules, login, reports, voice) |
| `deploy/`, `remote/` | Server deployment and isolated GPU training |
| `results/`, `remote_runs/` | Metrics from the reported runs (JSON) |

## Limits

- Audit agreement is measured against human labels, not field inspections. It is not proof of a legal violation.
- There are no curb-ramp or crosswalk classes, so crossing pedestrians count as "in roadway".
- One photo is not a street; aggregating several reports per block would make the map more reliable.
- BDD100K is licensed for educational, research and non-profit use. See [NOTICE.md](NOTICE.md).
