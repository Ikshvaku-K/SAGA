# SAGA (Street Accessibility and Guidance Audit): Project Details

**Street-level accessibility auditing with computer vision.** SAGA looks at ordinary street photos taken from a car, labels every pixel (road, sidewalk, person, car, bicycle and 14 other classes), and then checks *what each object is standing on*. From that it flags three accessibility problems, puts them on a map, and ranks them into a repair priority list for a city to inspect.

Status as of 10 Oct 2026: the dataset, baseline, audit engine, tests and web platform are complete. Fine-tuning ran to **step 1,000 of 8,000** on an Apple M4 Pro and is **paused, waiting for an NVIDIA RTX 6000-class GPU**. All numbers below are real measurements; the fine-tuned numbers come from the step-1,000 checkpoint.

---

## 1. The problem and why it matters

People who use wheelchairs, walkers, canes or strollers depend on a continuous, clear sidewalk. Three things routinely break that:

1. **Obstructions:** cars, trucks, motorcycles and bikes parked on the sidewalk narrow or block the path. A wheelchair user often has to turn back or go into the street.
2. **Pedestrians pushed into traffic:** when a sidewalk is blocked, missing, or buried in snow, people walk in the roadway.
3. **Missing sidewalks:** built-up streets with no sidewalk at all.

Cities are legally responsible for sidewalk access (under the Americans with Disabilities Act, cities must keep pedestrian routes accessible and plan fixes). Finding problems today relies on manual surveys and resident complaints, which are slow, expensive and uneven: neighbourhoods that complain less get fixed less.

**SAGA's idea:** vehicles already drive every street, and many carry dashcams. The same footage can be turned into a city-wide accessibility audit.

**Design principle: audit infrastructure, not people.** The output is a list of *places* to inspect and fix, never a ticket for a driver. This keeps the tool on the side of disabled residents without turning it into a surveillance or enforcement system that falls hardest on low-income neighbourhoods.

---

## 2. How it works

```
Dashcam frame (1280×720)
        │
        ▼
Semantic segmentation (SegFormer-B2)  ──►  19-class label map
        │                                  road, sidewalk, person, car, bicycle, …
        ▼
Ground-contact audit (rules)          ──►  findings per frame
        │                                  obstruction / pedestrian in road / no sidewalk
        ▼
Priority score (0-100) + GPS          ──►  map + repair priority list
```

**Why segmentation instead of object detection?** A detector draws boxes around cars and people but says nothing about the ground. The audit's key question is *what surface is under this object*, which needs a pixel-level label for road vs sidewalk. Segmentation gives both the objects and the ground in one pass.

---

## 3. Data

**Source:** Berkeley DeepDrive BDD100K, real dashcam video recorded mostly in New York and the San Francisco Bay Area, across day, night, rain and snow. Licence: educational, research and non-profit use.

**What we used:**

| Package | Contents | Size |
|---|---|---|
| 10K Images | 7,000 train, 1,000 val, 2,000 test photos, 1280×720 | 1.1 GB |
| Semantic segmentation labels | Pixel masks for train and val (19 classes, Cityscapes ids) | 0.2 GB |
| Info (GPS/IMU) | Used once to extract a GPS point per test image, then deleted | 4.8 GB |

**Splits (fixed, seed 0):**

| Split | Images | Source | Used for |
|---|---|---|---|
| Train | 6,500 | official train | learning |
| Holdout | 500 | official train | choosing the best checkpoint |
| **Test** | **1,000** | **official val** | **every number in this report; never used for training or selection** |
| Map | 747 | official test (unlabelled) | the platform's map: test images with a US GPS fix |

Keeping test separate from holdout matters: if the same images were used to pick the checkpoint *and* to report accuracy, the reported number would be optimistic.

**Data quality issues found and handled:**
- 6 of the 8,000 labelled images are stored rotated (portrait) against a landscape label mask. Brightness checks on sky vs road showed a 90° rotation realigns them; the loader fixes this automatically.
- Some annotations are inconsistent: parking lanes, and in one case a whole NYC street, are labelled "sidewalk". This puts a ceiling on how well any model can score on the obstruction flag (see §7).

---

## 4. Model

**SegFormer-B2** (27 M parameters), a transformer-based segmentation model, initialised from weights trained on **Cityscapes** (European city streets). Cityscapes and BDD100K use the same 19 classes with the same ids, so the pretrained model works on BDD with no changes. That gives a fair **baseline** to measure fine-tuning against.

**Training setup:** AdamW optimiser (learning rate 6e-5 for the backbone, 10× for the head), 300-step warm-up then linear decay, batch 8, random 512×512 crops, random scaling 0.5–2.0×, horizontal flips and brightness/contrast jitter. Evaluated on full 1280×720 frames.

**Hardware so far:** Apple M4 Pro (24 GB): 1.42 s per training step, 0.14 s per image for inference. A full 8,000-step run would take about 3.5 hours (§10 covers the GPU plan).

---

## 5. The audit engine

The rules run on the label map at 640×360 and are unit-tested (10 tests on synthetic street scenes with known answers).

| Flag | Rule | Why it signals an accessibility problem |
|---|---|---|
| **Sidewalk obstruction** | A car, truck, bus, motorcycle or bicycle whose *contact strip* (the 6 px directly below its lowest edge) is ≥ 60% sidewalk **and** with ≥ 40% sidewalk beside its lowest rows | A vehicle standing on the sidewalk narrows or blocks the path |
| **Pedestrian in roadway** | A person whose contact strip is ≥ 60% road, with road beside their feet | People walking in traffic, often because the sidewalk is blocked, missing or buried in snow |
| **No visible sidewalk** | ≥ 5% building, ≥ 10% road, ≤ 0.5% sidewalk, ≤ 25% vehicles in the left/right curb zones, ≤ 3% wall + fence | A built-up street where no sidewalk can be seen even though the curb is visible |

**Safeguards added after reviewing real output.** Each was added because a visual review of the rules on the human-labelled masks showed a systematic false alarm:

| Problem found | Fix | Effect on the 1,000 test labels |
|---|---|---|
| Cars on a cross street, seen *behind* a near-side sidewalk, read as "on the sidewalk" | Side check (the sidewalk must also be *beside* the object, not just below it), plus a larger minimum object size | Obstruction flags 70 → 42 |
| Tiny distant cars and cars cut off by the frame edge gave unreliable contact | Distance limit (object ≥ 35 px tall at 640×360) and frame-edge skip | Obstruction flags 42 → 14 |
| NYC streets where parked cars hide the sidewalk read as "no sidewalk" | Only call a sidewalk missing if the curb zones are visible; exclude highway barriers | No-sidewalk flags 183 → 32 |

**Priority score (0–100):** 40 for no visible sidewalk, +25 per obstruction (max 50), +15 per pedestrian in roadway (max 30), +15 extra when pedestrians are in the road *and* no sidewalk is visible (they have nowhere else to walk). The score is a transparent heuristic for ranking, not a validated severity scale.

---

## 6. Metrics: what each one measures and why it matters

SAGA is measured at three levels, because each can fail independently:

1. **Does the model see the street correctly?** Segmentation metrics.
2. **Does the model's view produce the right audit?** Audit-agreement metrics.
3. **Are the audit rules themselves right?** Rule review.

### 6.1 Segmentation metrics

**IoU (Intersection over Union), per class**

> IoU = TP / (TP + FP + FN), counted in pixels

For the "sidewalk" class: TP = pixels correctly called sidewalk, FP = pixels wrongly called sidewalk, FN = sidewalk pixels the model missed. IoU is 1.0 for a perfect match and penalises both over- and under-prediction.

*Why it matters here:* the audit lives or dies on a few classes.
- **Sidewalk** and **road** IoU decide whether the contact strip under each object is read correctly. Confusing the two flips an obstruction flag on or off.
- **Person, car, bicycle, motorcycle, truck, bus** IoU decide whether the objects are found at all, and whether their bottom edge (where they touch the ground) is in the right place.
- Classes like sky or vegetation matter much less to the concept.

**mIoU (mean IoU over the 19 classes)**

The average of the per-class IoUs. It is the standard single-number summary, and every class counts equally whatever its size, so rare classes (bicycle, rider) weigh as much as road.

*Why it matters:* it is the fairest headline for "how well does the model understand the street", and it is comparable with published BDD100K results. It is **not** enough on its own: a model can raise mIoU by improving sky and buildings while getting worse on sidewalk.

**Pixel accuracy**

The share of all labelled pixels classified correctly. Road, sky and buildings cover most of each frame, so this number is dominated by easy classes. It is reported for completeness but is the **least informative** metric for this project.

**Holdout mIoU (during training)**

mIoU on the 500 holdout images, checked every 1,000 steps to keep the best checkpoint. It is used only to choose the checkpoint, never as a reported result.

### 6.2 Audit-agreement metrics

Each test image is audited twice: once on the model's predicted mask and once on the annotators' mask. Treating the annotator-based flags as the reference, each flag gets:

| Metric | Formula | What it means for SAGA |
|---|---|---|
| **Precision** | flagged correctly / all flags raised | Of the places the model sends an inspector to, how many are real? Low precision wastes inspection time and erodes a city's trust in the tool |
| **Recall** | flagged correctly / all reference flags | Of the real problems, how many does the model catch? Low recall means hazards stay unfixed and the map under-represents those streets |
| **F1** | 2·P·R / (P + R) | One number that balances the two; used to compare models |
| **Support** | number of reference flags | How many real cases the score is based on. **Small support means a noisy score** (see 95% intervals below) |

For an accessibility audit, **precision is usually worth more than recall**. A city will act on a short, trustworthy list; a long list full of false alarms gets ignored. Missed cases also tend to be caught on later drives.

**Priority-score agreement:**
- **Score MAE** (mean absolute error) is the average gap between the priority score from the model's mask and from the human mask. Lower is better.
- **Spearman rank correlation** asks whether the model *ranks* frames in the same order as the human-label audit (1.0 = identical order). This is the metric closest to the product, since the repair priority list is a ranking.

**Important:** audit agreement measures how faithfully the model reproduces the audit the human labels would give. It does **not** prove that each flag is a real-world violation. That is what §6.3 is for.

### 6.3 Rule validity (visual review)

To judge the rules separately from the model, the rule output on the *human-labelled* masks was reviewed image by image:

| Flag | Reviewed | Plausible | What the errors were |
|---|---|---|---|
| Sidewalk obstruction | 14 | **8 (57%)** | Ambiguous annotations: parking lanes and one NYC street labelled as sidewalk |
| Pedestrian in roadway | 12 (sample) | **12 (100%)** | None. Most are people crossing, which the rule cannot tell apart from being forced off the sidewalk (no crosswalk class) |
| No visible sidewalk | 32 | not scored | Includes suburban roads without sidewalks and sidewalks buried in snow, plus frames where annotators did not mark a sidewalk |

### 6.4 Operational metrics

| Metric | Value (M4 Pro) | Why it matters |
|---|---|---|
| Inference time per 1280×720 frame | 0.14 s (≈0.3 s with flip averaging) | A city-scale audit is millions of frames; this sets processing cost |
| Training speed | 1.42 s per step (batch 8, 512×512) | Sets how much model you can train in a time budget |
| Map coverage | 747 US locations (618 New York, 120 SF Bay Area, 9 other US) | How much of a city the audit can speak to |

---

## 7. Results

All results are on the **1,000-image test set**, single-scale (no flip averaging) for a fair comparison.

### 7.1 Segmentation

| Metric | Cityscapes baseline | Fine-tuned, step 1,000 | Change |
|---|---|---|---|
| **mIoU (19 classes)** | 50.4% | **60.6%** | **+10.2** |
| Pixel accuracy | 89.3% | 92.7% | +3.4 |
| Holdout mIoU (selection set) | 43.5% | 55.8% | +12.3 |

Per-class IoU for the classes the audit depends on:

| Class | Baseline | Step 1,000 | Change | Role in the audit |
|---|---|---|---|---|
| road | 90.4 | 92.7 | +2.3 | Ground under pedestrians |
| **sidewalk** | 51.4 | **58.1** | **+6.7** | Ground under vehicles; the weakest key class |
| person | 65.0 | 73.5 | +8.5 | Pedestrian flag |
| rider | 39.5 | 55.2 | +15.8 | Kept separate from pedestrians |
| car | 84.2 | 88.3 | +4.0 | Obstruction flag |
| truck | 31.0 | 41.7 | +10.7 | Obstruction flag |
| bus | 28.0 | 71.3 | +43.3 | Obstruction flag |
| motorcycle | 50.7 | 58.9 | +8.2 | Obstruction flag |
| bicycle | 43.5 | 63.3 | +19.8 | Obstruction flag (parked bikes) |

Every class except "train" (almost never present) improved after only 12.5% of the planned training.

### 7.2 Audit agreement (model mask vs annotator mask)

| Flag | Support | Baseline P / R / F1 | Step 1,000 P / R / F1 |
|---|---|---|---|
| Pedestrian in roadway | 68 | 0.69 / 0.69 / 0.69 | **0.83 / 0.72 / 0.77** |
| No visible sidewalk | 32 | 0.18 / 0.31 / 0.23 | **0.33 / 0.41 / 0.36** |
| Sidewalk obstruction | 14 | 0.30 / 0.43 / 0.35 | 0.33 / 0.14 / 0.20 |

| Priority score | Baseline | Step 1,000 |
|---|---|---|
| Score MAE (lower is better) | 4.07 | **2.85** |
| Spearman rank correlation | 0.699 | **0.709** |

**95% confidence intervals** (Wilson) show how much each score can be trusted:

| Flag (step 1,000) | Recall | 95% interval | Precision | 95% interval |
|---|---|---|---|---|
| Pedestrian in roadway | 49/68 | 0.60–0.81 | 49/59 | 0.72–0.91 |
| No visible sidewalk | 13/32 | 0.26–0.58 | 13/40 | 0.20–0.48 |
| Sidewalk obstruction | 2/14 | 0.04–0.40 | 2/6 | 0.10–0.70 |

### 7.3 How to read these results

- **Segmentation improved clearly and broadly.** +10 mIoU after 1,000 steps, with the biggest gains on the classes the audit needs (bicycle, rider, bus, person, sidewalk).
- **The pedestrian flag is the strongest:** 83% precision, with a tight interval. This flag is ready to demonstrate.
- **"No visible sidewalk" improved but is still weak** (F1 0.36). It is the hardest flag, because one frame often cannot show whether a sidewalk exists.
- **The obstruction flag is unresolved.** Its recall fell from 6/14 to 2/14. With only 14 reference cases the intervals overlap heavily, so this may be noise, but it may also be real: the fine-tuned model may be labelling the ground under parked cars as road. This is the first thing to investigate after full training, together with the label noise found in §6.3. More reference cases (Project Sidewalk labels) are needed to measure this flag properly.

---

## 8. The platform

A local web app (FastAPI backend, plain HTML/JS frontend, Leaflet map), started with:

```bash
cd saga
.venv/bin/python -m uvicorn app.server:app --port 8010
```

| Tab | What it shows | Which metrics it surfaces |
|---|---|---|
| **Audit a photo** | Upload any street photo (or click a held-out sample): colour overlay, boxes on findings, priority score, scene coverage | Per-frame findings, priority score, inference time |
| **City map** | 747 audited US locations, coloured by priority; filters by flag and city; click for the frame and findings; top-25 repair list | Location counts, findings per flag, priority ranking |
| **Model report** | Baseline vs fine-tuned IoU per class, audit agreement, training curves, gallery of correct flags / false alarms / misses | §7 metrics |
| **Method & limits** | Rules, thresholds, the safeguards above, known limits | Rule definitions |

The platform will show fine-tuned results once full training finishes and `run_final.sh` has run. Until then it serves the Cityscapes baseline.

---

## 9. Limits and ethics

- **Agreement with labels is not real-world validity.** Flags are checked against annotator masks and a visual review, not against field inspections.
- **No crosswalk or curb-ramp classes** in the data. Crossing pedestrians count as "in roadway", and curb ramps (a core ADA item) need a different dataset, e.g. Project Sidewalk labels with Mapillary imagery.
- **One frame is not a street.** Aggregating several drives per block would make "no visible sidewalk" and obstruction flags much more reliable.
- **Privacy:** dashcam frames show faces and licence plates. Both must be blurred before any public deployment.
- **Licence:** BDD100K permits educational, research and non-profit use only.

---

## 10. Next step: training on an RTX 6000-class GPU

The code now picks an NVIDIA GPU automatically and uses bf16 mixed precision there. Holdout checks load data in-process, which fixes the hang that stopped the Mac run right after the step-1,000 check.

**Estimated training time** (not yet measured):

| GPU | 8,000 steps, current setup | vs M4 Pro |
|---|---|---|
| Quadro RTX 6000 (24 GB) | ~35–45 min | ~4–5× faster |
| RTX 6000 Ada (48 GB) | ~15–25 min | ~8–12× faster |
| RTX PRO 6000 Blackwell (96 GB) | ~10–15 min | ~12–15× faster |

**Recommended use of the extra capacity**, given the results above:
1. **Finish the current B2 run** (8,000 steps) for a like-for-like comparison with this report.
2. Then **SegFormer-B5 with 1024×512 crops for 20,000+ steps.** Larger crops keep small and distant objects (bikes, pedestrians, the ground line under parked cars) at higher resolution, which is exactly what the obstruction flag needs.

**Commands on the GPU machine** (after copying `saga/` with `data/10k`, `data/bdd100k_seg_maps`, `data/splits.json` and `data/gps_test.json`):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision transformers fastapi "uvicorn[standard]" python-multipart pillow numpy scipy pytest
.venv/bin/python -m pytest -q tests
# resume the B2 run from step 1,000 (or delete checkpoints/b2_bdd to start fresh)
.venv/bin/python scripts/train.py --iters 8000 --out checkpoints/b2_bdd --amp
./run_final.sh
```

---

## 11. File map

| Path | Purpose |
|---|---|
| `ag/data.py` | Dataset, splits, augmentation, rotation fix |
| `ag/model.py` | Model loading, prediction, mIoU evaluation |
| `ag/audit.py` | Accessibility rules and priority score |
| `ag/viz.py` | Colour overlays and finding boxes |
| `scripts/make_splits.py` | Fixed train / holdout / test split |
| `scripts/train.py` | Fine-tuning (resumable, best-on-holdout) |
| `scripts/eval_seg.py` | Segmentation metrics on any split |
| `scripts/eval_audit.py` | Audit precision / recall / F1, score MAE and Spearman |
| `scripts/gps_index.py` | GPS per test image (US only) |
| `scripts/predict_map.py` | Audits the map images for the platform |
| `scripts/make_gallery.py` | Correct / false-alarm / missed examples for the report |
| `run_final.sh` | Full final evaluation and platform data |
| `tests/test_audit.py` | Unit tests for the audit rules |
| `app/server.py`, `app/static/index.html` | The platform |
| `results/` | All metric files (`seg_*.json`, `audit_*.json`, `rule_review.json`) |
