# MLBB Draft AI v2

Real-time hero pick/ban assistant for Mobile Legends: Bang Bang.
Runs on PC, captures the MLBB screen (live or via scrcpy), detects
hero picks and bans using computer vision, and displays ranked
recommendations in a floating overlay.

---

## What's new in v2

| Area | v1 | v2 |
|---|---|---|
| Hero detection | Full-frame + x-position heuristic | 20 fixed slot crops, each detected independently |
| CV pipeline | Single backend | Template matching → YOLO fallback per slot |
| Confidence | Not enforced | Per-slot threshold (default 0.60), ignores low-confidence |
| Temporal stability | None | 7-frame sliding window majority vote + lock-in mechanism |
| Dataset | 20 heroes | **132 heroes** (full roster from ODS file) |
| Hero data | Static | Auto-reloads when `heroes.json` changes on disk |
| Scoring | 4-component flat | 6-component + **phase-aware dynamic weights** |
| Draft phases | None | BAN → EARLY → MID → LATE (weights shift per phase) |
| Enemy analysis | 4 archetypes | 6 archetypes (tank-heavy, squishy, cc-heavy, mobile, burst, sustain) |
| Tier system | None | S/A/B/C tier affects recommendation score |
| UI | Basic cards | Phase badge, enemy comp pills, per-rec risk badge, score bar |
| Performance | Basic threading | Change-detection skips unchanged frames/slots |
| Logging | stdout only | stdout + `mlbb_ai.log` file |

---

## Project structure

```
mlbb_ai_v2/
├── capture/
│   └── screen_capture.py     # mss/cv2 capture with change detection
├── config/
│   └── slot_config.py        # 20 fixed slot regions, resolution scaling
├── vision/
│   ├── hero_detector.py      # Slot cropper, hybrid detector, temporal filter
│   └── templates/            # Hero PNG icons (72×72px, add to enable template matching)
│   └── weights/              # mlbb_draft.pt (add to enable YOLO)
├── recommender/
│   └── engine.py             # Phase-aware scoring engine, 132-hero support
├── data/
│   ├── hero_db.py            # Thread-safe DB with auto-reload
│   └── heroes.json           # 132-hero dataset
├── ui/
│   └── overlay.py            # PyQt5 overlay (Tkinter fallback)
├── tests/
│   └── test_all.py           # 24-test suite (no pytest required)
├── main.py                   # Entry point
└── requirements.txt
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Demo mode (no game, no CV — recommended first run)
python main.py --mode mock

# 3. Live mode via scrcpy
scrcpy --window-title "MLBB"     # mirror phone screen to PC window
python main.py --mode auto --width 1280 --height 720

# 4. Specify exact capture region
python main.py --mode auto --region 0 0 1280 720

# 5. Run tests
python tests/test_all.py
```

---

## CLI reference

```
python main.py --help

  --mode    {auto|mock|template|yolo}  CV backend (default: mock)
  --fps     INT    Target capture FPS             (default: 10)
  --top     INT    Recommendations to show         (default: 3)
  --conf    FLOAT  Detection confidence threshold  (default: 0.60)
  --window  INT    Temporal filter window size     (default: 7)
  --width   INT    Capture window width            (default: 1280)
  --height  INT    Capture window height           (default: 720)
  --region  X Y W H  Override capture region
  --log-level {DEBUG|INFO|WARNING}                 (default: INFO)
```

---

## Calibrating slot positions

The 20 slot rectangles in `config/slot_config.py` are calibrated for
a 1280×720 scrcpy window. Other resolutions auto-scale from this
baseline. To fine-tune for your layout:

1. Take a screenshot during the draft phase.
2. Open in any image editor (Paint, GIMP, Photoshop).
3. Measure each hero-portrait bounding box (x, y, w, h).
4. Update `_ALLY_PICKS`, `_ENEMY_PICKS`, `_BANS` in `slot_config.py`.

Resolution presets (1280×720, 1920×1080, 2560×1440, 800×600) are
in `PRESETS` and auto-selected by `get_config(width, height)`.

---

## Adding template icons (enables offline CV)

1. Screenshot each hero portrait from the draft screen (~72×72 px).
2. Save as `vision/templates/<HeroName>.png`.
   - Name must exactly match the `name` field in `heroes.json`.
   - E.g. `vision/templates/Gusion.png`, `vision/templates/Yi Sun-shin.png`
3. Run with `--mode template` or `--mode auto`.

With 132 templates loaded, template matching runs at ~5–15 ms per frame on CPU.

---

## Training a YOLOv8 model

```bash
pip install ultralytics

# Annotate draft screenshots with hero bounding boxes (use LabelImg or Roboflow)
# Class names must match hero names in heroes.json

yolo train model=yolov8n.pt data=mlbb_draft.yaml epochs=100 imgsz=128
cp runs/detect/train/weights/best.pt vision/weights/mlbb_draft.pt
python main.py --mode yolo
```

---

## Updating hero data (patch changes)

Edit `data/heroes.json` directly. The database auto-reloads within 30 seconds
(no restart needed). To add a new hero:

```json
{
  "id": 133,
  "name": "NewHero",
  "role": "Fighter",
  "win_rate": 0.50,
  "pick_rate": 0.12,
  "ban_rate": 0.10,
  "tier": "B",
  "counters": [],
  "strong_against": [],
  "tags": []
}
```

Valid roles: `Tank`, `Fighter`, `Assassin`, `Mage`, `Marksman`, `Support`
Valid tiers: `S`, `A`, `B`, `C`

---

## Scoring formula

```
score = win_rate      × W_win_rate
      + counter_score × W_counter
      + team_fit      × W_team_fit
      + tier_bonus    × W_tier
      + synergy_bonus × W_synergy
      - risk          × W_risk
```

Weights shift by phase:

| Phase | win_rate | counter | team_fit | tier | synergy | risk |
|-------|----------|---------|----------|------|---------|------|
| BAN   | 0.20 | 0.20 | 0.30 | 0.15 | 0.10 | 0.05 |
| EARLY | 0.25 | 0.20 | 0.35 | 0.10 | 0.05 | 0.05 |
| MID   | 0.30 | 0.30 | 0.20 | 0.10 | 0.05 | 0.05 |
| LATE  | 0.20 | 0.45 | 0.15 | 0.08 | 0.07 | 0.05 |

---

## ML integration hook

`recommender/engine.py` contains `ml_win_probability()`:

```python
def ml_win_probability(state: DraftState, candidate: Hero) -> float:
    # TODO: load model, featurise state + candidate, return float in [0,1]
    return -1.0  # -1.0 = disabled, falls back to rule-based scoring
```

When this returns ≥ 0, it blends with the rule-based win_rate component:
`effective_wr = ml_prob × 0.6 + base_wr × 0.4`

---

## Performance

- `mss` screen capture: ~15–20 FPS, ~1 ms/frame overhead
- Frame change detection: skips CV if frame hash unchanged (~0.3 ms)
- Template matching (132 templates): ~8–18 ms per slot
- Full pipeline (10 slots with templates): ~100–200 ms
- Recommendation engine (132 candidates): ~2 ms
- YOLOv8n inference per slot: ~5 ms GPU / ~25 ms CPU
