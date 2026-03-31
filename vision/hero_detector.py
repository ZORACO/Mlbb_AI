"""
vision/hero_detector.py
------------------------
Production hero-detection pipeline.

Architecture
------------

                  ┌─────────────────────────────────┐
  Frame ──────►   │  SlotCropper                    │
                  │  Crops 20 fixed regions          │
                  └──────────┬──────────────────────┘
                             │ 20 sub-images (crops)
                             ▼
                  ┌─────────────────────────────────┐
                  │  HybridDetector (per-slot)       │
                  │  1. TemplateMatcher (fast)        │
                  │  2. YOLODetector   (fallback)     │
                  │  3. None           (low conf)     │
                  └──────────┬──────────────────────┘
                             │ SlotResult per slot
                             ▼
                  ┌─────────────────────────────────┐
                  │  TemporalFilter                  │
                  │  Sliding-window majority vote    │
                  │  Lock-in mechanism               │
                  └──────────┬──────────────────────┘
                             │ Stable DraftState
                             ▼
                  ┌─────────────────────────────────┐
                  │  DraftState (structured output)  │
                  └─────────────────────────────────┘

Key upgrades over v1
--------------------
* Slot-based detection — each pick/ban slot cropped independently
* Hybrid pipeline per slot: template matching → YOLO fallback
* Confidence thresholding (default 0.60) — ignores weak detections
* Temporal stabilisation: sliding window + lock-in for stable output
* Changed-region optimisation: skips slots whose pixel content is unchanged
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.slot_config import SlotConfig, SlotRect, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection result for a single slot
# ---------------------------------------------------------------------------

@dataclass
class SlotResult:
    """Detection result for one slot crop."""
    hero: Optional[str]      # None → nothing detected
    confidence: float        # 0.0 – 1.0
    source: str              # "template" | "yolo" | "mock" | "none"


# ---------------------------------------------------------------------------
# Full draft state (output)
# ---------------------------------------------------------------------------

@dataclass
class DraftState:
    """Structured, stable representation of the current draft."""
    ally_team:  List[str] = field(default_factory=list)
    enemy_team: List[str] = field(default_factory=list)
    bans:       List[str] = field(default_factory=list)
    confidences: Dict[str, float] = field(default_factory=dict)
    detected_at: float = field(default_factory=time.time)

    def all_taken(self) -> List[str]:
        return self.ally_team + self.enemy_team + self.bans

    def to_dict(self) -> dict:
        return {
            "ally_team":  self.ally_team,
            "enemy_team": self.enemy_team,
            "bans":       self.bans,
        }

    def mean_confidence(self) -> float:
        vals = list(self.confidences.values())
        return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Slot cropper
# ---------------------------------------------------------------------------

class SlotCropper:
    """Crop sub-images from a frame according to slot configuration."""

    def __init__(self, config: SlotConfig) -> None:
        self._cfg = config

    def crop(
        self, frame: np.ndarray, rect: SlotRect
    ) -> Optional[np.ndarray]:
        """
        Safely crop *rect* from *frame*.
        Returns None if the rect falls outside the frame boundaries.
        """
        x, y, w, h = rect
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def crop_all(
        self, frame: np.ndarray
    ) -> Dict[str, List[Optional[np.ndarray]]]:
        """
        Crop every slot and return a dict:
            {"ally": [img|None, ...], "enemy": [...], "bans": [...]}
        """
        return {
            group: [self.crop(frame, rect) for rect in rects]
            for group, rects in self._cfg.all_slots().items()
        }


# ---------------------------------------------------------------------------
# Template matching detector
# ---------------------------------------------------------------------------

class TemplateMatcher:
    """
    Matches per-slot crops against pre-built hero portrait templates.

    Templates live in:  vision/templates/<HeroName>.png
    Each template should be cropped to exactly the portrait area (~72×72px).
    """

    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
    _DEFAULT_THRESHOLD = 0.80

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold
        self._templates: Dict[str, np.ndarray] = {}
        self._gray_templates: Dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        import cv2

        if not os.path.isdir(self.TEMPLATE_DIR):
            logger.info(
                "Template dir not found (%s). Create it and add hero PNG files.",
                self.TEMPLATE_DIR,
            )
            return

        for fname in os.listdir(self.TEMPLATE_DIR):
            if not fname.lower().endswith(".png"):
                continue
            hero_name = os.path.splitext(fname)[0]
            path = os.path.join(self.TEMPLATE_DIR, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                self._templates[hero_name] = img
                self._gray_templates[hero_name] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        logger.info("TemplateMatcher: loaded %d templates.", len(self._templates))

    def detect(self, crop: np.ndarray) -> SlotResult:
        """
        Match *crop* against all templates using NCC.

        Returns the best match above threshold, or a 'none' result.
        """
        import cv2

        if not self._gray_templates or crop is None:
            return SlotResult(hero=None, confidence=0.0, source="none")

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        best_hero: Optional[str] = None
        best_score: float = 0.0

        for hero_name, tmpl_gray in self._gray_templates.items():
            # Resize template to match crop if sizes differ
            if tmpl_gray.shape != gray.shape:
                tmpl_resized = cv2.resize(
                    tmpl_gray,
                    (gray.shape[1], gray.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                tmpl_resized = tmpl_gray

            try:
                result = cv2.matchTemplate(gray, tmpl_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
            except cv2.error:
                continue

            if max_val > best_score:
                best_score = max_val
                best_hero = hero_name

        if best_score >= self._threshold and best_hero:
            return SlotResult(hero=best_hero, confidence=best_score, source="template")
        return SlotResult(hero=None, confidence=best_score, source="none")

    @property
    def has_templates(self) -> bool:
        return len(self._templates) > 0


# ---------------------------------------------------------------------------
# YOLO detector (optional)
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    Per-slot YOLO inference using a fine-tuned YOLOv8 model.

    The model must be trained to output hero names as class labels.
    Weights path: vision/weights/mlbb_draft.pt
    """

    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "mlbb_draft.pt")
    _DEFAULT_THRESHOLD = 0.50

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.isfile(self.WEIGHTS_PATH):
            logger.info(
                "YOLODetector: weights not found at %s. YOLO disabled.",
                self.WEIGHTS_PATH,
            )
            return
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(self.WEIGHTS_PATH)
            logger.info("YOLODetector: model loaded from %s.", self.WEIGHTS_PATH)
        except ImportError:
            logger.info("YOLODetector: ultralytics not installed. YOLO disabled.")

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def detect(self, crop: np.ndarray) -> SlotResult:
        """Run inference on a single slot crop."""
        if self._model is None or crop is None:
            return SlotResult(hero=None, confidence=0.0, source="none")

        results = self._model(crop, conf=self._threshold, verbose=False)
        best_hero: Optional[str] = None
        best_conf: float = 0.0

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_hero = self._model.names[int(box.cls[0])]

        if best_hero and best_conf >= self._threshold:
            return SlotResult(hero=best_hero, confidence=best_conf, source="yolo")
        return SlotResult(hero=None, confidence=best_conf, source="none")


# ---------------------------------------------------------------------------
# Hybrid per-slot detector
# ---------------------------------------------------------------------------

class HybridSlotDetector:
    """
    Combines template matching and YOLO into a single per-slot detector.

    Decision logic per slot:
        1. Run template matcher
        2. If conf ≥ CONF_HIGH → accept immediately
        3. Else if YOLO available → run YOLO
           a. If YOLO conf ≥ CONF_THRESHOLD → use YOLO result
           b. Else use whichever had higher confidence (if above min)
        4. If neither reaches CONF_THRESHOLD → return empty result
    """

    CONF_HIGH      = 0.85   # Threshold to skip YOLO fallback (template already good)
    CONF_THRESHOLD = 0.60   # Minimum acceptable confidence for any detection
    CONF_MIN_KEEP  = 0.40   # If above this but below CONF_THRESHOLD, keep as low-conf

    def __init__(
        self,
        template_matcher: TemplateMatcher,
        yolo_detector: Optional[YOLODetector] = None,
    ) -> None:
        self._tm = template_matcher
        self._yolo = yolo_detector

    def detect(self, crop: Optional[np.ndarray]) -> SlotResult:
        """Detect hero in a single cropped slot image."""
        if crop is None:
            return SlotResult(hero=None, confidence=0.0, source="none")

        # Step 1: template matching
        tm_result = self._tm.detect(crop)

        if tm_result.confidence >= self.CONF_HIGH:
            return tm_result                        # high-confidence template hit

        # Step 2: YOLO fallback
        if self._yolo and self._yolo.is_available:
            yolo_result = self._yolo.detect(crop)
            # Pick whichever detector is more confident
            best = max(tm_result, yolo_result, key=lambda r: r.confidence)
        else:
            best = tm_result

        # Step 3: threshold gate
        if best.confidence >= self.CONF_THRESHOLD:
            return best
        if best.confidence >= self.CONF_MIN_KEEP:
            # Keep with low-confidence flag (caller may choose to ignore)
            return SlotResult(hero=best.hero, confidence=best.confidence, source=best.source)

        return SlotResult(hero=None, confidence=0.0, source="none")


# ---------------------------------------------------------------------------
# Temporal stability filter
# ---------------------------------------------------------------------------

class TemporalFilter:
    """
    Stabilises per-slot detections across multiple frames using:

    1. Sliding window majority vote  — prevents single-frame noise
    2. Lock-in mechanism             — confirmed heroes resist eviction

    Parameters
    ----------
    window_size : int
        Number of past frames used for majority voting (default 7).
    lock_threshold : float
        Confidence above which a result is "locked in" immediately.
    unlock_votes : int
        Number of contradictory votes required to unlock a slot.
    """

    def __init__(
        self,
        n_ally: int = 5,
        n_enemy: int = 5,
        n_bans: int = 10,
        window_size: int = 7,
        lock_threshold: float = 0.85,
        unlock_votes: int = 4,
    ) -> None:
        total = n_ally + n_enemy + n_bans
        self._window_size = window_size
        self._lock_threshold = lock_threshold
        self._unlock_votes = unlock_votes

        # Per-slot circular buffers: stores (hero|None, confidence) tuples
        self._windows: List[deque] = [
            deque(maxlen=window_size) for _ in range(total)
        ]
        # Locked hero for each slot (None = not locked)
        self._locked: List[Optional[str]] = [None] * total

        self._n_ally = n_ally
        self._n_enemy = n_enemy
        self._n_bans = n_bans

    def update(
        self,
        ally_results:  List[SlotResult],
        enemy_results: List[SlotResult],
        ban_results:   List[SlotResult],
    ) -> DraftState:
        """
        Ingest new per-slot results and return a temporally-stable DraftState.
        """
        all_results = ally_results + enemy_results + ban_results
        stable: List[Optional[str]] = []
        confs: Dict[str, float] = {}

        for idx, result in enumerate(all_results):
            hero, conf = result.hero, result.confidence

            # Push to window
            self._windows[idx].append((hero, conf))

            # Check unlock FIRST (before potential re-lock)
            # N contradictory votes against the CURRENT locked hero evict it
            if self._locked[idx]:
                contradiction = sum(
                    1 for h, _ in self._windows[idx]
                    if h is not None and h != self._locked[idx]
                )
                if contradiction >= self._unlock_votes:
                    logger.debug(
                        "Slot %d: unlocking %s after %d contradictions",
                        idx, self._locked[idx], contradiction,
                    )
                    self._locked[idx] = None

            # Then check lock-in: high-confidence new detection → lock
            # Only lock if slot is currently unlocked (prevents re-lock thrash)
            if hero and conf >= self._lock_threshold and self._locked[idx] is None:
                self._locked[idx] = hero

            # If locked, use locked value
            if self._locked[idx]:
                stable_hero = self._locked[idx]
                stable_conf = max(
                    (c for h, c in self._windows[idx] if h == stable_hero),
                    default=conf,
                )
            else:
                # Majority vote from window
                candidates = [h for h, _ in self._windows[idx] if h is not None]
                if candidates:
                    winner, votes = Counter(candidates).most_common(1)[0]
                    vote_ratio = votes / len(self._windows[idx])
                    # Only accept if majority of window agrees
                    if vote_ratio >= 0.5:
                        stable_hero = winner
                        stable_conf = max(
                            (c for h, c in self._windows[idx] if h == winner),
                            default=conf,
                        )
                    else:
                        stable_hero = None
                        stable_conf = 0.0
                else:
                    stable_hero = None
                    stable_conf = 0.0

            stable.append(stable_hero)
            if stable_hero:
                confs[stable_hero] = round(stable_conf, 3)

        # Split back into groups
        na, ne = self._n_ally, self._n_enemy
        ally_heroes  = [h for h in stable[:na] if h]
        enemy_heroes = [h for h in stable[na:na+ne] if h]
        ban_heroes   = [h for h in stable[na+ne:] if h]

        return DraftState(
            ally_team=ally_heroes,
            enemy_team=enemy_heroes,
            bans=ban_heroes,
            confidences=confs,
        )

    def reset(self) -> None:
        """Clear all windows and locks (call on draft reset)."""
        for dq in self._windows:
            dq.clear()
        self._locked = [None] * len(self._locked)


# ---------------------------------------------------------------------------
# Mock detector (development / demo mode)
# ---------------------------------------------------------------------------

class MockDetector:
    """
    Produces a realistic advancing draft state without a live game.
    Used in --mode mock for UI development and testing.
    """

    _STATES = [
        DraftState(ally_team=[], enemy_team=[], bans=["Ling"]),
        DraftState(ally_team=[], enemy_team=[], bans=["Ling", "Fanny"]),
        DraftState(ally_team=["Chou"], enemy_team=["Gusion"], bans=["Ling", "Fanny"]),
        DraftState(ally_team=["Chou"], enemy_team=["Gusion", "Khufra"], bans=["Ling", "Fanny", "Atlas"]),
        DraftState(ally_team=["Chou", "Diggie"], enemy_team=["Gusion", "Khufra"], bans=["Ling", "Fanny", "Atlas"]),
        DraftState(ally_team=["Chou", "Diggie", "Granger"], enemy_team=["Gusion", "Khufra", "Layla"], bans=["Ling", "Fanny", "Atlas"]),
        DraftState(ally_team=["Chou", "Diggie", "Granger", "Karrie"], enemy_team=["Gusion", "Khufra", "Layla", "Esmeralda"], bans=["Ling", "Fanny", "Atlas"]),
        DraftState(ally_team=["Chou", "Diggie", "Granger", "Karrie", "Yve"], enemy_team=["Gusion", "Khufra", "Layla", "Esmeralda", "Lancelot"], bans=["Ling", "Fanny", "Atlas", "Phoveus"]),
    ]
    _TICK = 4.0

    def __init__(self) -> None:
        self._idx = 0
        self._last = time.time()

    def detect(self, frame: Optional[np.ndarray]) -> DraftState:  # type: ignore[override]
        now = time.time()
        if now - self._last >= self._TICK:
            self._idx = min(self._idx + 1, len(self._STATES) - 1)
            self._last = now
        s = self._STATES[self._idx]
        return DraftState(
            ally_team=list(s.ally_team),
            enemy_team=list(s.enemy_team),
            bans=list(s.bans),
            confidences={h: 0.95 for h in s.all_taken()},
            detected_at=now,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class DraftDetector:
    """
    Full detection pipeline: cropping → hybrid detection → temporal filter.

    This is the class callers should instantiate.  Construct once, call
    detect(frame) on each new frame.
    """

    def __init__(
        self,
        config: SlotConfig = DEFAULT_CONFIG,
        conf_threshold: float = 0.60,
        window_size: int = 7,
    ) -> None:
        self._cropper = SlotCropper(config)
        tm = TemplateMatcher(threshold=conf_threshold)
        yolo = YOLODetector()
        self._detector = HybridSlotDetector(tm, yolo if yolo.is_available else None)
        self._filter = TemporalFilter(
            n_ally=len(config.ally),
            n_enemy=len(config.enemy),
            n_bans=len(config.bans),
            window_size=window_size,
        )
        self._config = config

        # Change detection: per-slot hash cache to skip unchanged slots
        self._slot_hashes: Dict[int, str] = {}

    def detect(self, frame: np.ndarray) -> DraftState:
        """
        Run the full detection pipeline on one frame.

        Returns a temporally-stable DraftState.
        """
        crops = self._cropper.crop_all(frame)

        ally_results  = self._detect_group(crops["ally"],  0)
        enemy_results = self._detect_group(crops["enemy"], len(crops["ally"]))
        ban_results   = self._detect_group(
            crops["bans"],
            len(crops["ally"]) + len(crops["enemy"]),
        )

        return self._filter.update(ally_results, enemy_results, ban_results)

    def _detect_group(
        self,
        slot_crops: List[Optional[np.ndarray]],
        offset: int,
    ) -> List[SlotResult]:
        """
        Detect heroes in a group of slot crops.
        Skips slots whose content hash has not changed since last frame.
        """
        import hashlib

        results: List[SlotResult] = []
        for i, crop in enumerate(slot_crops):
            slot_id = offset + i
            if crop is None:
                results.append(SlotResult(hero=None, confidence=0.0, source="none"))
                continue

            # Check if slot changed
            slot_hash = hashlib.md5(crop.tobytes()).hexdigest()
            if self._slot_hashes.get(slot_id) == slot_hash:
                # No change — re-use last stable result from filter window
                window = self._filter._windows[slot_id]
                if window:
                    last_hero, last_conf = window[-1]
                    results.append(SlotResult(
                        hero=last_hero, confidence=last_conf, source="cached"
                    ))
                    continue
            self._slot_hashes[slot_id] = slot_hash

            results.append(self._detector.detect(crop))
        return results

    def reset(self) -> None:
        """Reset temporal filter and hash cache (call at start of new draft)."""
        self._filter.reset()
        self._slot_hashes.clear()
        logger.info("DraftDetector: state reset.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_detector(
    mode: str = "auto",
    config: SlotConfig = DEFAULT_CONFIG,
    conf_threshold: float = 0.60,
    window_size: int = 7,
) -> "DraftDetector | MockDetector":
    """
    Build and return the appropriate detector for the given mode.

    mode options
    ------------
    "mock"     → MockDetector (demo/testing, no CV)
    "auto"     → DraftDetector with best available CV backend
    "template" → DraftDetector (template-only, YOLO disabled)
    "yolo"     → DraftDetector (assert YOLO available or raise)
    """
    if mode == "mock":
        logger.info("Using MockDetector (demo mode).")
        return MockDetector()

    detector = DraftDetector(
        config=config,
        conf_threshold=conf_threshold,
        window_size=window_size,
    )

    if mode == "yolo":
        if not detector._detector._yolo or not detector._detector._yolo.is_available:
            raise RuntimeError(
                "YOLO mode requested but model not found at "
                f"{YOLODetector.WEIGHTS_PATH}"
            )

    logger.info(
        "DraftDetector ready. Templates=%d, YOLO=%s",
        len(detector._detector._tm._templates),
        detector._detector._yolo.is_available if detector._detector._yolo else False,
    )
    return detector
