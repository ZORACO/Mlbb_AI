"""
main.py
--------
MLBB Draft AI v2 — production entry point.

Thread architecture
-------------------

  Main thread (Qt/Tk GUI)
       │
       │ polls UI_QUEUE every 200ms via QTimer
       │
  ┌────┴─────────────────────────────────────────┐
  │  FramePump thread                            │
  │  Reads capturer.latest_frame at target FPS   │
  │  Pushes to FRAME_QUEUE only if frame changed │
  └────┬─────────────────────────────────────────┘
       │
  ┌────▼─────────────────────────────────────────┐
  │  CVThread                                    │
  │  Pulls frames from FRAME_QUEUE               │
  │  Runs DraftDetector (slot-based)             │
  │  Runs RecommendationEngine                   │
  │  Pushes UIPayload to UI_QUEUE                │
  └──────────────────────────────────────────────┘

Key production features
-----------------------
* Bounded queues (maxsize=4) — drops stale data instead of growing memory
* Change-detection short-circuit — CV thread skips unchanged frames
* CPU overload detection — logs warning when capturer throttles
* Graceful shutdown sequence on SIGINT/window close
* CLI flags for all tunable parameters
* Structured logging with timestamps
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

from capture.screen_capture import ScreenCapturer, Region
from config.slot_config import get_config
from data.hero_db import get_db
from recommender.engine import RecommendationEngine, DraftPhase
from ui.overlay import UIPayload, create_app_and_overlay
from vision.hero_detector import DraftState, build_detector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("mlbb_ai.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Inter-thread queues
# ---------------------------------------------------------------------------

FRAME_QUEUE: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=4)
UI_QUEUE:    queue.Queue[Optional[UIPayload]]   = queue.Queue(maxsize=4)

_SHUTDOWN = threading.Event()

# ---------------------------------------------------------------------------
# CV + Recommendation thread
# ---------------------------------------------------------------------------

class CVThread(threading.Thread):
    """
    Consumes frames from FRAME_QUEUE.
    Runs slot-based detection + recommendation engine.
    Pushes UIPayload to UI_QUEUE.
    """

    def __init__(
        self,
        detector_mode: str,
        top_n: int,
        conf_threshold: float,
        window_size: int,
        capturer: ScreenCapturer,
        screen_w: int,
        screen_h: int,
    ) -> None:
        super().__init__(name="CVThread", daemon=True)
        self.detector_mode  = detector_mode
        self.top_n          = top_n
        self.conf_threshold = conf_threshold
        self.window_size    = window_size
        self.capturer       = capturer
        self.screen_w       = screen_w
        self.screen_h       = screen_h

        self._last_state: Optional[DraftState] = None
        self._frame_count: int = 0
        self._detect_count: int = 0

    def run(self) -> None:
        # Build slot config for detected resolution
        slot_cfg = get_config(self.screen_w, self.screen_h)
        detector = build_detector(
            mode=self.detector_mode,
            config=slot_cfg,
            conf_threshold=self.conf_threshold,
            window_size=self.window_size,
        )
        engine = RecommendationEngine(db=get_db())

        logger.info(
            "CVThread ready. Detector=%s, SlotCfg=%dx%d, Threshold=%.2f, Window=%d",
            type(detector).__name__, self.screen_w, self.screen_h,
            self.conf_threshold, self.window_size,
        )

        while not _SHUTDOWN.is_set():
            try:
                frame = FRAME_QUEUE.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None:
                break  # sentinel

            self._frame_count += 1

            # ── Detection ──────────────────────────────────────────
            try:
                state = detector.detect(frame)
            except Exception as exc:
                logger.error("Detection error (frame %d): %s", self._frame_count, exc, exc_info=True)
                continue

            # ── Skip if draft unchanged ─────────────────────────────
            if self._state_equal(state, self._last_state):
                continue
            self._last_state = state
            self._detect_count += 1

            # ── Recommendation ──────────────────────────────────────
            try:
                recs = engine.recommend(state, top_n=self.top_n)
            except Exception as exc:
                logger.error("Recommendation error: %s", exc, exc_info=True)
                recs = []

            phase = DraftPhase.from_state(state)
            self._log_recs(recs, state, phase)

            # ── Push to UI ──────────────────────────────────────────
            payload = UIPayload(
                recommendations=recs,
                draft_state=state,
                fps=self.capturer.fps_actual,
                phase=phase,
                cpu_throttled=self.capturer.cpu_throttle_active,
            )
            self._push_ui(payload)

        logger.info(
            "CVThread stopped. Frames processed: %d, State changes: %d",
            self._frame_count, self._detect_count,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _state_equal(a: DraftState, b: Optional[DraftState]) -> bool:
        if b is None:
            return False
        return (
            a.ally_team  == b.ally_team  and
            a.enemy_team == b.enemy_team and
            a.bans       == b.bans
        )

    @staticmethod
    def _push_ui(payload: UIPayload) -> None:
        try:
            UI_QUEUE.put_nowait(payload)
        except queue.Full:
            try:
                UI_QUEUE.get_nowait()
            except queue.Empty:
                pass
            UI_QUEUE.put_nowait(payload)

    @staticmethod
    def _log_recs(recs, state: DraftState, phase: DraftPhase) -> None:
        logger.info(
            "Draft [%s] — Ally:%s  Enemy:%s  Bans:%s",
            phase.value.upper(),
            state.ally_team or "[]",
            state.enemy_team or "[]",
            state.bans or "[]",
        )
        for i, r in enumerate(recs, 1):
            logger.info(
                "  #%d %-16s [%s] score=%.3f conf=%d%% risk=%-4s — %s",
                i, r.hero.name, r.hero.tier,
                r.score, r.confidence_pct(), r.risk_label(),
                r.reason,
            )


# ---------------------------------------------------------------------------
# Frame pump (separate thread so capture never blocks CV)
# ---------------------------------------------------------------------------

def _pump_frames(capturer: ScreenCapturer, fps: int) -> None:
    """Push changed frames from capturer into FRAME_QUEUE at target FPS."""
    interval = 1.0 / max(1, fps)
    while not _SHUTDOWN.is_set():
        t0 = time.perf_counter()
        if capturer.frame_changed:
            frame = capturer.latest_frame
            if frame is not None:
                try:
                    FRAME_QUEUE.put_nowait(frame)
                except queue.Full:
                    pass  # drop — CV thread busy
        elapsed = time.perf_counter() - t0
        sleep = interval - elapsed
        if sleep > 0:
            time.sleep(sleep)


# ---------------------------------------------------------------------------
# UI polling helper
# ---------------------------------------------------------------------------

def _poll_ui(overlay) -> None:
    """Drain UI_QUEUE and apply latest payload to overlay. Non-blocking."""
    latest: Optional[UIPayload] = None
    while True:
        try:
            latest = UI_QUEUE.get_nowait()
        except queue.Empty:
            break
    if latest is not None:
        overlay.update_recommendations(latest)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLBB Draft AI v2 — Real-time hero recommendation system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["auto", "mock", "template", "yolo"],
        default="mock",
        help="CV detection backend",
    )
    p.add_argument("--fps",    type=int,   default=10,   help="Target capture FPS")
    p.add_argument("--top",    type=int,   default=3,    help="Recommendations to show")
    p.add_argument("--conf",   type=float, default=0.60, help="Detection confidence threshold")
    p.add_argument("--window", type=int,   default=7,    help="Temporal filter window size")
    p.add_argument("--width",  type=int,   default=1280, help="Capture window width")
    p.add_argument("--height", type=int,   default=720,  help="Capture window height")
    p.add_argument(
        "--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        default=None, help="Override capture region",
    )
    p.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING"],
        default="INFO", help="Log verbosity",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 60)
    logger.info("MLBB Draft AI v2 starting")
    logger.info("Mode=%s FPS=%d Conf=%.2f Window=%d", args.mode, args.fps, args.conf, args.window)
    logger.info("=" * 60)

    # Pre-load database (validates JSON, starts auto-reload thread)
    db = get_db()
    logger.info("Database: %r", db)

    # ── Screen capturer ────────────────────────────────────────────────
    if args.region:
        region = Region(*args.region)
    else:
        region = Region(x=0, y=0, width=args.width, height=args.height)

    capturer = ScreenCapturer(region=region, target_fps=args.fps)
    capturer.start()

    # ── Frame pump thread ──────────────────────────────────────────────
    pump = threading.Thread(
        target=_pump_frames,
        args=(capturer, args.fps),
        name="FramePump",
        daemon=True,
    )
    pump.start()

    # ── CV + Rec thread ────────────────────────────────────────────────
    cv_thread = CVThread(
        detector_mode=args.mode,
        top_n=args.top,
        conf_threshold=args.conf,
        window_size=args.window,
        capturer=capturer,
        screen_w=args.width,
        screen_h=args.height,
    )
    cv_thread.start()

    # ── UI (must run on main thread) ───────────────────────────────────
    app, overlay = create_app_and_overlay()

    def _shutdown(*_) -> None:
        logger.info("Shutdown requested.")
        _SHUTDOWN.set()
        capturer.stop()
        FRAME_QUEUE.put_nowait(None)
        if app:
            app.quit()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info(
        "Overlay launched. Drag to reposition. Right-click to hide. Ctrl-C to quit."
    )

    try:
        if app is not None:
            # PyQt5 path ─────────────────────────────────────────────
            from PyQt5.QtCore import QTimer
            overlay.show()

            poll_timer = QTimer()
            poll_timer.timeout.connect(lambda: _poll_ui(overlay))
            poll_timer.start(150)   # 150ms polling — smooth without thrashing

            # FPS watchdog — logs a warning if capture falls below 3 FPS
            def _watchdog() -> None:
                if capturer.fps_actual < 3.0 and capturer._running:
                    logger.warning(
                        "Capture FPS low (%.1f). Check CPU load or reduce --fps.",
                        capturer.fps_actual,
                    )
            wd_timer = QTimer()
            wd_timer.timeout.connect(_watchdog)
            wd_timer.start(5000)

            exit_code = app.exec_()

        else:
            # Tkinter path ────────────────────────────────────────────
            overlay.show()

            def _tk_poll() -> None:
                if _SHUTDOWN.is_set():
                    overlay._root.destroy()
                    return
                _poll_ui(overlay)
                overlay._root.after(150, _tk_poll)

            overlay._root.after(150, _tk_poll)
            overlay.mainloop()
            exit_code = 0

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down.")
        exit_code = 0
    finally:
        _shutdown()
        cv_thread.join(timeout=3.0)
        logger.info("MLBB Draft AI v2 stopped cleanly.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
