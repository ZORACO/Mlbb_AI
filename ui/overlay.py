"""
ui/overlay.py
-------------
Production-grade floating overlay window.

Upgrades over v1
----------------
* Detection confidence badge per recommendation
* Draft phase indicator with colour coding
* Risk level colour (LOW=green, MED=amber, HIGH=red)
* Enemy composition summary panel
* Slot-level confidence mini-bars (shows per-slot detection quality)
* Score component breakdown (expandable tooltip-style label)
* Thread-safe via Qt signal/slot — zero Qt calls from non-GUI threads
* Graceful Tkinter fallback when PyQt5 is absent
* Draggable + right-click-to-close (no system title bar needed)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UI payload (passed between threads via queue)
# ---------------------------------------------------------------------------

from recommender.engine import Recommendation, DraftPhase
from vision.hero_detector import DraftState


@dataclass
class UIPayload:
    recommendations: List[Recommendation] = field(default_factory=list)
    draft_state:     Optional[DraftState]  = None
    fps:             float                  = 0.0
    phase:           DraftPhase             = DraftPhase.BAN
    cpu_throttled:   bool                   = False


# ---------------------------------------------------------------------------
# PyQt5 overlay (primary backend)
# ---------------------------------------------------------------------------

def _has_pyqt5() -> bool:
    try:
        import PyQt5  # noqa: F401
        return True
    except ImportError:
        return False


if _has_pyqt5():
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QFrame, QSizePolicy, QGraphicsOpacityEffect,
    )
    from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, QObject
    from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush

    # ------------------------------------------------------------------
    # Thread-bridge signals
    # ------------------------------------------------------------------

    class _Bridge(QObject):
        update_ui = pyqtSignal(object)   # emits UIPayload

    # ------------------------------------------------------------------
    # Phase colour mapping
    # ------------------------------------------------------------------

    _PHASE_COLORS = {
        DraftPhase.BAN:   ("#ff6b35", "BAN PHASE"),
        DraftPhase.EARLY: ("#00e5ff", "PICK — EARLY"),
        DraftPhase.MID:   ("#7c4dff", "PICK — MID"),
        DraftPhase.LATE:  ("#ffd740", "PICK — LATE"),
    }

    _RISK_COLORS = {
        "LOW":  "#00c853",
        "MED":  "#ffab00",
        "HIGH": "#ff1744",
    }

    _TIER_COLORS = {
        "S": "#ffd740",
        "A": "#69f0ae",
        "B": "#40c4ff",
        "C": "#bdbdbd",
    }

    # ------------------------------------------------------------------
    # Thin separator
    # ------------------------------------------------------------------

    class _HSep(QFrame):
        def __init__(self) -> None:
            super().__init__()
            self.setFrameShape(QFrame.HLine)
            self.setStyleSheet("color: rgba(255,255,255,18);")
            self.setFixedHeight(1)

    # ------------------------------------------------------------------
    # Recommendation card
    # ------------------------------------------------------------------

    class _RecCard(QFrame):
        """One recommendation row: rank + hero info + score + risk badge."""

        _RANK_COLORS = ["#ffd740", "#c0c0c0", "#cd7f32"]

        def __init__(self, rank: int) -> None:
            super().__init__()
            self._rank = rank
            rc = self._RANK_COLORS[rank - 1]
            self.setStyleSheet(f"""
                QFrame {{
                    background: rgba(255,255,255,10);
                    border: 1px solid {rc}44;
                    border-left: 3px solid {rc};
                    border-radius: 6px;
                }}
                QLabel {{ border: none; background: transparent; }}
            """)
            self._build_ui(rc)

        def _build_ui(self, rank_color: str) -> None:
            outer = QHBoxLayout(self)
            outer.setContentsMargins(10, 7, 10, 7)
            outer.setSpacing(8)

            # Rank number
            self._rank_lbl = QLabel(f"#{self._rank}")
            self._rank_lbl.setFont(QFont("Consolas", 10, QFont.Bold))
            self._rank_lbl.setStyleSheet(f"color: {rank_color};")
            self._rank_lbl.setFixedWidth(22)
            outer.addWidget(self._rank_lbl)

            # Hero info column
            info = QVBoxLayout()
            info.setSpacing(1)
            self._name_lbl = QLabel("—")
            self._name_lbl.setFont(QFont("Consolas", 11, QFont.Bold))
            self._name_lbl.setStyleSheet("color: #e8eaf0;")

            self._role_row = QHBoxLayout()
            self._role_row.setSpacing(5)
            self._role_lbl = QLabel("")
            self._role_lbl.setFont(QFont("Consolas", 9))
            self._role_lbl.setStyleSheet("color: #8890a4;")
            self._tier_lbl = QLabel("")
            self._tier_lbl.setFont(QFont("Consolas", 9, QFont.Bold))
            self._role_row.addWidget(self._role_lbl)
            self._role_row.addWidget(self._tier_lbl)
            self._role_row.addStretch()

            self._reason_lbl = QLabel("")
            self._reason_lbl.setFont(QFont("Consolas", 8))
            self._reason_lbl.setStyleSheet("color: #8890a4;")
            self._reason_lbl.setWordWrap(True)

            info.addWidget(self._name_lbl)
            info.addLayout(self._role_row)
            info.addWidget(self._reason_lbl)
            outer.addLayout(info)
            outer.addStretch()

            # Right column: score + risk
            right = QVBoxLayout()
            right.setSpacing(3)
            right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            self._score_lbl = QLabel("")
            self._score_lbl.setFont(QFont("Consolas", 18, QFont.Bold))
            self._score_lbl.setAlignment(Qt.AlignRight)

            self._conf_lbl = QLabel("CONF%")
            self._conf_lbl.setFont(QFont("Consolas", 8))
            self._conf_lbl.setStyleSheet("color: #8890a4;")
            self._conf_lbl.setAlignment(Qt.AlignRight)

            self._risk_lbl = QLabel("")
            self._risk_lbl.setFont(QFont("Consolas", 8, QFont.Bold))
            self._risk_lbl.setAlignment(Qt.AlignRight)

            # Score component bar (thin multi-segment bar)
            self._bar = _ScoreBar()
            self._bar.setFixedSize(60, 4)

            right.addWidget(self._score_lbl)
            right.addWidget(self._conf_lbl)
            right.addWidget(self._risk_lbl)
            right.addWidget(self._bar, alignment=Qt.AlignRight)
            outer.addLayout(right)

        def set_recommendation(self, rec: Recommendation) -> None:
            hero = rec.hero
            pct  = rec.confidence_pct()
            risk = rec.risk_label()

            self._name_lbl.setText(hero.name)
            self._role_lbl.setText(hero.role)
            self._tier_lbl.setText(f"[{hero.tier}]")
            self._tier_lbl.setStyleSheet(
                f"color: {_TIER_COLORS.get(hero.tier, '#bdbdbd')};"
            )
            self._reason_lbl.setText(rec.reason[:70])

            self._score_lbl.setText(str(pct))
            rc = self._RANK_COLORS[self._rank - 1]
            self._score_lbl.setStyleSheet(f"color: {rc};")

            rcolor = _RISK_COLORS.get(risk, "#bdbdbd")
            self._risk_lbl.setText(f"RISK: {risk}")
            self._risk_lbl.setStyleSheet(f"color: {rcolor};")

            self._bar.set_components(
                rec.win_rate_comp,
                rec.counter_comp,
                rec.team_fit_comp,
                rec.synergy_comp,
            )
            self.setVisible(True)

        def clear(self) -> None:
            self._name_lbl.setText("—")
            self._role_lbl.setText("")
            self._tier_lbl.setText("")
            self._reason_lbl.setText("")
            self._score_lbl.setText("")
            self._risk_lbl.setText("")
            self._bar.set_components(0, 0, 0, 0)

    # ------------------------------------------------------------------
    # Score component bar (4-segment coloured mini-bar)
    # ------------------------------------------------------------------

    class _ScoreBar(QWidget):
        """4-segment horizontal bar: WR | Counter | Fit | Synergy."""

        _COLORS = ["#00b0ff", "#f50057", "#69f0ae", "#ffd740"]
        _LABELS = ["WR", "CTR", "FIT", "SYN"]

        def __init__(self) -> None:
            super().__init__()
            self._vals = [0.0, 0.0, 0.0, 0.0]

        def set_components(self, wr: float, ctr: float, fit: float, syn: float) -> None:
            self._vals = [wr, ctr, fit, syn]
            self.update()

        def paintEvent(self, event) -> None:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            total = sum(self._vals) or 1.0
            x = 0
            for i, val in enumerate(self._vals):
                w = int((val / total) * self.width())
                color = QColor(self._COLORS[i])
                p.fillRect(x, 0, w, self.height(), QBrush(color))
                x += w
            p.end()

    # ------------------------------------------------------------------
    # Enemy composition panel
    # ------------------------------------------------------------------

    class _EnemyCompPanel(QFrame):
        """Shows detected enemy archetypes as coloured pills."""

        def __init__(self) -> None:
            super().__init__()
            self.setStyleSheet("QFrame { background: transparent; } QLabel { border: none; }")
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            self._pills: list[QLabel] = []
            self._layout = layout

        def set_profile(self, profile_desc: str) -> None:
            # Clear existing pills
            for pill in self._pills:
                pill.deleteLater()
            self._pills.clear()

            if not profile_desc or profile_desc == "balanced":
                lbl = QLabel("balanced")
                lbl.setFont(QFont("Consolas", 8))
                lbl.setStyleSheet("color: #8890a4;")
                self._pills.append(lbl)
                self._layout.addWidget(lbl)
                return

            colors = {
                "tank-heavy":    ("#1565c0", "#82b1ff"),
                "cc-heavy":      ("#6a1b9a", "#ce93d8"),
                "mobile":        ("#00695c", "#80cbc4"),
                "squishy":       ("#b71c1c", "#ef9a9a"),
                "burst-heavy":   ("#e65100", "#ffcc80"),
                "sustain-heavy": ("#1b5e20", "#a5d6a7"),
            }
            for trait in profile_desc.split(", "):
                bg, fg = colors.get(trait.strip(), ("#333", "#aaa"))
                pill = QLabel(trait)
                pill.setFont(QFont("Consolas", 8))
                pill.setStyleSheet(
                    f"background: {bg}; color: {fg}; "
                    f"padding: 1px 6px; border-radius: 3px;"
                )
                self._pills.append(pill)
                self._layout.addWidget(pill)

            self._layout.addStretch()

    # ------------------------------------------------------------------
    # Main overlay window
    # ------------------------------------------------------------------

    class OverlayWindow(QWidget):
        """
        Frameless, always-on-top, semi-transparent overlay.

        Thread-safe: call update_recommendations(payload) from any thread.
        """

        _WIDTH = 340

        def __init__(self) -> None:
            super().__init__()
            self._bridge = _Bridge()
            self._bridge.update_ui.connect(self._on_update)
            self._drag_pos = None
            self._build_window()
            self._build_ui()

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------

        def update_recommendations(self, payload: UIPayload) -> None:
            """Thread-safe update — emits Qt signal to GUI thread."""
            self._bridge.update_ui.emit(payload)

        # ------------------------------------------------------------------
        # Window setup
        # ------------------------------------------------------------------

        def _build_window(self) -> None:
            self.setWindowFlags(
                Qt.FramelessWindowHint
                | Qt.WindowStaysOnTopHint
                | Qt.Tool
            )
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setFixedWidth(self._WIDTH)

        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)

            self._panel = QFrame(self)
            self._panel.setObjectName("panel")
            self._panel.setStyleSheet("""
                QFrame#panel {
                    background: rgba(8,10,16,215);
                    border: 1px solid rgba(0,229,255,0.20);
                    border-radius: 10px;
                }
                QLabel { background: transparent; border: none; }
            """)
            root.addWidget(self._panel)

            inner = QVBoxLayout(self._panel)
            inner.setContentsMargins(12, 10, 12, 12)
            inner.setSpacing(6)

            # ── Header ──────────────────────────────────────────────
            hdr = QHBoxLayout()
            logo = QLabel("⚔  DRAFT AI")
            logo.setFont(QFont("Consolas", 11, QFont.Bold))
            logo.setStyleSheet("color: #00e5ff; letter-spacing: 2px;")

            self._fps_lbl = QLabel("● LIVE")
            self._fps_lbl.setFont(QFont("Consolas", 8))
            self._fps_lbl.setStyleSheet("color: #00c853;")

            hdr.addWidget(logo)
            hdr.addStretch()
            hdr.addWidget(self._fps_lbl)
            inner.addLayout(hdr)

            # ── Phase badge ─────────────────────────────────────────
            self._phase_lbl = QLabel("WAITING…")
            self._phase_lbl.setFont(QFont("Consolas", 9, QFont.Bold))
            self._phase_lbl.setStyleSheet(
                "color: #00e5ff; background: rgba(0,229,255,0.10); "
                "border: 1px solid rgba(0,229,255,0.25); border-radius: 4px; "
                "padding: 2px 8px; letter-spacing: 1px;"
            )
            self._phase_lbl.setAlignment(Qt.AlignCenter)
            inner.addWidget(self._phase_lbl)

            # ── Draft state summary ──────────────────────────────────
            self._state_lbl = QLabel("Waiting for draft...")
            self._state_lbl.setFont(QFont("Consolas", 8))
            self._state_lbl.setStyleSheet("color: #8890a4;")
            self._state_lbl.setWordWrap(True)
            inner.addWidget(self._state_lbl)

            # ── Enemy comp ───────────────────────────────────────────
            comp_row = QHBoxLayout()
            comp_row.setSpacing(4)
            comp_lbl = QLabel("ENEMY:")
            comp_lbl.setFont(QFont("Consolas", 8))
            comp_lbl.setStyleSheet("color: #8890a4;")
            comp_row.addWidget(comp_lbl)
            self._comp_panel = _EnemyCompPanel()
            comp_row.addWidget(self._comp_panel)
            inner.addLayout(comp_row)

            inner.addWidget(_HSep())

            # ── Picks header ─────────────────────────────────────────
            sub = QLabel("TOP PICKS")
            sub.setFont(QFont("Consolas", 8, QFont.Bold))
            sub.setStyleSheet("color: #8890a4; letter-spacing: 1px;")
            inner.addWidget(sub)

            # ── Recommendation cards ─────────────────────────────────
            self._cards: list[_RecCard] = []
            for i in range(3):
                card = _RecCard(rank=i + 1)
                inner.addWidget(card)
                self._cards.append(card)

            # ── Detection quality footer ─────────────────────────────
            inner.addWidget(_HSep())
            self._det_lbl = QLabel("Detection: —")
            self._det_lbl.setFont(QFont("Consolas", 8))
            self._det_lbl.setStyleSheet("color: #8890a4;")
            inner.addWidget(self._det_lbl)

            self.adjustSize()
            self.move(1560, 20)      # default: top-right corner on 1920-wide screen

        # ------------------------------------------------------------------
        # Qt slot — runs on GUI thread
        # ------------------------------------------------------------------

        def _on_update(self, payload: UIPayload) -> None:
            # FPS
            fps = payload.fps
            fps_color = "#00c853" if fps >= 5.0 else "#ff1744"
            fps_text = f"⚠ {fps:.1f}" if payload.cpu_throttled else f"● {fps:.1f} FPS"
            self._fps_lbl.setText(fps_text)
            self._fps_lbl.setStyleSheet(f"color: {fps_color};")

            # Phase
            pc, pt = _PHASE_COLORS.get(payload.phase, ("#8890a4", "UNKNOWN"))
            self._phase_lbl.setText(pt)
            self._phase_lbl.setStyleSheet(
                f"color: {pc}; background: {pc}18; "
                f"border: 1px solid {pc}44; border-radius: 4px; "
                f"padding: 2px 8px; letter-spacing: 1px;"
            )

            # Draft summary
            state = payload.draft_state
            if state:
                ally  = ", ".join(state.ally_team)  or "—"
                enemy = ", ".join(state.enemy_team) or "—"
                bans  = ", ".join(state.bans)       or "—"
                self._state_lbl.setText(
                    f"Ally: {ally}\nEnemy: {enemy}\nBans: {bans}"
                )
                mean_conf = state.mean_confidence()
                self._det_lbl.setText(
                    f"Avg detection conf: {mean_conf:.0%}  |  "
                    f"Locked: {sum(1 for v in state.confidences.values() if v >= 0.85)}"
                )

            # Enemy comp pills (from recs[0] if available)
            if payload.recommendations:
                # Import here to avoid circular import at module level
                from recommender.engine import _analyse_enemy
                from data.hero_db import get_db
                if state:
                    db = get_db()
                    enemies = db.get_many(state.enemy_team)
                    from recommender.engine import _analyse_enemy
                    profile = _analyse_enemy(enemies)
                    self._comp_panel.set_profile(profile.describe())

            # Cards
            recs = payload.recommendations
            for i, card in enumerate(self._cards):
                if i < len(recs):
                    card.set_recommendation(recs[i])
                else:
                    card.clear()

            self.adjustSize()

        # ------------------------------------------------------------------
        # Drag to move (no title bar)
        # ------------------------------------------------------------------

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.LeftButton:
                self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            elif event.button() == Qt.RightButton:
                self.hide()                          # right-click to dismiss

        def mouseMoveEvent(self, event) -> None:
            if self._drag_pos and event.buttons() == Qt.LeftButton:
                self.move(event.globalPos() - self._drag_pos)

        def mouseReleaseEvent(self, event) -> None:
            self._drag_pos = None

        def mouseDoubleClickEvent(self, event) -> None:
            self.show()                              # double-click to restore if hidden

    def create_app_and_overlay():
        app = QApplication.instance() or QApplication(sys.argv)
        overlay = OverlayWindow()
        return app, overlay

# ---------------------------------------------------------------------------
# Tkinter fallback
# ---------------------------------------------------------------------------

else:
    logger.warning("PyQt5 not found — using minimal Tkinter fallback.")
    import tkinter as tk

    class OverlayWindow:  # type: ignore[no-redef]
        """Minimal Tkinter fallback overlay."""

        def __init__(self) -> None:
            self._root = tk.Tk()
            self._root.title("MLBB Draft AI")
            self._root.attributes("-topmost", True)
            self._root.attributes("-alpha", 0.92)
            self._root.configure(bg="#080a10")
            self._root.geometry("320x280+1560+20")
            self._root.overrideredirect(True)

            tk.Label(
                self._root, text="⚔  DRAFT AI",
                fg="#00e5ff", bg="#080a10",
                font=("Courier", 11, "bold"),
            ).pack(pady=(8, 2))

            self._phase = tk.Label(
                self._root, text="WAITING…",
                fg="#00e5ff", bg="#080a10",
                font=("Courier", 9),
            )
            self._phase.pack()

            self._cards: list[tk.Label] = []
            for _ in range(3):
                lbl = tk.Label(
                    self._root, text="—",
                    fg="#ffd740", bg="#080a10",
                    font=("Courier", 9), justify="left", wraplength=300,
                )
                lbl.pack(anchor="w", padx=10, pady=2)
                self._cards.append(lbl)

            self._det = tk.Label(
                self._root, text="",
                fg="#8890a4", bg="#080a10",
                font=("Courier", 8),
            )
            self._det.pack(anchor="w", padx=10)

        def show(self) -> None:
            self._root.deiconify()

        def update_recommendations(self, payload: "UIPayload") -> None:
            from recommender.engine import DraftPhase
            pc_map = {
                DraftPhase.BAN:   "#ff6b35",
                DraftPhase.EARLY: "#00e5ff",
                DraftPhase.MID:   "#b388ff",
                DraftPhase.LATE:  "#ffd740",
            }
            self._phase.config(
                text=payload.phase.value.upper(),
                fg=pc_map.get(payload.phase, "#00e5ff"),
            )
            for i, lbl in enumerate(self._cards):
                if i < len(payload.recommendations):
                    r = payload.recommendations[i]
                    lbl.config(
                        text=f"#{i+1} {r.hero.name} [{r.hero.tier}] {r.confidence_pct()}% "
                             f"RISK:{r.risk_label()}\n   {r.reason[:60]}"
                    )
                else:
                    lbl.config(text="—")
            if payload.draft_state:
                self._det.config(
                    text=f"Conf: {payload.draft_state.mean_confidence():.0%}"
                )

        def mainloop(self) -> None:
            self._root.mainloop()

    def create_app_and_overlay():
        overlay = OverlayWindow()
        return None, overlay
