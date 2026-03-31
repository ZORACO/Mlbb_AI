"""
config/slot_config.py
---------------------
Fixed screen-region definitions for every draft slot.

All coordinates are for a 1280×720 scrcpy window (the default).
Adjust SCREEN_W / SCREEN_H and call recalculate_slots() at runtime
if the window is resized.

Slot layout (MLBB standard draft screen):
  - Ally picks  : 5 slots on the LEFT side
  - Enemy picks : 5 slots on the RIGHT side
  - Bans        : 10 slots along the TOP centre (5 per side)

Each slot is a (x, y, w, h) tuple in pixels.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Slot descriptor
# ---------------------------------------------------------------------------

SlotRect = Tuple[int, int, int, int]   # (x, y, width, height)


@dataclass
class SlotConfig:
    """
    Complete slot layout for one screen resolution.

    Attributes
    ----------
    screen_w, screen_h : int
        Reference resolution this layout was calibrated for.
    ally : list of SlotRect
        5 pick slots for the local (ally) team, top to bottom.
    enemy : list of SlotRect
        5 pick slots for the enemy team, top to bottom.
    bans : list of SlotRect
        Up to 10 ban slots: first 5 = ally bans, last 5 = enemy bans.
    """
    screen_w: int
    screen_h: int
    ally:  List[SlotRect] = field(default_factory=list)
    enemy: List[SlotRect] = field(default_factory=list)
    bans:  List[SlotRect] = field(default_factory=list)

    def scale_to(self, target_w: int, target_h: int) -> "SlotConfig":
        """Return a new SlotConfig scaled to a different resolution."""
        sx = target_w / self.screen_w
        sy = target_h / self.screen_h

        def _scale(rects: List[SlotRect]) -> List[SlotRect]:
            return [
                (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
                for x, y, w, h in rects
            ]

        return SlotConfig(
            screen_w=target_w,
            screen_h=target_h,
            ally=_scale(self.ally),
            enemy=_scale(self.enemy),
            bans=_scale(self.bans),
        )

    def all_slots(self) -> Dict[str, List[SlotRect]]:
        """Return all slot groups as a dict."""
        return {"ally": self.ally, "enemy": self.enemy, "bans": self.bans}


# ---------------------------------------------------------------------------
# Calibrated layout — 1280 × 720 (scrcpy default, MLBB draft screen)
#
# HOW TO CALIBRATE:
#   1. Take a screenshot during the draft phase.
#   2. Open in any image editor.
#   3. Measure each hero-portrait bounding box.
#   4. Update the values below.
#
# Current values are approximate — good enough for template matching.
# Fine-tune with the calibration tool:  python -m tools.calibrate
# ---------------------------------------------------------------------------

# Ally pick slots (left side, portrait area only — excludes hero name text)
_ALLY_PICKS: List[SlotRect] = [
    (30,  160, 100, 100),   # slot 0  (first pick)
    (30,  270, 100, 100),   # slot 1
    (30,  380, 100, 100),   # slot 2
    (30,  490, 100, 100),   # slot 3
    (30,  600, 100, 100),   # slot 4  (fifth pick)
]

# Enemy pick slots (right side, mirrored)
_ENEMY_PICKS: List[SlotRect] = [
    (1150, 160, 100, 100),
    (1150, 270, 100, 100),
    (1150, 380, 100, 100),
    (1150, 490, 100, 100),
    (1150, 600, 100, 100),
]

# Ban slots — top row, 5 ally bans left of centre + 5 enemy bans right
_BANS: List[SlotRect] = [
    # Ally bans (left of centre)
    (230,  40, 72, 72),
    (310,  40, 72, 72),
    (390,  40, 72, 72),
    (470,  40, 72, 72),
    (550,  40, 72, 72),
    # Enemy bans (right of centre)
    (660,  40, 72, 72),
    (740,  40, 72, 72),
    (820,  40, 72, 72),
    (900,  40, 72, 72),
    (980,  40, 72, 72),
]

# Default config — 1280 × 720
DEFAULT_CONFIG = SlotConfig(
    screen_w=1280,
    screen_h=720,
    ally=_ALLY_PICKS,
    enemy=_ENEMY_PICKS,
    bans=_BANS,
)


# ---------------------------------------------------------------------------
# Preset library for common resolutions
# ---------------------------------------------------------------------------

PRESETS: Dict[str, SlotConfig] = {
    "1280x720":  DEFAULT_CONFIG,
    "1920x1080": DEFAULT_CONFIG.scale_to(1920, 1080),
    "2560x1440": DEFAULT_CONFIG.scale_to(2560, 1440),
    "800x600":   DEFAULT_CONFIG.scale_to(800,  600),
}


def get_config(width: int, height: int) -> SlotConfig:
    """
    Return slot config for the given resolution.
    Exact match first; falls back to scaling from the 1280×720 baseline.
    """
    key = f"{width}x{height}"
    if key in PRESETS:
        return PRESETS[key]
    return DEFAULT_CONFIG.scale_to(width, height)
