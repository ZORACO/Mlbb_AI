"""
data/hero_db.py
---------------
Production data layer for 132+ heroes.

Key upgrades over v1
--------------------
* Supports full hero roster from heroes.json (currently 132)
* Auto-reloads when the JSON file changes on disk (watchdog-lite polling)
* Exposes tier, patch version, pick/ban rates for scoring
* Thread-safe singleton with RLock
* Strict field validation on load — bad entries are logged and skipped
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hero data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Hero:
    """Immutable hero record.  All fields are present after load()."""
    id: int
    name: str
    role: str
    win_rate: float
    pick_rate: float
    ban_rate: float
    tier: str                           # S / A / B / C
    counters: tuple[str, ...]
    strong_against: tuple[str, ...]
    tags: tuple[str, ...]

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    def is_counter_to(self, enemy_name: str) -> bool:
        """Return True if this hero is strong against *enemy_name*."""
        return enemy_name in self.strong_against

    def is_countered_by(self, enemy_name: str) -> bool:
        """Return True if *enemy_name* is listed in this hero's counters."""
        return enemy_name in self.counters

    def primary_role(self) -> str:
        """First role segment before any '/' separator."""
        return self.role.split("/")[0].strip()

    def all_roles(self) -> List[str]:
        """All roles for multi-role heroes (e.g. 'Tank/Mage' → ['Tank','Mage'])."""
        return [r.strip() for r in self.role.split("/")]

    def tier_value(self) -> int:
        """Numeric tier: S=4, A=3, B=2, C=1 — for scoring."""
        return {"S": 4, "A": 3, "B": 2, "C": 1}.get(self.tier.upper(), 2)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class HeroDatabase:
    """
    Thread-safe, auto-reloading hero database.

    Usage
    -----
    db = HeroDatabase()              # loads heroes.json from same directory
    hero = db.get("Gusion")          # O(1) lookup
    tanks = db.by_role("Tank")       # role index
    avail = db.available(["Gusion"]) # all heroes not in exclusion list

    Auto-reload
    -----------
    The database polls the JSON file every `reload_interval` seconds.
    If the mtime changes it reloads transparently — useful for live
    patch-data updates without restarting the process.
    """

    _DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "heroes.json")

    def __init__(
        self,
        json_path: Optional[str] = None,
        reload_interval: float = 30.0,
    ) -> None:
        self._path = json_path or self._DEFAULT_PATH
        self._reload_interval = reload_interval
        self._lock = threading.RLock()

        # Indexes
        self._by_name: Dict[str, Hero] = {}
        self._by_role: Dict[str, List[Hero]] = {}
        self._meta: dict = {}
        self._mtime: float = 0.0

        # Load synchronously on init
        self._load()

        # Background reload thread
        if reload_interval > 0:
            t = threading.Thread(
                target=self._reload_loop,
                name="HeroDBReload",
                daemon=True,
            )
            t.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Hero]:
        """Case-insensitive name lookup.  Returns None if not found."""
        with self._lock:
            return self._by_name.get(name.lower())

    def get_many(self, names: List[str]) -> List[Hero]:
        """Look up a list of names; silently skip unknowns."""
        with self._lock:
            return [h for n in names if (h := self._by_name.get(n.lower()))]

    def by_role(self, role: str) -> List[Hero]:
        """All heroes whose primary role matches (case-insensitive)."""
        with self._lock:
            return list(self._by_role.get(role.capitalize(), []))

    def all_heroes(self) -> List[Hero]:
        """All heroes as a flat list (alphabetical by name)."""
        with self._lock:
            return sorted(self._by_name.values(), key=lambda h: h.name)

    def available(self, excluded: List[str]) -> List[Hero]:
        """Heroes not in the excluded list (picks + bans)."""
        excl: Set[str] = {n.lower() for n in excluded}
        with self._lock:
            return [h for h in self._by_name.values() if h.name.lower() not in excl]

    def roles_covered(self, names: List[str]) -> Set[str]:
        """Union of all roles covered by the given hero names."""
        roles: Set[str] = set()
        for h in self.get_many(names):
            roles.update(h.all_roles())
        return roles

    def patch_version(self) -> str:
        with self._lock:
            return self._meta.get("patch", "unknown")

    def hero_count(self) -> int:
        with self._lock:
            return len(self._by_name)

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Parse JSON and rebuild all indexes."""
        try:
            mtime = os.path.getmtime(self._path)
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("HeroDatabase: failed to load %s — %s", self._path, exc)
            return

        new_by_name: Dict[str, Hero] = {}
        new_by_role: Dict[str, List[Hero]] = {}
        errors = 0

        for entry in raw.get("heroes", []):
            try:
                hero = Hero(
                    id=int(entry["id"]),
                    name=str(entry["name"]),
                    role=str(entry["role"]),
                    win_rate=float(entry.get("win_rate", 0.50)),
                    pick_rate=float(entry.get("pick_rate", 0.12)),
                    ban_rate=float(entry.get("ban_rate", 0.10)),
                    tier=str(entry.get("tier", "B")),
                    counters=tuple(entry.get("counters", [])),
                    strong_against=tuple(entry.get("strong_against", [])),
                    tags=tuple(entry.get("tags", [])),
                )
                key = hero.name.lower()
                new_by_name[key] = hero
                primary = hero.primary_role()
                new_by_role.setdefault(primary, []).append(hero)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed hero entry %s: %s", entry, exc)
                errors += 1

        with self._lock:
            self._by_name = new_by_name
            self._by_role = new_by_role
            self._meta = raw.get("meta", {})
            self._mtime = mtime

        logger.info(
            "HeroDatabase loaded %d heroes (patch=%s, errors=%d)",
            len(new_by_name), self._meta.get("patch", "?"), errors,
        )

    def _reload_loop(self) -> None:
        """Daemon thread: poll for file changes and hot-reload."""
        while True:
            time.sleep(self._reload_interval)
            try:
                mtime = os.path.getmtime(self._path)
                if mtime != self._mtime:
                    logger.info("HeroDatabase: detected change in %s — reloading.", self._path)
                    self._load()
            except OSError:
                pass  # file briefly unavailable during write

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_name)

    def __repr__(self) -> str:
        return f"<HeroDatabase heroes={len(self)} patch={self.patch_version()}>"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DB_INSTANCE: Optional[HeroDatabase] = None
_DB_LOCK = threading.Lock()


def get_db(json_path: Optional[str] = None) -> HeroDatabase:
    """Return the module-level singleton, creating it if necessary."""
    global _DB_INSTANCE
    with _DB_LOCK:
        if _DB_INSTANCE is None:
            _DB_INSTANCE = HeroDatabase(json_path=json_path)
    return _DB_INSTANCE
