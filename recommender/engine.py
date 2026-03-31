"""
recommender/engine.py
----------------------
Production recommendation engine — hybrid rule-based + scoring.

Upgrades over v1
----------------
* Phase-aware dynamic weight adjustment
  Early picks → weight team_fit higher (role gaps matter most)
  Late picks  → weight counter_score higher (specific counters matter)
* Richer enemy composition analysis (6 archetypes detected)
* Tier bonus: S/A tier heroes get a small score boost
* Synergy bonus: rewards compositions with multiple cc or engage heroes
* ML hook: ml_win_probability() integrates seamlessly when implemented
* Full 132-hero support via HeroDatabase
* Deterministic scoring — no randomness in output

Scoring formula (base):
    score = win_rate      * W_WIN_RATE
          + counter_score * W_COUNTER
          + team_fit      * W_TEAM_FIT
          + tier_bonus    * W_TIER
          + synergy_bonus * W_SYNERGY
          - risk          * W_RISK

Weights shift by draft phase — see PhaseWeights below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.hero_db import Hero, HeroDatabase, get_db
from vision.hero_detector import DraftState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Draft phase
# ---------------------------------------------------------------------------

class DraftPhase(Enum):
    """
    Maps pick count to draft phase for weight adjustment.
    BAN  : during ban selection (no picks yet)
    EARLY: first 2 picks — role composition matters most
    MID  : picks 3–4 — starting to counter-pick
    LATE : 5th pick  — last pick, full counter-pick mode
    """
    BAN   = "ban"
    EARLY = "early"
    MID   = "mid"
    LATE  = "late"

    @classmethod
    def from_state(cls, state: DraftState) -> "DraftPhase":
        total_picks = len(state.ally_team) + len(state.enemy_team)
        if total_picks == 0:
            return cls.BAN
        if total_picks <= 4:
            return cls.EARLY
        if total_picks <= 7:
            return cls.MID
        return cls.LATE


# ---------------------------------------------------------------------------
# Phase-aware weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseWeights:
    win_rate:  float = 0.30
    counter:   float = 0.25
    team_fit:  float = 0.25
    tier:      float = 0.10
    synergy:   float = 0.05
    risk:      float = 0.05  # subtracted


PHASE_WEIGHTS: Dict[DraftPhase, PhaseWeights] = {
    DraftPhase.BAN: PhaseWeights(
        win_rate=0.20, counter=0.20, team_fit=0.30,
        tier=0.15, synergy=0.10, risk=0.05,
    ),
    DraftPhase.EARLY: PhaseWeights(
        win_rate=0.25, counter=0.20, team_fit=0.35,
        tier=0.10, synergy=0.05, risk=0.05,
    ),
    DraftPhase.MID: PhaseWeights(
        win_rate=0.30, counter=0.30, team_fit=0.20,
        tier=0.10, synergy=0.05, risk=0.05,
    ),
    DraftPhase.LATE: PhaseWeights(
        win_rate=0.20, counter=0.45, team_fit=0.15,
        tier=0.08, synergy=0.07, risk=0.05,
    ),
}


# ---------------------------------------------------------------------------
# Recommendation output
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """Single hero recommendation with full scoring breakdown."""
    hero:             Hero
    score:            float
    win_rate_comp:    float
    counter_comp:     float
    team_fit_comp:    float
    tier_comp:        float
    synergy_comp:     float
    risk_comp:        float
    reason:           str
    phase:            DraftPhase
    weights:          PhaseWeights

    def confidence_pct(self) -> int:
        return round(self.score * 100)

    def risk_label(self) -> str:
        if self.risk_comp >= 0.6:
            return "HIGH"
        if self.risk_comp >= 0.3:
            return "MED"
        return "LOW"

    def __lt__(self, other: "Recommendation") -> bool:
        return self.score < other.score


# ---------------------------------------------------------------------------
# Enemy composition analysis
# ---------------------------------------------------------------------------

@dataclass
class EnemyProfile:
    """Classified enemy team composition."""
    tank_heavy:   bool = False   # ≥2 tanks
    squishy:      bool = False   # ≥3 mm/mage
    cc_heavy:     bool = False   # ≥3 cc tags
    mobile:       bool = False   # ≥2 mobile tags
    burst_heavy:  bool = False   # ≥3 burst tags
    sustain_heavy: bool = False  # ≥2 sustain tags

    def describe(self) -> str:
        traits = []
        if self.tank_heavy:   traits.append("tank-heavy")
        if self.squishy:      traits.append("squishy")
        if self.cc_heavy:     traits.append("cc-heavy")
        if self.mobile:       traits.append("mobile")
        if self.burst_heavy:  traits.append("burst-heavy")
        if self.sustain_heavy: traits.append("sustain-heavy")
        return ", ".join(traits) if traits else "balanced"


def _analyse_enemy(enemy_heroes: List[Hero]) -> EnemyProfile:
    if not enemy_heroes:
        return EnemyProfile()
    roles = [h.primary_role() for h in enemy_heroes]
    tags:  List[str] = []
    for h in enemy_heroes:
        tags.extend(h.tags)
    return EnemyProfile(
        tank_heavy=roles.count("Tank") >= 2,
        squishy=(roles.count("Marksman") + roles.count("Mage")) >= 3,
        cc_heavy=tags.count("cc") >= 3,
        mobile=tags.count("mobile") >= 2,
        burst_heavy=tags.count("burst") >= 3,
        sustain_heavy=tags.count("sustain") >= 2,
    )


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------

_ROLE_PRIORITY = ["Tank", "Marksman", "Mage", "Support", "Assassin", "Fighter"]


def _missing_roles(ally_heroes: List[Hero]) -> List[str]:
    covered = set()
    for h in ally_heroes:
        covered.update(h.all_roles())
    return [r for r in _ROLE_PRIORITY if r not in covered]


def _counter_score(candidate: Hero, enemy_heroes: List[Hero]) -> float:
    """How effectively *candidate* counters the enemy lineup. Returns [0,1]."""
    if not enemy_heroes:
        return 0.5
    raw = 0.0
    for enemy in enemy_heroes:
        if candidate.is_counter_to(enemy.name):
            raw += 1.0
        if candidate.is_countered_by(enemy.name):
            raw -= 0.5
    # Normalise to [0,1] — max raw = len(enemies), min raw = -len(enemies)*0.5
    max_raw = len(enemy_heroes)
    min_raw = -len(enemy_heroes) * 0.5
    span = max_raw - min_raw
    return max(0.0, min(1.0, (raw - min_raw) / span))


def _team_fit(
    candidate: Hero,
    ally_heroes: List[Hero],
    missing_roles: List[str],
) -> float:
    """How well *candidate* fills team composition needs. Returns [0,1]."""
    candidate_roles = set(candidate.all_roles())

    # Full score if candidate fills a prioritised missing role
    for i, role in enumerate(missing_roles):
        if role in candidate_roles:
            return 1.0 - i * 0.05  # slight decay for less-urgent gaps

    # Synergy via shared tags
    ally_tags: set[str] = set()
    for h in ally_heroes:
        ally_tags.update(h.tags)
    shared = {"cc", "engage", "burst", "poke"} & set(candidate.tags) & ally_tags
    return min(0.7, 0.3 + len(shared) * 0.15)


def _tier_bonus(candidate: Hero) -> float:
    """[0,1] bonus based on hero tier. S=1.0, A=0.75, B=0.50, C=0.25."""
    return {4: 1.0, 3: 0.75, 2: 0.50, 1: 0.25}.get(candidate.tier_value(), 0.50)


def _synergy_score(
    candidate: Hero,
    ally_heroes: List[Hero],
    enemy_profile: EnemyProfile,
) -> float:
    """Reward for counter-composition awareness."""
    score = 0.0
    tags = set(candidate.tags)

    if enemy_profile.tank_heavy and "true-damage" in tags:
        score += 1.0
    if enemy_profile.cc_heavy and "immune" in tags:
        score += 1.0
    if enemy_profile.cc_heavy and "anti-cc" in tags:
        score += 0.8
    if enemy_profile.mobile and "anti-mobile" in tags:
        score += 1.0
    if enemy_profile.squishy and "burst" in tags:
        score += 0.5
    if enemy_profile.sustain_heavy and "anti-regen" in tags:
        score += 0.8

    return min(1.0, score)


def _risk(candidate: Hero) -> float:
    """[0,1] risk score — penalises high-skill-cap and low win-rate heroes."""
    risk = 0.0
    if "high-skill" in candidate.tags:
        risk += 0.50
    if candidate.win_rate < 0.49:
        risk += 0.30
    elif candidate.win_rate < 0.50:
        risk += 0.10
    return min(1.0, risk)


# ---------------------------------------------------------------------------
# ML hook (future integration point)
# ---------------------------------------------------------------------------

def ml_win_probability(state: DraftState, candidate: Hero) -> float:
    """
    Placeholder for a trained ML model.

    Expected interface
    ------------------
    Input  : DraftState + candidate Hero
    Output : float ∈ [0,1] — predicted win probability

    Returns -1.0 to signal "not available" — the engine falls back to
    rule-based scoring when this function returns < 0.

    Integration guide
    -----------------
    1. Train a gradient-boosted or neural model on ranked match data.
    2. Featurise: one-hot encode hero identities + role flags + tier.
    3. Load model once (outside this function) and cache globally.
    4. Replace `return -1.0` with actual inference.
    """
    # TODO: load model, featurise state + candidate, return probability
    return -1.0


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """
    Generates ranked hero recommendations for any draft state.

    Usage
    -----
    engine = RecommendationEngine()
    recs = engine.recommend(state, top_n=3)
    """

    def __init__(self, db: Optional[HeroDatabase] = None) -> None:
        self.db = db or get_db()

    def recommend(
        self,
        state: DraftState,
        top_n: int = 3,
    ) -> List[Recommendation]:
        """
        Generate *top_n* hero recommendations given the current draft state.

        Parameters
        ----------
        state : DraftState
            Current ally/enemy/ban information.
        top_n : int
            Number of recommendations to return (default 3).

        Returns
        -------
        List[Recommendation] sorted by composite score descending.
        """
        phase     = DraftPhase.from_state(state)
        weights   = PHASE_WEIGHTS[phase]

        ally_heroes  = self.db.get_many(state.ally_team)
        enemy_heroes = self.db.get_many(state.enemy_team)
        enemy_profile = _analyse_enemy(enemy_heroes)

        missing_roles = _missing_roles(ally_heroes)
        excluded      = state.all_taken()
        candidates    = self.db.available(excluded)

        if not candidates:
            return []

        logger.debug(
            "Recommend: phase=%s | missing_roles=%s | enemy=%s",
            phase.value, missing_roles, enemy_profile.describe(),
        )

        recs: List[Recommendation] = []

        for hero in candidates:
            # --- Component scores ----------------------------------------
            wr   = hero.win_rate
            ctr  = _counter_score(hero, enemy_heroes)
            fit  = _team_fit(hero, ally_heroes, missing_roles)
            tier = _tier_bonus(hero)
            syn  = _synergy_score(hero, ally_heroes, enemy_profile)
            rsk  = _risk(hero)

            # --- Optional ML override ------------------------------------
            ml_prob = ml_win_probability(state, hero)
            if ml_prob >= 0:
                wr = ml_prob * 0.6 + wr * 0.4

            # --- Composite score -----------------------------------------
            score = (
                wr   * weights.win_rate
                + ctr  * weights.counter
                + fit  * weights.team_fit
                + tier * weights.tier
                + syn  * weights.synergy
                - rsk  * weights.risk
            )
            score = max(0.0, min(1.0, score))

            reason = self._build_reason(
                hero, ctr, fit, syn, missing_roles, enemy_profile, phase,
            )

            recs.append(Recommendation(
                hero=hero,
                score=round(score, 4),
                win_rate_comp=wr,
                counter_comp=ctr,
                team_fit_comp=fit,
                tier_comp=tier,
                synergy_comp=syn,
                risk_comp=rsk,
                reason=reason,
                phase=phase,
                weights=weights,
            ))

        recs.sort(reverse=True, key=lambda r: r.score)
        return recs[:top_n]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reason(
        hero: Hero,
        ctr: float,
        fit: float,
        syn: float,
        missing_roles: List[str],
        enemy_profile: EnemyProfile,
        phase: DraftPhase,
    ) -> str:
        parts: List[str] = []

        # Role fit
        candidate_roles = set(hero.all_roles())
        for role in missing_roles:
            if role in candidate_roles:
                parts.append(f"fills missing {role} slot")
                break

        # Counter synergy
        if ctr >= 0.75:
            parts.append("hard counters enemy lineup")
        elif ctr >= 0.60:
            parts.append("favorable matchup vs enemies")
        elif ctr <= 0.30:
            parts.append("watch out — enemies counter this hero")

        # Composition awareness
        if syn >= 0.8:
            if enemy_profile.tank_heavy and "true-damage" in hero.tags:
                parts.append("true damage shreds tanky enemies")
            if enemy_profile.cc_heavy and "immune" in hero.tags:
                parts.append("immune negates CC-heavy composition")
            if enemy_profile.cc_heavy and "anti-cc" in hero.tags:
                parts.append("hard counters enemy CC")
            if enemy_profile.mobile and "anti-mobile" in hero.tags:
                parts.append("punishes mobile enemy picks")
            if enemy_profile.sustain_heavy and "anti-regen" in hero.tags:
                parts.append("anti-regen vs sustain comp")

        # Phase context
        if phase == DraftPhase.LATE and not parts:
            parts.append(f"best late counter pick ({hero.tier} tier)")

        if not parts:
            parts.append(
                f"{hero.tier}-tier {hero.primary_role().lower()} "
                f"({hero.win_rate:.0%} WR)"
            )

        return "; ".join(parts[:3])   # cap at 3 clauses
