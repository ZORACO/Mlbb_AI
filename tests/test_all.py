"""
tests/test_all.py
-----------------
Full test suite for MLBB Draft AI v2.

Covers:
  - HeroDatabase (132-hero load, indexing, auto-reload, edge cases)
  - SlotConfig (scaling, coverage, region math)
  - TemplateMatcher + HybridSlotDetector (unit-level, with mock crops)
  - TemporalFilter (sliding window, lock-in, unlock)
  - DraftDetector integration (mock frame)
  - RecommendationEngine (all phases, scoring, edge cases)
  - EnemyProfile composition analysis
  - UIPayload construction

Run with:
    python -m pytest tests/ -v
    or
    python tests/test_all.py  (standalone)
"""

from __future__ import annotations

import os
import sys
import time
import json
import tempfile
import threading
import numpy as np

# Make project root importable when running as standalone script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

from data.hero_db import HeroDatabase, Hero, get_db
from config.slot_config import SlotConfig, DEFAULT_CONFIG, get_config
from vision.hero_detector import (
    DraftState, MockDetector, TemporalFilter,
    SlotResult, SlotCropper, HybridSlotDetector,
    TemplateMatcher, DraftDetector,
)
from recommender.engine import (
    RecommendationEngine, Recommendation, DraftPhase,
    EnemyProfile, PHASE_WEIGHTS,
    _counter_score, _team_fit, _tier_bonus, _risk,
    _synergy_score, _analyse_enemy, _missing_roles,
)


@pytest.fixture(scope="module")
def db():
    return HeroDatabase()


@pytest.fixture(scope="module")
def engine(db):
    return RecommendationEngine(db=db)


@pytest.fixture
def basic_state():
    return DraftState(
        ally_team=["Chou"],
        enemy_team=["Gusion", "Khufra"],
        bans=["Ling", "Fanny"],
    )


@pytest.fixture
def full_state():
    return DraftState(
        ally_team=["Chou", "Diggie", "Granger", "Karrie"],
        enemy_team=["Gusion", "Khufra", "Layla", "Esmeralda"],
        bans=["Ling", "Fanny", "Atlas"],
    )


@pytest.fixture
def black_frame():
    """Synthetic blank frame simulating a captured screen."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# ===========================================================================
# 1. HeroDatabase
# ===========================================================================

class TestHeroDatabase:

    def test_loads_all_heroes(self, db):
        assert len(db) == 132, f"Expected 132 heroes, got {len(db)}"

    def test_get_by_name_exact(self, db):
        hero = db.get("Gusion")
        assert hero is not None
        assert hero.name == "Gusion"
        assert hero.role == "Assassin"

    def test_get_case_insensitive(self, db):
        assert db.get("gusion") == db.get("GUSION") == db.get("GuSiOn")

    def test_get_missing_returns_none(self, db):
        assert db.get("NotARealHero") is None

    def test_get_hyphenated_name(self, db):
        hero = db.get("Yi Sun-shin")
        assert hero is not None

    def test_get_spaced_name(self, db):
        hero = db.get("Popol and Kupa")
        assert hero is not None

    def test_by_role_tank(self, db):
        tanks = db.by_role("Tank")
        assert len(tanks) >= 10
        for h in tanks:
            assert "Tank" in h.role

    def test_by_role_all_roles_present(self, db):
        for role in ["Tank", "Fighter", "Assassin", "Mage", "Marksman", "Support"]:
            assert len(db.by_role(role)) > 0, f"No heroes found for role {role}"

    def test_available_excludes_banned(self, db):
        excluded = ["Gusion", "Khufra", "Ling"]
        avail = db.available(excluded)
        names = {h.name for h in avail}
        for name in excluded:
            assert name not in names

    def test_available_count(self, db):
        avail = db.available(["Gusion"])
        assert len(avail) == 131

    def test_roles_covered(self, db):
        roles = db.roles_covered(["Gusion", "Tigreal"])
        assert "Assassin" in roles
        assert "Tank" in roles

    def test_hero_fields_valid(self, db):
        for hero in db.all_heroes():
            assert 0 <= hero.win_rate <= 1, f"{hero.name} win_rate out of range"
            assert 0 <= hero.pick_rate <= 1
            assert 0 <= hero.ban_rate <= 1
            assert hero.tier in ("S", "A", "B", "C"), f"{hero.name} bad tier"
            assert hero.id >= 1

    def test_primary_role(self, db):
        esme = db.get("Esmeralda")
        assert esme is not None
        # Esmeralda is Tank in our dataset
        assert esme.primary_role() in ["Tank", "Fighter", "Mage"]

    def test_tier_value_ordering(self, db):
        gusion = db.get("Gusion")
        layla  = db.get("Layla")
        assert gusion is not None and layla is not None
        # S-tier Gusion should have higher tier_value than B-tier Layla
        assert gusion.tier_value() > layla.tier_value()

    def test_auto_reload_on_change(self, db):
        """Verify database reloads when JSON mtime changes."""
        orig_count = len(db)
        orig_mtime = db._mtime
        # Touch the file to force reload check
        path = db._path
        os.utime(path, None)
        db._load()
        assert len(db) == orig_count  # same data, just reloaded

    def test_thread_safe_concurrent_reads(self, db):
        """Concurrent reads must not cause race conditions."""
        errors: list[Exception] = []

        def _read():
            try:
                for name in ["Gusion", "Atlas", "Chou", "Layla", "Ling"]:
                    _ = db.get(name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_read) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == [], f"Thread errors: {errors}"


# ===========================================================================
# 2. SlotConfig
# ===========================================================================

class TestSlotConfig:

    def test_default_counts(self):
        cfg = DEFAULT_CONFIG
        assert len(cfg.ally)  == 5
        assert len(cfg.enemy) == 5
        assert len(cfg.bans)  == 10

    def test_all_slots_dict(self):
        slots = DEFAULT_CONFIG.all_slots()
        assert set(slots.keys()) == {"ally", "enemy", "bans"}

    def test_scale_to_1080p(self):
        cfg = DEFAULT_CONFIG.scale_to(1920, 1080)
        assert cfg.screen_w == 1920
        # First ally slot x should scale by 1.5 from 30
        assert cfg.ally[0][0] == int(30 * 1920 / 1280)

    def test_get_config_preset(self):
        cfg = get_config(1280, 720)
        assert cfg.screen_w == 1280
        assert cfg.screen_h == 720

    def test_get_config_custom(self):
        cfg = get_config(800, 600)
        assert cfg.screen_w == 800

    def test_rects_within_screen(self):
        cfg = DEFAULT_CONFIG
        for group in [cfg.ally, cfg.enemy, cfg.bans]:
            for x, y, w, h in group:
                assert x >= 0 and y >= 0
                assert x + w <= cfg.screen_w
                assert y + h <= cfg.screen_h


# ===========================================================================
# 3. SlotCropper
# ===========================================================================

class TestSlotCropper:

    def test_crop_returns_correct_size(self, black_frame):
        cropper = SlotCropper(DEFAULT_CONFIG)
        rect = (0, 0, 100, 100)
        crop = cropper.crop(black_frame, rect)
        assert crop is not None
        assert crop.shape == (100, 100, 3)

    def test_crop_out_of_bounds_returns_none(self, black_frame):
        cropper = SlotCropper(DEFAULT_CONFIG)
        rect = (9999, 9999, 100, 100)  # completely outside
        crop = cropper.crop(black_frame, rect)
        assert crop is None

    def test_crop_all_returns_groups(self, black_frame):
        cropper = SlotCropper(DEFAULT_CONFIG)
        crops = cropper.crop_all(black_frame)
        assert "ally" in crops and "enemy" in crops and "bans" in crops
        assert len(crops["ally"])  == 5
        assert len(crops["enemy"]) == 5
        assert len(crops["bans"])  == 10


# ===========================================================================
# 4. TemporalFilter
# ===========================================================================

class TestTemporalFilter:

    def _make_results(self, heroes: list[str | None], conf: float = 0.90) -> list[SlotResult]:
        return [
            SlotResult(hero=h, confidence=conf if h else 0.0, source="template")
            for h in heroes
        ]

    def test_empty_state_produces_empty_draft(self):
        tf = TemporalFilter(n_ally=5, n_enemy=5, n_bans=10, window_size=5)
        empty = self._make_results([None] * 5)
        state = tf.update(empty, empty, self._make_results([None] * 10))
        assert state.ally_team == []
        assert state.enemy_team == []
        assert state.bans == []

    def test_stable_detection_produces_locked_hero(self):
        tf = TemporalFilter(n_ally=5, n_enemy=5, n_bans=10, window_size=5)
        ally_with_gusion = self._make_results(["Gusion"] + [None]*4)
        enemy_empty = self._make_results([None]*5)
        bans_empty = self._make_results([None]*10)

        state = None
        for _ in range(3):
            state = tf.update(ally_with_gusion, enemy_empty, bans_empty)
        assert "Gusion" in state.ally_team

    def test_lock_in_on_high_confidence(self):
        tf = TemporalFilter(n_ally=5, n_enemy=5, n_bans=10, window_size=5, lock_threshold=0.85)
        # Single high-confidence detection should lock immediately
        results = [SlotResult(hero="Atlas", confidence=0.92, source="template")] + \
                  [SlotResult(hero=None, confidence=0.0, source="none")] * 4
        state = tf.update(results, self._make_results([None]*5), self._make_results([None]*10))
        assert "Atlas" in state.ally_team
        assert tf._locked[0] == "Atlas"

    def test_unlock_on_contradiction(self):
        tf = TemporalFilter(
            n_ally=5, n_enemy=5, n_bans=10,
            window_size=5, lock_threshold=0.85, unlock_votes=3,
        )
        # Lock Atlas
        lock_r = [SlotResult("Atlas", 0.95, "template")] + [SlotResult(None, 0, "none")]*4
        tf.update(lock_r, self._make_results([None]*5), self._make_results([None]*10))
        assert tf._locked[0] == "Atlas"

        # Contradict with a different hero 3 times
        contra_r = [SlotResult("Khufra", 0.85, "template")] + [SlotResult(None, 0, "none")]*4
        for _ in range(4):
            tf.update(contra_r, self._make_results([None]*5), self._make_results([None]*10))
        assert tf._locked[0] is None  # should be unlocked

    def test_majority_vote_filters_noise(self):
        tf = TemporalFilter(n_ally=5, n_enemy=5, n_bans=10, window_size=7)
        empty_e = self._make_results([None]*5)
        empty_b = self._make_results([None]*10)

        # 4 votes for Chou, 2 votes for noise, 1 vote for noise
        for hero in ["Chou", "Chou", "Chou", "Chou", "Ling", "Fanny", "Chou"]:
            r = [SlotResult(hero, 0.75, "template")] + [SlotResult(None,0,"none")]*4
            state = tf.update(r, empty_e, empty_b)
        assert "Chou" in state.ally_team

    def test_reset_clears_state(self):
        tf = TemporalFilter(n_ally=5, n_enemy=5, n_bans=10, window_size=5)
        r = [SlotResult("Gusion", 0.95, "template")] + [SlotResult(None,0,"none")]*4
        tf.update(r, self._make_results([None]*5), self._make_results([None]*10))
        tf.reset()
        assert tf._locked[0] is None
        assert len(tf._windows[0]) == 0


# ===========================================================================
# 5. MockDetector
# ===========================================================================

class TestMockDetector:

    def test_returns_draft_state(self):
        d = MockDetector()
        state = d.detect(None)
        assert isinstance(state, DraftState)

    def test_advances_over_time(self):
        d = MockDetector()
        d._TICK = 0.05
        s1 = d.detect(None)
        time.sleep(0.1)
        s2 = d.detect(None)
        total1 = len(s1.all_taken())
        total2 = len(s2.all_taken())
        assert total2 >= total1

    def test_final_state_has_full_teams(self):
        d = MockDetector()
        d._idx = len(d._STATES) - 1
        state = d.detect(None)
        assert len(state.ally_team) > 0
        assert len(state.enemy_team) > 0


# ===========================================================================
# 6. Scoring components
# ===========================================================================

class TestScoringComponents:

    def test_counter_score_hard_counter(self, db):
        """Diggie should score very high vs Khufra (Diggie counters Khufra)."""
        diggie = db.get("Diggie")
        khufra = db.get("Khufra")
        assert diggie and khufra
        score = _counter_score(diggie, [khufra])
        assert score >= 0.7, f"Expected high counter score, got {score:.3f}"

    def test_counter_score_countered(self, db):
        """Layla is countered by Gusion — should score low."""
        layla  = db.get("Layla")
        gusion = db.get("Gusion")
        assert layla and gusion
        score = _counter_score(layla, [gusion])
        assert score < 0.5, f"Expected low counter score, got {score:.3f}"

    def test_counter_score_neutral_no_enemies(self, db):
        """No enemies → neutral 0.5."""
        hero = db.get("Chou")
        assert hero
        assert _counter_score(hero, []) == 0.5

    def test_counter_score_in_range(self, db):
        """Counter score must always be in [0,1]."""
        for hero in db.all_heroes()[:20]:
            enemies = db.get_many(["Gusion", "Khufra", "Atlas"])
            score = _counter_score(hero, enemies)
            assert 0.0 <= score <= 1.0, f"{hero.name} counter score {score} out of range"

    def test_team_fit_fills_role(self, db):
        """Tigreal should get max fit when team has no Tank."""
        tigreal = db.get("Tigreal")
        ally = db.get_many(["Gusion", "Layla"])
        missing = _missing_roles(ally)
        assert "Tank" in missing
        score = _team_fit(tigreal, ally, missing)
        assert score >= 0.9, f"Expected max fit for Tank, got {score:.3f}"

    def test_team_fit_role_already_covered(self, db):
        """Tigreal should get lower fit when team already has a tank."""
        tigreal = db.get("Tigreal")
        ally = db.get_many(["Khufra", "Layla"])   # Khufra covers Tank
        missing = _missing_roles(ally)
        assert "Tank" not in missing
        score = _team_fit(tigreal, ally, missing)
        assert score < 0.9

    def test_tier_bonus_ordering(self, db):
        """S > A > B tier bonus ordering must hold."""
        gusion  = db.get("Gusion")   # S
        chou    = db.get("Chou")     # A
        layla   = db.get("Layla")    # B
        assert gusion and chou and layla
        assert _tier_bonus(gusion) > _tier_bonus(chou)
        assert _tier_bonus(chou)   > _tier_bonus(layla)

    def test_risk_high_skill(self, db):
        """Fanny (high-skill tag) must have high risk."""
        fanny = db.get("Fanny")
        assert fanny
        assert _risk(fanny) >= 0.5

    def test_risk_safe_hero(self, db):
        """Tigreal (no high-skill, decent WR) should have low risk."""
        tigreal = db.get("Tigreal")
        assert tigreal
        assert _risk(tigreal) < 0.4

    def test_risk_always_in_range(self, db):
        for hero in db.all_heroes():
            r = _risk(hero)
            assert 0.0 <= r <= 1.0

    def test_synergy_true_damage_vs_tanks(self, db):
        """Karrie (true-damage) should get synergy bonus vs tank-heavy enemy."""
        karrie = db.get("Karrie")
        ally   = db.get_many(["Chou"])
        profile = EnemyProfile(tank_heavy=True)
        score = _synergy_score(karrie, ally, profile)
        assert score > 0.0

    def test_synergy_immune_vs_cc_heavy(self, db):
        """Lancelot (immune) should get synergy vs cc-heavy enemy."""
        lance = db.get("Lancelot")
        ally  = db.get_many(["Chou"])
        profile = EnemyProfile(cc_heavy=True)
        score = _synergy_score(lance, ally, profile)
        assert score > 0.0


# ===========================================================================
# 7. Enemy composition analysis
# ===========================================================================

class TestEnemyAnalysis:

    def test_tank_heavy(self, db):
        enemies = db.get_many(["Tigreal", "Franco", "Khufra"])
        profile = _analyse_enemy(enemies)
        assert profile.tank_heavy

    def test_squishy(self, db):
        enemies = db.get_many(["Layla", "Kagura", "Yve"])
        profile = _analyse_enemy(enemies)
        assert profile.squishy

    def test_empty_enemy_default(self):
        profile = _analyse_enemy([])
        assert not profile.tank_heavy
        assert not profile.squishy

    def test_describe_balanced(self):
        profile = EnemyProfile()
        assert profile.describe() == "balanced"

    def test_describe_multiple_traits(self):
        profile = EnemyProfile(tank_heavy=True, cc_heavy=True)
        desc = profile.describe()
        assert "tank-heavy" in desc
        assert "cc-heavy" in desc


# ===========================================================================
# 8. RecommendationEngine
# ===========================================================================

class TestRecommendationEngine:

    def test_returns_top_n(self, engine, basic_state):
        recs = engine.recommend(basic_state, top_n=3)
        assert len(recs) == 3

    def test_returns_fewer_when_pool_small(self, engine):
        # Almost all heroes taken
        db = get_db()
        all_heroes = db.all_heroes()
        taken = [h.name for h in all_heroes[:-2]]
        state = DraftState(ally_team=taken[:2], enemy_team=taken[2:4], bans=taken[4:])
        recs = engine.recommend(state, top_n=3)
        assert len(recs) <= 3

    def test_no_taken_heroes_in_recs(self, engine, basic_state):
        taken = set(basic_state.all_taken())
        for r in engine.recommend(basic_state):
            assert r.hero.name not in taken

    def test_scores_in_range(self, engine, basic_state):
        for r in engine.recommend(basic_state):
            assert 0.0 <= r.score <= 1.0

    def test_sorted_descending(self, engine, basic_state):
        recs = engine.recommend(basic_state)
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_reason_non_empty(self, engine, basic_state):
        for r in engine.recommend(basic_state):
            assert r.reason.strip() != ""

    def test_early_phase_uses_early_weights(self, engine, basic_state):
        recs = engine.recommend(basic_state)
        for r in recs:
            assert r.phase == DraftPhase.EARLY
            assert r.weights == PHASE_WEIGHTS[DraftPhase.EARLY]

    def test_late_phase_weights_counter_more(self, engine, full_state):
        recs = engine.recommend(full_state, top_n=3)
        for r in recs:
            assert r.phase == DraftPhase.LATE
            # Late weights counter at 0.45
            assert r.weights.counter == 0.45

    def test_empty_draft_returns_recs(self, engine):
        state = DraftState()
        recs = engine.recommend(state, top_n=3)
        assert len(recs) == 3
        for r in recs:
            assert r.phase == DraftPhase.BAN

    def test_confidence_pct_in_range(self, engine, basic_state):
        for r in engine.recommend(basic_state):
            pct = r.confidence_pct()
            assert 0 <= pct <= 100

    def test_risk_label_valid(self, engine, basic_state):
        valid = {"LOW", "MED", "HIGH"}
        for r in engine.recommend(basic_state):
            assert r.risk_label() in valid

    def test_tank_heavy_enemy_suggests_true_damage(self, engine, db):
        """When enemy is tank-heavy, at least one rec should have true-damage tag."""
        state = DraftState(
            ally_team=[],
            enemy_team=["Tigreal", "Franco", "Khufra"],
            bans=[],
        )
        recs = engine.recommend(state, top_n=5)
        any_true_damage = any("true-damage" in r.hero.tags for r in recs)
        # Not required to be #1, but should appear in top 5
        assert any_true_damage, "Expected a true-damage hero in recs vs tank-heavy"

    def test_cc_heavy_enemy_suggests_immune_or_anticc(self, engine, db):
        """CC-heavy enemy should surface immune/anti-cc heroes."""
        state = DraftState(
            ally_team=[],
            enemy_team=["Tigreal", "Franco", "Atlas"],  # 3 cc tanks
            bans=[],
        )
        recs = engine.recommend(state, top_n=5)
        any_immune = any(
            "immune" in r.hero.tags or "anti-cc" in r.hero.tags
            for r in recs
        )
        assert any_immune


# ===========================================================================
# 9. DraftPhase detection
# ===========================================================================

class TestDraftPhase:

    def test_ban_phase(self):
        assert DraftPhase.from_state(DraftState()) == DraftPhase.BAN

    def test_early_phase(self):
        s = DraftState(ally_team=["Chou"], enemy_team=["Gusion"])
        assert DraftPhase.from_state(s) == DraftPhase.EARLY

    def test_mid_phase(self):
        s = DraftState(
            ally_team=["Chou","Diggie","Granger"],
            enemy_team=["Gusion","Khufra"],
        )
        assert DraftPhase.from_state(s) == DraftPhase.MID

    def test_late_phase(self):
        s = DraftState(
            ally_team=["Chou","Diggie","Granger","Karrie","Yve"],
            enemy_team=["Gusion","Khufra","Layla","Esmeralda","Lancelot"],
        )
        assert DraftPhase.from_state(s) == DraftPhase.LATE


# ===========================================================================
# 10. Integration smoke tests
# ===========================================================================

class TestIntegration:

    def test_mock_detector_feeds_engine(self):
        detector = MockDetector()
        engine   = RecommendationEngine(db=get_db())

        detector._idx = 4   # advance to a mid-draft state
        state = detector.detect(None)
        recs  = engine.recommend(state, top_n=3)

        assert len(recs) > 0
        taken = set(state.all_taken())
        for r in recs:
            assert r.hero.name not in taken

    def test_full_sequence_no_crash(self):
        """Simulate an entire draft from ban phase to last pick."""
        detector = MockDetector()
        engine   = RecommendationEngine(db=get_db())

        for i in range(len(MockDetector._STATES)):
            detector._idx = i
            state = detector.detect(None)
            recs  = engine.recommend(state, top_n=3)
            assert isinstance(recs, list)

    def test_database_consistent_with_detector_output(self):
        """All hero names produced by MockDetector must exist in the database."""
        db  = get_db()
        det = MockDetector()
        for i in range(len(MockDetector._STATES)):
            det._idx = i
            state = det.detect(None)
            for name in state.all_taken():
                assert db.get(name) is not None, f"Hero '{name}' not found in DB"


# ===========================================================================
# Standalone runner (no pytest needed)
# ===========================================================================

def _run_standalone() -> None:
    """Run all tests without pytest, printing pass/fail to stdout."""
    import traceback

    db      = HeroDatabase()
    engine  = RecommendationEngine(db=db)
    b_state = DraftState(ally_team=["Chou"], enemy_team=["Gusion", "Khufra"], bans=["Ling", "Fanny"])
    f_state = DraftState(
        ally_team=["Chou","Diggie","Granger","Karrie"],
        enemy_team=["Gusion","Khufra","Layla","Esmeralda"],
        bans=["Ling","Fanny","Atlas"],
    )
    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    suites = [
        TestHeroDatabase(),
        TestSlotConfig(),
        TestSlotCropper(),
        TestTemporalFilter(),
        TestMockDetector(),
        TestScoringComponents(),
        TestEnemyAnalysis(),
        TestRecommendationEngine(),
        TestDraftPhase(),
        TestIntegration(),
    ]

    total = passed = failed = 0

    for suite in suites:
        suite_name = type(suite).__name__
        print(f"\n{'─'*50}")
        print(f"  {suite_name}")
        print(f"{'─'*50}")

        for method_name in [m for m in dir(suite) if m.startswith("test_")]:
            total += 1
            method = getattr(suite, method_name)
            try:
                # Inject fixtures based on parameter names
                import inspect
                sig = inspect.signature(method)
                kwargs = {}
                for param in sig.parameters:
                    if param == "db":           kwargs["db"] = db
                    elif param == "engine":     kwargs["engine"] = engine
                    elif param == "basic_state": kwargs["basic_state"] = b_state
                    elif param == "full_state":  kwargs["full_state"] = f_state
                    elif param == "black_frame": kwargs["black_frame"] = black_frame
                method(**kwargs)
                print(f"  ✓  {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ✗  {method_name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'═'*50}")
    print(f"  Results: {passed}/{total} passed", "✓" if failed == 0 else f"  ({failed} FAILED)")
    print(f"{'═'*50}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_standalone()
