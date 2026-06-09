"""Tests for Task 15: ResultScout + self-evaluation loop.

The self-eval loop is the agent's "learning" step:
  1. ResultScout polls a results source (Phase 1: fixture; Phase 2: API)
  2. For every new Result, look up the latest prediction for that match_id
  3. Compute accuracy (was_correct), Brier score, top_factor_hit
  4. Persist the result row so the eval harness can aggregate later
  5. Loop until either (a) every tracked prediction is decided, or
     (b) max_iterations is reached.

Pure helpers (brier, top_factor_hit) are testable without a DB.
The integration is tested end-to-end with a fixture-based scout and
a couple of pre-seeded predictions.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_brier_perfect_prediction_is_zero():
    from soccer_agent.eval.scoring import brier
    p = {"home": 1.0, "draw": 0.0, "away": 0.0}
    assert brier(p, "home") == pytest.approx(0.0)


def test_brier_worst_prediction_is_one():
    """Worst case for a 3-class Brier: 100% on one wrong class, 0% on actual.
    With one-hot y and sum((p-y)^2) / 2:
      actual=home, predicted={home:0, draw:1, away:0}
      one-hot: y={home:1, draw:0, away:0}
      sum = (0-1)^2 + (1-0)^2 + (0-0)^2 = 1+1+0 = 2
      / 2 = 1.0
    So the 3-class Brier max is 1.0 (2.0 raw -> 1.0 after /2).
    The 2-class Brier max is 0.25; do not confuse the two.
    """
    from soccer_agent.eval.scoring import brier
    p = {"home": 0.0, "draw": 1.0, "away": 0.0}
    assert brier(p, "home") == pytest.approx(1.0)


def test_brier_handles_missing_keys_as_zero():
    """A reasoner that emitted only {home, draw} shouldn't crash brier."""
    from soccer_agent.eval.scoring import brier
    p = {"home": 0.6, "draw": 0.4}  # no "away" key
    val = brier(p, "draw")
    assert 0.0 < val < 1.0


def test_top_factor_hit_matches_when_factor_sign_agrees_with_outcome():
    from soccer_agent.eval.scoring import top_factor_hit
    from soccer_agent.models import Factor, ReasonerOutput
    out = ReasonerOutput(
        reasoner="numeric",
        pick="home",
        probs={"home": 0.6, "draw": 0.2, "away": 0.2},
        confidence=0.4,
        rationale="x",
        factors=[
            Factor(name="elo_gap", value=120.0, sign="positive", weight=1.0),
            Factor(name="form_delta", value=0.3, sign="positive", weight=0.5),
        ],
    )
    # Actual home win → positive factors should count as hits.
    assert top_factor_hit(out, "home") is True


def test_top_factor_hit_false_when_no_factors_agree():
    from soccer_agent.eval.scoring import top_factor_hit
    from soccer_agent.models import Factor, ReasonerOutput
    out = ReasonerOutput(
        reasoner="numeric",
        pick="home",
        probs={"home": 0.6, "draw": 0.2, "away": 0.2},
        confidence=0.4,
        rationale="x",
        factors=[
            # All factors point AWAY — but the agent picked home, so no hit.
            Factor(name="home_injuries", value=2.0, sign="negative", weight=1.0),
        ],
    )
    assert top_factor_hit(out, "home") is False


def test_top_factor_hit_returns_none_when_no_factors():
    from soccer_agent.eval.scoring import top_factor_hit
    from soccer_agent.models import ReasonerOutput
    out = ReasonerOutput(
        reasoner="numeric",
        pick="home",
        probs={"home": 0.6, "draw": 0.2, "away": 0.2},
        confidence=0.4,
        rationale="x",
        factors=[],
    )
    assert top_factor_hit(out, "home") is None


# ---------------------------------------------------------------------------
# ResultScout
# ---------------------------------------------------------------------------


def test_result_scout_returns_empty_when_no_results_yet():
    """A scout backed by an empty store returns no Results."""
    from soccer_agent.eval.scout import ResultScout
    scout = ResultScout(provider=lambda since: [])
    assert scout.fetch_new_results(since=datetime(1970, 1, 1, tzinfo=timezone.utc)) == []


def test_result_scout_returns_results_after_watermark():
    from soccer_agent.eval.scout import ResultScout
    from soccer_agent.models import Result
    r1 = Result(match_id="m1", home_goals=1, away_goals=0, decided_at=datetime(2025, 5, 1, tzinfo=timezone.utc))
    r2 = Result(match_id="m2", home_goals=2, away_goals=2, decided_at=datetime(2025, 5, 2, tzinfo=timezone.utc))
    provider = lambda since: [r1, r2]
    scout = ResultScout(provider=provider)
    out = scout.fetch_new_results(since=datetime(1970, 1, 1, tzinfo=timezone.utc))
    # Scout sorts newest-first so the caller can update its watermark
    # after each successful batch.
    assert [r.match_id for r in out] == ["m2", "m1"]


def test_result_scout_filters_results_before_watermark():
    """Results older than `since` are filtered out (caller-managed watermark)."""
    from soccer_agent.eval.scout import ResultScout
    from soccer_agent.models import Result
    old = Result(match_id="m0", home_goals=1, away_goals=0, decided_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    new = Result(match_id="m1", home_goals=1, away_goals=0, decided_at=datetime(2025, 5, 1, tzinfo=timezone.utc))
    provider = lambda since: [old, new] if since.year == 2025 else []
    scout = ResultScout(provider=provider)
    out = scout.fetch_new_results(since=datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert [r.match_id for r in out] == ["m1"]


def test_result_scout_fixture_provider_reads_from_dir():
    """Phase 1 default: read results from a directory of <match_id>.json files."""
    import json
    from pathlib import Path
    from soccer_agent.eval.scout import fixture_provider
    tmp = Path("results_dir")
    tmp.mkdir(exist_ok=True)
    (tmp / "ucl_final_2025.json").write_text(json.dumps({
        "match_id": "ucl_final_2025",
        "home_goals": 2,
        "away_goals": 1,
        "decided_at": "2025-05-30T22:00:00Z",
    }))
    try:
        provider = fixture_provider(tmp)
        results = provider(since=datetime(1970, 1, 1, tzinfo=timezone.utc))
        assert len(results) == 1
        assert results[0].match_id == "ucl_final_2025"
        assert results[0].home_goals == 2
    finally:
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()


# ---------------------------------------------------------------------------
# Agent.evaluate_all (loop)
# ---------------------------------------------------------------------------


def test_evaluate_all_marks_predictions_correct(tmp_path):
    """Pre-seed 2 predictions; insert 2 matching results; loop → both correct."""
    import json
    from soccer_agent.agent import PredictionAgent
    from soccer_agent.tools import default_registry
    from soccer_agent.tools._fixtures import fixture_path
    from soccer_agent.db import Database
    from soccer_agent.eval.scout import ResultScout
    from soccer_agent.models import (
        Match, Team, Prediction, ReasonerOutput, Factor,
    )

    # 1. Seed a minimal fixture so tools don't blow up (we don't care about
    #    signals here — we're testing the self-eval loop, not the predictor).
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / "venues").mkdir()
    (fx / "venues" / "venue_v1.json").write_text(json.dumps({
        "id": "v1", "name": "v1", "city": "x", "country": "x",
        "is_neutral": False, "is_dome": False, "altitude_m": 0, "lat": 0, "lon": 0,
    }))

    db_path = tmp_path / "agent.db"
    db = Database(db_path=db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        db_path=db_path,
    )

    # 2. Pre-seed two predictions directly (faster than running the agent
    #    twice; we're testing the self-eval loop, not the predictor).
    preds = [
        Prediction(
            prediction_id="p1",
            match_id="m_home",
            created_at=datetime(2025, 5, 1, tzinfo=timezone.utc),
            signals={},
            reasoner_outputs=[
                ReasonerOutput(reasoner="numeric", pick="home",
                               probs={"home": 0.6, "draw": 0.2, "away": 0.2},
                               confidence=0.4, rationale="x", factors=[]),
            ],
            final_pick="home",
            final_probs={"home": 0.6, "draw": 0.2, "away": 0.2},
            final_confidence=0.4,
            final_rationale="x",
        ),
        Prediction(
            prediction_id="p2",
            match_id="m_away",
            created_at=datetime(2025, 5, 1, tzinfo=timezone.utc),
            signals={},
            reasoner_outputs=[
                ReasonerOutput(reasoner="numeric", pick="away",
                               probs={"home": 0.2, "draw": 0.2, "away": 0.6},
                               confidence=0.4, rationale="x", factors=[]),
            ],
            final_pick="away",
            final_probs={"home": 0.2, "draw": 0.2, "away": 0.6},
            final_confidence=0.4,
            final_rationale="x",
        ),
    ]
    for p in preds:
        db.insert_prediction({
            "prediction_id": p.prediction_id,
            "match_id": p.match_id,
            "created_at": p.created_at,
            "signals": p.signals,
            "reasoner_outputs": [ro.model_dump(mode="json") for ro in p.reasoner_outputs],
            "final_pick": p.final_pick,
            "final_probs": p.final_probs,
            "final_confidence": p.final_confidence,
            "final_rationale": p.final_rationale,
            "warnings": p.warnings,
            "model_versions": p.model_versions,
        })

    # 3. A scout that returns one Result per match.
    from soccer_agent.models import Result
    results = [
        Result(match_id="m_home", home_goals=2, away_goals=0, decided_at=datetime(2025, 5, 30, tzinfo=timezone.utc)),
        Result(match_id="m_away", home_goals=1, away_goals=3, decided_at=datetime(2025, 5, 30, tzinfo=timezone.utc)),
    ]
    scout = ResultScout(provider=lambda since: results)

    # 4. Run the loop.
    n_scored = agent.evaluate_all(scout)
    assert n_scored == 2

    # 5. Re-load both predictions (joined with results) and assert
    #    self-eval fields were written.
    rows = db.list_predictions(limit=10)
    by_id = {r["prediction_id"]: r for r in rows}
    p1 = by_id["p1"]
    p2 = by_id["p2"]
    assert p1["was_correct"] == 1
    assert p2["was_correct"] == 1
    assert p1["home_goals"] == 2 and p1["away_goals"] == 0
    assert p2["home_goals"] == 1 and p2["away_goals"] == 3
    # Brier must be a finite float in [0, 0.5] for 3-class.
    # (list_predictions exposes the joined brier as `result_brier`.)
    assert 0.0 <= float(p1["result_brier"]) <= 0.5
    assert 0.0 <= float(p2["result_brier"]) <= 0.5


def test_evaluate_all_skips_match_without_prediction(tmp_path):
    """A result with no matching prediction must be a no-op, not a crash."""
    from soccer_agent.agent import PredictionAgent
    from soccer_agent.tools import default_registry
    from soccer_agent.eval.scout import ResultScout
    from soccer_agent.models import Result
    from datetime import datetime, timezone
    agent = PredictionAgent(registry=default_registry(), db_path=tmp_path / "x.db")
    scout = ResultScout(provider=lambda since: [
        Result(match_id="ghost", home_goals=1, away_goals=1, decided_at=datetime(2025, 5, 30, tzinfo=timezone.utc)),
    ])
    n = agent.evaluate_all(scout)
    assert n == 0


def test_evaluate_all_idempotent(tmp_path):
    """Running the loop twice with the same scout must not double-count."""
    from soccer_agent.agent import PredictionAgent
    from soccer_agent.tools import default_registry
    from soccer_agent.db import Database
    from soccer_agent.eval.scout import ResultScout
    from soccer_agent.models import (
        Prediction, Result, ReasonerOutput,
    )
    from datetime import datetime, timezone

    db_path = tmp_path / "x.db"
    agent = PredictionAgent(registry=default_registry(), db_path=db_path)
    db = Database(db_path=db_path)
    db.insert_prediction({
        "prediction_id": "p1",
        "match_id": "m1",
        "created_at": datetime(2025, 5, 1, tzinfo=timezone.utc),
        "signals": {},
        "reasoner_outputs": [{
            "reasoner": "numeric", "pick": "home",
            "probs": {"home": 1.0, "draw": 0.0, "away": 0.0},
            "confidence": 1.0, "rationale": "x", "factors": [],
        }],
        "final_pick": "home",
        "final_probs": {"home": 1.0, "draw": 0.0, "away": 0.0},
        "final_confidence": 1.0,
        "final_rationale": "x",
        "warnings": [],
        "model_versions": {"reasoner": "numeric"},
    })
    scout = ResultScout(provider=lambda since: [
        Result(match_id="m1", home_goals=3, away_goals=0,
               decided_at=datetime(2025, 5, 30, tzinfo=timezone.utc)),
    ])
    n1 = agent.evaluate_all(scout)
    n2 = agent.evaluate_all(scout)
    assert n1 == 1
    # Second run: result row already present, no new scoring.
    assert n2 == 0
