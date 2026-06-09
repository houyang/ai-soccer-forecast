"""Tests for Pydantic models in soccer_agent.models."""

import pytest
from pydantic import ValidationError

from soccer_agent.models import (
    Factor,
    Match,
    MatchContext,
    Prediction,
    ReasonerOutput,
    Result,
    Signal,
    Team,
    ToolErrorPayload,
)


def test_team_minimal():
    t = Team(id="man_city", name="Manchester City")
    assert t.id == "man_city"
    assert t.country == "ENG"  # default


def test_match_requires_home_and_away_and_id():
    with pytest.raises(ValidationError):
        Match(home=Team(id="a", name="A"))  # type: ignore[call-arg]
    m = Match(
        match_id="m1",
        home=Team(id="a", name="A"),
        away=Team(id="b", name="B"),
        kickoff="2026-05-30T20:00:00Z",
        venue_id="puskas_arena",
        competition="UCL",
    )
    assert m.competition == "UCL"
    assert m.match_id == "m1"


def test_match_kickoff_must_be_iso():
    with pytest.raises(ValidationError):
        Match(
            home=Team(id="a", name="A"),
            away=Team(id="b", name="B"),
            kickoff="not-a-date",
            venue_id="x",
            competition="UCL",
        )


def test_reasoner_output_confidence_bounded_to_unit_interval():
    # v2 schema: confidence is 0..1, not 0..100
    with pytest.raises(ValidationError):
        ReasonerOutput(
            reasoner="numeric",
            pick="home",
            probs={"home": 0.6, "draw": 0.2, "away": 0.2},
            confidence=1.5,  # > 1.0 must be rejected
            rationale="x",
        )
    with pytest.raises(ValidationError):
        ReasonerOutput(
            reasoner="numeric",
            pick="home",
            probs={"home": 0.6, "draw": 0.2, "away": 0.2},
            confidence=-0.1,  # < 0 must be rejected
            rationale="x",
        )
    # boundaries
    r0 = ReasonerOutput(
        reasoner="numeric", pick="home",
        probs={"home": 1.0, "draw": 0.0, "away": 0.0},
        confidence=0.0, rationale="x",
    )
    assert r0.confidence == 0.0
    r1 = ReasonerOutput(
        reasoner="numeric", pick="home",
        probs={"home": 1.0, "draw": 0.0, "away": 0.0},
        confidence=1.0, rationale="x",
    )
    assert r1.confidence == 1.0


def test_reasoner_output_probs_must_sum_to_one_within_tolerance():
    with pytest.raises(ValidationError):
        ReasonerOutput(
            reasoner="numeric",
            pick="home",
            probs={"home": 0.5, "draw": 0.1, "away": 0.1},  # sums to 0.7
            confidence=0.5,
            rationale="x",
        )


def test_factor_sign_required():
    f = Factor(name="elo_delta", value=12.0, sign="positive", weight=0.3)
    assert f.sign == "positive"
    with pytest.raises(ValidationError):
        Factor(name="x", value=0.0, sign="up", weight=0.0)  # type: ignore[arg-type]


def test_signal_and_context():
    sig = Signal(tool="form_recent", data={"home": {"points": 9}})
    ctx = MatchContext(
        match=Match(
            match_id="m1",
            home=Team(id="a", name="A"),
            away=Team(id="b", name="B"),
            kickoff="2026-05-30T20:00:00Z",
            venue_id="x",
            competition="UCL",
        ),
        signals={"form_recent": sig},
    )
    assert ctx.signals["form_recent"].tool == "form_recent"


def test_prediction_persists_reasoner_outputs_and_warnings():
    p = Prediction(
        prediction_id="p1",
        match_id="m1",
        created_at="2026-05-30T10:00:00Z",
        signals={},
        reasoner_outputs=[
            ReasonerOutput(
                reasoner="numeric",
                pick="home",
                probs={"home": 0.6, "draw": 0.2, "away": 0.2},
                confidence=0.6,
                rationale="home favourite",
            )
        ],
        final_pick="home",
        final_probs={"home": 0.6, "draw": 0.2, "away": 0.2},
        final_confidence=0.6,
        final_rationale="home favourite",
        warnings=[],
        model_versions={"numeric": "v0.1"},
    )
    assert p.final_pick == "home"
    assert p.warnings == []


def test_result_score():
    r = Result(match_id="m1", home_goals=2, away_goals=1, decided_at="2026-05-30T22:00:00Z")
    assert r.winner == "home"
    r2 = Result(match_id="m2", home_goals=1, away_goals=1, decided_at="2026-05-30T22:00:00Z")
    assert r2.winner == "draw"
    r3 = Result(match_id="m3", home_goals=0, away_goals=2, decided_at="2026-05-30T22:00:00Z")
    assert r3.winner == "away"


def test_tool_error_payload():
    e = ToolErrorPayload(source="weather_venue", message="no fixture for x", retriable=False)
    assert e.source == "weather_venue"
    assert e.retriable is False
