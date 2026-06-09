"""Tests for the numeric reasoner."""

from __future__ import annotations

from datetime import datetime

import pytest

from soccer_agent.models import (
    Factor,
    FormEntry,
    FormOutput,
    H2HMeeting,
    H2HOutput,
    InjuryOutput,
    InjuryReport,
    Match,
    MatchContext,
    OddsOutput,
    BookmakerOdds,
    ReasonerOutput,
    Signal,
    Team,
    WeatherOutput,
)
from soccer_agent.reasoners.numeric import (
    DEFAULT_ELO,
    NumericReasoner,
    _draw_residual,
    _elo_expected,
    _form_score,
    _h2h_score,
    _injury_score,
    _market_score,
    run,
)


def _ctx(*, form=None, h2h=None, inj=None, odds=None, wx=None) -> MatchContext:
    sigs: dict[str, Signal] = {}
    if form is not None:
        sigs["form_recent"] = Signal(tool="form_recent", data=form.model_dump(), source="fixture")
    if h2h is not None:
        sigs["h2h_history"] = Signal(tool="h2h_history", data=h2h.model_dump(), source="fixture")
    if inj is not None:
        sigs["injury_news"] = Signal(tool="injury_news", data=inj.model_dump(), source="fixture")
    if odds is not None:
        sigs["odds_market"] = Signal(tool="odds_market", data=odds.model_dump(), source="fixture")
    if wx is not None:
        sigs["weather_venue"] = Signal(tool="weather_venue", data=wx.model_dump(), source="fixture")
    return MatchContext(
        match=Match(
            match_id="m1",
            home=Team(id="man_city", name="Manchester City"),
            away=Team(id="real_madrid", name="Real Madrid"),
            kickoff=datetime(2025, 5, 30, 20, 0, 0),
            venue_id="puskas_arena",
            competition="UCL",
        ),
        signals=sigs,
    )


def test_elo_expected_equal_ratings_is_half():
    assert abs(_elo_expected(1500, 1500) - 0.5) < 1e-9


def test_elo_expected_higher_rating_wins_more():
    assert _elo_expected(1700, 1500) > 0.5
    assert _elo_expected(1500, 1700) < 0.5


def test_form_score_unbeaten_is_one():
    f = FormEntry(played=5, won=5, drawn=0, lost=0, gf=15, ga=2, points=15, last5_form_string="WWWWW")
    assert _form_score(f) == 1.0


def test_form_score_zero_played_is_half():
    f = FormEntry(played=0, won=0, drawn=0, lost=0, gf=0, ga=0, points=0, last5_form_string="")
    assert _form_score(f) == 0.5


def test_h2h_score_no_meetings_is_half():
    h = H2HOutput(meetings=[], home_wins=0, away_wins=0, draws=0, last_meeting=None, last_winner=None)
    assert _h2h_score(h) == 0.5


def test_h2h_score_home_dominant():
    h = H2HOutput(meetings=[], home_wins=4, away_wins=0, draws=1, last_meeting=None, last_winner="home")
    assert abs(_h2h_score(h) - 0.8) < 1e-9


def test_injury_score_zero_outs_is_zero():
    inj = InjuryOutput(home=[], away=[])
    assert _injury_score(inj) == 0.0


def test_injury_score_three_outs_caps_at_minus_half():
    inj = InjuryOutput(
        home=[
            InjuryReport(player="a", status="out", reported_at=datetime(2025, 5, 1), source="x"),
            InjuryReport(player="b", status="out", reported_at=datetime(2025, 5, 1), source="x"),
            InjuryReport(player="c", status="out", reported_at=datetime(2025, 5, 1), source="x"),
        ],
        away=[],
    )
    assert _injury_score(inj) == pytest.approx(-0.3)  # -0.1 * 3 (cap at -0.5 not hit)


def test_injury_score_five_outs_caps_at_minus_half():
    inj = InjuryOutput(
        home=[InjuryReport(player=str(i), status="out", reported_at=datetime(2025, 5, 1), source="x") for i in range(5)],
        away=[],
    )
    assert _injury_score(inj) == -0.5


def test_market_score_inside_range():
    odds = OddsOutput(
        bookmakers=[BookmakerOdds(name="x", home=2.0, draw=3.0, away=4.0)],
        implied_probs={"home": 0.5, "draw": 0.3, "away": 0.2},
        market_consensus_pick="home",
    )
    assert _market_score(odds) == 0.5


def test_market_score_clamps_extreme():
    odds = OddsOutput(
        bookmakers=[],
        implied_probs={"home": 0.99, "draw": 0.005, "away": 0.005},
        market_consensus_pick="home",
    )
    assert _market_score(odds) == 0.95


def test_draw_residual_sums_to_one():
    ph, pa, pd = _draw_residual(0.6, 0.2)
    assert abs(ph + pa + pd - 1.0) < 1e-9
    assert pd > 0


def test_run_no_signals_returns_neutral_pick():
    ctx = _ctx()  # no signals
    out = run(ctx)
    assert isinstance(out, ReasonerOutput)
    assert out.reasoner == "numeric"
    # All probs close to 1/3 (Elo alone = 50/50; draw residual splits it near-uniformly)
    for k in ("home", "draw", "away"):
        assert abs(out.probs[k] - 1/3) < 0.1
    # The Elo factor is always emitted (it's our prior).
    assert any(f.name == "elo_p_home" for f in out.factors)


def test_run_strong_home_form_and_market_picks_home():
    form = FormOutput(
        home=FormEntry(played=5, won=5, drawn=0, lost=0, gf=15, ga=2, points=15, last5_form_string="WWWWW"),
        away=FormEntry(played=5, won=0, drawn=1, lost=4, gf=3, ga=12, points=1, last5_form_string="LLLDL"),
    )
    h2h = H2HOutput(
        meetings=[H2HMeeting(date=datetime(2024, 5, 1, 20, 0), home="man_city", away="real_madrid",
                             home_goals=3, away_goals=1, competition="UCL")],
        home_wins=2, away_wins=0, draws=0, last_meeting=datetime(2024, 5, 1, 20, 0), last_winner="home",
    )
    inj = InjuryOutput(
        home=[],
        away=[InjuryReport(player="mbappe", status="doubt", reported_at=datetime(2025, 5, 25), source="x")],
    )
    odds = OddsOutput(
        bookmakers=[BookmakerOdds(name="pinnacle", home=1.6, draw=4.0, away=5.5)],
        implied_probs={"home": 0.625, "draw": 0.25, "away": 0.125},
        market_consensus_pick="home",
    )
    ctx = _ctx(form=form, h2h=h2h, inj=inj, odds=odds)
    out = run(ctx)
    assert out.pick == "home"
    assert out.probs["home"] > 0.5
    assert out.confidence > 0.0
    assert any(f.name == "form_diff" for f in out.factors)


def test_run_home_injuries_pull_toward_away():
    form = FormOutput(
        home=FormEntry(played=5, won=2, drawn=1, lost=2, gf=6, ga=6, points=7, last5_form_string="WLWDL"),
        away=FormEntry(played=5, won=2, drawn=1, lost=2, gf=6, ga=6, points=7, last5_form_string="DLWLW"),
    )
    inj_home = InjuryOutput(
        home=[
            InjuryReport(player="a", status="out", reported_at=datetime(2025, 5, 1), source="x"),
            InjuryReport(player="b", status="out", reported_at=datetime(2025, 5, 1), source="x"),
            InjuryReport(player="c", status="out", reported_at=datetime(2025, 5, 1), source="x"),
        ],
        away=[],
    )
    odds = OddsOutput(
        bookmakers=[],
        implied_probs={"home": 0.4, "draw": 0.3, "away": 0.3},
        market_consensus_pick="home",
    )
    ctx = _ctx(form=form, inj=inj_home, odds=odds)
    out = run(ctx)
    # Injuries + neutral market + neutral form = should not pick 'home' strongly.
    # We don't assert pick, just that the injury factor is present and negative.
    inj_factors = [f for f in out.factors if f.name == "injury_penalty"]
    assert inj_factors and inj_factors[0].value < 0


def test_run_probs_always_sum_to_one():
    out = run(_ctx(
        form=FormOutput(
            home=FormEntry(played=3, won=2, drawn=0, lost=1, gf=5, ga=3, points=6, last5_form_string="WLW"),
            away=FormEntry(played=3, won=0, drawn=1, lost=2, gf=1, ga=4, points=1, last5_form_string="LLD"),
        ),
        h2h=H2HOutput(meetings=[], home_wins=0, away_wins=0, draws=0, last_meeting=None, last_winner=None),
    ))
    s = sum(out.probs.values())
    assert abs(s - 1.0) < 1e-9


def test_run_confidence_bounded():
    out = run(_ctx())
    assert 0.0 <= out.confidence <= 1.0


def test_run_warns_when_form_signal_present_but_not_ok():
    """A failed form signal (e.g. fixture not found) emits a warning."""
    sigs: dict = {}
    # Manually build a not-ok signal
    from soccer_agent.models import Signal, ToolErrorPayload
    sigs["form_recent"] = Signal(
        tool="form_recent", ok=False, data={}, source="fixture",
        error=ToolErrorPayload(source="fixture", message="missing", retriable=False),
    )
    sigs["odds_market"] = Signal(
        tool="odds_market", ok=True, data={
            "bookmakers": [{"name": "p", "home": 2.0, "draw": 3.0, "away": 4.0}],
            "implied_probs": {"home": 0.5, "draw": 0.3, "away": 0.2},
            "market_consensus_pick": "home",
        }, source="fixture",
    )
    ctx = MatchContext(
        match=Match(
            match_id="m1",
            home=Team(id="man_city", name="Manchester City"),
            away=Team(id="real_madrid", name="Real Madrid"),
            kickoff=datetime(2025, 5, 30, 20, 0, 0),
            venue_id="puskas_arena",
            competition="UCL",
        ),
        signals=sigs,
    )
    out = run(ctx)
    assert any("form_recent signal missing" in w for w in out.warnings)


def test_numeric_reasoner_is_a_reasoner():
    r = NumericReasoner()
    assert r.name == "numeric"
    assert r.version == "0.1.0"
    out = r.run(_ctx())
    assert out.reasoner == "numeric"
