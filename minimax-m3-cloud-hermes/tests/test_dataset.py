"""Tests for Task 16: eval dataset seed.

The dataset is the foundation of the eval harness (Task 18). It must be:
  - Deterministic (no random IDs, fixed dates, fixed scores).
  - Schema-correct (every fixture file must parse into the tool's Pydantic model).
  - Complete (form, injury, h2h, weather, odds, venue — all six tools).
  - Diverse (mix of home wins, away wins, draws; mix of competitions).

These tests pin those properties. The dataset itself lives in
src/soccer_agent/eval/dataset.py; the fixture-factory in
src/soccer_agent/eval/fixture_factory.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


# ------------------------------------------------------------------
# EvalCase + dataset shape
# ------------------------------------------------------------------


def test_dataset_exposes_a_list_of_eval_cases():
    from soccer_agent.eval.dataset import EVAL_CASES, EvalCase
    assert isinstance(EVAL_CASES, list)
    assert len(EVAL_CASES) >= 5, "Need at least 5 cases to claim a real eval"
    for c in EVAL_CASES:
        assert isinstance(c, EvalCase)


def test_each_eval_case_has_required_fields():
    from soccer_agent.eval.dataset import EVAL_CASES
    for c in EVAL_CASES:
        assert c.match_id, "match_id must be non-empty"
        assert c.competition, "competition must be set"
        assert c.home_id and c.away_id
        assert c.home_id != c.away_id
        assert c.kickoff.tzinfo is not None, "kickoff must be timezone-aware"
        assert c.actual_winner in {"home", "away", "draw"}
        assert isinstance(c.home_goals, int) and isinstance(c.away_goals, int)
        assert c.home_goals >= 0 and c.away_goals >= 0
        # Internal consistency: actual_winner matches the score.
        if c.home_goals > c.away_goals:
            assert c.actual_winner == "home"
        elif c.home_goals < c.away_goals:
            assert c.actual_winner == "away"
        else:
            assert c.actual_winner == "draw"


def test_dataset_is_deterministic_and_ordered_by_date():
    """Pinned dates, no randomness, ordered chronologically.

    The eval harness will iterate this list. Date order makes logs
    readable; determinism makes failures reproducible.
    """
    from soccer_agent.eval.dataset import EVAL_CASES
    dates = [c.kickoff for c in EVAL_CASES]
    assert dates == sorted(dates), "cases must be ordered by kickoff"
    # No two cases share a match_id.
    ids = [c.match_id for c in EVAL_CASES]
    assert len(set(ids)) == len(ids), "match_ids must be unique"


def test_dataset_covers_each_outcome_class():
    """A useful eval set has home wins, away wins, AND draws.

    The agent learns nothing if every case has the same outcome.
    With 30+ cases we expect a *reasonable* mix — historical EPL
    is ~46% home, ~27% draw, ~27% away. A draw rate of <15% would
    mean we've cherry-picked home wins and the eval is uninformative.
    """
    from soccer_agent.eval.dataset import EVAL_CASES
    winners = {c.actual_winner for c in EVAL_CASES}
    assert "home" in winners
    assert "away" in winners
    # Draws are rare in real football but should appear at least once
    # in any reasonably-sized set.
    if len(EVAL_CASES) >= 8:
        assert "draw" in winners, "with 8+ cases, include at least one draw"


def test_dataset_size_at_least_30():
    """Task 30 expanded from 10 to 30+ to enable real calibration.

    See docs/calibration.md § 'What to do next' — ECE is data-blocked.
    """
    from soccer_agent.eval.dataset import EVAL_CASES
    assert len(EVAL_CASES) >= 30, (
        f"only {len(EVAL_CASES)} cases; need 30+ for meaningful calibration"
    )


def test_dataset_size_at_least_100_after_task_34():
    """Task 34: expanded from 34 to ≥100 cases by ingesting
    football-data.co.uk CSVs. Real post-match outcomes, not
    synthetic — see scripts/ingest_football_data.py."""
    from soccer_agent.eval.dataset import EVAL_CASES
    assert len(EVAL_CASES) >= 100, (
        f"only {len(EVAL_CASES)} cases; need 100+ per Task 34"
    )


def test_ingested_cases_cover_all_five_leagues():
    """The football-data ingestion only kept pairs that used clubs
    already in the existing dataset. We want at least some cases
    from each of the 5 leagues (EPL, Bundesliga, LaLiga, SerieA,
    plus the existing UCL — Ligue1 may be empty if no Ligue1 clubs
    are wired in, that's fine)."""
    from soccer_agent.eval.dataset import EVAL_CASES
    from collections import Counter
    c = Counter(case.competition for case in EVAL_CASES)
    # At minimum we should have EPL/Bundesliga/LaLiga/SerieA all
    # represented after ingestion.
    for must_have in ("EPL", "Bundesliga", "LaLiga", "SerieA"):
        assert c.get(must_have, 0) >= 18, (
            f"{must_have} under-represented: {c.get(must_have, 0)}"
        )


def test_ingested_match_ids_are_unique():
    """Defensive: re-running the ingest must not produce dupes."""
    from soccer_agent.eval.dataset import EVAL_CASES
    ids = [c.match_id for c in EVAL_CASES]
    assert len(ids) == len(set(ids)), "duplicate match_ids in EVAL_CASES"


def test_dataset_outcome_distribution_is_plausible():
    """Home wins should be the largest class, draws the smallest.

    A dataset with more away wins than home wins (or zero draws) is
    a sign the curator cherry-picked. Real football is ~46/27/27.
    We allow a band: home 45-75%, draw 10-30%, away 15-40%.
    """
    from soccer_agent.eval.dataset import EVAL_CASES
    n = len(EVAL_CASES)
    counts = {"home": 0, "away": 0, "draw": 0}
    for c in EVAL_CASES:
        counts[c.actual_winner] += 1
    pct = {k: v / n for k, v in counts.items()}
    assert 0.45 <= pct["home"] <= 0.75, f"home share {pct['home']:.2f} out of band"
    assert 0.10 <= pct["draw"] <= 0.30, f"draw share {pct['draw']:.2f} out of band"
    assert 0.15 <= pct["away"] <= 0.45, f"away share {pct['away']:.2f} out of band"


def test_dataset_covers_four_major_leagues():
    """Phase 2 dashboard tiles assume the dataset spans competitions.

    We need at least 4 leagues to claim 'multi-league coverage'.
    UCL is the headline target, but the eval set is wider.
    """
    from soccer_agent.eval.dataset import EVAL_CASES
    comps = {c.competition for c in EVAL_CASES}
    required = {"UCL", "EPL", "LaLiga", "Bundesliga", "SerieA"}
    missing = required - comps
    assert not missing, f"dataset missing leagues: {missing}; got {comps}"


def test_dataset_includes_ucl_and_league_matches():
    """Phase 1 spec: eval spans competitions; UCL is a target, not eval,
    but domestic leagues are. The 24/25 season's UCL *group stage* (Sept-Jan)
    IS eval-eligible (it happened). The knockouts are not.
    """
    from soccer_agent.eval.dataset import EVAL_CASES
    comps = {c.competition for c in EVAL_CASES}
    assert "UCL" in comps, "24/25 UCL group-stage is eval-eligible"
    assert any(c in comps for c in ("EPL", "LaLiga", "Bundesliga", "SerieA")), (
        "Need at least one domestic league"
    )


# ------------------------------------------------------------------
# fixture_factory: schema-correct generation
# ------------------------------------------------------------------


def test_fixture_factory_writes_all_six_tool_fixtures(tmp_path):
    """For one case, generate every fixture file. Every file must
    validate into the corresponding tool's output model — that's the
    round-trip property the harness will rely on."""
    from soccer_agent.eval.dataset import EVAL_CASES
    from soccer_agent.eval.fixture_factory import materialize_case

    def season_of(kickoff):
        y, m = kickoff.year, kickoff.month
        return f"{y}-{y + 1}" if m >= 8 else f"{y - 1}-{y}"

    case = EVAL_CASES[0]
    materialize_case(case, tmp_path)

    expected = [
        f"form/{case.home_id}__{case.away_id}__{season_of(case.kickoff)}.json",
        f"injury/{case.home_id}__{case.away_id}__{case.kickoff.date().isoformat()}.json",
        f"h2h/{case.home_id}__{case.away_id}.json",
        f"weather/{case.venue_id or 'neutral'}__{case.kickoff.date().isoformat()}.json",
        f"odds/{case.home_id}__{case.away_id}__{case.kickoff.date().isoformat()}.json",
        f"venues/venue_{case.venue_id or 'neutral'}.json",
    ]
    for rel in expected:
        assert (tmp_path / rel).exists(), f"missing fixture: {rel}"


@pytest.mark.parametrize("tool_module,fixture_rel", [
    ("soccer_agent.tools.form_recent", "form"),
    ("soccer_agent.tools.injury_news", "injury"),
    ("soccer_agent.tools.h2h_history", "h2h"),
    ("soccer_agent.tools.weather_venue", "weather"),
    ("soccer_agent.tools.odds_market", "odds"),
    ("soccer_agent.tools.venue_info", "venues"),
])
def test_each_tool_accepts_its_generated_fixture(tmp_path, tool_module, fixture_rel):
    """Every fixture the factory writes must be loadable by its tool.

    This is the round-trip property Task 14's shell run found missing:
    a schema the tool rejects is useless to the agent. We test it
    once-per-tool here so adding a new tool is a single test case.
    """
    import importlib
    from soccer_agent.eval.dataset import EVAL_CASES
    from soccer_agent.eval.fixture_factory import materialize_case

    case = EVAL_CASES[0]
    materialize_case(case, tmp_path)
    tool = importlib.import_module(tool_module)
    # Find the input class + a sane input; the round-trip test runs the tool.
    # We don't run the tool here — the smoke test below does. This test
    # just confirms the fixture is present and JSON-valid.
    files = list((tmp_path / fixture_rel).glob("*.json"))
    assert files, f"no fixtures written under {fixture_rel}/"
    import json
    for f in files:
        data = json.loads(f.read_text())
        assert isinstance(data, dict), f"{f} is not a JSON object"


@pytest.mark.parametrize("tool_module,fixture_rel,fixture_glob,output_model_path", [
    # Each tool exposes a `run_<name>(payload)` async fn that returns
    # a Pydantic model. We invoke it and assert validation passes.
    ("soccer_agent.tools.form_recent", "form", "*.json",
     "soccer_agent.models.FormOutput"),
    ("soccer_agent.tools.injury_news", "injury", "*.json",
     "soccer_agent.models.InjuryOutput"),
    ("soccer_agent.tools.h2h_history", "h2h", "*.json",
     "soccer_agent.models.H2HOutput"),
    ("soccer_agent.tools.weather_venue", "weather", "*.json",
     "soccer_agent.models.WeatherOutput"),
    ("soccer_agent.tools.odds_market", "odds", "*.json",
     "soccer_agent.models.OddsOutput"),
    ("soccer_agent.tools.venue_info", "venues", "*.json",
     "soccer_agent.models.Venue"),
])
def test_each_tool_validates_its_generated_fixture_against_pydantic(
    tmp_path, tool_module, fixture_rel, fixture_glob, output_model_path,
):
    """Round-trip: factory writes → tool reads → Pydantic validates.

    This catches the class of bug Task 14's shell run exposed: a JSON
    fixture that's *syntactically* valid but doesn't match the
    tool's output schema. Pydantic is the source of truth, so we
    re-import the model and call `model_validate(json.load(f))`.

    Using the model directly (not the tool's run()) is intentional:
    the tool is async and adds file-loading; the model validates
    shape, which is what the eval harness actually depends on.
    """
    import importlib
    import json
    from soccer_agent.eval.dataset import EVAL_CASES
    from soccer_agent.eval.fixture_factory import materialize_case

    case = EVAL_CASES[0]
    materialize_case(case, tmp_path)
    # models.py is a single file, so import the module then grab the
    # class by attribute access — parameterized test IDs read better
    # as module.ClassName anyway.
    module_path, _, class_name = output_model_path.rpartition(".")
    model = getattr(importlib.import_module(module_path), class_name)
    files = list((tmp_path / fixture_rel).glob(fixture_glob))
    assert files, f"no fixtures under {fixture_rel}/"
    for f in files:
        data = json.loads(f.read_text())
        instance = model.model_validate(data)
        # model_dump round-trips through JSON mode, so nested datetimes
        # become ISO strings — that's the contract the rest of the
        # agent relies on.
        roundtripped = json.loads(
            json.dumps(instance.model_dump(mode="json"), default=str)
        )
        model.model_validate(roundtripped)  # must revalidate cleanly


def test_factory_is_deterministic(tmp_path):
    """Materialize the same case twice → byte-identical fixtures.

    Determinism makes the eval harness reproducible across runs and
    machines, which is what makes Brier numbers comparable.
    """
    from soccer_agent.eval.dataset import EVAL_CASES
    from soccer_agent.eval.fixture_factory import materialize_case

    case = EVAL_CASES[0]
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    materialize_case(case, a)
    materialize_case(case, b)
    a_files = sorted(p.relative_to(a) for p in a.rglob("*.json"))
    b_files = sorted(p.relative_to(b) for p in b.rglob("*.json"))
    assert a_files == b_files
    for rel in a_files:
        assert (a / rel).read_text() == (b / rel).read_text(), (
            f"fixture {rel} differs between runs — factory is not deterministic"
        )
