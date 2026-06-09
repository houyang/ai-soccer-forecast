"""Tests for the Elo wiring in PredictionAgent (Task 27).

The agent now accepts an `elo_state_path` and injects the loaded
state into the MatchContext so the numeric reasoner uses real
per-team ratings.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from soccer_agent.agent import PredictionAgent
from soccer_agent.db import init_db
from soccer_agent.elo import EloState
from soccer_agent.reasoners.llm import LLMReasoner
from soccer_agent.reasoners.numeric import NumericReasoner
from soccer_agent.tools import default_registry


def _write_elo_state(path: Path, state: EloState) -> None:
    state.to_json(path)


def test_agent_loads_elo_state_from_path(tmp_path):
    """When given a path to a pre-computed state, the agent should
    use it in the context for the numeric reasoner."""
    db = tmp_path / "agent.db"
    init_db(db)
    elo = EloState(home_advantage=0.0)
    elo.ensure("man_city")
    elo.ensure("real_madrid")
    # Man City utterly dominant at home
    elo.ratings["man_city"].home = 1800.0
    elo.ratings["real_madrid"].away = 1500.0
    elo_path = tmp_path / "elo.json"
    _write_elo_state(elo_path, elo)

    agent = PredictionAgent(
        registry=default_registry(),
        # Use the numeric reasoner directly so we can read its output
        reasoner=NumericReasoner(),
        db_path=db,
        elo_state_path=elo_path,
    )
    # Sanity: the agent holds the state
    assert agent.elo_state.ratings["man_city"].home == 1800.0


def test_agent_falls_back_to_fresh_state(tmp_path):
    """No path provided and no env var → fresh empty state, not a
    crash."""
    db = tmp_path / "agent.db"
    init_db(db)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=NumericReasoner(),
        db_path=db,
    )
    # Fresh state: 0 ratings
    assert len(agent.elo_state.ratings) == 0
    # K-factor is the default
    assert agent.elo_state.k == EloState().k


def test_agent_uses_env_var_for_elo_path(tmp_path, monkeypatch):
    """SOCCER_AGENT_ELO_STATE is honored when no explicit path is
    given."""
    elo = EloState()
    elo.ensure("a")
    elo.ratings["a"].home = 1900.0
    elo_path = tmp_path / "from_env.json"
    _write_elo_state(elo_path, elo)
    monkeypatch.setenv("SOCCER_AGENT_ELO_STATE", str(elo_path))

    db = tmp_path / "agent.db"
    init_db(db)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=NumericReasoner(),
        db_path=db,
    )
    assert agent.elo_state.ratings["a"].home == 1900.0


def test_agent_predict_uses_elo_state_in_context(tmp_path):
    """End-to-end: a strong home-Elo team should tilt the numeric
    reasoner's pick. We use NumericReasoner (no LLM) so we can read
    the deterministic output."""
    from datetime import datetime
    from soccer_agent.models import Match, Team

    db = tmp_path / "agent.db"
    init_db(db)
    elo = EloState(home_advantage=0.0)
    # Make man_city look unbeatable at home
    elo.ensure("man_city")
    elo.ensure("real_madrid")
    elo.ratings["man_city"].home = 1900.0
    elo.ratings["man_city"].away = 1500.0
    elo.ratings["real_madrid"].home = 1500.0
    elo.ratings["real_madrid"].away = 1500.0
    elo_path = tmp_path / "elo.json"
    _write_elo_state(elo_path, elo)

    match = Match(
        match_id="ucl-25-final",
        competition="UCL",
        kickoff=datetime(2025, 5, 30, 20, 0, 0),
        home=Team(id="man_city", name="Manchester City"),
        away=Team(id="real_madrid", name="Real Madrid"),
        venue_id="puskas_arena",
    )

    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=NumericReasoner(),
        secondary_reasoner=None,  # disable blending
        db_path=db,
        elo_state_path=elo_path,
    )
    pred = asyncio.run(agent.predict(match))
    # With a 400-elo home-rating gap (1900 vs 1500) the agent should
    # confidently pick man_city (home).
    assert pred.pick == "home"
    assert pred.final_probs["home"] > pred.final_probs["away"]
