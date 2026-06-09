"""Tests for the prompt-iteration sweep harness (Task 25).

A "sweep" runs the same eval dataset under N prompt candidates and
reports which one scores best. This is the loop that powers
`scripts/prompt_iterate.py`.

These tests use a tiny, deterministic stub LLM so they finish in
milliseconds. The real script uses a live ollama/openai client and is
expected to be slow on a CPU host.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from soccer_agent.eval.dataset import EvalCase
from soccer_agent.eval.prompt_sweep import (
    PromptCandidate,
    SweepResult,
    run_prompt_sweep,
)
from soccer_agent.llm import LLMClient, LLMResult


# ---- a stub LLM that always returns the same response --------------------
class _ConstantClient:
    """Returns a fixed pick/probs/rationale regardless of input."""

    def __init__(self, pick: str = "home", probs: dict[str, float] | None = None,
                 confidence: float = 0.5, rationale: str = "stub"):
        self._pick = pick
        self._probs = probs or {"home": 0.5, "draw": 0.25, "away": 0.25}
        self._confidence = confidence
        self._rationale = rationale
        self.last_system: str = ""
        self.name = "stub_constant"

    def complete(self, system: str, user: str) -> LLMResult:
        self.last_system = system
        return LLMResult(
            raw="<stub>",
            parsed={
                "pick": self._pick,
                "probs": self._probs,
                "confidence": self._confidence,
                "rationale": self._rationale,
            },
            model="stub",
        )


def _make_case(match_id: str, winner: str) -> EvalCase:
    """Build a minimal EvalCase that asserts `winner` from the score."""
    from datetime import datetime, timezone
    if winner == "home":
        hg, ag = 2, 1
    elif winner == "away":
        hg, ag = 1, 2
    else:
        hg, ag = 1, 1
    return EvalCase(
        match_id=match_id,
        competition="EPL", round="regular",
        home_id="team_a", away_id="team_b",
        venue_id="",
        kickoff=datetime(2024, 10, 1, 16, 0, tzinfo=timezone.utc),
        home_goals=hg, away_goals=ag,
    )


# ---- fixtures -----------------------------------------------------------

@pytest.fixture
def tiny_eval():
    """A 4-case mini dataset: 2 home wins, 1 away win, 1 draw."""
    return [
        _make_case("m1", "home"),
        _make_case("m2", "home"),
        _make_case("m3", "away"),
        _make_case("m4", "draw"),
    ]


@pytest.fixture
def candidate_home_picker() -> PromptCandidate:
    return PromptCandidate(name="always_home", system_prompt="always pick home")


@pytest.fixture
def candidate_away_picker() -> PromptCandidate:
    return PromptCandidate(name="always_away", system_prompt="always pick away")


# ---- the actual contract tests -----------------------------------------

def test_prompt_sweep_runs_each_candidate(tiny_eval, tmp_path,
                                            candidate_home_picker,
                                            candidate_away_picker):
    """The sweep must produce a row per (candidate, case)."""
    result = run_prompt_sweep(
        candidates=[candidate_home_picker, candidate_away_picker],
        eval_cases=tiny_eval,
        client_factory=lambda: _ConstantClient(pick="home"),
        db_path=tmp_path / "sweep.db",
    )
    assert isinstance(result, SweepResult)
    assert len(result.per_candidate) == 2


def test_prompt_sweep_picks_best_candidate_by_accuracy(
        tiny_eval, tmp_path,
        candidate_home_picker, candidate_away_picker):
    """When two candidates produce different accuracies, the higher one wins.

    On this dataset (2 home, 1 away, 1 draw):
      - home-picker gets 2/4 (the two home wins)
      - away-picker gets 1/4 (the one away win)
    The test passes a per-candidate client factory so each prompt gets
    a client that follows its instruction.
    """
    def factory_for(candidate: PromptCandidate):
        if candidate.name == "always_home":
            return _ConstantClient(pick="home", probs={"home": 0.8, "draw": 0.1, "away": 0.1})
        if candidate.name == "always_away":
            return _ConstantClient(pick="away", probs={"home": 0.1, "draw": 0.1, "away": 0.8})
        raise AssertionError(f"unknown candidate {candidate.name}")

    def client_factory():
        # The default factory is unused because run_prompt_sweep()
        # builds clients per candidate via factory_for(). But we still
        # need to pass something callable.
        return _ConstantClient(pick="home")

    # Use a richer per-candidate factory by monkey-patching the call.
    # Easier: rewrite the sweep to take a per-candidate factory. The
    # simpler path for this test is to vary the candidate names so we
    # can drive the right client.
    result = _run_sweep_with_per_candidate_clients(
        candidates=[candidate_home_picker, candidate_away_picker],
        eval_cases=tiny_eval,
        per_candidate_client={
            "always_home": _ConstantClient(pick="home", probs={"home": 0.8, "draw": 0.1, "away": 0.1}),
            "always_away": _ConstantClient(pick="away", probs={"home": 0.1, "draw": 0.1, "away": 0.8}),
        },
        db_path=tmp_path / "sweep.db",
    )
    assert result.best.name == "always_home", (
        f"expected always_home to win; got {result.best.name}; "
        f"per-candidate: {result.per_candidate}"
    )
    assert result.per_candidate["always_home"]["accuracy"] == 0.5
    assert result.per_candidate["always_away"]["accuracy"] == 0.25


def _run_sweep_with_per_candidate_clients(
    candidates, eval_cases, per_candidate_client, db_path,
):
    """Helper: run the sweep giving each candidate its own client instance.

    Wraps `run_prompt_sweep` with a tiny shim. Kept here so the contract
    test stays self-contained.
    """
    from soccer_agent.eval.prompt_sweep import (
        PromptCandidate, run_prompt_sweep,
    )
    # Build a wrapper that returns the per-candidate client the first
    # time it's asked for that name, and a new copy of the same client
    # on subsequent calls.
    used = set()
    def factory():
        # Find the first candidate not yet satisfied.
        for c in candidates:
            if c.name not in used:
                used.add(c.name)
                return per_candidate_client[c.name]
        # Fallback for any further call (e.g. tests that don't need this).
        return per_candidate_client[candidates[0].name]
    return run_prompt_sweep(
        candidates=candidates,
        eval_cases=eval_cases,
        client_factory=factory,
        db_path=db_path,
    )


def test_prompt_sweep_writes_a_summary_file(tiny_eval, tmp_path,
                                              candidate_home_picker):
    """The script must persist a run summary so we can compare sweeps."""
    out = tmp_path / "summary.json"
    run_prompt_sweep(
        candidates=[candidate_home_picker],
        eval_cases=tiny_eval,
        client_factory=lambda: _ConstantClient(pick="home"),
        db_path=tmp_path / "sweep.db",
        output=out,
    )
    assert out.exists()
    import json as jsonlib
    data = jsonlib.loads(out.read_text())
    assert "per_candidate" in data
    assert "best" in data
    assert data["best"] == "always_home"


def test_prompt_sweep_handles_empty_candidate_list(tiny_eval, tmp_path):
    """Degenerate case: zero candidates returns an empty SweepResult."""
    result = run_prompt_sweep(
        candidates=[],
        eval_cases=tiny_eval,
        client_factory=lambda: _ConstantClient(pick="home"),
        db_path=tmp_path / "sweep.db",
    )
    assert result.per_candidate == {}
    assert result.best is None


def test_prompt_sweep_persists_per_candidate_predictions(
        tiny_eval, tmp_path, candidate_home_picker):
    """Each candidate's predictions live in its own per-candidate DB
    (sweep__<candidate_name>.db) so candidates don't clobber each other."""
    run_prompt_sweep(
        candidates=[candidate_home_picker],
        eval_cases=tiny_eval,
        client_factory=lambda: _ConstantClient(pick="home"),
        db_path=tmp_path / "sweep.db",
    )
    cand_db = tmp_path / "sweep__always_home.db"
    assert cand_db.exists(), f"per-candidate DB not created: {cand_db}"
    from soccer_agent.db import Database
    rows = Database(str(cand_db)).list_predictions(limit=1000)
    assert len(rows) >= len(tiny_eval)
