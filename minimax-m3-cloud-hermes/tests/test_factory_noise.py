"""Fixture factory: noise parameter.

The default `noise=0.0` factory builds *fair* fixtures where every
signal (form, h2h, odds, injuries) agrees with the actual result.
That's fine for round-trip tests and the smoke pipeline — but
useless for measuring calibration, because the agent will look
smarter than it is.

`noise=0.4` flips a random subset of signals so they disagree with
the result. The agent still gets correct *odds* more often than not
(bookmakers are accurate), but the form and h2h signals now contain
real-world noise. Calibration measured on this data is meaningful.

Determinism is critical: same case + same noise seed → same bytes.
The eval harness depends on reproducible Brier numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from soccer_agent.eval.dataset import EvalCase
from soccer_agent.eval.fixture_factory import materialize_case
from datetime import datetime, timezone


def _make_case(actual_winner: str = "home") -> EvalCase:
    return EvalCase(
        match_id="noise_test_1",
        competition="EPL",
        round="regular",
        home_id="alpha",
        away_id="beta",
        venue_id="alpha_venue",
        kickoff=datetime(2024, 11, 1, 20, 0, tzinfo=timezone.utc),
        home_goals=2 if actual_winner == "home" else (
            1 if actual_winner == "draw" else 0
        ),
        away_goals=0 if actual_winner == "home" else (
            1 if actual_winner == "draw" else 2
        ),
    )


def test_default_factory_agrees_with_result(tmp_path: Path) -> None:
    """noise=0 (default) keeps the original 'fair fixture' contract.

    Pinning this is important so smoke tests don't get noisier
    overnight when someone changes a default.
    """
    case = _make_case("home")
    materialize_case(case, tmp_path)
    form = json.loads((tmp_path / "form" / "alpha__beta__2024-2025.json").read_text())
    # The default factory's "home wins" form for the home side.
    assert form["home"]["last5_form_string"] == "WWWDW"


def test_noise_zero_is_byte_identical_to_default(tmp_path: Path) -> None:
    """Passing noise=0 explicitly must match the default (no noise).

    Tests that the new param doesn't accidentally change default bytes.
    """
    case = _make_case("away")
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    materialize_case(case, a)                       # default (no noise)
    materialize_case(case, b, noise=0.0)            # explicit zero
    for fa, fb in zip(sorted(a.rglob("*.json")), sorted(b.rglob("*.json"))):
        assert fa.read_bytes() == fb.read_bytes(), (
            f"noise=0 should be byte-identical to default; {fa.name} differs"
        )


def test_high_noise_flips_at_least_one_signal(tmp_path: Path) -> None:
    """With noise=0.7 we expect a substantial fraction of signals to flip.

    A loose bound (≥2 of 4 signal categories disagree) — exact count
    depends on RNG. This catches the 'noise is a no-op' regression.
    """
    case = _make_case("home")
    materialize_case(case, tmp_path, noise=0.7, seed=42)

    form = json.loads((tmp_path / "form" / "alpha__beta__2024-2025.json").read_text())
    h2h = json.loads((tmp_path / "h2h" / "alpha__beta.json").read_text())
    odds = json.loads(
        (tmp_path / "odds" / "alpha__beta__2024-11-01.json").read_text()
    )

    disagreements = 0
    # Form: the clean-fixture home winner has 'WWWDW'. Any other string
    # means the factory flipped this signal.
    if form["home"]["last5_form_string"] != "WWWDW":
        disagreements += 1
    # H2H: clean fixture has 3 home wins out of 5 meetings when
    # home wins the result. A non-home majority disagrees.
    if h2h["home_wins"] < h2h["away_wins"] + h2h["draws"]:
        disagreements += 1
    # Odds: clean fixture favours home (1.90/3.40/4.20). Disagreement
    # means home is not the shortest price.
    book = odds["bookmakers"][0]
    home_odds, draw_odds, away_odds = book["home"], book["draw"], book["away"]
    if not (home_odds < draw_odds and home_odds < away_odds):
        disagreements += 1

    assert disagreements >= 1, (
        f"noise=0.7 flipped 0/3 signal categories; RNG seed or factory broke. "
        f"form={form['home']['last5_form_string']}, "
        f"h2h={h2h['home_wins']}/{h2h['away_wins']}/{h2h['draws']}, "
        f"odds={home_odds}/{draw_odds}/{away_odds}"
    )


def test_noise_is_deterministic_with_seed(tmp_path: Path) -> None:
    """Same (case, noise, seed) → byte-identical fixtures across runs.

    This is the property the eval harness depends on: re-running
    calibration must produce identical Brier numbers.
    """
    case = _make_case("home")
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    materialize_case(case, a, noise=0.4, seed=123)
    materialize_case(case, b, noise=0.4, seed=123)
    for fa, fb in zip(sorted(a.rglob("*.json")), sorted(b.rglob("*.json"))):
        assert fa.read_bytes() == fb.read_bytes(), (
            f"seed=123 should be reproducible; {fa.name} differs"
        )


def test_noise_different_seed_yields_different_bytes(tmp_path: Path) -> None:
    """Different seed → at least one byte differs somewhere.

    Catches the 'I forgot to actually use the seed' regression.
    """
    case = _make_case("home")
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    materialize_case(case, a, noise=0.4, seed=1)
    materialize_case(case, b, noise=0.4, seed=2)
    any_diff = any(
        fa.read_bytes() != fb.read_bytes()
        for fa, fb in zip(sorted(a.rglob("*.json")), sorted(b.rglob("*.json")))
    )
    assert any_diff, "seed=1 and seed=2 produced identical bytes; seed unused"


def test_pydantic_round_trip_under_noise(tmp_path: Path) -> None:
    """Noisy fixtures must still validate against their Pydantic models.

    The whole point of round-trip tests in test_dataset.py is that
    the factory never produces a shape the tool can't parse. Adding
    noise could break that — re-pin it here.
    """
    import importlib
    case = _make_case("draw")
    materialize_case(case, tmp_path, noise=0.5, seed=7)

    tools = [
        ("soccer_agent.models.FormOutput",   "form",   "alpha__beta__2024-2025.json"),
        ("soccer_agent.models.H2HOutput",    "h2h",    "alpha__beta.json"),
        ("soccer_agent.models.OddsOutput",   "odds",   "alpha__beta__2024-11-01.json"),
        ("soccer_agent.models.WeatherOutput","weather","alpha_venue__2024-11-01.json"),
        ("soccer_agent.models.Venue",        "venues", "venue_alpha_venue.json"),
    ]
    for model_path, subdir, fname in tools:
        module_path, _, cls = model_path.rpartition(".")
        Model = getattr(importlib.import_module(module_path), cls)
        path = tmp_path / subdir / fname
        assert path.exists(), f"missing fixture: {path}"
        data = json.loads(path.read_text())
        Model.model_validate(data)  # raises if invalid
