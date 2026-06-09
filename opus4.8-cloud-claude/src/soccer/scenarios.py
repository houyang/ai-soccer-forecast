from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from soccer.harness import Scenario
from soccer.models import MatchRef, MatchResult
from soccer.registry import build_fixture_registry

_FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"

_KICK = datetime(2026, 7, 19, 19, 0, tzinfo=UTC)


def _ref(
    match_id: str,
    competition: str,
    home: str,
    away: str,
    venue_id: str,
    season: str,
    kickoff: datetime,
) -> MatchRef:
    return MatchRef(
        id=match_id,
        competition=competition,
        home=home,
        away=away,
        kickoff=kickoff,
        venue_id=venue_id,
        season=season,
    )


_SCENARIO_MATCHES: dict[str, list[MatchRef]] = {
    "wc-2026-final": [
        _ref("wc-final", "FIFA World Cup", "France", "Brazil", "metlife", "2026", _KICK),
    ],
    "ucl-2025-26": [
        _ref(
            "ucl-1",
            "UEFA Champions League",
            "Real Madrid",
            "Manchester City",
            "bernabeu",
            "2025-26",
            datetime(2026, 2, 18, 20, 0, tzinfo=UTC),
        ),
        _ref(
            "ucl-2",
            "UEFA Champions League",
            "Bayern Munich",
            "Arsenal",
            "allianz",
            "2025-26",
            datetime(2026, 2, 25, 20, 0, tzinfo=UTC),
        ),
        _ref(
            "ucl-3",
            "UEFA Champions League",
            "Inter",
            "PSG",
            "giuseppe-meazza",
            "2025-26",
            datetime(2026, 3, 4, 20, 0, tzinfo=UTC),
        ),
    ],
}

_SCENARIO_RESULTS: dict[str, dict[str, MatchResult]] = {
    "wc-2026-final": {
        "wc-final": MatchResult(
            match_id="wc-final", home_goals=2, away_goals=1, status="finished", source="fixture"
        ),
    },
    "ucl-2025-26": {
        "ucl-1": MatchResult(
            match_id="ucl-1", home_goals=3, away_goals=1, status="finished", source="fixture"
        ),
        "ucl-2": MatchResult(
            match_id="ucl-2", home_goals=1, away_goals=1, status="finished", source="fixture"
        ),
        "ucl-3": MatchResult(
            match_id="ucl-3", home_goals=0, away_goals=2, status="finished", source="fixture"
        ),
    },
}

SCENARIO_NAMES: tuple[str, ...] = ("ucl-2025-26", "wc-2026-final")


def load_scenario(name: str) -> Scenario:
    if name not in _SCENARIO_MATCHES:
        raise KeyError(f"unknown scenario: {name}")
    fixture_path = _FIXTURE_DIR / f"{name}.json"
    registry = build_fixture_registry(fixture_path)
    return Scenario(
        name=name,
        registry=registry,
        matches=_SCENARIO_MATCHES[name],
        results=_SCENARIO_RESULTS[name],
    )
