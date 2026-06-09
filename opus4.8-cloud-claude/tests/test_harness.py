import json
from datetime import UTC, datetime
from pathlib import Path

from soccer.agent import PredictionAgent
from soccer.harness import Scenario, run_scenario
from soccer.models import MatchRef, MatchResult
from soccer.reasoning.fake import DeterministicReasoner
from soccer.registry import ToolRegistry, build_fixture_registry

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=UTC)
NOW = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)


def _ref(mid: str) -> MatchRef:
    return MatchRef(
        id=mid,
        competition="UCL",
        home="A",
        away="B",
        kickoff=KICK,
        venue_id="v1",
        season="2025-26",
    )


def _registry(tmp_path: Path) -> ToolRegistry:
    payload = {
        "form": {
            "A": {
                "team": "A",
                "last_n": ["W", "W"],
                "gf": 6,
                "ga": 1,
                "points": 18,
                "streak": "W2",
            },
            "B": {"team": "B", "last_n": ["L"], "gf": 1, "ga": 4, "points": 3, "streak": "L1"},
        },
        "injuries": {
            "A": {"team": "A", "out": [], "doubtful": []},
            "B": {"team": "B", "out": [], "doubtful": []},
        },
        "h2h": {
            "A|B": {
                "home": "A",
                "away": "B",
                "meetings": [],
                "home_wins": 1,
                "draws": 0,
                "away_wins": 0,
            }
        },
        "weather": {
            "v1": {
                "venue_id": "v1",
                "temp_c": 9.0,
                "wind_kph": 4.0,
                "precip_mm": 0.0,
                "condition": "clear",
            }
        },
        "venue": {
            "v1": {
                "venue_id": "v1",
                "name": "S",
                "city": "C",
                "surface": "grass",
                "capacity": 1000,
                "altitude_m": 0,
                "home_advantage_hint": 0.1,
            }
        },
        "odds": {
            "m1": {"bookmaker": "b", "home": 1.7, "draw": 3.8, "away": 5.0},
            "m2": {"bookmaker": "b", "home": 1.7, "draw": 3.8, "away": 5.0},
        },
        "results": {},
    }
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    return build_fixture_registry(path)


def test_run_scenario_reports_metrics(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    agent = PredictionAgent(registry=registry, reasoner=DeterministicReasoner(), clock=lambda: NOW)
    scenario = Scenario(
        name="t",
        registry=registry,
        matches=[_ref("m1"), _ref("m2")],
        results={
            "m1": MatchResult(
                match_id="m1", home_goals=2, away_goals=0, status="finished", source="fixture"
            ),
            "m2": MatchResult(
                match_id="m2", home_goals=0, away_goals=1, status="finished", source="fixture"
            ),
        },
    )
    report = run_scenario(scenario, agent)
    assert report.n == 2
    assert 0.0 <= report.accuracy <= 1.0
    assert report.accuracy == 0.5  # m1 home hit, m2 away miss
    assert report.mean_brier > 0
    assert report.market_baseline.mean_log_loss > 0
    assert len(report.per_match) == 2
    # edge = our log loss minus market log loss
    assert report.edge_vs_market == (report.mean_log_loss - report.market_baseline.mean_log_loss)
