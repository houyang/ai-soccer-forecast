from __future__ import annotations

from dataclasses import replace

from soccer.worldcup.adjust import (
    CAP_TOTAL,
    TeamAdjustment,
    compute_adjustments,
    parse_formation,
)
from soccer.worldcup.entities import Lineup, WorldCup
from soccer.worldcup.ranking import rank_all


def test_parse_formation() -> None:
    assert parse_formation("4-3-3") == (4, 3)
    assert parse_formation("5-4-1") == (5, 1)
    assert parse_formation("") is None
    assert parse_formation("nonsense") is None


def test_no_played_match_means_no_adjustment(sample_world_cup: WorldCup) -> None:
    adj = compute_adjustments(sample_world_cup, rank_all(sample_world_cup))
    assert adj == {}


def _play(wc: WorldCup, home_goals: int, away_goals: int) -> WorldCup:
    match = replace(wc.matches[0], home_goals=home_goals, away_goals=away_goals)
    return replace(wc, matches=(match,))


def test_overperformance_gives_positive_momentum(sample_world_cup: WorldCup) -> None:
    # England (home, the favourite) thrashes Mexico beyond expectation -> positive momentum.
    wc = _play(sample_world_cup, 5, 0)
    adj = compute_adjustments(wc, rank_all(wc))
    assert adj[1].momentum > 0
    assert adj[2].momentum < 0  # Mexico under-performed


def test_rating_delta_is_capped(sample_world_cup: WorldCup) -> None:
    wc = _play(sample_world_cup, 8, 0)
    adj = compute_adjustments(wc, rank_all(wc))
    assert abs(adj[1].rating_delta) <= CAP_TOTAL + 1e-9


def test_formation_lean_from_lineup(sample_world_cup: WorldCup) -> None:
    wc = _play(sample_world_cup, 1, 0)
    wc = replace(wc, lineups=(Lineup(9001, 1, "3-4-3", (1, 2), ()),))
    adj = compute_adjustments(wc, rank_all(wc))
    # 3 at the back -> negative defensive lean (less solid); 3 up top -> neutral attack.
    assert adj[1].defense_lean < 0
    assert adj[1].attack_lean == 0.0


def test_team_adjustment_defaults_to_zero() -> None:
    assert TeamAdjustment() == TeamAdjustment(0.0, 0.0, 0.0, 0.0, 0.0)
