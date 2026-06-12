from __future__ import annotations

from datetime import UTC, datetime

import pytest

from soccer.worldcup.entities import (
    Club,
    Coach,
    League,
    NationalTeam,
    Player,
    WcMatch,
    WorldCup,
)


def _player(pid: int, team: int, club: int, goals: int, rating: float, pos: str) -> Player:
    return Player(
        id=pid,
        name=f"P{pid}",
        age=27,
        position=pos,
        club_id=club,
        goals=goals,
        rating=rating,
        appearances=30,
        wc_team_id=team,
    )


@pytest.fixture
def sample_world_cup() -> WorldCup:
    """Two teams in one group, a strong side and a weak side, with full sub-entities."""
    leagues = {
        10: League(10, "Premier League", "England", 20, 380, 40000.0),
        20: League(20, "Liga MX", "Mexico", 18, 306, 25000.0),
    }
    clubs = {
        100: Club(100, "Big Club", "England", 10, 28, 6, 4, 3),
        200: Club(200, "Small Club", "Mexico", 20, 10, 8, 20, 0),
    }
    players = {
        1: _player(1, 1, 100, 25, 7.8, "Attacker"),
        2: _player(2, 1, 100, 5, 7.4, "Midfielder"),
        3: _player(3, 2, 200, 6, 6.6, "Attacker"),
        4: _player(4, 2, 200, 1, 6.4, "Defender"),
    }
    coaches = {
        500: Coach(500, "Strong Coach", 55, 8, 1, 1, 2, 1),
        600: Coach(600, "Weak Coach", 48, 2, 2, 6, 0, 2),
    }
    teams = {
        1: NationalTeam(
            id=1,
            name="England",
            group="Group A",
            confederation="UEFA",
            is_host=False,
            player_ids=(1, 2),
            coach_id=500,
            recent_w=8,
            recent_d=1,
            recent_l=1,
        ),
        2: NationalTeam(
            id=2,
            name="Mexico",
            group="Group A",
            confederation="CONCACAF",
            is_host=True,
            player_ids=(3, 4),
            coach_id=600,
            recent_w=3,
            recent_d=2,
            recent_l=5,
        ),
    }
    matches = (
        WcMatch(
            fixture_id=9001,
            matchday=1,
            group="Group A",
            home_id=1,
            away_id=2,
            kickoff=datetime(2026, 6, 12, 19, 0, tzinfo=UTC),
            venue="MetLife Stadium / East Rutherford",
            home_goals=None,
            away_goals=None,
        ),
    )
    return WorldCup(
        leagues=leagues,
        clubs=clubs,
        players=players,
        coaches=coaches,
        teams=teams,
        matches=matches,
    )
