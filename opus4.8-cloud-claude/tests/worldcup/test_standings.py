# tests/worldcup/test_standings.py
from __future__ import annotations

from datetime import UTC, datetime

from soccer.worldcup.entities import NationalTeam, WcMatch, WorldCup
from soccer.worldcup.standings import group_tables, team_labels


def _team(tid: int, group: str) -> NationalTeam:
    return NationalTeam(
        id=tid,
        name=f"T{tid}",
        group=group,
        confederation="UEFA",
        is_host=False,
        player_ids=(),
        coach_id=None,
        recent_w=0,
        recent_d=0,
        recent_l=0,
    )


def _match(fid: int, h: int, a: int, hg: int, ag: int) -> WcMatch:
    return WcMatch(
        fixture_id=fid,
        matchday=1,
        group="Group A",
        home_id=h,
        away_id=a,
        kickoff=datetime(2026, 6, 11, tzinfo=UTC),
        venue="v",
        home_goals=hg,
        away_goals=ag,
        round_name="Group Stage - 1",
    )


def _wc(teams: dict[int, NationalTeam], matches: tuple[WcMatch, ...]) -> WorldCup:
    return WorldCup(teams=teams, matches=matches)


def test_points_order_decides_rank() -> None:
    teams = {1: _team(1, "Group A"), 2: _team(2, "Group A"), 3: _team(3, "Group A")}
    # team1 beats 2 and 3; team2 beats 3
    matches = (_match(1, 1, 2, 2, 0), _match(2, 1, 3, 1, 0), _match(3, 2, 3, 3, 1))
    table = group_tables(_wc(teams, matches))["Group A"]
    assert [r.team_id for r in table] == [1, 2, 3]
    assert [r.rank for r in table] == [1, 2, 3]
    assert team_labels(_wc(teams, matches))[1] == "1A"
    assert team_labels(_wc(teams, matches))[2] == "2A"


def test_goal_difference_breaks_equal_points() -> None:
    teams = {1: _team(1, "Group A"), 2: _team(2, "Group A"), 3: _team(3, "Group A")}
    # team1 and team2 both beat team3, lose nothing else; team1 by more goals
    matches = (_match(1, 1, 3, 5, 0), _match(2, 2, 3, 1, 0), _match(3, 1, 2, 0, 0))
    table = group_tables(_wc(teams, matches))["Group A"]
    assert table[0].team_id == 1  # better GD
    assert table[0].gd == 5
    assert table[1].team_id == 2


def test_head_to_head_breaks_equal_points_and_gd() -> None:
    # Two teams identical on points, GD, GF; head-to-head decides.
    teams = {1: _team(1, "Group A"), 2: _team(2, "Group A")}
    matches = (_match(1, 1, 2, 2, 1),)  # team1 won the only meeting
    table = group_tables(_wc(teams, matches))["Group A"]
    assert table[0].team_id == 1


def test_ignores_knockout_and_unplayed_matches() -> None:
    teams = {1: _team(1, "Group A"), 2: _team(2, "Group A")}
    ko = WcMatch(
        fixture_id=9,
        matchday=0,
        group="",
        home_id=1,
        away_id=2,
        kickoff=datetime(2026, 6, 28, tzinfo=UTC),
        venue="v",
        home_goals=None,
        away_goals=None,
        round_name="Round of 32",
    )
    unplayed = WcMatch(
        fixture_id=10,
        matchday=2,
        group="Group A",
        home_id=1,
        away_id=2,
        kickoff=datetime(2026, 6, 16, tzinfo=UTC),
        venue="v",
        home_goals=None,
        away_goals=None,
        round_name="Group Stage - 2",
    )
    played = _match(1, 1, 2, 1, 0)
    table = group_tables(_wc(teams, (played, ko, unplayed)))["Group A"]
    assert table[0].played == 1  # only the played group match counts
