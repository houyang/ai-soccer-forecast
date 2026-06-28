from __future__ import annotations

from soccer.worldcup.entities import Club, Coach, WorldCup


def test_world_cup_round_trips(sample_world_cup: WorldCup) -> None:
    restored = WorldCup.from_dict(sample_world_cup.to_dict())
    assert restored.to_dict() == sample_world_cup.to_dict()


def test_squad_and_groups(sample_world_cup: WorldCup) -> None:
    squad = sample_world_cup.squad(1)
    assert {p.id for p in squad} == {1, 2}
    groups = sample_world_cup.groups()
    assert list(groups) == ["Group A"]
    assert {t.name for t in groups["Group A"]} == {"England", "Mexico"}


def test_club_win_rate() -> None:
    club = Club(1, "C", "X", None, wins=6, draws=2, losses=2, titles=0)
    assert club.played == 10
    assert club.win_rate == 0.6


def test_coach_win_rate_handles_zero_games() -> None:
    coach = Coach(1, "C", 50, wins=0, draws=0, losses=0, titles=0, team_id=1)
    assert coach.win_rate == 0.0


def test_match_played_flag(sample_world_cup: WorldCup) -> None:
    assert sample_world_cup.matches[0].played is False


def test_empty_world_cup_round_trips() -> None:
    wc = WorldCup()
    assert WorldCup.from_dict(wc.to_dict()).to_dict() == wc.to_dict()


def test_lineup_round_trips() -> None:
    from soccer.worldcup.entities import Lineup

    lu = Lineup(
        fixture_id=9001,
        team_id=1,
        formation="4-3-3",
        start_ids=(1, 2, 3),
        sub_ids=(4, 5),
    )
    assert Lineup.from_dict(lu.to_dict()) == lu


def test_world_cup_round_trips_lineups() -> None:
    from soccer.worldcup.entities import Lineup

    wc = WorldCup(lineups=(Lineup(9001, 1, "4-3-3", (1, 2), (3,)),))
    restored = WorldCup.from_dict(wc.to_dict())
    assert restored.lineups == wc.lineups


def test_wcmatch_round_name_defaults_empty_and_roundtrips() -> None:
    from datetime import UTC, datetime

    from soccer.worldcup.entities import WcMatch

    m = WcMatch(
        fixture_id=1,
        matchday=1,
        group="Group A",
        home_id=10,
        away_id=20,
        kickoff=datetime(2026, 6, 11, 19, 0, tzinfo=UTC),
        venue="Estadio Azteca / Mexico City",
        home_goals=2,
        away_goals=0,
    )
    assert m.round_name == ""
    assert m.to_dict()["round_name"] == ""


def test_wcmatch_from_dict_without_round_name_is_empty() -> None:
    from soccer.worldcup.entities import WcMatch

    raw = {
        "fixture_id": 1,
        "matchday": 1,
        "group": "Group A",
        "home_id": 10,
        "away_id": 20,
        "kickoff": "2026-06-11T19:00:00+00:00",
        "venue": "v",
        "home_goals": None,
        "away_goals": None,
    }
    assert WcMatch.from_dict(raw).round_name == ""


def test_wcmatch_knockout_round_name_roundtrips() -> None:
    from datetime import UTC, datetime

    from soccer.worldcup.entities import WcMatch

    m = WcMatch(
        fixture_id=99,
        matchday=0,
        group="",
        home_id=10,
        away_id=20,
        kickoff=datetime(2026, 6, 28, 19, 0, tzinfo=UTC),
        venue="SoFi Stadium",
        home_goals=None,
        away_goals=None,
        round_name="Round of 32",
    )
    assert WcMatch.from_dict(m.to_dict()).round_name == "Round of 32"
