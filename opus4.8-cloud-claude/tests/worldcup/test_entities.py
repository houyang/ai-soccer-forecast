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
