from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from soccer.worldcup.entities import Lineup, WcMatch, WorldCup
from soccer.worldcup.lineup import (
    DEFAULT_FORMATION,
    formation_slots,
    preferred_formation,
    project_lineup,
)
from soccer.worldcup.ranking import rank_all


def test_formation_slots_parses_defenders_mids_forwards() -> None:
    assert formation_slots("4-3-3") == (4, 3, 3)
    assert formation_slots("4-2-3-1") == (4, 5, 1)
    assert formation_slots("nonsense") == (4, 3, 3)


def test_preferred_formation_picks_most_common(sample_world_cup: WorldCup) -> None:
    wc = replace(
        sample_world_cup,
        lineups=(
            Lineup(9001, 1, "3-5-2", (1, 2), ()),
            Lineup(9002, 1, "3-5-2", (1, 2), ()),
            Lineup(9003, 1, "4-4-2", (1, 2), ()),
        ),
    )
    assert preferred_formation(wc, 1) == "3-5-2"
    assert preferred_formation(wc, 999) == DEFAULT_FORMATION


def test_project_lineup_uses_confirmed_lineup(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    wc = replace(sample_world_cup, lineups=(Lineup(9001, 1, "3-5-2", (1, 2), (3,)),))
    lu = project_lineup(wc, rankings, 1, 9001)
    assert lu.source == "confirmed"
    assert lu.formation == "3-5-2"
    assert lu.start_ids == (1, 2)
    assert lu.sub_ids == (3,)
    assert lu.source_matchday is None


def test_project_lineup_falls_back_to_prior_matchday(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    md2 = WcMatch(
        fixture_id=9002,
        matchday=2,
        group="Group A",
        home_id=1,
        away_id=2,
        kickoff=datetime(2026, 6, 18, 19, 0, tzinfo=UTC),
        venue="venue",
        home_goals=None,
        away_goals=None,
    )
    wc = replace(
        sample_world_cup,
        matches=sample_world_cup.matches + (md2,),
        lineups=(Lineup(9001, 1, "4-4-2", (1, 2), ()),),
    )
    lu = project_lineup(wc, rankings, 1, 9002)
    assert lu.source == "prior"
    assert lu.source_matchday == 1
    assert lu.formation == "4-4-2"
    assert lu.start_ids == (1, 2)


def test_project_lineup_projects_from_squad_on_matchday_one(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    lu = project_lineup(sample_world_cup, rankings, 1, 9001)
    assert lu.source == "projected"
    assert lu.formation == DEFAULT_FORMATION
    # team 1's squad is players (1, 2); both start, no subs left.
    assert set(lu.start_ids) == {1, 2}
    assert lu.sub_ids == ()


def test_project_lineup_unknown_fixture_raises(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    with pytest.raises(ValueError, match="not found"):
        project_lineup(sample_world_cup, rankings, 1, 123456)
