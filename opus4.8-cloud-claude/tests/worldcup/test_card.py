from __future__ import annotations

import pytest

from soccer.worldcup.card import build_card
from soccer.worldcup.entities import WorldCup
from soccer.worldcup.ranking import rank_all


def test_build_card_assembles_both_teams(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    card = build_card(sample_world_cup, rankings, 9001)
    assert card.fixture_id == 9001
    assert card.group == "Group A"
    assert card.home.name == "England"
    assert card.away.name == "Mexico"
    assert card.home.source == "projected"  # no lineups in the sample dataset
    assert card.home.coach_name == "Strong Coach"
    assert card.home.coach_record == (8, 1, 1)
    assert len(card.home.starters) >= 1
    assert len(card.top_scorelines) == 3
    assert card.prediction.fixture_id == 9001


def test_build_card_to_dict_has_expected_keys(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    card = build_card(sample_world_cup, rankings, 9001)
    data = card.to_dict()
    assert {"fixture_id", "group", "kickoff", "venue", "home", "away", "prediction"} <= set(data)
    assert {"name", "formation", "starters", "subs", "source"} <= set(data["home"])
    assert {"player_id", "name", "position", "rating"} <= set(data["home"]["starters"][0])


def test_build_card_unknown_fixture_raises(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    with pytest.raises(ValueError, match="not found"):
        build_card(sample_world_cup, rankings, 4242)
