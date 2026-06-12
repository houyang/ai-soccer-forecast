from __future__ import annotations

from soccer.worldcup.entities import WorldCup
from soccer.worldcup.ranking import _minmax, rank_all, top_n


def test_minmax_handles_constant_values() -> None:
    assert _minmax({1: 5.0, 2: 5.0}) == {1: 0.5, 2: 0.5}
    assert _minmax({}) == {}
    assert _minmax({1: 0.0, 2: 10.0}) == {1: 0.0, 2: 1.0}


def test_rank_all_orders_each_tier(sample_world_cup: WorldCup) -> None:
    ranks = rank_all(sample_world_cup)
    # Premier League (more attendance + stronger country) outranks Liga MX.
    assert ranks.leagues[10] > ranks.leagues[20]
    # Big Club (top division, high win rate) outranks Small Club.
    assert ranks.clubs[100] > ranks.clubs[200]
    # The elite forward is the top-ranked player.
    assert top_n(ranks.players, 1)[0][0] == 1
    # The successful coach with the stronger squad outranks the other.
    assert ranks.coaches[500] > ranks.coaches[600]
    # England outranks Mexico despite Mexico's host bonus.
    assert ranks.teams[1] > ranks.teams[2]


def test_all_scores_within_bounds(sample_world_cup: WorldCup) -> None:
    ranks = rank_all(sample_world_cup)
    for table in (ranks.leagues, ranks.clubs, ranks.players, ranks.coaches, ranks.teams):
        assert all(0.0 <= v <= 100.0 for v in table.values())


def test_host_bonus_raises_team_score(sample_world_cup: WorldCup) -> None:
    from dataclasses import replace

    base = rank_all(sample_world_cup).teams[2]
    no_host = replace(sample_world_cup.teams[2], is_host=False)
    wc2 = WorldCup(
        leagues=sample_world_cup.leagues,
        clubs=sample_world_cup.clubs,
        players=sample_world_cup.players,
        coaches=sample_world_cup.coaches,
        teams={1: sample_world_cup.teams[1], 2: no_host},
        matches=sample_world_cup.matches,
    )
    assert base > rank_all(wc2).teams[2]


def test_empty_dataset_ranks_empty() -> None:
    ranks = rank_all(WorldCup())
    assert ranks.leagues == {} and ranks.teams == {}
