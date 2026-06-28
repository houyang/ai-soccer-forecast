from __future__ import annotations

from soccer.worldcup.entities import WorldCup
from soccer.worldcup.predict import advance_prob, predict_knockout
from soccer.worldcup.ranking import rank_all


def test_advancement_probs_sum_to_one(sample_world_cup: WorldCup) -> None:
    wc = sample_world_cup
    ranks = rank_all(wc)
    pred = predict_knockout(wc, ranks, home_id=1, away_id=2, match_no=104, round_name="Final")
    assert abs(pred.p_home_advance + pred.p_away_advance - 1.0) < 1e-9
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-9


def test_stronger_team_is_favoured_to_advance(sample_world_cup: WorldCup) -> None:
    wc = sample_world_cup
    ranks = rank_all(wc)
    # team 1 is the strong side in the fixture
    assert advance_prob(wc, ranks, 1, 2) > 0.5
    assert advance_prob(wc, ranks, 2, 1) < 0.5


def test_equal_teams_advance_is_half() -> None:
    # Build a symmetric two-team world cup so ratings tie exactly.
    from soccer.worldcup.entities import NationalTeam, WorldCup

    def t(i: int) -> NationalTeam:
        return NationalTeam(
            id=i,
            name=f"T{i}",
            group="Group A",
            confederation="UEFA",
            is_host=False,
            player_ids=(),
            coach_id=None,
            recent_w=3,
            recent_d=1,
            recent_l=1,
        )

    wc = WorldCup(teams={1: t(1), 2: t(2)})
    ranks = rank_all(wc)
    assert abs(advance_prob(wc, ranks, 1, 2) - 0.5) < 1e-6


def test_expected_extra_time_flag_set_for_evenly_matched(sample_world_cup: WorldCup) -> None:
    wc = sample_world_cup
    ranks = rank_all(wc)
    # Same team vs itself-strength: draw is the plurality outcome -> flag true.
    pred = predict_knockout(wc, ranks, home_id=1, away_id=1)
    assert pred.expected_extra_time is True


def test_advancement_probs_sum_to_one_for_awkward_matchup() -> None:
    """Prove rounding invariant for non-symmetric matchup with non-clean advance prob."""
    from soccer.worldcup.entities import NationalTeam, WorldCup

    def t(i: int, w: int, d: int, losses: int) -> NationalTeam:
        return NationalTeam(
            id=i,
            name=f"T{i}",
            group="Group A",
            confederation="UEFA",
            is_host=False,
            player_ids=(),
            coach_id=None,
            recent_w=w,
            recent_d=d,
            recent_l=losses,
        )

    # Create a 2-team world cup with asymmetric records to generate a non-clean advance prob.
    wc = WorldCup(teams={1: t(1, 5, 1, 1), 2: t(2, 2, 2, 3)})
    ranks = rank_all(wc)
    pred = predict_knockout(wc, ranks, home_id=1, away_id=2)
    # Advancement probs must sum to exactly 1.0 after rounding.
    assert abs(pred.p_home_advance + pred.p_away_advance - 1.0) < 1e-9
