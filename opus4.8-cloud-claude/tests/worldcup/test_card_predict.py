from __future__ import annotations

from soccer.worldcup.adjust import adjustment_for_match
from soccer.worldcup.entities import WorldCup
from soccer.worldcup.lineup import ProjectedLineup
from soccer.worldcup.predict import predict_match, predict_one, top_scorelines
from soccer.worldcup.ranking import rank_all


def test_adjustment_for_match_reflects_formation_lean(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    attacking = ProjectedLineup(1, "3-3-4", (1, 2), (), "projected", None)
    adj = adjustment_for_match(sample_world_cup, rankings, 1, attacking)
    # 3-3-4 -> 4 forwards (attack lean up), 3 defenders (defense lean down).
    assert adj.attack_lean > 0.0
    assert adj.defense_lean < 0.0


def test_predict_one_degrades_to_baseline_without_lineups(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    baseline = predict_match(sample_world_cup, rankings, 9001)
    one = predict_one(sample_world_cup, rankings, 9001, None, None)
    # No played matches + no lineup => zero adjustment => identical to the baseline forecast.
    assert one.lambda_home == baseline.lambda_home
    assert one.lambda_away == baseline.lambda_away
    assert one.p_home == baseline.p_home


def test_predict_one_attacking_home_raises_home_xg(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    baseline = predict_match(sample_world_cup, rankings, 9001)
    attacking = ProjectedLineup(1, "3-3-4", (1, 2), (), "projected", None)
    one = predict_one(sample_world_cup, rankings, 9001, attacking, None)
    assert one.lambda_home > baseline.lambda_home


def test_top_scorelines_is_sorted_and_bounded() -> None:
    tops = top_scorelines(1.4, 1.1, n=3)
    assert len(tops) == 3
    probs = [p for _, _, p in tops]
    assert probs == sorted(probs, reverse=True)
    assert 0.0 < sum(probs) < 1.0
    home, away, _ = tops[0]
    assert isinstance(home, int) and isinstance(away, int)
