from __future__ import annotations

import math

from soccer.worldcup.entities import WorldCup
from soccer.worldcup.predict import (
    HOST_HOME_FIELD,
    MAX_GOALS,
    _effective_rating,
    _is_hot,
    _modal_score,
    _outcome_probs,
    _scoreline_matrix,
    predict_group_stage,
)
from soccer.worldcup.ranking import rank_all


def test_scoreline_matrix_is_a_distribution() -> None:
    matrix = _scoreline_matrix(1.3, 1.3)
    total = sum(p for row in matrix for p in row)
    assert abs(total - 1.0) < 1e-9


def test_rho_zero_reproduces_independent_poisson() -> None:
    lam_home, lam_away = 2.1, 0.7
    matrix = _scoreline_matrix(lam_home, lam_away, rho=0.0)

    def pmf(lam: float, k: int) -> float:
        return math.exp(-lam) * lam**k / math.factorial(k)

    raw = [
        [pmf(lam_home, i) * pmf(lam_away, j) for j in range(MAX_GOALS + 1)]
        for i in range(MAX_GOALS + 1)
    ]
    total = sum(p for row in raw for p in row)
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            assert abs(matrix[i][j] - raw[i][j] / total) < 1e-12


def test_draw_correction_raises_draw_probability() -> None:
    # Dixon-Coles correction must lift P(draw) vs the independent-Poisson baseline.
    corrected = _outcome_probs(_scoreline_matrix(1.3, 1.3))[1]
    independent = _outcome_probs(_scoreline_matrix(1.3, 1.3, rho=0.0))[1]
    assert corrected > independent


def test_equal_lambdas_are_symmetric_and_draw_modal() -> None:
    matrix = _scoreline_matrix(1.3, 1.3)
    p_home, p_draw, p_away = _outcome_probs(matrix)
    assert abs(p_home - p_away) < 1e-9
    assert _modal_score(matrix)[0] == _modal_score(matrix)[1]


def test_supremacy_shifts_probability_to_stronger_side() -> None:
    strong = _scoreline_matrix(2.1, 0.7)
    p_home, _, p_away = _outcome_probs(strong)
    assert p_home > p_away


def test_host_home_field_boosts_effective_rating(sample_world_cup: WorldCup) -> None:
    # Mexico (id 2) is the host; at home its effective rating gains the home-field bonus.
    home = _effective_rating(sample_world_cup, 2, 70.0, is_home=True, venue="Estadio Azteca")
    away = _effective_rating(sample_world_cup, 2, 70.0, is_home=False, venue="Estadio Azteca")
    assert home - away == HOST_HOME_FIELD


def test_hot_venue_detection() -> None:
    assert _is_hot("Hard Rock Stadium / Miami")
    assert not _is_hot("BC Place / Vancouver")


def test_predict_remaining_only_unplayed_and_shifts_lambda(sample_world_cup: WorldCup) -> None:
    from dataclasses import replace

    from soccer.worldcup.adjust import TeamAdjustment
    from soccer.worldcup.predict import predict_remaining

    rankings = rank_all(sample_world_cup)
    # Baseline (no adjustments) for fixture 9001.
    base = predict_remaining(sample_world_cup, rankings, {})
    assert len(base) == 1  # the single match is unplayed
    base_lambda_home = base[0].lambda_home

    # Boost England (home, id 1) -> its lambda_home should rise vs baseline.
    boosted = predict_remaining(sample_world_cup, rankings, {1: TeamAdjustment(rating_delta=5.0)})
    assert boosted[0].lambda_home > base_lambda_home
    assert boosted[0].home_adjustment == 5.0

    # A played match drops out of the remaining set.
    played = replace(sample_world_cup.matches[0], home_goals=1, away_goals=0)
    wc_played = replace(sample_world_cup, matches=(played,))
    assert predict_remaining(wc_played, rank_all(wc_played), {}) == []


def test_predict_group_stage_covers_all_matches(sample_world_cup: WorldCup) -> None:
    preds = predict_group_stage(sample_world_cup, rank_all(sample_world_cup))
    assert len(preds) == len(sample_world_cup.matches)
    pred = preds[0]
    # England (stronger) at home should be favoured over Mexico.
    assert pred.p_home > pred.p_away
    # probabilities are stored rounded to 4 dp, so allow rounding slack
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-3
    assert pred.score_home >= 0 and pred.score_away >= 0
