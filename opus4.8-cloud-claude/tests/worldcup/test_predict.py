from __future__ import annotations

from soccer.worldcup.entities import WorldCup
from soccer.worldcup.predict import (
    HOST_HOME_FIELD,
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
    assert abs(total - 1.0) < 1e-3


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


def test_predict_group_stage_covers_all_matches(sample_world_cup: WorldCup) -> None:
    preds = predict_group_stage(sample_world_cup, rank_all(sample_world_cup))
    assert len(preds) == len(sample_world_cup.matches)
    pred = preds[0]
    # England (stronger) at home should be favoured over Mexico.
    assert pred.p_home > pred.p_away
    # probabilities are stored rounded to 4 dp, so allow rounding slack
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-3
    assert pred.score_home >= 0 and pred.score_away >= 0
