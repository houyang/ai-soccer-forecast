# tests/test_worldcup_predict.py
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength
from soccer_agent.worldcup.lineup import project_lineup
from soccer_agent.worldcup.predict import (
    predict_match,
    predict_one,
    scoreline_matrix,
    top_scorelines,
)
from soccer_agent.worldcup.ranking import rank_all


def _setup():
    wc = load_worldcup()
    r = rank_all(wc)
    f = compute_forms(wc)
    return wc, r, recalibrated_strength(wc, r, f), f


def test_probs_sum_to_one_and_floor():
    wc, r, s, f = _setup()
    m = next(m for m in wc.matches if m.matchday == 0)
    hlu = project_lineup(wc, r, m.home_id, m.fixture_id)
    alu = project_lineup(wc, r, m.away_id, m.fixture_id)
    pred = predict_one(wc, r, s, f, m.fixture_id, hlu, alu)
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-9
    assert pred.lambda_home >= 0.18 and pred.lambda_away >= 0.18


def test_predict_match_is_fixture_agnostic():
    wc, r, s, f = _setup()
    m = next(m for m in wc.matches if m.matchday == 0)
    hlu = project_lineup(wc, r, m.home_id, 0)
    alu = project_lineup(wc, r, m.away_id, 0)
    pred = predict_match(wc, r, s, f, m.home_id, m.away_id, hlu, alu,
                         venue="Neutral", group="Knockout", round_name="Round of 32")
    assert pred.home_id == m.home_id
    assert pred.venue == "Neutral"
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-9


def test_attack_defense_shapes_lambda():
    wc, r, s, f = _setup()
    # The team with the best group-stage attack should produce a higher lambda when home
    # than the team with the worst attack, all else equal.
    by_attack = sorted(f.values(), key=lambda x: x.attack, reverse=True)
    strong = by_attack[0]
    weak = by_attack[-1]
    hlu = project_lineup(wc, r, strong.team_id, 0)
    alu = project_lineup(wc, r, weak.team_id, 0)
    pred = predict_match(wc, r, s, f, strong.team_id, weak.team_id, hlu, alu, venue="Neutral")
    assert pred.lambda_home > pred.lambda_away


def test_modal_score_is_matrix_argmax():
    lh, la = 1.5, 1.2
    mat = scoreline_matrix(lh, la)
    tops = top_scorelines(lh, la, 3)
    assert abs(tops[0][2] - max(max(row) for row in mat)) < 1e-9
