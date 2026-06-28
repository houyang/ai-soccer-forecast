# tests/test_worldcup_predict.py
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.ranking import rank_all
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength
from soccer_agent.worldcup.lineup import project_lineup
from soccer_agent.worldcup.predict import predict_one, top_scorelines, scoreline_matrix


def _strengths():
    wc = load_worldcup()
    r = rank_all(wc)
    f = compute_forms(wc)
    return wc, r, recalibrated_strength(wc, r, f)


def test_probs_sum_to_one():
    wc, r, s = _strengths()
    m = next(m for m in wc.matches if m.matchday == 0)
    hlu = project_lineup(wc, r, m.home_id, m.fixture_id)
    alu = project_lineup(wc, r, m.away_id, m.fixture_id)
    pred = predict_one(wc, r, s, m.fixture_id, hlu, alu)
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-9
    assert pred.lambda_home >= 0.18 and pred.lambda_away >= 0.18


def test_modal_score_is_matrix_argmax():
    lh, la = 1.5, 1.2
    mat = scoreline_matrix(lh, la)
    pred = predict_one  # noqa: F841 (just to ensure import path works)
    tops = top_scorelines(lh, la, 3)
    best_h, best_a, best_p = tops[0]
    assert abs(best_p - max(max(row) for row in mat)) < 1e-9


def test_stronger_team_favored():
    wc, r, s = _strengths()
    m = next(m for m in wc.matches if m.matchday == 0)
    hlu = project_lineup(wc, r, m.home_id, m.fixture_id)
    alu = project_lineup(wc, r, m.away_id, m.fixture_id)
    pred = predict_one(wc, r, s, m.fixture_id, hlu, alu)
    if s[m.home_id] - s[m.away_id] > 10:
        assert pred.p_home > pred.p_away
