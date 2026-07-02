from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.forecast import forecast_bracket
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength
from soccer_agent.worldcup.ranking import rank_all


def _setup():
    wc = load_worldcup()
    r = rank_all(wc)
    f = compute_forms(wc)
    return wc, r, recalibrated_strength(wc, r, f), f


def test_forecast_has_all_rounds_and_counts():
    wc, r, s, f = _setup()
    fc = forecast_bracket(wc, r, s, f)
    assert len(fc.rounds["R32"]) == 16
    assert len(fc.rounds["R16"]) == 8
    assert len(fc.rounds["QF"]) == 4
    assert len(fc.rounds["SF"]) == 2
    assert len(fc.rounds["Final"]) == 1
    assert len(fc.rounds["3rd"]) == 1
    assert fc.champion_id in wc.teams


def test_every_future_round_match_has_two_teams():
    wc, r, s, f = _setup()
    fc = forecast_bracket(wc, r, s, f)
    for rnd in ("R16", "QF", "SF", "Final", "3rd"):
        for bm in fc.rounds[rnd]:
            assert bm.home_id is not None and bm.away_id is not None
            assert bm.prediction is not None
            assert bm.advancing_id in (bm.home_id, bm.away_id)


def test_champion_won_the_final():
    wc, r, s, f = _setup()
    fc = forecast_bracket(wc, r, s, f)
    final = fc.rounds["Final"][0]
    assert fc.champion_id == final.advancing_id
    # runner-up is the loser of the final
    assert fc.runner_up_id == (final.away_id if final.advancing_id == final.home_id else final.home_id)


def test_to_dict_is_json_serializable():
    import json
    wc, r, s, f = _setup()
    fc = forecast_bracket(wc, r, s, f)
    json.dumps(fc.to_dict())
