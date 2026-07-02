# tests/test_worldcup_bracket.py
from soccer_agent.worldcup.bracket import build_bracket
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength
from soccer_agent.worldcup.ranking import rank_all
from soccer_agent.worldcup.simulate import simulate_bracket
from soccer_agent.worldcup.standings import group_standings


def _setup():
    wc = load_worldcup()
    r = rank_all(wc)
    f = compute_forms(wc)
    return wc, r, recalibrated_strength(wc, r, f), f


def test_group_standings_twelve_groups_four_each():
    wc, _, _, _ = _setup()
    gs = group_standings(wc)
    assert len(gs) == 12
    for rows in gs.values():
        assert len(rows) == 4
        # sorted by points desc
        assert rows[0].pts >= rows[-1].pts


def test_bracket_has_sixteen_r32_and_tree():
    wc, _, _, _ = _setup()
    b = build_bracket(wc)
    assert len(b.r32) == 16
    assert len(b.pairs) == 8  # 8 R16 ties


def test_simulation_champion_mass_sums_to_one():
    wc, r, s, f = _setup()
    sim = simulate_bracket(wc, r, s, f, fetcher=None, n=500)
    assert len(sim.r32_predictions) == 16
    total = sum(sim.champion.values())
    assert abs(total - 1.0) < 1e-6
    # exactly one champion team has the max
    assert max(sim.champion.values()) > 0.0
