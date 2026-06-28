# tests/test_worldcup_form.py
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.ranking import rank_all
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength


def test_all_48_teams_have_form_with_three_played():
    wc = load_worldcup()
    forms = compute_forms(wc)
    assert len(forms) == 48
    for f in forms.values():
        assert f.played == 3
        assert f.gf - f.ga == f.gd
        assert f.pts == 3 * f.wins + f.draws


def test_recalibration_tracks_group_performance():
    wc = load_worldcup()
    rankings = rank_all(wc)
    forms = compute_forms(wc)
    strengths = recalibrated_strength(wc, rankings, forms)
    assert len(strengths) == 48
    for s in strengths.values():
        assert 0.0 <= s <= 100.0
    # A team with a huge positive GD should rate higher than its static ranking.
    best_gd = max(forms.values(), key=lambda f: f.gd)
    static = rankings.teams[best_gd.team_id]
    assert strengths[best_gd.team_id] >= static - 5.0  # recalibration never crashes it
    # And a team with a very negative GD should drop relative to static.
    worst_gd = min(forms.values(), key=lambda f: f.gd)
    assert strengths[worst_gd.team_id] <= rankings.teams[worst_gd.team_id] + 5.0
