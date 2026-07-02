# tests/test_worldcup_reference.py
from soccer_agent.worldcup.reference import country_strength


def test_known_countries_ordered():
    assert country_strength("Brazil") > country_strength("New Zealand")
    assert country_strength("France") > 80.0
    assert country_strength("Germany") > 75.0


def test_unknown_is_neutral():
    assert country_strength("Atlantis") == 50.0


def test_all_wc_teams_have_a_value():
    from soccer_agent.worldcup.dataset import load_worldcup
    wc = load_worldcup()
    for t in wc.teams.values():
        assert 0.0 <= country_strength(t.name) <= 100.0
