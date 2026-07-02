# tests/test_worldcup_ranking.py
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.ranking import rank_all, top_n


def test_rankings_deterministic_and_bounded():
    wc = load_worldcup()
    r1 = rank_all(wc)
    r2 = rank_all(wc)
    assert r1.teams == r2.teams
    for score in r1.teams.values():
        assert 0.0 <= score <= 100.0
    for score in r1.players.values():
        assert 0.0 <= score <= 100.0


def test_top_teams_make_sense():
    wc = load_worldcup()
    r = rank_all(wc)
    top = top_n(r.teams, 5)
    assert len(top) == 5
    names = [wc.teams[tid].name for tid, _ in top]
    # At least three traditional powers in the top 5.
    powers = {"Argentina", "France", "Brazil", "England", "Spain", "Germany", "Portugal"}
    assert len(set(names) & powers) >= 3


def test_hosts_get_bonus():
    wc = load_worldcup()
    r = rank_all(wc)
    # USA/Mexico/Canada are hosts; they should rate respectably.
    for tid, team in wc.teams.items():
        if team.is_host:
            assert r.teams[tid] >= 55.0
