# tests/test_worldcup_entities.py
from soccer_agent.worldcup.dataset import load_worldcup


def test_load_worldcup_keys_and_squad():
    wc = load_worldcup()
    assert len(wc.teams) == 48
    assert len(wc.players) == 1248
    assert len(wc.coaches) == 48
    any_team = next(iter(wc.teams.values()))
    squad = wc.squad(any_team.id)
    assert 20 <= len(squad) <= 30
    assert all(p.wc_team_id == any_team.id for p in squad)


def test_groups_are_four_teams_each():
    wc = load_worldcup()
    groups = wc.groups()
    assert len(groups) == 12
    for name, teams in groups.items():
        assert name.startswith("Group ")
        assert len(teams) == 4


def test_matches_round_trip():
    wc = load_worldcup()
    r32 = [m for m in wc.matches if m.matchday == 0]
    assert len(r32) == 16
    assert all(m.home_goals is None for m in r32)
    group_played = [m for m in wc.matches if m.matchday in (1, 2, 3) and m.played]
    assert len(group_played) == 72
