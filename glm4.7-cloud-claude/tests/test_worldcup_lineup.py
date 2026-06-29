# tests/test_worldcup_lineup.py
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.lineup import formation_slots, project_lineup
from soccer_agent.worldcup.ranking import rank_all


def test_formation_slots():
    assert formation_slots("4-3-3") == (4, 3, 3)
    assert formation_slots("4-2-3-1") == (4, 5, 1)
    assert formation_slots("3-5-2") == (3, 5, 2)
    assert formation_slots("nonsense") == (4, 3, 3)


def test_projected_lineup_has_eleven_starters_seven_subs():
    wc = load_worldcup()
    rankings = rank_all(wc)
    any_team = next(iter(wc.teams.values()))
    lu = project_lineup(wc, rankings, any_team.id, fixture_id=0)
    assert len(lu.start_ids) == 11
    assert len(lu.sub_ids) == 7
    assert lu.formation in {"4-3-3", "4-2-3-1", "4-1-4-1", "3-5-2", "4-4-2", "3-4-3", "5-3-2"}
    assert lu.source == "projected"  # no fetcher, empty dataset lineups


def test_projected_starters_match_formation_shape():
    wc = load_worldcup()
    rankings = rank_all(wc)
    any_team = next(iter(wc.teams.values()))
    lu = project_lineup(wc, rankings, any_team.id, fixture_id=0)
    starters = [wc.players[pid] for pid in lu.start_ids]
    from collections import Counter
    pos = Counter(p.position for p in starters)
    d, m, f = formation_slots(lu.formation)
    assert pos["Goalkeeper"] == 1
    assert pos["Defender"] == d
    assert pos["Midfielder"] + pos["Attacker"] == m + f  # allow M/F flex
