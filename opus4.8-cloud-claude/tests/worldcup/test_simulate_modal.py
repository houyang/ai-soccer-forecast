from __future__ import annotations

from datetime import UTC, datetime

from soccer.worldcup.bracket import build_bracket
from soccer.worldcup.entities import NationalTeam, WcMatch, WorldCup
from soccer.worldcup.ranking import rank_all
from soccer.worldcup.simulate import run_modal_bracket
from soccer.worldcup.standings import team_labels


def _wc_from_labels(all_labels: list[str]) -> WorldCup:
    teams: dict[int, NationalTeam] = {}
    matches: list[WcMatch] = []
    # one team per label; group/rank encoded so team_labels reproduces the label.
    # Build a 3-team mini group per letter so ranks 1,2,3 fall out by constructed results.
    by_letter: dict[str, list[int]] = {}
    next_id = 1
    label_to_id: dict[str, int] = {}  # noqa: F841
    for label in all_labels:
        rank, letter = int(label[0]), label[1]
        tid = next_id
        next_id += 1
        label_to_id[label] = tid
        teams[tid] = NationalTeam(
            id=tid,
            name=label,
            group=f"Group {letter}",
            confederation="UEFA",
            is_host=False,
            player_ids=(),
            coach_id=None,
            recent_w=4 - rank,
            recent_d=0,
            recent_l=rank - 1,
        )
        by_letter.setdefault(letter, []).append(tid)
    # group matches that yield the intended rank order (higher rank id beats lower)
    fid = 1
    for letter, ids in by_letter.items():
        ids_sorted = sorted(ids, key=lambda t: teams[t].name)  # 1x,2x,3x
        for i in range(len(ids_sorted)):
            for j in range(i + 1, len(ids_sorted)):
                matches.append(
                    WcMatch(
                        fixture_id=fid,
                        matchday=1,
                        group=f"Group {letter}",
                        home_id=ids_sorted[i],
                        away_id=ids_sorted[j],
                        kickoff=datetime(2026, 6, 11, tzinfo=UTC),
                        venue="v",
                        home_goals=2,
                        away_goals=0,
                        round_name="Group Stage - 1",
                    )
                )
                fid += 1
    return WorldCup(teams=teams, matches=tuple(matches))


def _add_r32(wc: WorldCup) -> WorldCup:
    from soccer.worldcup.bracket import R32_ANCHORS

    labels = team_labels(wc)
    inv = {v: k for k, v in labels.items()}
    thirds = iter(sorted(t for t in labels.values() if t.startswith("3")))
    r32: list[WcMatch] = []
    fid = 5000
    for _no, anchors in R32_ANCHORS.items():
        anchor_list = list(anchors)
        if len(anchor_list) == 2:
            h, a = inv[anchor_list[0]], inv[anchor_list[1]]
        else:
            h, a = inv[anchor_list[0]], inv[next(thirds)]
        r32.append(
            WcMatch(
                fixture_id=fid,
                matchday=0,
                group="",
                home_id=h,
                away_id=a,
                kickoff=datetime(2026, 6, 28, tzinfo=UTC),
                venue="",
                home_goals=None,
                away_goals=None,
                round_name="Round of 32",
            )
        )
        fid += 1
    return WorldCup(teams=wc.teams, matches=wc.matches + tuple(r32))


def test_modal_bracket_is_complete_and_deterministic() -> None:
    wc = _add_r32(
        _wc_from_labels(
            [f"{r}{c}" for c in "ABCDEFGHIJKL" for r in (1, 2)] + [f"3{c}" for c in "CDEFGHIJ"]
        )
    )
    ranks = rank_all(wc)
    ties = build_bracket(wc, team_labels(wc))
    preds, podium = run_modal_bracket(wc, ranks, ties)
    assert [p.match_no for p in preds] == sorted(range(73, 105))
    assert podium.champion_id != podium.runner_up_id
    # determinism: same inputs -> same champion
    preds2, podium2 = run_modal_bracket(wc, ranks, ties)
    assert podium2.champion_id == podium.champion_id
    # the final's winner is the champion
    final = next(p for p in preds if p.match_no == 104)
    winner = final.home_id if final.p_home_advance >= 0.5 else final.away_id
    assert podium.champion_id == winner
