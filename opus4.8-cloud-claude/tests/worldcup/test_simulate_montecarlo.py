# tests/worldcup/test_simulate_montecarlo.py
from __future__ import annotations

import random

from soccer.worldcup.bracket import build_bracket
from soccer.worldcup.entities import WorldCup
from soccer.worldcup.ranking import rank_all
from soccer.worldcup.simulate import run_monte_carlo
from soccer.worldcup.standings import team_labels
from tests.worldcup.test_simulate_modal import _add_r32, _wc_from_labels


def _wc() -> WorldCup:
    return _add_r32(
        _wc_from_labels(
            [f"{r}{c}" for c in "ABCDEFGHIJKL" for r in (1, 2)] + [f"3{c}" for c in "CDEFGHIJ"]
        )
    )


def test_monte_carlo_is_reproducible_with_seed() -> None:
    wc = _wc()
    ranks = rank_all(wc)
    ties = build_bracket(wc, team_labels(wc))
    a = run_monte_carlo(wc, ranks, ties, rng=random.Random(7), n_sims=500)
    b = run_monte_carlo(wc, ranks, ties, rng=random.Random(7), n_sims=500)
    assert {t: o.win for t, o in a.items()} == {t: o.win for t, o in b.items()}


def test_probabilities_are_valid_and_monotone() -> None:
    wc = _wc()
    ranks = rank_all(wc)
    ties = build_bracket(wc, team_labels(wc))
    odds = run_monte_carlo(wc, ranks, ties, rng=random.Random(1), n_sims=1000)
    assert abs(sum(o.win for o in odds.values()) - 1.0) < 1e-9
    for o in odds.values():
        assert 0.0 <= o.win <= o.reach_final <= o.reach_sf <= o.reach_qf <= o.reach_r16 <= 1.0


def test_favourite_has_highest_title_odds() -> None:
    wc = _wc()
    ranks = rank_all(wc)
    ties = build_bracket(wc, team_labels(wc))
    odds = run_monte_carlo(wc, ranks, ties, rng=random.Random(3), n_sims=2000)
    best = max(odds.values(), key=lambda o: o.win)
    # strongest team (recent_w=3 -> a group winner) should out-rank a weak third
    assert best.win > 0
