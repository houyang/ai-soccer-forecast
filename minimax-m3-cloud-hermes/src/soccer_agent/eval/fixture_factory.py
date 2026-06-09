"""Fixture factory — deterministic JSON files for an EvalCase.

For each case in the dataset, write a complete set of fixture files
under the tool's expected layout:

    <root>/
        form/<home>__<away>__<YYYY-YYYY>.json   (4-digit season derived from kickoff)
        injury/<home>__<away>__<kickoff_date>.json
        h2h/<home>__<away>.json
        weather/<venue_id>__<date>.json
        odds/<home>__<away>__<kickoff_date>.json
        venues/venue_<venue_id>.json

Every file is JSON-valid and Pydantic-valid against the tool's output
model.

# Noise model

The factory takes a `noise` parameter (0..1) and a `seed` (int).
`noise=0.0` (the default) produces a *fair* fixture where every
signal — form, h2h, odds — agrees with the actual result. That's
useful for round-trip / smoke tests.

`noise=0.4` flips each of three signal categories (form, h2h, odds)
with probability `noise`, swapping which side the signal favors.
This is the regime calibration runs in: the agent sees realistic,
noisy data and is judged on whether it can extract the truth from
noise. Injuries and weather are not flipped — they're rare signals
in real life and flipping them breaks plausibility.

The factory is intentionally a pure function: same input → same bytes.
That's the property `test_factory_is_deterministic` and
`test_noise_is_deterministic_with_seed` pin.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import EvalCase


# 5 last-5 strings, one per "form quality" tier. Worst first.
_FORM_TIERS = [
    "LLLLL",  # 0 pts/5 — terrible
    "LLDLL",  # 4 pts/5 — poor
    "DWDLD",  # 8 pts/5 — average
    "WWDLW",  # 13 pts/5 — good
    "WWWDW",  # 13 pts/5 — strong (winner of the noisy default)
]


def _form_points(last5: str) -> int:
    return last5.count("W") * 3 + last5.count("D")


def _form_to_stats(last5: str) -> dict[str, int]:
    """Derive played/won/drawn/lost/gf/ga/points from a 5-char form string."""
    won = last5.count("W")
    drawn = last5.count("D")
    lost = last5.count("L")
    played = len(last5)
    gf = won * 2 + drawn + 1
    ga = lost * 2 + (played - won - lost) + 1
    return {
        "played": played,
        "won": won,
        "drawn": drawn,
        "lost": lost,
        "gf": gf,
        "ga": ga,
        "points": _form_points(last5),
    }


def _flip_form_tier(tier_index: int, rng: random.Random) -> int:
    """Pick a *different* tier from the one given.

    Used when noise flips the form signal — the side that was
    supposed to be strong gets a weak tier and vice versa.
    """
    candidates = [i for i in range(len(_FORM_TIERS)) if i != tier_index]
    return rng.choice(candidates)


def _serialize(obj: Any) -> str:
    """JSON with a fixed shape — no trailing spaces, stable key order."""
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


def _form_for(case: EvalCase, side: str, noise: float, rng: random.Random) -> dict[str, Any]:
    """Build a FormEntry. With `noise=0`, agrees with the actual result.

    Tiers (worst -> best): LLLLL, LLDLL, DWDLD, WWDLW, WWWDW.
    Home winner → home tier 4, away tier 1. Away winner → mirror.
    Draw → both tier 2. With probability `noise` per *side* (only the
    favorite side is at risk of being flipped, to keep the model
    plausible), the side's tier is replaced with a random *different*
    tier, breaking the signal-result agreement.
    """
    # Pick the "natural" tier for each side, in (0..4).
    if side == "home":
        natural = 4 if case.actual_winner == "home" else (
            1 if case.actual_winner == "away" else 2
        )
        can_flip = case.actual_winner in ("home", "away")
    else:  # away
        natural = 4 if case.actual_winner == "away" else (
            1 if case.actual_winner == "home" else 2
        )
        can_flip = case.actual_winner in ("home", "away")

    if can_flip and noise > 0 and rng.random() < noise:
        tier_index = _flip_form_tier(natural, rng)
    else:
        tier_index = natural

    last5 = _FORM_TIERS[tier_index]
    stats = _form_to_stats(last5)
    team_id = case.home_id if side == "home" else case.away_id
    return {
        "team_id": team_id,
        "as_of": case.kickoff.date().isoformat(),
        "last5_form_string": last5,
        **stats,
    }


def _injury_for(case: EvalCase, side: str) -> list[dict[str, Any]]:
    """Mostly empty injury lists — most teams are healthy. The losing
    side gets a single doubt so the agent has *something* to weigh,
    without dominating the decision.

    Returns a list of InjuryReport dicts, matching the
    `InjuryOutput` Pydantic model.
    """
    out: list[dict[str, Any]] = []
    if side == "home" and case.actual_winner == "away":
        out.append({
            "player": "key_player_home",
            "status": "doubt",
            "reported_at": case.kickoff.date().isoformat() + "T08:00:00Z",
            "source": "team_press",
            "summary": "Late fitness test.",
        })
    elif side == "away" and case.actual_winner == "home":
        out.append({
            "player": "key_player_away",
            "status": "doubt",
            "reported_at": case.kickoff.date().isoformat() + "T08:00:00Z",
            "source": "team_press",
            "summary": "Late fitness test.",
        })
    return out


def _h2h(case: EvalCase, noise: float, rng: random.Random) -> dict[str, Any]:
    """5 prior meetings distributed by winner. Noise may flip the
    majority side (i.e. the team that won the case can have an
    unfavorable h2h record)."""
    meetings: list[dict[str, Any]] = []
    base_year = case.kickoff.year - 1
    if case.actual_winner == "home":
        outcomes = ["home", "away", "home", "draw", "home"]
    elif case.actual_winner == "away":
        outcomes = ["away", "home", "away", "draw", "away"]
    else:
        outcomes = ["draw", "draw", "home", "away", "draw"]

    # With probability `noise` (only when there IS a non-draw winner),
    # invert every meeting outcome — the side that won the result
    # gets a hostile h2h instead of a friendly one.
    if case.actual_winner in ("home", "away") and noise > 0 and rng.random() < noise:
        flip = {"home": "away", "away": "home", "draw": "draw"}
        outcomes = [flip[o] for o in outcomes]

    for i, w in enumerate(outcomes):
        d = datetime(base_year - i, 10, 1, tzinfo=timezone.utc)
        if w == "home":
            hg, ag = 2, 1
        elif w == "away":
            hg, ag = 0, 1
        else:
            hg, ag = 1, 1
        meetings.append({
            "date": d.isoformat(),
            "home": case.home_id,
            "away": case.away_id,
            "home_goals": hg,
            "away_goals": ag,
            "competition": "UCL" if i % 2 == 0 else "Friendly",
        })
    last = meetings[0]
    if last["home_goals"] > last["away_goals"]:
        last_winner = "home"
    elif last["home_goals"] < last["away_goals"]:
        last_winner = "away"
    else:
        last_winner = "draw"
    return {
        "meetings": meetings,
        "home_wins": sum(1 for m in meetings if m["home_goals"] > m["away_goals"]),
        "away_wins": sum(1 for m in meetings if m["home_goals"] < m["away_goals"]),
        "draws": sum(1 for m in meetings if m["home_goals"] == m["away_goals"]),
        "last_meeting": last["date"],
        "last_winner": last_winner,
        "as_of": case.kickoff.date().isoformat(),
    }


def _weather(case: EvalCase) -> dict[str, Any]:
    """A reasonable Western-European spring/autumn evening."""
    return {
        "venue_id": case.venue_id or "neutral",
        "date": case.kickoff.date().isoformat(),
        "temp_c": 14.0,
        "precip_mm": 0.0,
        "wind_kph": 7.0,
        "conditions": "clear",
        "is_dome": False,
        "playability_risk": "low",
        "as_of": case.kickoff.date().isoformat(),
    }


def _odds(case: EvalCase, noise: float, rng: random.Random) -> dict[str, Any]:
    """Bookmaker odds. With noise, the favorite can be the *wrong* side.

    Clean (noise=0) — odds favor the actual winner. Noisy — odds
    favor the *other* side. The market_consensus_pick field follows
    the same flip (it's what an agent would naively read off the
    odds API).
    """
    if case.actual_winner == "home":
        h, d, a = 1.90, 3.40, 4.20
    elif case.actual_winner == "away":
        h, d, a = 2.10, 3.30, 3.50
    else:
        h, d, a = 2.90, 3.20, 2.90

    pick = case.actual_winner
    if case.actual_winner in ("home", "away") and noise > 0 and rng.random() < noise:
        # Flip the favorite to the *other* side. The actual-winner side
        # drifts to ~4.0+ so it's clearly the underdog. The Pinnacle
        # line in the bookmakers list below uses `+/-0.05` so it
        # automatically follows whatever the bet365 line is.
        if case.actual_winner == "home":
            h, d, a = 4.20, 3.40, 1.90
        else:
            h, d, a = 1.90, 3.30, 3.50
        pick = "away" if case.actual_winner == "home" else "home"

    # Convert to implied probabilities (multiplicative devig happens
    # downstream in the odds tool — factory just produces raw odds).
    inv = 1.0 / h + 1.0 / d + 1.0 / a
    ip = {k: (1.0 / v) / inv for k, v in [("home", h), ("draw", d), ("away", a)]}
    return {
        "bookmakers": [
            {"name": "bet365", "home": h, "draw": d, "away": a},
            {"name": "pinnacle", "home": h - 0.05, "draw": d, "away": a + 0.05},
        ],
        "implied_probs": {k: round(v, 4) for k, v in ip.items()},
        "market_consensus_pick": pick,
        "as_of": case.kickoff.date().isoformat(),
    }


def _venue(case: EvalCase) -> dict[str, Any]:
    """Plausible venue record; nothing fancy."""
    return {
        "id": case.venue_id or "neutral",
        "name": case.venue_id.replace("_", " ").title() if case.venue_id else "Neutral",
        "city": "Unknown",
        "country": "XX",
        "capacity": 50000,
        "surface": "grass",
        "is_neutral": case.venue_id == "",
        "altitude_m": 0,
        "is_dome": False,
        "lat": 0.0,
        "lon": 0.0,
        "as_of": case.kickoff.date().isoformat(),
    }


def materialize_case(
    case: EvalCase,
    root: Path,
    *,
    noise: float = 0.0,
    seed: int | None = None,
) -> None:
    """Write the six fixture files for one case under `root/`.

    `noise` (0..1) is the probability that each of the three "signal"
    categories (form, h2h, odds) gets flipped so it disagrees with
    the actual result. `noise=0.0` (default) preserves the original
    "fair fixture" contract.

    `seed` makes the flipping deterministic — same (case, noise,
    seed) → same bytes. The eval harness relies on this for
    reproducible Brier numbers.

    The directory structure matches what the tool loaders expect.
    Existing files are overwritten (the factory must be idempotent
    so the eval harness can re-run).
    """
    if noise < 0 or noise > 1:
        raise ValueError(f"noise must be in [0, 1], got {noise}")

    # Per-case RNG so the order of (form-side, h2h, odds) flips is
    # independent of how many other cases the caller has materialized.
    # We mix the user-supplied seed with the match_id so different
    # cases in a sweep don't all flip in the same way.
    base = seed if seed is not None else 0
    seed_material = f"{base}:{case.match_id}".encode("utf-8")
    derived_seed = int.from_bytes(seed_material[:8].ljust(8, b"\0"), "big")
    rng = random.Random(derived_seed)

    date_str = case.kickoff.date().isoformat()
    # Form fixture is *per (home, away, season)* — old code hard-coded
    # "2024-2025" so any 23/24 case would 404. Task 34 added 72 cases
    # that span two seasons, so we derive the suffix from kickoff.
    # 4-digit form per `_season_for` (tests/test_agent.py).
    y = case.kickoff.year
    m = case.kickoff.month
    next_y_short = (y + 1) % 100
    season = f"{y}-{y + 1}" if m >= 8 else f"{y - 1}-{y}"
    form_path = root / "form" / f"{case.home_id}__{case.away_id}__{season}.json"
    injury_path = root / "injury" / f"{case.home_id}__{case.away_id}__{date_str}.json"
    h2h_path = root / "h2h" / f"{case.home_id}__{case.away_id}.json"
    weather_path = root / "weather" / f"{case.venue_id or 'neutral'}__{date_str}.json"
    odds_path = root / "odds" / f"{case.home_id}__{case.away_id}__{date_str}.json"
    venue_path = root / "venues" / f"venue_{case.venue_id or 'neutral'}.json"

    form_path.parent.mkdir(parents=True, exist_ok=True)
    injury_path.parent.mkdir(parents=True, exist_ok=True)
    h2h_path.parent.mkdir(parents=True, exist_ok=True)
    weather_path.parent.mkdir(parents=True, exist_ok=True)
    odds_path.parent.mkdir(parents=True, exist_ok=True)
    venue_path.parent.mkdir(parents=True, exist_ok=True)

    form_path.write_text(_serialize({
        "home": _form_for(case, "home", noise, rng),
        "away": _form_for(case, "away", noise, rng),
    }))
    injury_path.write_text(_serialize({
        "home": _injury_for(case, "home"),
        "away": _injury_for(case, "away"),
    }))
    h2h_path.write_text(_serialize(_h2h(case, noise, rng)))
    weather_path.write_text(_serialize(_weather(case)))
    odds_path.write_text(_serialize(_odds(case, noise, rng)))
    venue_path.write_text(_serialize(_venue(case)))


def materialize_all(
    cases: list[EvalCase],
    root: Path,
    *,
    noise: float = 0.0,
    seed: int | None = None,
) -> None:
    """Materialize every case under the same root. Idempotent."""
    for c in cases:
        materialize_case(c, root, noise=noise, seed=seed)
