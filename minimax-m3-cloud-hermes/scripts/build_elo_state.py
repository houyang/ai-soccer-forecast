#!/usr/bin/env python3
"""Pre-compute an Elo state from a JSONL file of historical matches.

This is the bridge between raw historical data and the agent's
pre-match prediction. Run this on the host (fast CPU, lots of past
matches), ship the resulting `elo_state.json` to the sandbox or
production, and the agent will load it via `elo_state_path` or
`SOCCER_AGENT_ELO_STATE`.

Usage:

    python scripts/build_elo_state.py \
        --matches path/to/past_matches.jsonl \
        --out data/elo_state.json

Each input row must be a JSON object with at least::

    {
      "date": "2024-08-15",
      "home_id": "man_city",
      "away_id": "arsenal",
      "home_goals": 2,
      "away_goals": 0
    }

The rows may be in any order; the script sorts by `date`. Teams not
present in the data are auto-registered at 1500 when first seen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

# Allow running this script from the repo root without installing.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soccer_agent.elo import (  # noqa: E402
    DEFAULT_FORM_WINDOW,
    DEFAULT_HOME_ADVANTAGE,
    DEFAULT_K,
    EloState,
    update_state,
    MatchResult,
)


def _iter_matches(path: Path) -> Iterable[MatchResult]:
    """Yield MatchResult rows from a JSONL file, sorted by date."""
    rows: list[dict] = []
    with path.open() as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln}: bad JSON: {e}") from e
            required = {"home_id", "away_id", "home_goals", "away_goals"}
            missing = required - set(row)
            if missing:
                raise SystemExit(
                    f"{path}:{ln}: missing fields {missing} in row {row!r}"
                )
            rows.append(row)
    rows.sort(key=lambda r: r.get("date", ""))
    for r in rows:
        yield MatchResult(
            home_id=str(r["home_id"]),
            away_id=str(r["away_id"]),
            home_goals=int(r["home_goals"]),
            away_goals=int(r["away_goals"]),
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build an Elo state file from historical matches.",
    )
    p.add_argument(
        "--matches", required=True, type=Path,
        help="Path to a JSONL file of historical matches.",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Output path for the JSON-serialized EloState.",
    )
    p.add_argument(
        "--k", type=float, default=DEFAULT_K,
        help="Per-game K-factor (default: %(default)s).",
    )
    p.add_argument(
        "--home-advantage", type=float, default=DEFAULT_HOME_ADVANTAGE,
        help="Home advantage in Elo points (default: %(default)s).",
    )
    p.add_argument(
        "--form-window", type=int, default=DEFAULT_FORM_WINDOW,
        help="Form window size in matches (default: %(default)s).",
    )
    p.add_argument(
        "--report", action="store_true",
        help="Print a short summary of the resulting state to stdout.",
    )
    args = p.parse_args()

    if not args.matches.exists():
        print(f"matches file not found: {args.matches}", file=sys.stderr)
        return 2

    state = EloState(
        k=args.k,
        home_advantage=args.home_advantage,
        form_window=args.form_window,
    )
    n = 0
    for m in _iter_matches(args.matches):
        update_state(state, m)
        n += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    state.to_json(args.out)
    print(f"Wrote Elo state ({n} matches, {len(state.ratings)} teams) to {args.out}")
    if args.report:
        top = sorted(
            state.ratings.items(),
            key=lambda kv: kv[1].overall,
            reverse=True,
        )
        print("\nTop 10 teams by overall Elo:")
        for tid, r in top[:10]:
            print(f"  {tid:>20s}  overall={r.overall:7.1f}  "
                  f"home={r.home:7.1f}  away={r.away:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
