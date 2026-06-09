"""Ingest football-data.co.uk CSVs into the agent's EvalCase set.

Task 34: expand the eval from 34 → ≥100 cases. Source: the public
football-data.co.uk CSV dumps (5 leagues × 2 seasons = 10 files).
The agent is only evaluated on matches where both clubs already
exist as IDs in the existing dataset — that keeps fixture
materialization consistent with the cases written by hand.

Usage:
  python scripts/ingest_football_data.py \\
      --external data/external \\
      --target  src/soccer_agent/eval/dataset.py
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Map football-data.co.uk team display names → agent canonical IDs.
# Only included for clubs that already appear in the existing
# EVAL_CASES set, so the rest of the pipeline (form, H2H, fixture
# materialization) keeps working unchanged.
TEAM_MAP: dict[str, dict[str, str]] = {
    "E0": {  # EPL
        "Arsenal": "arsenal", "Aston Villa": "aston_villa",
        "Bournemouth": "bournemouth", "Brentford": "brentford",
        "Brighton": "brighton", "Chelsea": "chelsea",
        "Crystal Palace": "crystal_palace", "Everton": "everton",
        "Fulham": "fulham", "Ipswich": "ipswich",
        "Leicester": "leicester", "Liverpool": "liverpool",
        "Man City": "man_city", "Man United": "man_utd",
        "Newcastle": "newcastle", "Nott'm Forest": "nottingham",
        "Southampton": "southampton", "Spurs": "spurs",
        "Tottenham": "spurs",  # alias used in some seasons
        "West Ham": "west_ham", "Wolves": "wolves",
        "Wolverhampton": "wolves",  # alias
    },
    "D1": {  # Bundesliga
        "Bayern Munich": "bayern", "Dortmund": "dortmund",
        "RB Leipzig": "rb_leipzig", "Leverkusen": "leverkusen",
        "Stuttgart": "stuttgart", "Wolfsburg": "wolfsburg",
        "Frankfurt": "frankfurt", "Hoffenheim": "hoffenheim",
        "Freiburg": "freiburg", "Union Berlin": "union_berlin",
        "Mainz": "mainz", "Werder Bremen": "werder_bremen",
        "Augsburg": "augsburg", "Gladbach": "gladbach",
        "Mönchengladbach": "gladbach",
        "FC Koln": "koln", "Köln": "koln", "Hertha": "hertha",
        "Schalke": "schalke", "Bochum": "bochum",
    },
    "SP1": {  # La Liga
        "Real Madrid": "real_madrid", "Barcelona": "barca",
        "Atletico Madrid": "atletico", "Sevilla": "sevilla",
        "Real Betis": "real_betis", "Valencia": "valencia",
        "Villarreal": "villarreal", "Athletic Club": "athletic",
        "Real Sociedad": "real_sociedad", "Getafe": "getafe",
        "Osasuna": "osasuna", "Celta": "celta",
        "Celta Vigo": "celta", "Mallorca": "mallorca",
        "Girona": "girona", "Rayo Vallecano": "rayo",
        "Las Palmas": "las_palmas", "Alaves": "alaves",
        "Espanyol": "espanyol", "Leganes": "leganes",
        "Valladolid": "valladolid",
    },
    "I1": {  # Serie A
        "Inter": "inter", "AC Milan": "milan", "Milan": "milan",
        "Juventus": "juventus", "Napoli": "napoli",
        "Roma": "roma", "Lazio": "lazio", "Atalanta": "atalanta",
        "Fiorentina": "fiorentina", "Bologna": "bologna",
        "Torino": "torino", "Genoa": "genoa",
        "Monza": "monza", "Verona": "verona", "Empoli": "empoli",
        "Cagliari": "cagliari", "Lecce": "lecce",
        "Udinese": "udinese", "Sassuolo": "sassuolo",
        "Salernitana": "salernitana", "Frosinone": "frosinone",
        "Parma": "parma", "Como": "como", "Venezia": "venezia",
    },
    "F1": {  # Ligue 1
        "Paris SG": "psg", "PSG": "psg",
        "Marseille": "marseille", "Monaco": "monaco",
        "Lyon": "lyon", "Lille": "lille", "Rennes": "rennes",
        "Nice": "nice", "Lens": "lens", "Strasbourg": "strasbourg",
        "Montpellier": "montpellier", "Nantes": "nantes",
        "Reims": "reims", "Toulouse": "toulouse",
        "Brest": "brest", "Le Havre": "le_havre",
        "Auxerre": "auxerre", "Angers": "angers",
        "Saint-Etienne": "saint_etienne",
    },
}

COMPETITION_FROM_CODE = {
    "E0": "EPL", "D1": "Bundesliga",
    "SP1": "LaLiga", "I1": "SerieA", "F1": "Ligue1",
}


@dataclass(frozen=True)
class RawRow:
    code: str
    date: str  # "16/08/2024"
    home: str
    away: str
    home_goals: int
    away_goals: int


def parse_csv(path: Path) -> Iterable[RawRow]:
    """Yield rows from a football-data.co.uk CSV.

    The CSVs have a UTF-8 BOM and dd/mm/YYYY dates. We only need
    the post-match columns: Date, HomeTeam, AwayTeam, FTHG, FTAG.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                hg = int(r["FTHG"])
                ag = int(r["FTAG"])
            except (KeyError, ValueError):
                continue
            yield RawRow(
                code=path.stem.split("_")[0],
                date=r["Date"],
                home=r["HomeTeam"],
                away=r["AwayTeam"],
                home_goals=hg,
                away_goals=ag,
            )


def slugify(s: str) -> str:
    """Used only for match_id generation."""
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    return s.strip("_")


def date_to_utc_kickoff(d: str) -> datetime:
    """Football-data dates are dd/mm/yyyy. We stamp kickoff at
    20:00 UTC (typical European evening slot) so the same match_id
    is stable across re-ingestion."""
    dt = datetime.strptime(d, "%d/%m/%Y")
    return dt.replace(hour=20, tzinfo=timezone.utc)


def is_valid_pair(
    code: str, home: str, away: str, existing_ids: set[str]
) -> bool:
    h = TEAM_MAP.get(code, {}).get(home)
    a = TEAM_MAP.get(code, {}).get(away)
    return bool(h and a and h != a and h in existing_ids and a in existing_ids)


def ingest(
    external_dir: Path,
    existing_ids: set[str],
    per_comp_cap: int = 18,
    skip_dates_after: str | None = "2025-08-01",
) -> list[dict]:
    """Read all CSVs, filter to (home, away) pairs already known
    to the agent, and yield dicts shaped for EvalCase.

    Each dict has all the fields `EvalCase(...)` requires.
    `cap=18` keeps a single league from dominating the eval set
    (still leaves room for cross-competition diversity).
    """
    cutoff = (
        datetime.strptime(skip_dates_after, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        if skip_dates_after
        else None
    )
    by_comp: dict[str, list[dict]] = defaultdict(list)
    for csv_path in sorted(external_dir.glob("*_*.csv")):
        for row in parse_csv(csv_path):
            h = TEAM_MAP.get(row.code, {}).get(row.home)
            a = TEAM_MAP.get(row.code, {}).get(row.away)
            if not (h and a) or h == a:
                continue
            if h not in existing_ids or a not in existing_ids:
                continue
            ko = date_to_utc_kickoff(row.date)
            if cutoff and ko >= cutoff:
                continue
            comp = COMPETITION_FROM_CODE[row.code]
            mid = f"fd_{slugify(comp)}_{row.date.replace('/', '-')}_{slugify(h)}_v_{slugify(a)}"
            by_comp[comp].append({
                "match_id": mid,
                "competition": comp,
                "round": "regular",
                "home_id": h,
                "away_id": a,
                "venue_id": "",
                "kickoff": ko,
                "home_goals": row.home_goals,
                "away_goals": row.away_goals,
            })
    out: list[dict] = []
    for comp, rows in by_comp.items():
        # Dedupe by match_id (alias teams like "Spurs"/"Tottenham"
        # would otherwise double-count), then take the most recent
        # per_comp_cap rows.
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in sorted(rows, key=lambda x: x["kickoff"]):
            if r["match_id"] in seen:
                continue
            seen.add(r["match_id"])
            deduped.append(r)
        out.extend(deduped[-per_comp_cap:])
    return sorted(out, key=lambda r: r["kickoff"])


def existing_match_ids() -> set[str]:
    """Load the current dataset.py and return the IDs of all cases
    it defines, so we can avoid duplicates when we re-emit."""
    from soccer_agent.eval.dataset import _EVAL_CASES_UNSORTED
    return {c.match_id for c in _EVAL_CASES_UNSORTED}


def existing_team_ids() -> set[str]:
    from soccer_agent.eval.dataset import _EVAL_CASES_UNSORTED
    s: set[str] = set()
    for c in _EVAL_CASES_UNSORTED:
        s.add(c.home_id)
        s.add(c.away_id)
    return s


def render_case(c: dict) -> str:
    ko = c["kickoff"]
    return (
        f'    EvalCase(\n'
        f'        match_id="{c["match_id"]}",\n'
        f'        competition="{c["competition"]}", round="{c["round"]}",\n'
        f'        home_id="{c["home_id"]}", away_id="{c["away_id"]}",\n'
        f'        venue_id="{c["venue_id"]}",\n'
        f'        kickoff=datetime({ko.year}, {ko.month}, {ko.day}, {ko.hour}, 0, tzinfo=timezone.utc),\n'
        f'        home_goals={c["home_goals"]}, away_goals={c["away_goals"]},\n'
        f'    ),'
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", type=Path, default=Path("data/external"))
    ap.add_argument("--per-comp-cap", type=int, default=18)
    ap.add_argument("--cutoff", default="2025-08-01")
    ap.add_argument("--out", type=Path, default=Path("data/ingested_cases.txt"))
    args = ap.parse_args()

    if not args.external.is_dir():
        print(f"no external data at {args.external}")
        return 1

    existing = existing_match_ids()
    team_ids = existing_team_ids()
    new = ingest(
        args.external,
        team_ids,
        per_comp_cap=args.per_comp_cap,
        skip_dates_after=args.cutoff,
    )
    # Drop any that already exist (idempotency).
    new = [c for c in new if c["match_id"] not in existing]

    print(f"discovered {len(new)} new cases (existing {len(existing)})")
    by_comp: dict[str, int] = defaultdict(int)
    for c in new:
        by_comp[c["competition"]] += 1
    for k, v in sorted(by_comp.items()):
        print(f"  {k}: +{v}")

    lines = [render_case(c) for c in new]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
