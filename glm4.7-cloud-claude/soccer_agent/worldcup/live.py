"""Optional API-Sports v3 lineup fetcher with an on-disk cache.

Never raises on network/key problems: returns None so callers fall back to projected
lineups. The key is read only from the environment (git-ignored .env).
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

from soccer_agent.worldcup.entities import Lineup, WorldCup

BASE_URL = "https://v3.football.api-sports.io"
_CACHE_NAME = "lineups_fixture={}.json"


def _ssl_context() -> ssl.SSLContext | None:
    """A CA-trust SSL context. macOS framework Python lacks system CAs, so prefer certifi."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def parse_lineup_response(payload: dict, fixture_id: int) -> list[Lineup]:
    """Parse an API-Sports /fixtures/lineups response into Lineup objects."""
    out: list[Lineup] = []
    for side in payload.get("response", []):
        team = side.get("team") or {}
        start = tuple(p["player"]["id"] for p in side.get("startXI", []) if p.get("player", {}).get("id"))
        subs = tuple(p["player"]["id"] for p in side.get("substitutes", []) if p.get("player", {}).get("id"))
        out.append(Lineup(
            fixture_id=fixture_id,
            team_id=int(team.get("id", 0)),
            formation=str(side.get("formation") or "4-3-3"),
            start_ids=start,
            sub_ids=subs,
        ))
    return out


class LineupFetcher:
    def __init__(self, cache_dir: str | Path | None = None, timeout: float = 10.0):
        repo_root = Path(__file__).resolve().parents[2]
        self.cache_dir = Path(cache_dir) if cache_dir else repo_root / "data" / "live"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    @property
    def _key(self) -> str | None:
        return os.getenv("API_FOOTBALL_KEY")

    def fetch_fixture_lineups(self, fixture_id: int) -> list[Lineup] | None:
        """Return lineups for a fixture (cached on disk after first fetch). None if unavailable."""
        cache_path = self.cache_dir / _CACHE_NAME.format(fixture_id)
        if cache_path.exists():
            return parse_lineup_response(json.loads(cache_path.read_text()), fixture_id)
        key = self._key
        if not key:
            return None
        url = f"{BASE_URL}/fixtures/lineups?fixture={fixture_id}"
        req = urllib.request.Request(url, headers={"x-apisports-key": key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=_ssl_context()) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None
        cache_path.write_text(json.dumps(payload))
        return parse_lineup_response(payload, fixture_id)

    def recent_team_lineup(self, wc: WorldCup, team_id: int) -> Lineup | None:
        """Most-recent *played* WC match lineup for a team, or None."""
        played = sorted(
            (m for m in wc.matches if m.played and team_id in (m.home_id, m.away_id)),
            key=lambda m: m.kickoff,
        )
        for m in reversed(played):
            lineups = self.fetch_fixture_lineups(m.fixture_id)
            if not lineups:
                continue
            for lu in lineups:
                if lu.team_id == team_id and lu.start_ids:
                    return lu
        return None
