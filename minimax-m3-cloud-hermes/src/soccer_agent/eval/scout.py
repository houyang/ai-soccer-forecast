"""ResultScout — fetches new match results since a watermark.

A scout is a thin layer over a "provider" callable. The provider takes
a `since: datetime` and returns a list of `Result` objects. The scout
adds the small bookkeeping needed for the agent's self-eval loop:

  - filtering (already done by the provider, but we double-check)
  - ordering (most recent first, so the loop can update its watermark
    after each successful batch)
  - a factory for the Phase 1 fixture-based provider

Phase 2 will add live providers (API-Football, football-data.org, etc.)
behind the same interface.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from ..models import Result

Provider = Callable[[datetime], list[Result]]


class ResultScout:
    """Thin wrapper around a results provider."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def fetch_new_results(self, since: datetime) -> list[Result]:
        """Return all results the provider knows about, decided at or after `since`.

        The provider may return anything (caller-defined semantics);
        the scout filters, sorts, and validates.
        """
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        raw = self._provider(since)
        # Pydantic re-validates each row — cheap, catches schema drift.
        results = [r if isinstance(r, Result) else Result.model_validate(r) for r in raw]
        # Filter to only results decided at or after the watermark.
        results = [r for r in results if r.decided_at >= since]
        # Sort newest first so the caller can update its watermark
        # after each successful batch.
        results.sort(key=lambda r: r.decided_at, reverse=True)
        return results


def fixture_provider(directory: Path | str) -> Provider:
    """Phase 1 provider: read results from a directory of JSON files.

    Each file is named `<match_id>.json` and has shape:
        {
          "match_id": "ucl_final_2025",
          "home_goals": 2,
          "away_goals": 1,
          "decided_at": "2025-05-30T22:00:00Z"
        }
    Files that fail to parse are skipped (logged in the self-eval loop
    summary, not raised — a single bad fixture must not stop the loop).
    """
    directory = Path(directory)

    def _provider(since: datetime) -> list[Result]:
        if not directory.exists():
            return []
        out: list[Result] = []
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                r = Result.model_validate(data)
                if r.decided_at >= since:
                    out.append(r)
            except Exception:  # noqa: BLE001 — best-effort loader
                continue
        return out

    return _provider
