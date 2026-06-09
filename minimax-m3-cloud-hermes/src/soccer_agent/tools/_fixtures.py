"""Fixture loader.

Tools fall back to deterministic JSON fixtures when live APIs are absent
or fail. Fixtures are looked up by the fixture key the tool decides
(usually `(home_id, away_id, season)`), and they live under
`SOCCER_AGENT_FIXTURES_DIR` (default `fixtures/`) organised by tool name.

Layout:
    fixtures/
        teams/team_<id>.json            # one file per team
        venues/venue_<id>.json
        form/<home>__<away>__<season>.json
        injury/<home>__<away>__<kickoff_date>.json
        h2h/<home>__<away>.json
        weather/<venue_id>__<date>.json
        odds/<home>__<away>__<kickoff_date>.json

The loader returns `None` when a fixture is missing — that signals the
caller to treat the tool as failed (no fixture for this match).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import get_settings

# File names: use double-underscore as separator so a single underscore
# can appear inside a team id. Forbidden characters: '/', '\0', '__' inside parts.
_SAFE_PART = re.compile(r"^[A-Za-z0-9_.\-]+$")


class FixtureNotFound(LookupError):
    """Raised when a fixture is missing. The caller decides the policy."""


def _sanitize(part: str) -> str:
    if not _SAFE_PART.match(part):
        raise ValueError(f"unsafe fixture key part: {part!r}")
    return part


def fixture_path(*parts: str) -> Path:
    """Build a path under the configured fixtures dir; assert safety."""
    if not parts:
        raise ValueError("at least one fixture part required")
    safe_parts = [_sanitize(p) for p in parts]
    base = Path(get_settings().fixtures_dir)
    return base.joinpath(*safe_parts)


def load_json(*parts: str, default: Any = None) -> Any:
    """Load a JSON fixture. Returns `default` (None) if missing.

    `default=None` (the "not found" sentinel) is distinct from a real
    value of None — fixtures are dicts/lists/strings, never None.
    """
    path = fixture_path(*parts)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_or_raise(*parts: str) -> Any:
    """Like load_json but raise FixtureNotFound on miss — for cases where
    the caller truly requires the fixture (e.g. eval harness)."""
    data = load_json(*parts)
    if data is None:
        raise FixtureNotFound("/".join(parts))
    return data


def ensure_fixtures_dir(*parts: str) -> Path:
    """Create the fixture directory tree. Returns the path."""
    path = fixture_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(*parts: str, data: Any) -> Path:
    """Write a fixture file (for seeding/regeneration scripts)."""
    if not parts:
        raise ValueError("at least one fixture part required")
    real = fixture_path(*parts)
    real.parent.mkdir(parents=True, exist_ok=True)
    with real.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return real
