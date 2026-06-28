# soccer_agent/worldcup/dataset.py
"""Locate and load the cached World Cup dataset."""
from __future__ import annotations

import json
from pathlib import Path

from soccer_agent.worldcup.entities import WorldCup

# This module's file: soccer_agent/worldcup/dataset.py -> repo root is 3 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = _REPO_ROOT / "data" / "worldcup-2026.json"


def load_worldcup(path: str | Path | None = None) -> WorldCup:
    """Load the cached dataset, defaulting to data/worldcup-2026.json under the repo root."""
    p = Path(path) if path else DATA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"World Cup dataset not found at {p}. Copy it from opus4.8-cloud-claude/data/."
        )
    return WorldCup.from_dict(json.loads(p.read_text()))
