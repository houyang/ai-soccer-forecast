# src/soccer/tools/fixtures.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from soccer.tools.base import MissingFixtureKey, ToolError


class FixtureStore:
    """Loads a single JSON file of the form {section: {key: payload}}."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        try:
            self._data: dict[str, Any] = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolError(f"cannot load fixture {self._path}: {exc}") from exc

    def get(self, section: str, key: str) -> Any:
        if section not in self._data:
            raise ToolError(f"fixture missing section {section!r}")
        bucket = self._data[section]
        if not isinstance(bucket, dict):
            raise ToolError(f"fixture section {section!r} is not an object")
        try:
            return bucket[key]
        except KeyError as exc:
            raise MissingFixtureKey(f"fixture missing {section}/{key}") from exc
