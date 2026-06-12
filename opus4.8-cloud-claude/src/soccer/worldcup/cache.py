"""On-disk JSON cache for API responses ("stored locally first").

Each logical API page is cached as one JSON file under ``root``. Keys are sanitized to
safe filenames; long keys fall back to a hash so paths stay within OS limits. The cache is
deliberately dumb: it stores and returns already-parsed JSON values and never knows about
the API shape.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_SAFE = re.compile(r"[^A-Za-z0-9._=-]+")


def _filename(key: str) -> str:
    slug = _SAFE.sub("_", key).strip("_")
    if len(slug) > 120:
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        slug = f"{slug[:100]}_{digest}"
    return f"{slug}.json"


class JsonCache:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        return self._root / _filename(key)

    def load(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def store(self, key: str, value: Any) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(json.dumps(value), encoding="utf-8")

    def has(self, key: str) -> bool:
        return self._path(key).exists()
