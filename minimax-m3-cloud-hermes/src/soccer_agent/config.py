"""Environment-driven configuration.

All knobs are read from env vars. The agent never hard-codes paths or
keys, so the same code runs in tests, dev, and CI without changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = _env(name, ",".join(default))
    return [s.strip() for s in raw.split(",") if s.strip()]


@dataclass(frozen=True)
class Settings:
    # Storage
    db_path: str = field(default_factory=lambda: _env("SOCCER_AGENT_DB_PATH", "data/soccer_agent.db"))

    # Fixtures (deterministic tool fallbacks)
    fixtures_dir: str = field(default_factory=lambda: _env("SOCCER_AGENT_FIXTURES_DIR", "fixtures"))

    # LLM
    llm_api_key: str | None = field(default_factory=lambda: os.environ.get("SOCCER_AGENT_LLM_API_KEY") or None)
    llm_base_url: str = field(default_factory=lambda: _env("SOCCER_AGENT_LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_model: str = field(default_factory=lambda: _env("SOCCER_AGENT_LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(default_factory=lambda: _env_float("SOCCER_AGENT_LLM_TEMPERATURE", 0.2))

    # HTTP
    http_timeout: int = field(default_factory=lambda: _env_int("SOCCER_AGENT_HTTP_TIMEOUT", 10))
    http_retries: int = field(default_factory=lambda: _env_int("SOCCER_AGENT_HTTP_RETRIES", 2))

    # Reasoners
    reasoners: list[str] = field(default_factory=lambda: _env_list("SOCCER_AGENT_REASONERS", ["numeric"]))
    final_pick_policy: str = field(default_factory=lambda: _env("SOCCER_AGENT_FINAL_PICK_POLICY", "numeric"))

    # API
    api_host: str = field(default_factory=lambda: _env("SOCCER_AGENT_API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("SOCCER_AGENT_API_PORT", 8000))

    # ResultScout
    scout_poll_seconds: int = field(default_factory=lambda: _env_int("SOCCER_AGENT_SCOUT_POLL_SECONDS", 300))

    def ensure_paths(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    s = Settings()
    s.ensure_paths()
    return s
