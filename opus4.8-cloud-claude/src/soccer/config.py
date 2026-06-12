from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_PROVIDER_MODES = {"fixture", "http"}
_REASONERS = {"fake", "ollama"}


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    ollama_host: str
    ollama_model: str
    ollama_timeout: float
    provider_mode: str
    reasoner: str
    api_football_base_url: str
    api_football_key: str | None
    prediction_dir: Path

    @classmethod
    def from_env(cls) -> AppConfig:
        provider_mode = os.environ.get("SOCCER_PROVIDER_MODE", "fixture")
        reasoner = os.environ.get("SOCCER_REASONER", "fake")
        if provider_mode not in _PROVIDER_MODES:
            raise ValueError(f"SOCCER_PROVIDER_MODE must be one of {_PROVIDER_MODES}")
        if reasoner not in _REASONERS:
            raise ValueError(f"SOCCER_REASONER must be one of {_REASONERS}")
        return cls(
            data_dir=Path(os.environ.get("SOCCER_DATA_DIR", "./data")),
            ollama_host=os.environ.get("SOCCER_OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.environ.get("SOCCER_OLLAMA_MODEL", "gemma4:12b-mlx"),
            ollama_timeout=float(os.environ.get("SOCCER_OLLAMA_TIMEOUT", "60")),
            provider_mode=provider_mode,
            reasoner=reasoner,
            api_football_base_url=os.environ.get(
                "SOCCER_API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io"
            ),
            api_football_key=os.environ.get("SOCCER_API_FOOTBALL_KEY") or None,
            prediction_dir=Path(os.environ.get("SOCCER_PREDICTION_DIR", "./perdiction")),
        )
