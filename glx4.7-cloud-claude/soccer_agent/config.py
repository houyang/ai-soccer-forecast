import os
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Config:
    database_url: str
    anthropic_api_key: str
    api_football_key: str
    odds_api_key: str | None = None
    openweather_api_key: str | None = None
    llm_model: str = "claude-3-5-sonnet-20240620"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    prediction_schedule: str = "0 */6 * * *"
    evaluation_schedule: str = "0 8 * * *"
    metrics_schedule: str = "0 9 * * 1"
    metrics_port: int = 9090
    tracing_enabled: bool = True

    # API URLs
    api_football_base_url: str = "https://api-football-v1.p.rapidapi.com"
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    openweather_base_url: str = "https://api.openweathermap.org/data/2.5"


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config(
            database_url=_get_required_env("DATABASE_URL"),
            anthropic_api_key=_get_required_env("ANTHROPIC_API_KEY"),
            api_football_key=_get_required_env("API_FOOTBALL_KEY"),
            odds_api_key=os.getenv("ODDS_API_KEY"),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
            llm_model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20240620"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            prediction_schedule=os.getenv("PREDICTION_SCHEDULE", "0 */6 * * *"),
            evaluation_schedule=os.getenv("EVALUATION_SCHEDULE", "0 8 * * *"),
            metrics_schedule=os.getenv("METRICS_SCHEDULE", "0 9 * * 1"),
            metrics_port=int(os.getenv("METRICS_PORT", "9090")),
            tracing_enabled=os.getenv("TRACING_ENABLED", "true").lower() == "true",
        )
    return _config


def _get_required_env(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"{key} is required")
    return value


def reset_config() -> None:
    """Reset config for testing purposes"""
    global _config
    _config = None