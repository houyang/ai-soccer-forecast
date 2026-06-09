import pytest
from soccer_agent.config import get_config, reset_config


def test_config_loads_from_env(monkeypatch):
    reset_config()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
    monkeypatch.setenv("API_FOOTBALL_KEY", "api_football_key")
    monkeypatch.setenv("LLM_MODEL", "claude-3-5-sonnet-20240620")

    config = get_config()
    assert config.database_url == "postgresql://test"
    assert config.anthropic_api_key == "test_key"
    assert config.api_football_key == "api_football_key"
    assert config.llm_model == "claude-3-5-sonnet-20240620"


def test_config_missing_required_key(monkeypatch):
    reset_config()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL is required"):
        get_config()