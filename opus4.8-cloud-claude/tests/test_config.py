from pathlib import Path

import pytest

from soccer.config import AppConfig


def test_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in [
        "SOCCER_DATA_DIR",
        "SOCCER_OLLAMA_HOST",
        "SOCCER_OLLAMA_MODEL",
        "SOCCER_OLLAMA_TIMEOUT",
        "SOCCER_PROVIDER_MODE",
        "SOCCER_REASONER",
    ]:
        monkeypatch.delenv(var, raising=False)
    cfg = AppConfig.from_env()
    assert cfg.data_dir == Path("./data")
    assert cfg.ollama_model == "gemma4:12b-mlx"
    assert cfg.provider_mode == "fixture"
    assert cfg.reasoner == "fake"


def test_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOCCER_OLLAMA_MODEL", "other:7b")
    monkeypatch.setenv("SOCCER_REASONER", "ollama")
    monkeypatch.setenv("SOCCER_OLLAMA_TIMEOUT", "30")
    cfg = AppConfig.from_env()
    assert cfg.ollama_model == "other:7b"
    assert cfg.reasoner == "ollama"
    assert cfg.ollama_timeout == 30.0


def test_invalid_reasoner_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOCCER_REASONER", "bogus")
    with pytest.raises(ValueError):
        AppConfig.from_env()
