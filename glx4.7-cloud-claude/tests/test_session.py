import pytest
from unittest.mock import patch, AsyncMock
from soccer_agent.db.session import get_async_session, init_db
from soccer_agent.config import get_config, reset_config


@pytest.mark.asyncio
async def test_get_async_session(monkeypatch):
    reset_config()
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
    monkeypatch.setenv("API_FOOTBALL_KEY", "test_key")

    session_gen = get_async_session()
    assert session_gen is not None


@pytest.mark.asyncio
async def test_init_db(monkeypatch):
    reset_config()
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
    monkeypatch.setenv("API_FOOTBALL_KEY", "test_key")

    # Should not raise
    await init_db()
    await close_db()


async def close_db():
    from soccer_agent.db.session import close_db
    await close_db()