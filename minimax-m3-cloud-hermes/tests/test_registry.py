"""Tests for the tool protocol and ToolRegistry."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field

from soccer_agent.tools import BaseTool, ToolError, ToolRegistry, ToolResult


# -- fakes --------------------------------------------------------------------


class _In(BaseModel):
    x: int = Field(ge=0)


class _Out(BaseModel):
    doubled: int


class _Doubler:
    name = "doubler"
    description = "doubles x"
    input_model = _In
    output_model = _Out

    def __init__(self):
        self.calls = 0

    async def run(self, payload: _In) -> _Out:  # type: ignore[override]
        self.calls += 1
        return _Out(doubled=payload.x * 2)


class _Flaky:
    name = "flaky"
    description = "fails twice, then succeeds"
    input_model = _In
    output_model = _Out
    fail_with: Exception = ToolError(source="live", message="boom", retriable=True)
    fail_times: int = 2
    calls: int = 0

    async def run(self, payload: _In) -> _Out:  # type: ignore[override]
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.fail_with
        return _Out(doubled=42)


class _Slow:
    name = "slow"
    description = "sleeps longer than the timeout"
    input_model = _In
    output_model = _Out

    async def run(self, payload: _In) -> _Out:  # type: ignore[override]
        await asyncio.sleep(0.2)
        return _Out(doubled=payload.x)


class _BadOut:
    name = "bad_out"
    description = "returns wrong shape"
    input_model = _In
    output_model = _Out

    async def run(self, payload: _In) -> _Out:  # type: ignore[override]
        # Return a model that doesn't match _Out (caller ignores type hint).
        class _Wrong(BaseModel):
            tripled: int

        return _Wrong(tripled=1)  # type: ignore[return-value]


# -- tests --------------------------------------------------------------------


def test_registry_register_and_get():
    reg = ToolRegistry()
    d = _Doubler()
    reg.register(d)
    assert reg.get("doubler") is d
    assert "doubler" in reg.names


def test_registry_rejects_duplicate_names():
    reg = ToolRegistry()
    reg.register(_Doubler())
    with pytest.raises(ValueError):
        reg.register(_Doubler())


def test_registry_unknown_tool_returns_failure_not_raises():
    reg = ToolRegistry()
    res = asyncio.run(reg.run("nope", _In(x=1)))
    assert res.ok is False
    assert res.source == "registry"
    assert "unknown tool" in (res.error or "")


def test_registry_happy_path_returns_validated_output():
    reg = ToolRegistry()
    d = _Doubler()
    reg.register(d)
    res = asyncio.run(reg.run("doubler", _In(x=5)))
    assert res.ok is True
    assert res.data is not None
    assert res.data.doubled == 10
    assert res.source == "live"
    assert res.attempts == 1


def test_registry_retries_on_retriable_tool_error():
    reg = ToolRegistry(default_retries=2, default_timeout=2.0)
    f = _Flaky()
    reg.register(f)
    res = asyncio.run(reg.run("flaky", _In(x=1)))
    assert res.ok is True
    assert res.attempts == 3  # failed twice, then succeeded
    assert f.calls == 3


def test_registry_does_not_retry_non_retriable_error():
    reg = ToolRegistry(default_retries=3, default_timeout=2.0)

    class _NonRetriable(BaseTool):
        name = "nr"
        description = "x"
        input_model = _In
        output_model = _Out

        async def run(self, payload):
            raise ToolError(source="live", message="nope", retriable=False)

    reg.register(_NonRetriable())
    res = asyncio.run(reg.run("nr", _In(x=1)))
    assert res.ok is False
    assert res.attempts == 1


def test_registry_timeout_triggers_retry():
    reg = ToolRegistry(default_retries=1, default_timeout=0.05)
    reg.register(_Slow())
    res = asyncio.run(reg.run("slow", _In(x=1)))
    assert res.ok is False
    assert "timeout" in (res.error or "").lower()
    assert res.attempts == 2  # 1 retry


def test_registry_captures_unexpected_exception():
    reg = ToolRegistry(default_retries=0)

    class _Boom(BaseTool):
        name = "boom"
        description = "x"
        input_model = _In
        output_model = _Out

        async def run(self, payload):
            raise RuntimeError("kapow")

    reg.register(_Boom())
    res = asyncio.run(reg.run("boom", _In(x=1)))
    assert res.ok is False
    assert "kapow" in (res.error or "")
    assert res.source == "tool"


def test_registry_validates_payload_against_input_model():
    reg = ToolRegistry()
    reg.register(_Doubler())

    class _BadIn(BaseModel):
        x: int = "not-an-int"  # type: ignore[assignment]

    res = asyncio.run(reg.run("doubler", _BadIn()))
    assert res.ok is False
    assert "payload does not match" in (res.error or "")


def test_registry_validates_output_against_output_model():
    reg = ToolRegistry()
    reg.register(_BadOut())
    res = asyncio.run(reg.run("bad_out", _In(x=1)))
    assert res.ok is False
    assert "output did not match" in (res.error or "")


def test_toolresult_dataclass_construction():
    r = ToolResult(ok=True, source="live", data=_Out(doubled=2), duration_ms=3, attempts=1)
    assert r.ok and r.data and r.data.doubled == 2
    r2 = ToolResult(ok=False, source="tool", error="x")
    assert not r2.ok and r2.error == "x"


def test_toolerror_carries_source_and_retriable():
    e = ToolError(source="live", message="m", retriable=False)
    assert e.source == "live"
    assert e.retriable is False
    assert "m" in str(e)
