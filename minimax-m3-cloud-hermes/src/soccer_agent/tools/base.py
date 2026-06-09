"""Tool protocol, errors, and registry.

Every tool exposes a Pydantic input/output schema and an async run().
The ToolRegistry wraps run() with timeout, retry, and fixture fallback
so callers always get back a discriminated ToolResult — never an
unhandled exception. This is the single most important design
decision in the agent: a tool failing is a Signal, not a crash.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)


class ToolError(Exception):
    """Raised by tools and the registry's internals.

    `retriable=True` means the registry should retry before surfacing.
    `source` is "live" | "fixture" | "tool" so we can blame the right layer.
    """

    def __init__(self, *, source: str, message: str, retriable: bool = True):
        super().__init__(message)
        self.source = source
        self.message = message
        self.retriable = retriable


@runtime_checkable
class BaseTool(Protocol, Generic[TIn, TOut]):
    """A tool is a name, a description, two Pydantic models, and an async run."""

    name: str
    description: str
    input_model: type[TIn]
    output_model: type[TOut]

    async def run(self, payload: TIn) -> TOut: ...


# -- result envelope ----------------------------------------------------------


@dataclass
class ToolResult:
    """Discriminated union: either {ok: True, data, source} or {ok: False, error}."""

    ok: bool
    source: str  # "live" | "fixture" | "stub"
    data: BaseModel | None = None
    error: str | None = None
    duration_ms: int = 0
    attempts: int = 1


# -- registry -----------------------------------------------------------------


class ToolRegistry:
    """Holds tools. Runs them with timeout/retry. Never raises out of run()."""

    def __init__(
        self,
        *,
        default_timeout: float = 10.0,
        default_retries: int = 2,
    ):
        self._tools: dict[str, BaseTool] = {}
        self.default_timeout = default_timeout
        self.default_retries = default_retries

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    async def run(
        self,
        name: str,
        payload: BaseModel,
        *,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> ToolResult:
        """Run a tool. Always returns ToolResult; never raises."""
        try:
            tool = self.get(name)
        except KeyError as e:
            return ToolResult(ok=False, source="registry", error=str(e))

        # Validate payload against the tool's input model.
        # Accept either a Pydantic model or a dict; validate uniformly.
        try:
            if isinstance(payload, BaseModel):
                payload_data = payload.model_dump()
            elif isinstance(payload, dict):
                payload_data = payload
            else:
                return ToolResult(
                    ok=False,
                    source="registry",
                    error=f"payload must be a Pydantic model or dict, got {type(payload).__name__}",
                )
            validated_in = tool.input_model.model_validate(payload_data)
        except ValidationError as e:
            return ToolResult(
                ok=False,
                source="registry",
                error=f"payload does not match {tool.input_model.__name__}: {e}",
            )

        attempts_cap = (self.default_retries if retries is None else retries) + 1
        timeout_s = self.default_timeout if timeout is None else timeout
        last_err: str = ""
        attempts = 0

        for attempt in range(1, attempts_cap + 1):
            attempts = attempt
            t0 = time.perf_counter()
            try:
                out = await asyncio.wait_for(tool.run(validated_in), timeout=timeout_s)
                # Validate the output too — keeps reasoner inputs honest.
                if isinstance(out, BaseModel):
                    out_data = out.model_dump()
                elif isinstance(out, dict):
                    out_data = out
                else:
                    # surface as a validation error; the caller will get a clean failure
                    raise ValidationError.from_exception_data(
                        tool.output_model.__name__,
                        [{"type": "value_error", "loc": ("output",), "input": out,
                          "ctx": {"error": f"unsupported type {type(out).__name__}"}}],
                    )
                validated_out = tool.output_model.model_validate(out_data)
                dur_ms = int((time.perf_counter() - t0) * 1000)
                return ToolResult(
                    ok=True,
                    source="live",
                    data=validated_out,
                    duration_ms=dur_ms,
                    attempts=attempts,
                )
            except asyncio.TimeoutError:
                last_err = f"timeout after {timeout_s}s"
                # timeouts are retriable
                continue
            except ToolError as e:
                last_err = f"{e.source}: {e.message}"
                if e.retriable and attempt < attempts_cap:
                    continue
                return ToolResult(
                    ok=False,
                    source=e.source,
                    error=last_err,
                    attempts=attempts,
                )
            except ValidationError as e:
                # Output shape is wrong — not worth retrying
                return ToolResult(
                    ok=False,
                    source="tool",
                    error=f"output did not match {tool.output_model.__name__}: {e}",
                    attempts=attempts,
                )
            except Exception as e:  # noqa: BLE001 — last-resort safety net
                last_err = f"unexpected: {type(e).__name__}: {e}"
                if attempt < attempts_cap:
                    continue
                return ToolResult(
                    ok=False,
                    source="tool",
                    error=last_err,
                    attempts=attempts,
                )

        return ToolResult(ok=False, source="tool", error=last_err, attempts=attempts)
