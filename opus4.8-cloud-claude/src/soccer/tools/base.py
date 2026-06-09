# src/soccer/tools/base.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolError(Exception):
    """Raised by a provider when data cannot be obtained."""


class MissingFixtureKey(ToolError):
    """Raised when a key is absent within an existing fixture section."""


@dataclass(frozen=True)
class Tool:
    """Uniform view of a capability for a future model-driven selection loop."""

    name: str
    description: str
    # Callable[..., Any] is intentional: providers have heterogeneous
    # signatures (get_form, get_odds, get_result, ...) under one uniform view.
    call: Callable[..., Any]
