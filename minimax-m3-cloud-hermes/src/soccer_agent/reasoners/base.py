"""Reasoner protocol.

A reasoner consumes a MatchContext (Match + all Signals) and emits a
ReasonerOutput. Two reasoners run side by side in the orchestrator:

  - numeric:  deterministic, free, reproducible baseline
  - llm:      reads the same context, produces a rationale and probs

The eval harness scores both. The numeric reasoner is the floor the
LLM must beat to justify its API cost.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import MatchContext, ReasonerOutput


@runtime_checkable
class Reasoner(Protocol):
    name: str
    description: str
    version: str

    def run(self, context: MatchContext) -> ReasonerOutput: ...


def normalize_probs(p: dict[str, float]) -> dict[str, float]:
    """Renormalize a probs dict to sum to 1.0 (handles rounding drift)."""
    s = sum(p.values())
    if s <= 0:
        return {"home": 1/3, "draw": 1/3, "away": 1/3}
    return {k: v / s for k, v in p.items()}
