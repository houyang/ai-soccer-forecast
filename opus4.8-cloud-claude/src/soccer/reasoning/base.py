from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from soccer.models import MatchDossier, MatchResult, Outcome, Prediction


class ReasonerError(Exception):
    """Raised when a reasoner produces unusable output."""


@dataclass(frozen=True)
class ReasonResult:
    probs: dict[Outcome, float]
    confidence: float
    rationale: str


class Reasoner(Protocol):
    name: str

    def predict(self, dossier: MatchDossier) -> ReasonResult: ...

    def self_evaluate(self, prediction: Prediction, result: MatchResult) -> str: ...
