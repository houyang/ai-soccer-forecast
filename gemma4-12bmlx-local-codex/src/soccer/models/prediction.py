from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class FormResult:
    date: str
    score: str
    opponent: str
    result: str  # "W", "D", "L"

@dataclass(frozen=True)
class Prediction:
    match_id: str
    predicted_outcome: str  # "HomeWin", "AwayWin", "Draw"
    rationale: str
    confidence_score: float  # 0.0 to 1.0
    source_data_summary: List[str] # Summaries of what was looked up
