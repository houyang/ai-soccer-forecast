from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass(frozen=True)
class Team:
    name: str
    short_name: str

@dataclass(frozen=True)
class Match:
    home_team: Team
    away_team: Team
    timestamp: datetime
    venue: Optional[str] = None
