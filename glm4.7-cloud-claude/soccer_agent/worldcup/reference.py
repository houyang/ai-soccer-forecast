# soccer_agent/worldcup/reference.py
"""Static 0-100 pedigree scores per national team (pre-tournament strength prior).

Values reflect historical World Cup pedigree and current federation standing. They are a
prior only; group-stage results override them in `form.recalibrated_strength`.
"""
from __future__ import annotations

_NEUTRAL = 50.0

# Curated pedigree (0-100). Tuned so traditional powers lead; minnows trail.
_STRENGTH: dict[str, float] = {
    "Argentina": 94.0, "France": 93.0, "Brazil": 90.0, "England": 89.0,
    "Spain": 88.0, "Germany": 87.0, "Portugal": 86.0, "Netherlands": 85.0,
    "Belgium": 82.0, "Croatia": 81.0, "Italy": 80.0,
    "Colombia": 78.0, "Uruguay": 78.0, "Morocco": 77.0, "Mexico": 76.0,
    "USA": 75.0, "Switzerland": 74.0, "Japan": 74.0, "Senegal": 73.0,
    "Ecuador": 71.0, "Australia": 70.0, "South Korea": 70.0, "Sweden": 70.0,
    "Norway": 72.0, "Austria": 71.0, "Czech Republic": 70.0, "Türkiye": 71.0,
    "Ivory Coast": 70.0, "Ghana": 69.0, "Egypt": 69.0, "Tunisia": 67.0,
    "Iran": 68.0, "Saudi Arabia": 65.0, "Iraq": 63.0, "Jordan": 60.0,
    "Qatar": 61.0, "Uzbekistan": 62.0, "Canada": 71.0, "Panama": 60.0,
    "Paraguay": 68.0, "Scotland": 69.0, "Wales": 70.0,
    "Algeria": 68.0, "Cape Verde Islands": 58.0, "Congo DR": 60.0,
    "Curaçao": 57.0, "Haiti": 58.0, "New Zealand": 55.0, "Bosnia & Herzegovina": 64.0,
    "South Africa": 66.0,
}


def country_strength(name: str) -> float:
    """Return a 0-100 pedigree score for a national team name; 50.0 if unknown."""
    return _STRENGTH.get(name, _NEUTRAL)
