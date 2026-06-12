"""Curated reference constants for signals the API does not expose directly.

These are static, well-known public approximations (country footballing pedigree, league
average attendance, confederation membership). They are deliberately coarse and serve as
documented inputs to the ranking models, with neutral defaults for anything unlisted so an
unknown entity never crashes a ranking -- it just lands mid-table.
"""

from __future__ import annotations

# Confederation per country (used for host/travel adjustments and national-team grouping).
CONFEDERATION: dict[str, str] = {
    # CONCACAF (host confederation)
    "USA": "CONCACAF",
    "Canada": "CONCACAF",
    "Mexico": "CONCACAF",
    "Costa Rica": "CONCACAF",
    "Panama": "CONCACAF",
    "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF",
    "Haiti": "CONCACAF",
    "Curaçao": "CONCACAF",
    # CONMEBOL
    "Brazil": "CONMEBOL",
    "Argentina": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Chile": "CONMEBOL",
    "Peru": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # UEFA
    "France": "UEFA",
    "Spain": "UEFA",
    "Germany": "UEFA",
    "England": "UEFA",
    "Portugal": "UEFA",
    "Netherlands": "UEFA",
    "Italy": "UEFA",
    "Belgium": "UEFA",
    "Croatia": "UEFA",
    "Denmark": "UEFA",
    "Switzerland": "UEFA",
    "Austria": "UEFA",
    "Sweden": "UEFA",
    "Poland": "UEFA",
    "Serbia": "UEFA",
    "Ukraine": "UEFA",
    "Wales": "UEFA",
    "Scotland": "UEFA",
    "Norway": "UEFA",
    "Czech Republic": "UEFA",
    "Bosnia & Herzegovina": "UEFA",
    "Hungary": "UEFA",
    "Greece": "UEFA",
    "Türkiye": "UEFA",
    # CAF
    "Morocco": "CAF",
    "Senegal": "CAF",
    "Nigeria": "CAF",
    "Egypt": "CAF",
    "Algeria": "CAF",
    "Tunisia": "CAF",
    "Cameroon": "CAF",
    "Ghana": "CAF",
    "Ivory Coast": "CAF",
    "South Africa": "CAF",
    "Mali": "CAF",
    "Cape Verde Islands": "CAF",
    "Congo DR": "CAF",
    # AFC
    "Japan": "AFC",
    "South Korea": "AFC",
    "Iran": "AFC",
    "Saudi Arabia": "AFC",
    "Australia": "AFC",
    "Qatar": "AFC",
    "Iraq": "AFC",
    "Uzbekistan": "AFC",
    "Jordan": "AFC",
    "United Arab Emirates": "AFC",
    # OFC
    "New Zealand": "OFC",
}

HOST_COUNTRIES: frozenset[str] = frozenset({"USA", "Canada", "Mexico"})

# Country footballing pedigree / strength on a 0-100 scale (history + recent stature).
# Doubles as the "national soccer history" and "country strength" ranking input.
COUNTRY_STRENGTH: dict[str, float] = {
    "Brazil": 96,
    "Argentina": 95,
    "France": 95,
    "Spain": 92,
    "England": 90,
    "Germany": 88,
    "Portugal": 88,
    "Netherlands": 86,
    "Italy": 85,
    "Belgium": 82,
    "Croatia": 80,
    "Uruguay": 80,
    "Colombia": 78,
    "Morocco": 76,
    "Mexico": 74,
    "Switzerland": 73,
    "USA": 72,
    "Denmark": 72,
    "Japan": 72,
    "Senegal": 71,
    "Serbia": 70,
    "Ecuador": 69,
    "South Korea": 69,
    "Austria": 68,
    "Ukraine": 67,
    "Poland": 67,
    "Nigeria": 67,
    "Sweden": 66,
    "Peru": 64,
    "Egypt": 64,
    "Algeria": 64,
    "Chile": 64,
    "Iran": 64,
    "Australia": 63,
    "Paraguay": 62,
    "Türkiye": 64,
    "Canada": 62,
    "Tunisia": 60,
    "Ivory Coast": 62,
    "Ghana": 61,
    "Cameroon": 61,
    "Norway": 64,
    "Scotland": 64,
    "Congo DR": 59,
    "Saudi Arabia": 58,
    "Qatar": 57,
    "Costa Rica": 58,
    "Panama": 56,
    "Uzbekistan": 55,
    "Iraq": 55,
    "Jordan": 52,
    "Cape Verde Islands": 52,
    "South Africa": 56,
    "New Zealand": 52,
    "Bosnia & Herzegovina": 62,
    "Czech Republic": 66,
    "Mali": 58,
    "Curaçao": 50,
    "Haiti": 50,
}

DEFAULT_COUNTRY_STRENGTH = 48.0

# Approximate average league attendance (most recent season), keyed by API league name.
LEAGUE_ATTENDANCE: dict[str, int] = {
    "Premier League": 40000,
    "Bundesliga": 43000,
    "La Liga": 30000,
    "Primera Division": 30000,
    "Serie A": 30000,
    "Ligue 1": 27000,
    "Major League Soccer": 23000,
    "Liga MX": 25000,
    "Eredivisie": 19000,
    "Primeira Liga": 14000,
    "Championship": 20000,
    "Serie A Brazil": 27000,
    "Brasileirao": 27000,
    "Liga Profesional Argentina": 22000,
    "Pro League": 26000,
    "Saudi Pro League": 9000,
    "Super Lig": 18000,
    "Scottish Premiership": 19000,
    "Jupiler Pro League": 19000,
    "Liga Portugal": 14000,
}

DEFAULT_LEAGUE_ATTENDANCE = 11000


def country_strength(country: str) -> float:
    return COUNTRY_STRENGTH.get(country, DEFAULT_COUNTRY_STRENGTH)


def confederation(country: str) -> str:
    return CONFEDERATION.get(country, "UEFA")


def league_attendance(name: str) -> int:
    return LEAGUE_ATTENDANCE.get(name, DEFAULT_LEAGUE_ATTENDANCE)
