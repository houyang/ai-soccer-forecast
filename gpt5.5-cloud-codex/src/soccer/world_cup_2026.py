"""World Cup 2026 profile loading, rankings, and score predictions."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast

from soccer.models import Outcome

JsonObject = dict[str, object]

DEFAULT_WORLD_CUP_DATA_DIR = Path("data/api-football/world-cup-2026")
HOST_COUNTRIES = {"Canada", "Mexico", "United States"}
HOT_WEATHER_CITIES = {
    "Atlanta",
    "Arlington",
    "Dallas",
    "Houston",
    "Kansas City",
    "Miami",
    "Miami Gardens",
    "Monterrey",
}
HOST_CITY_COUNTRIES = {
    "Atlanta": "United States",
    "Arlington": "United States",
    "Boston": "United States",
    "Dallas": "United States",
    "East Rutherford": "United States",
    "Guadalajara": "Mexico",
    "Houston": "United States",
    "Inglewood": "United States",
    "Kansas City": "United States",
    "Los Angeles": "United States",
    "Mexico City": "Mexico",
    "Miami": "United States",
    "Miami Gardens": "United States",
    "Monterrey": "Mexico",
    "New York New Jersey": "United States",
    "New York/New Jersey": "United States",
    "Philadelphia": "United States",
    "Santa Clara": "United States",
    "San Francisco Bay Area": "United States",
    "Seattle": "United States",
    "Toronto": "Canada",
    "Vancouver": "Canada",
}
AMERICAS_TEAMS = {
    "Argentina",
    "Brazil",
    "Canada",
    "Colombia",
    "Curacao",
    "Ecuador",
    "Haiti",
    "Mexico",
    "Panama",
    "Paraguay",
    "United States",
    "Uruguay",
}
WARM_WEATHER_TEAMS = {
    "Algeria",
    "Argentina",
    "Brazil",
    "Cabo Verde",
    "Colombia",
    "Congo DR",
    "Curacao",
    "Ecuador",
    "Egypt",
    "Ghana",
    "Haiti",
    "Iran",
    "Iraq",
    "Ivory Coast",
    "Jordan",
    "Mexico",
    "Morocco",
    "Panama",
    "Paraguay",
    "Qatar",
    "Saudi Arabia",
    "Senegal",
    "South Africa",
    "Tunisia",
    "Uruguay",
    "Uzbekistan",
}
COUNTRY_HISTORY_SCORES = {
    "Argentina": 98.0,
    "Austria": 72.0,
    "Australia": 65.0,
    "Belgium": 84.0,
    "Brazil": 100.0,
    "Canada": 62.0,
    "Cabo Verde": 52.0,
    "Colombia": 80.0,
    "Congo DR": 58.0,
    "Croatia": 86.0,
    "Curacao": 45.0,
    "Czechia": 74.0,
    "Ecuador": 72.0,
    "Egypt": 70.0,
    "England": 90.0,
    "France": 96.0,
    "Germany": 97.0,
    "Ghana": 70.0,
    "Haiti": 48.0,
    "Iran": 70.0,
    "Iraq": 58.0,
    "Ivory Coast": 73.0,
    "Japan": 78.0,
    "Jordan": 55.0,
    "Korea Republic": 76.0,
    "Mexico": 77.0,
    "Morocco": 79.0,
    "Netherlands": 89.0,
    "New Zealand": 52.0,
    "Norway": 72.0,
    "Panama": 56.0,
    "Paraguay": 72.0,
    "Portugal": 89.0,
    "Qatar": 61.0,
    "Saudi Arabia": 62.0,
    "Scotland": 68.0,
    "Senegal": 76.0,
    "South Africa": 61.0,
    "Spain": 93.0,
    "Sweden": 79.0,
    "Switzerland": 78.0,
    "Tunisia": 67.0,
    "Turkiye": 74.0,
    "United States": 73.0,
    "Uruguay": 88.0,
    "Uzbekistan": 57.0,
}


@dataclass(frozen=True)
class PlayerProfile:
    """A tournament player with the club-season statistics used for ranking."""

    player_id: int
    name: str
    national_team: str
    age: int | None
    club_id: int | None
    club: str | None
    league_id: int | None
    league: str | None
    position: str | None
    goals: int
    average_rating: float | None


@dataclass(frozen=True)
class CoachProfile:
    """A national-team coach with record and trophy context."""

    coach_id: int
    name: str
    national_team: str
    age: int | None
    wins: int
    draws: int
    losses: int
    titles_won: int
    career_teams: tuple[str, ...]


@dataclass(frozen=True)
class ClubProfile:
    """A club represented by at least one World Cup 2026 player."""

    club_id: int
    name: str
    country: str
    league_id: int
    league: str
    wins: int
    draws: int
    losses: int
    major_titles_won: int
    fifa_player_count: int


@dataclass(frozen=True)
class LeagueProfile:
    """A league represented by clubs with World Cup 2026 players."""

    league_id: int
    name: str
    country: str
    total_teams: int
    matches_played: int
    average_attendance: int
    fifa_player_count: int


@dataclass(frozen=True)
class NationalTeamProfile:
    """A World Cup 2026 national team profile."""

    team_id: int | None
    name: str
    group: str | None
    roster: tuple[int, ...]
    coaching_staff: tuple[int, ...]
    recent_wins: int
    recent_draws: int
    recent_losses: int
    goals_for: int
    goals_against: int
    history_score: float


@dataclass(frozen=True)
class GroupStageMatch:
    """A first-round group stage match."""

    match_id: str
    group: str | None
    round_number: int | None
    home_team: str
    away_team: str
    kickoff: datetime
    venue_name: str
    venue_city: str
    venue_country: str


@dataclass(frozen=True)
class CompletedGroupMatch:
    """A completed group-stage match result used for tournament updates."""

    match_id: str
    group: str | None
    round_number: int | None
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    outcome: Outcome


@dataclass(frozen=True)
class TeamTournamentUpdate:
    """In-tournament evidence from completed matches and tactical snapshots."""

    team: str
    matches: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    formations: tuple[str, ...]
    starter_ids: tuple[int, ...]
    substitute_ids: tuple[int, ...]

    @property
    def points_per_match(self) -> float:
        if self.matches == 0:
            return 0.0
        return ((self.wins * 3) + self.draws) / self.matches

    @property
    def goal_difference_per_match(self) -> float:
        if self.matches == 0:
            return 0.0
        return (self.goals_for - self.goals_against) / self.matches


@dataclass(frozen=True)
class WorldCupDataSet:
    """Normalized World Cup 2026 data loaded from local API-Football snapshots."""

    players: dict[int, PlayerProfile]
    coaches: dict[int, CoachProfile]
    clubs: dict[int, ClubProfile]
    leagues: dict[int, LeagueProfile]
    national_teams: dict[str, NationalTeamProfile]
    matches: tuple[GroupStageMatch, ...]
    completed_matches: dict[str, CompletedGroupMatch]
    team_updates: dict[str, TeamTournamentUpdate]
    external_factors: JsonObject


@dataclass(frozen=True)
class WorldCupRankings:
    """All model rankings on a 0-100 scale."""

    leagues: dict[int, float]
    clubs: dict[int, float]
    players: dict[int, float]
    coaches: dict[int, float]
    national_teams: dict[str, float]


@dataclass(frozen=True)
class WorldCupScorePrediction:
    """A deterministic final-score prediction for a group stage match."""

    match_id: str
    group: str | None
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    outcome: Outcome
    confidence: float
    home_adjusted_rating: float
    away_adjusted_rating: float
    home_tournament_adjustment: float
    away_tournament_adjustment: float
    home_expected_goals: float
    away_expected_goals: float
    rationale: str


def load_world_cup_dataset(
    snapshot_dir: Path = DEFAULT_WORLD_CUP_DATA_DIR,
    *,
    expected_team_count: int | None = None,
    completed_round_limit: int | None = None,
    completed_before: datetime | None = None,
) -> WorldCupDataSet:
    """Load normalized World Cup data from a local API-Football snapshot directory."""

    fixtures_path = snapshot_dir / "fixtures_world_cup.json"
    if not fixtures_path.exists():
        raise FileNotFoundError(
            f"Missing {fixtures_path}. Run fetch-world-cup-data before loading predictions."
        )

    external_factors = _read_optional_json(snapshot_dir / "external_factors.json")
    fixtures_payload = _read_json(fixtures_path)
    matches, team_ids, group_by_team = _matches_from_payload(fixtures_payload)
    completed_matches = _completed_matches_from_payload(
        fixtures_payload,
        max_round=completed_round_limit,
        before=completed_before,
    )

    team_payload = _read_optional_json(snapshot_dir / "teams_world_cup.json")
    team_ids.update(_team_ids_from_teams_payload(team_payload))

    standings_payload = _read_optional_json(snapshot_dir / "standings_world_cup.json")
    group_by_team.update(_groups_from_standings_payload(standings_payload))
    matches = _matches_with_group_mapping(matches, group_by_team)

    recent_forms = {
        team_name: _recent_team_form(snapshot_dir, team_id)
        for team_name, team_id in team_ids.items()
        if team_id is not None
    }
    players, rosters = _load_players(snapshot_dir, team_ids)
    coaches, coaching_staff = _load_coaches(snapshot_dir, team_ids, recent_forms)
    leagues = _load_leagues(snapshot_dir, players, external_factors)
    clubs = _load_clubs(snapshot_dir, players, external_factors, leagues)
    national_teams = _load_national_teams(
        team_ids,
        group_by_team,
        rosters,
        coaching_staff,
        recent_forms,
        external_factors,
    )
    team_updates = _load_team_tournament_updates(
        snapshot_dir,
        matches,
        completed_matches,
    )

    if expected_team_count is not None and len(national_teams) != expected_team_count:
        raise ValueError(
            f"Expected {expected_team_count} national teams, loaded {len(national_teams)}"
        )

    return WorldCupDataSet(
        players=players,
        coaches=coaches,
        clubs=clubs,
        leagues=leagues,
        national_teams=national_teams,
        matches=matches,
        completed_matches=completed_matches,
        team_updates=team_updates,
        external_factors=external_factors,
    )


def rank_world_cup_entities(dataset: WorldCupDataSet) -> WorldCupRankings:
    """Rank leagues, clubs, players, coaches, and national teams from 0 to 100."""

    league_scores = _rank_leagues(dataset)
    club_scores = _rank_clubs(dataset, league_scores)
    player_scores = _rank_players(dataset, league_scores, club_scores)
    coach_scores = _rank_coaches(dataset)
    national_team_scores = _rank_national_teams(
        dataset,
        league_scores,
        club_scores,
        player_scores,
        coach_scores,
    )
    return WorldCupRankings(
        leagues=league_scores,
        clubs=club_scores,
        players=player_scores,
        coaches=coach_scores,
        national_teams=national_team_scores,
    )


def predict_group_stage_scores(
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    *,
    remaining_only: bool = False,
) -> tuple[WorldCupScorePrediction, ...]:
    """Predict final scores for every first-round group stage match in the dataset."""

    predictions: list[WorldCupScorePrediction] = []
    for match in sorted(dataset.matches, key=lambda item: item.kickoff):
        if remaining_only and match.match_id in dataset.completed_matches:
            continue
        home_base = rankings.national_teams.get(match.home_team, 50.0)
        away_base = rankings.national_teams.get(match.away_team, 50.0)
        home_location = _location_adjustment(match.home_team, match, is_home=True)
        away_location = _location_adjustment(match.away_team, match, is_home=False)
        home_tournament = _tournament_adjustment(dataset, rankings, match.home_team)
        away_tournament = _tournament_adjustment(dataset, rankings, match.away_team)
        home_rating = _clamp_score(home_base + home_location + home_tournament)
        away_rating = _clamp_score(away_base + away_location + away_tournament)

        home_xg, away_xg = _expected_goals(dataset, match, home_rating, away_rating)
        home_score = _score_from_expected_goals(home_xg)
        away_score = _score_from_expected_goals(away_xg)
        rating_edge = home_rating - away_rating
        if home_score == away_score and abs(rating_edge) >= 10.0:
            if rating_edge > 0:
                home_score += 1
            else:
                away_score += 1

        outcome = _outcome_for_score(home_score, away_score)
        confidence = round(min(0.82, 0.34 + abs(rating_edge) / 140), 3)
        rationale = (
            f"{match.home_team} adjusted rating {home_rating:.1f} vs "
            f"{match.away_team} {away_rating:.1f}; expected goals "
            f"{home_xg:.2f}-{away_xg:.2f}. Location adjustment is "
            f"{home_location:+.1f}/{away_location:+.1f} and tournament "
            f"update is {home_tournament:+.1f}/{away_tournament:+.1f} "
            f"for {match.venue_city}, {match.venue_country}. "
            f"{_team_update_summary(dataset, rankings, match.home_team)} "
            f"{_team_update_summary(dataset, rankings, match.away_team)}"
        )
        predictions.append(
            WorldCupScorePrediction(
                match_id=match.match_id,
                group=match.group,
                home_team=match.home_team,
                away_team=match.away_team,
                home_score=home_score,
                away_score=away_score,
                outcome=outcome,
                confidence=confidence,
                home_adjusted_rating=round(home_rating, 3),
                away_adjusted_rating=round(away_rating, 3),
                home_tournament_adjustment=round(home_tournament, 3),
                away_tournament_adjustment=round(away_tournament, 3),
                home_expected_goals=round(home_xg, 3),
                away_expected_goals=round(away_xg, 3),
                rationale=rationale,
            )
        )
    return tuple(predictions)


def prediction_to_json(prediction: WorldCupScorePrediction) -> JsonObject:
    """Return a JSON-ready representation of a score prediction."""

    return {
        "match_id": prediction.match_id,
        "group": prediction.group,
        "home_team": prediction.home_team,
        "away_team": prediction.away_team,
        "home_score": prediction.home_score,
        "away_score": prediction.away_score,
        "outcome": prediction.outcome.value,
        "confidence": prediction.confidence,
        "home_adjusted_rating": prediction.home_adjusted_rating,
        "away_adjusted_rating": prediction.away_adjusted_rating,
        "home_tournament_adjustment": prediction.home_tournament_adjustment,
        "away_tournament_adjustment": prediction.away_tournament_adjustment,
        "home_expected_goals": prediction.home_expected_goals,
        "away_expected_goals": prediction.away_expected_goals,
        "rationale": prediction.rationale,
    }


def render_group_stage_markdown(
    predictions: Iterable[WorldCupScorePrediction],
    *,
    title: str = "FIFA 2026 World Cup Group Stage Predictions",
) -> str:
    """Render score predictions as one Markdown table per World Cup group."""

    grouped: dict[str, list[WorldCupScorePrediction]] = {}
    for prediction in predictions:
        grouped.setdefault(prediction.group or "Ungrouped", []).append(prediction)

    lines = [f"# {title}", ""]
    for group in sorted(grouped, key=_group_sort_key):
        lines.extend(
            [
                f"## {_markdown_text(group)}",
                "",
                "| Match | Home | Score | Away | Pick | Confidence |",
                "|---|---|---:|---|---|---:|",
            ]
        )
        for index, prediction in enumerate(grouped[group], start=1):
            lines.append(
                "| "
                f"{index} | "
                f"{_markdown_text(prediction.home_team)} | "
                f"{prediction.home_score}-{prediction.away_score} | "
                f"{_markdown_text(prediction.away_team)} | "
                f"{_markdown_text(_prediction_pick_label(prediction))} | "
                f"{prediction.confidence:.0%} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _matches_from_payload(
    payload: JsonObject,
) -> tuple[tuple[GroupStageMatch, ...], dict[str, int | None], dict[str, str | None]]:
    matches: list[GroupStageMatch] = []
    team_ids: dict[str, int | None] = {}
    group_by_team: dict[str, str | None] = {}
    for item in _response_items(payload):
        group = _group_from_fixture(item)
        if not _is_group_stage_fixture(item, group):
            continue
        home_name, home_id = _fixture_team(item, "home")
        away_name, away_id = _fixture_team(item, "away")
        if home_name is None or away_name is None:
            continue
        team_ids.setdefault(home_name, home_id)
        team_ids.setdefault(away_name, away_id)
        group_by_team[home_name] = group
        group_by_team[away_name] = group

        fixture = _mapping(item.get("fixture")) or item
        venue = _mapping(fixture.get("venue")) or _mapping(item.get("venue")) or {}
        match_id = _string_value(fixture.get("id")) or _string_value(item.get("match_id"))
        if match_id is None:
            match_id = f"{_slug(home_name)}-{_slug(away_name)}"
        kickoff = _datetime_value(fixture.get("date")) or _datetime_value(item.get("kickoff"))
        venue_city = _first_text(venue, ("city", "hostCity")) or "Unknown city"
        matches.append(
            GroupStageMatch(
                match_id=f"wc-2026-{_slug(match_id)}",
                group=group,
                round_number=_round_number_from_fixture(item),
                home_team=home_name,
                away_team=away_name,
                kickoff=kickoff or datetime(2026, 6, 11, tzinfo=UTC),
                venue_name=_first_text(venue, ("name", "venueName")) or "Unknown venue",
                venue_city=venue_city,
                venue_country=(
                    _first_text(venue, ("country", "countryName"))
                    or HOST_CITY_COUNTRIES.get(venue_city, "Unknown country")
                ),
            )
        )
    return tuple(matches), team_ids, group_by_team


def _team_ids_from_teams_payload(payload: JsonObject) -> dict[str, int | None]:
    team_ids: dict[str, int | None] = {}
    for item in _response_items(payload):
        team = _mapping(item.get("team")) or item
        name = _first_text(team, ("name", "country"))
        if name is not None:
            team_ids[name] = _int_value(team.get("id"))
    return team_ids


def _groups_from_standings_payload(payload: JsonObject) -> dict[str, str]:
    groups: dict[str, str] = {}
    for item in _response_items(payload):
        league = _mapping(item.get("league")) or item
        standings = league.get("standings")
        if not isinstance(standings, list):
            continue
        for table in standings:
            if not isinstance(table, list):
                continue
            for row in table:
                standing = _mapping(row)
                if not standing:
                    continue
                team = _mapping(standing.get("team")) or {}
                team_name = _first_text(team, ("name",))
                group = _specific_group_label(_first_text(standing, ("group",)))
                if team_name is not None and group is not None:
                    groups[team_name] = group
    return groups


def _matches_with_group_mapping(
    matches: tuple[GroupStageMatch, ...],
    group_by_team: Mapping[str, str | None],
) -> tuple[GroupStageMatch, ...]:
    updated: list[GroupStageMatch] = []
    for match in matches:
        group = (
            match.group or group_by_team.get(match.home_team) or group_by_team.get(match.away_team)
        )
        updated.append(
            GroupStageMatch(
                match_id=match.match_id,
                group=group,
                round_number=match.round_number,
                home_team=match.home_team,
                away_team=match.away_team,
                kickoff=match.kickoff,
                venue_name=match.venue_name,
                venue_city=match.venue_city,
                venue_country=match.venue_country,
            )
        )
    return tuple(updated)


def _completed_matches_from_payload(
    payload: JsonObject,
    *,
    max_round: int | None,
    before: datetime | None,
) -> dict[str, CompletedGroupMatch]:
    completed: dict[str, CompletedGroupMatch] = {}
    completed_statuses = {"FT", "AET", "PEN"}
    for item in _response_items(payload):
        group = _group_from_fixture(item)
        if not _is_group_stage_fixture(item, group):
            continue
        round_number = _round_number_from_fixture(item)
        if max_round is not None and (round_number is None or round_number > max_round):
            continue
        fixture = _mapping(item.get("fixture")) or item
        kickoff = _datetime_value(fixture.get("date")) or _datetime_value(item.get("kickoff"))
        if before is not None and kickoff is not None and kickoff >= before:
            continue
        status = _mapping(fixture.get("status")) or {}
        if _first_text(status, ("short",)) not in completed_statuses:
            continue
        home_name, _home_id = _fixture_team(item, "home")
        away_name, _away_id = _fixture_team(item, "away")
        home_score, away_score = _fixture_score(item)
        match_id = _string_value(fixture.get("id")) or _string_value(item.get("match_id"))
        if (
            home_name is None
            or away_name is None
            or home_score is None
            or away_score is None
            or match_id is None
        ):
            continue
        normalized_match_id = f"wc-2026-{_slug(match_id)}"
        completed[normalized_match_id] = CompletedGroupMatch(
            match_id=normalized_match_id,
            group=group,
            round_number=round_number,
            home_team=home_name,
            away_team=away_name,
            home_score=home_score,
            away_score=away_score,
            outcome=_outcome_for_score(home_score, away_score),
        )
    return completed


def _load_team_tournament_updates(
    snapshot_dir: Path,
    matches: Iterable[GroupStageMatch],
    completed_matches: Mapping[str, CompletedGroupMatch],
) -> dict[str, TeamTournamentUpdate]:
    match_by_id = {match.match_id: match for match in matches}
    builders: dict[str, _TeamTournamentUpdateBuilder] = {}
    for completed in completed_matches.values():
        _record_completed_match_update(builders, completed.home_team, completed, is_home=True)
        _record_completed_match_update(builders, completed.away_team, completed, is_home=False)

        match = match_by_id.get(completed.match_id)
        if match is None:
            continue
        fixture_id = _fixture_id_from_match_id(match.match_id)
        if fixture_id is None:
            continue
        _record_lineup_updates(snapshot_dir, fixture_id, builders)
        _record_substitution_updates(snapshot_dir, fixture_id, builders)

    return {
        team: TeamTournamentUpdate(
            team=team,
            matches=builder.matches,
            wins=builder.wins,
            draws=builder.draws,
            losses=builder.losses,
            goals_for=builder.goals_for,
            goals_against=builder.goals_against,
            formations=tuple(
                formation
                for formation, count in sorted(
                    builder.formations.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                for _index in range(count)
            ),
            starter_ids=tuple(builder.starter_ids),
            substitute_ids=tuple(builder.substitute_ids),
        )
        for team, builder in builders.items()
    }


@dataclass
class _TeamTournamentUpdateBuilder:
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    formations: Counter[str] = field(default_factory=Counter)
    starter_ids: list[int] = field(default_factory=list)
    substitute_ids: list[int] = field(default_factory=list)


def _record_completed_match_update(
    builders: dict[str, _TeamTournamentUpdateBuilder],
    team_name: str,
    completed: CompletedGroupMatch,
    *,
    is_home: bool,
) -> None:
    builder = builders.setdefault(team_name, _TeamTournamentUpdateBuilder())
    builder.matches += 1
    team_score = completed.home_score if is_home else completed.away_score
    opponent_score = completed.away_score if is_home else completed.home_score
    builder.goals_for += team_score
    builder.goals_against += opponent_score
    if team_score > opponent_score:
        builder.wins += 1
    elif team_score < opponent_score:
        builder.losses += 1
    else:
        builder.draws += 1


def _record_lineup_updates(
    snapshot_dir: Path,
    fixture_id: int,
    builders: dict[str, _TeamTournamentUpdateBuilder],
) -> None:
    payload = _read_optional_json(snapshot_dir / f"fixture_{fixture_id}_lineups.json")
    for item in _response_items(payload):
        team = _mapping(item.get("team")) or {}
        team_name = _first_text(team, ("name",))
        if team_name is None:
            continue
        builder = builders.setdefault(team_name, _TeamTournamentUpdateBuilder())
        formation = _first_text(item, ("formation",))
        if formation is not None:
            builder.formations[formation] += 1
        builder.starter_ids.extend(_player_ids_from_lineup_items(item.get("startXI")))


def _record_substitution_updates(
    snapshot_dir: Path,
    fixture_id: int,
    builders: dict[str, _TeamTournamentUpdateBuilder],
) -> None:
    payload = _read_optional_json(snapshot_dir / f"fixture_{fixture_id}_events.json")
    for event in _response_items(payload):
        if (_first_text(event, ("type",)) or "").lower() != "subst":
            continue
        team = _mapping(event.get("team")) or {}
        team_name = _first_text(team, ("name",))
        if team_name is None:
            continue
        builder = builders.setdefault(team_name, _TeamTournamentUpdateBuilder())
        for key in ("player", "assist"):
            player = _mapping(event.get(key)) or {}
            player_id = _int_value(player.get("id"))
            if player_id is not None:
                builder.substitute_ids.append(player_id)


def _player_ids_from_lineup_items(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    player_ids: list[int] = []
    for lineup_item in value:
        item = _mapping(lineup_item)
        if not item:
            continue
        player = _mapping(item.get("player")) or item
        player_id = _int_value(player.get("id"))
        if player_id is not None:
            player_ids.append(player_id)
    return tuple(player_ids)


def _fixture_score(item: JsonObject) -> tuple[int | None, int | None]:
    goals = _mapping(item.get("goals")) or {}
    home_score = _int_value(goals.get("home"))
    away_score = _int_value(goals.get("away"))
    if home_score is not None and away_score is not None:
        return home_score, away_score

    score = _mapping(item.get("score")) or {}
    fulltime = _mapping(score.get("fulltime")) or {}
    return _int_value(fulltime.get("home")), _int_value(fulltime.get("away"))


def _fixture_id_from_match_id(match_id: str) -> int | None:
    prefix = "wc-2026-"
    if not match_id.startswith(prefix):
        return None
    return _int_value(match_id.removeprefix(prefix))


def _load_players(
    snapshot_dir: Path,
    team_ids: Mapping[str, int | None],
) -> tuple[dict[int, PlayerProfile], dict[str, tuple[int, ...]]]:
    players: dict[int, PlayerProfile] = {}
    rosters: dict[str, tuple[int, ...]] = {}
    for team_name, team_id in team_ids.items():
        if team_id is None:
            rosters[team_name] = ()
            continue
        squad_path = snapshot_dir / f"team_{team_id}_squad.json"
        roster_ids: list[int] = []
        for squad_player in _squad_players(_read_optional_json(squad_path)):
            player_id = _int_value(squad_player.get("id"))
            if player_id is None:
                continue
            roster_ids.append(player_id)
            players[player_id] = _player_profile_from_snapshots(
                snapshot_dir,
                player_id,
                team_id,
                team_name,
                squad_player,
            )
        rosters[team_name] = tuple(roster_ids)
    return players, rosters


def _player_profile_from_snapshots(
    snapshot_dir: Path,
    player_id: int,
    national_team_id: int,
    national_team: str,
    squad_player: JsonObject,
) -> PlayerProfile:
    profile_payload = _read_optional_json(snapshot_dir / f"player_{player_id}_profile.json")
    statistics_payload = _read_optional_json(snapshot_dir / f"player_{player_id}_statistics.json")
    profile_player = _first_player_mapping(profile_payload)
    statistics_player = _first_player_mapping(statistics_payload)
    statistic = _select_player_statistic(statistics_payload, national_team_id)
    goals = _nested_int(statistic, "goals", "total") or 0
    games = _mapping(statistic.get("games")) or {}
    team = _mapping(statistic.get("team")) or {}
    league = _mapping(statistic.get("league")) or {}
    return PlayerProfile(
        player_id=player_id,
        name=(
            _first_text(statistics_player, ("name", "firstname"))
            or _first_text(profile_player, ("name", "firstname"))
            or _first_text(squad_player, ("name",))
            or f"player-{player_id}"
        ),
        national_team=national_team,
        age=(
            _int_value(statistics_player.get("age"))
            or _int_value(profile_player.get("age"))
            or _int_value(squad_player.get("age"))
        ),
        club_id=_int_value(team.get("id")),
        club=_first_text(team, ("name",)),
        league_id=_int_value(league.get("id")),
        league=_first_text(league, ("name",)),
        position=_first_text(squad_player, ("position",)) or _first_text(games, ("position",)),
        goals=goals,
        average_rating=_float_value(games.get("rating")),
    )


def _load_coaches(
    snapshot_dir: Path,
    team_ids: Mapping[str, int | None],
    recent_forms: Mapping[str, tuple[int, int, int, int, int]],
) -> tuple[dict[int, CoachProfile], dict[str, tuple[int, ...]]]:
    coaches: dict[int, CoachProfile] = {}
    coaching_staff: dict[str, tuple[int, ...]] = {}
    for team_name, team_id in team_ids.items():
        if team_id is None:
            coaching_staff[team_name] = ()
            continue
        coaches_path = snapshot_dir / f"team_{team_id}_coaches.json"
        coach_ids: list[int] = []
        for coach_item in _response_items(_read_optional_json(coaches_path)):
            coach_id = _int_value(coach_item.get("id"))
            if coach_id is None:
                continue
            wins, draws, losses, _goals_for, _goals_against = recent_forms.get(
                team_name,
                (0, 0, 0, 0, 0),
            )
            record = _mapping(coach_item.get("record"))
            if record:
                wins = _int_value(record.get("wins")) or wins
                draws = _int_value(record.get("draws")) or draws
                losses = _int_value(record.get("losses")) or losses
            coach_ids.append(coach_id)
            coaches[coach_id] = CoachProfile(
                coach_id=coach_id,
                name=_first_text(coach_item, ("name", "firstname")) or f"coach-{coach_id}",
                national_team=team_name,
                age=_int_value(coach_item.get("age")),
                wins=wins,
                draws=draws,
                losses=losses,
                titles_won=_trophy_count(snapshot_dir / f"coach_{coach_id}_trophies.json"),
                career_teams=_coach_career_teams(coach_item),
            )
        coaching_staff[team_name] = tuple(coach_ids)
    return coaches, coaching_staff


def _load_clubs(
    snapshot_dir: Path,
    players: Mapping[int, PlayerProfile],
    external_factors: JsonObject,
    leagues: Mapping[int, LeagueProfile],
) -> dict[int, ClubProfile]:
    profiles: dict[int, ClubProfile] = {}
    for club_id in sorted({player.club_id for player in players.values() if player.club_id}):
        represented = [player for player in players.values() if player.club_id == club_id]
        primary_player = represented[0]
        league_id = primary_player.league_id
        if league_id is None:
            continue
        stats_path = snapshot_dir / f"club_{club_id}_statistics_league_{league_id}.json"
        stats = _first_response_mapping(_read_optional_json(stats_path))
        fixtures = _mapping(stats.get("fixtures")) or {}
        wins = _nested_int(fixtures, "wins", "total") or 0
        draws = _nested_int(fixtures, "draws", "total") or 0
        losses = _nested_int(fixtures, "loses", "total") or 0
        team = _mapping(stats.get("team")) or {}
        league = leagues.get(league_id)
        club_name = _first_text(team, ("name",)) or primary_player.club or f"club-{club_id}"
        profiles[club_id] = ClubProfile(
            club_id=club_id,
            name=club_name,
            country=league.country if league else "Unknown country",
            league_id=league_id,
            league=league.name if league else primary_player.league or f"league-{league_id}",
            wins=wins,
            draws=draws,
            losses=losses,
            major_titles_won=int(
                _factor_number(
                    external_factors,
                    "club_major_titles",
                    str(club_id),
                    club_name,
                    default=0.0,
                )
            ),
            fifa_player_count=len(represented),
        )
    return profiles


def _load_leagues(
    snapshot_dir: Path,
    players: Mapping[int, PlayerProfile],
    external_factors: JsonObject,
) -> dict[int, LeagueProfile]:
    profiles: dict[int, LeagueProfile] = {}
    league_ids = sorted({player.league_id for player in players.values() if player.league_id})
    for league_id in league_ids:
        league_payload = _read_optional_json(snapshot_dir / f"league_{league_id}.json")
        league_item = _first_response_mapping(league_payload)
        league = _mapping(league_item.get("league")) or league_item
        country = _mapping(league_item.get("country")) or {}
        league_name = _first_text(league, ("name",)) or _league_name_from_players(
            players,
            league_id,
        )
        country_name = _first_text(country, ("name",)) or "Unknown country"
        standings = _read_optional_json(snapshot_dir / f"league_{league_id}_standings.json")
        fixtures = _read_optional_json(snapshot_dir / f"league_{league_id}_fixtures.json")
        profiles[league_id] = LeagueProfile(
            league_id=league_id,
            name=league_name,
            country=country_name,
            total_teams=_standing_team_count(standings),
            matches_played=len(_response_items(fixtures)),
            average_attendance=int(
                _factor_number(
                    external_factors,
                    "league_average_attendance",
                    str(league_id),
                    league_name,
                    default=0.0,
                )
            ),
            fifa_player_count=sum(
                1 for player in players.values() if player.league_id == league_id
            ),
        )
    return profiles


def _load_national_teams(
    team_ids: Mapping[str, int | None],
    group_by_team: Mapping[str, str | None],
    rosters: Mapping[str, tuple[int, ...]],
    coaching_staff: Mapping[str, tuple[int, ...]],
    recent_forms: Mapping[str, tuple[int, int, int, int, int]],
    external_factors: JsonObject,
) -> dict[str, NationalTeamProfile]:
    profiles: dict[str, NationalTeamProfile] = {}
    for team_name, team_id in sorted(team_ids.items()):
        wins, draws, losses, goals_for, goals_against = recent_forms.get(
            team_name,
            (0, 0, 0, 0, 0),
        )
        profiles[team_name] = NationalTeamProfile(
            team_id=team_id,
            name=team_name,
            group=group_by_team.get(team_name),
            roster=rosters.get(team_name, ()),
            coaching_staff=coaching_staff.get(team_name, ()),
            recent_wins=wins,
            recent_draws=draws,
            recent_losses=losses,
            goals_for=goals_for,
            goals_against=goals_against,
            history_score=_factor_number(
                external_factors,
                "country_history",
                team_name,
                _normalized_country_key(team_name),
                default=_default_country_history(team_name),
            ),
        )
    return profiles


def _rank_leagues(dataset: WorldCupDataSet) -> dict[int, float]:
    fifa_player_counts = _scale_values(
        {
            league_id: float(league.fifa_player_count)
            for league_id, league in dataset.leagues.items()
        }
    )
    attendances = _scale_values(
        {
            league_id: float(league.average_attendance)
            for league_id, league in dataset.leagues.items()
        }
    )
    sizes = _scale_values(
        {
            league_id: float(league.total_teams + league.matches_played / 10)
            for league_id, league in dataset.leagues.items()
        }
    )
    scores: dict[int, float] = {}
    for league_id, league in dataset.leagues.items():
        country_strength = _factor_number(
            dataset.external_factors,
            "country_strength",
            league.country,
            _normalized_country_key(league.country),
            default=_default_country_history(league.country),
        )
        scores[league_id] = round(
            _clamp_score(
                fifa_player_counts.get(league_id, 0.0) * 0.35
                + attendances.get(league_id, 0.0) * 0.25
                + country_strength * 0.25
                + sizes.get(league_id, 0.0) * 0.15
            ),
            3,
        )
    return scores


def _rank_clubs(
    dataset: WorldCupDataSet,
    league_scores: Mapping[int, float],
) -> dict[int, float]:
    player_counts = _scale_values(
        {club_id: float(club.fifa_player_count) for club_id, club in dataset.clubs.items()}
    )
    title_counts = _scale_values(
        {club_id: float(club.major_titles_won) for club_id, club in dataset.clubs.items()}
    )
    scores: dict[int, float] = {}
    for club_id, club in dataset.clubs.items():
        scores[club_id] = round(
            _clamp_score(
                league_scores.get(club.league_id, 45.0) * 0.40
                + player_counts.get(club_id, 0.0) * 0.25
                + _record_score(club.wins, club.draws, club.losses) * 0.25
                + title_counts.get(club_id, 0.0) * 0.10
            ),
            3,
        )
    return scores


def _rank_players(
    dataset: WorldCupDataSet,
    league_scores: Mapping[int, float],
    club_scores: Mapping[int, float],
) -> dict[int, float]:
    goal_scores = _scale_values(
        {player_id: float(player.goals) for player_id, player in dataset.players.items()}
    )
    rating_scores = _scale_values(
        {
            player_id: player.average_rating if player.average_rating is not None else 6.5
            for player_id, player in dataset.players.items()
        }
    )
    scores: dict[int, float] = {}
    for player_id, player in dataset.players.items():
        club_score = club_scores.get(player.club_id or -1, 45.0)
        league_score = league_scores.get(player.league_id or -1, 45.0)
        scores[player_id] = round(
            _clamp_score(
                club_score * 0.25
                + league_score * 0.20
                + goal_scores.get(player_id, 0.0) * 0.30
                + rating_scores.get(player_id, 50.0) * 0.25
            ),
            3,
        )
    return scores


def _rank_coaches(dataset: WorldCupDataSet) -> dict[int, float]:
    title_scores = _scale_values(
        {coach_id: float(coach.titles_won) for coach_id, coach in dataset.coaches.items()}
    )
    scores: dict[int, float] = {}
    for coach_id, coach in dataset.coaches.items():
        team = dataset.national_teams.get(coach.national_team)
        team_strength = 50.0
        if team is not None:
            team_strength = team.history_score * 0.55 + _national_form_score(team) * 0.45
        scores[coach_id] = round(
            _clamp_score(
                _record_score(coach.wins, coach.draws, coach.losses) * 0.45
                + title_scores.get(coach_id, 0.0) * 0.30
                + team_strength * 0.25
            ),
            3,
        )
    return scores


def _rank_national_teams(
    dataset: WorldCupDataSet,
    league_scores: Mapping[int, float],
    club_scores: Mapping[int, float],
    player_scores: Mapping[int, float],
    coach_scores: Mapping[int, float],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for team_name, team in dataset.national_teams.items():
        player_quality = _average(
            player_scores[player_id] for player_id in team.roster if player_id in player_scores
        )
        coach_quality = _average(
            coach_scores[coach_id] for coach_id in team.coaching_staff if coach_id in coach_scores
        )
        domestic_leagues = [
            score
            for league_id, score in league_scores.items()
            if _same_country(dataset.leagues[league_id].country, team_name)
        ]
        domestic_clubs = [
            score
            for club_id, score in club_scores.items()
            if _same_country(dataset.clubs[club_id].country, team_name)
        ]
        domestic_strength = _average([*domestic_leagues, *domestic_clubs])
        scores[team_name] = round(
            _clamp_score(
                team.history_score * 0.20
                + (domestic_strength if domestic_strength > 0 else 50.0) * 0.20
                + (player_quality if player_quality > 0 else 50.0) * 0.35
                + (coach_quality if coach_quality > 0 else 50.0) * 0.10
                + _national_form_score(team) * 0.15
            ),
            3,
        )
    return scores


def _expected_goals(
    dataset: WorldCupDataSet,
    match: GroupStageMatch,
    home_rating: float,
    away_rating: float,
) -> tuple[float, float]:
    rating_edge = home_rating - away_rating
    home_attack = _team_goal_bonus(dataset, match.home_team)
    away_attack = _team_goal_bonus(dataset, match.away_team)
    home_xg = _clamp(1.25 + rating_edge * 0.025 + home_attack, 0.2, 4.2)
    away_xg = _clamp(1.15 - rating_edge * 0.023 + away_attack, 0.2, 4.2)
    return home_xg, away_xg


def _team_goal_bonus(dataset: WorldCupDataSet, team_name: str) -> float:
    team = dataset.national_teams.get(team_name)
    if team is None:
        return 0.0
    goals = sorted(
        (
            dataset.players[player_id].goals
            for player_id in team.roster
            if player_id in dataset.players
        ),
        reverse=True,
    )
    if not goals:
        return 0.0
    return min(0.45, sum(goals[:5]) / max(len(goals[:5]), 1) / 40)


def _location_adjustment(team_name: str, match: GroupStageMatch, *, is_home: bool) -> float:
    adjustment = 0.0
    if _same_country(team_name, match.venue_country):
        adjustment += 5.0 if is_home else 4.0
    elif match.venue_country in HOST_COUNTRIES:
        adjustment += 1.5 if _normalized_country_key(team_name) in AMERICAS_TEAMS else -0.8

    if match.venue_city in HOT_WEATHER_CITIES:
        adjustment += 1.0 if _normalized_country_key(team_name) in WARM_WEATHER_TEAMS else -0.5
    return adjustment


def _tournament_adjustment(
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    team_name: str,
) -> float:
    update = dataset.team_updates.get(team_name)
    if update is None or update.matches == 0:
        return 0.0

    result_adjustment = (
        (update.points_per_match - 1.0) * 2.4
        + update.goal_difference_per_match * 1.1
        + ((update.goals_for / update.matches) - 1.25) * 0.5
        - ((update.goals_against / update.matches) - 1.25) * 0.35
    )
    starter_adjustment, substitute_adjustment = _lineup_quality_adjustments(
        dataset,
        rankings,
        team_name,
    )
    formation_adjustment = _formation_adjustment(update)
    return round(
        _clamp(
            result_adjustment + starter_adjustment + substitute_adjustment + formation_adjustment,
            -10.0,
            10.0,
        ),
        3,
    )


def _lineup_quality_adjustments(
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    team_name: str,
) -> tuple[float, float]:
    update = dataset.team_updates.get(team_name)
    team = dataset.national_teams.get(team_name)
    if update is None or team is None:
        return 0.0, 0.0

    roster_average = _player_score_average(rankings, team.roster)
    if roster_average <= 0:
        roster_average = 50.0

    starter_average = _player_score_average(rankings, update.starter_ids)
    substitute_average = _player_score_average(rankings, update.substitute_ids)
    starter_adjustment = 0.0
    substitute_adjustment = 0.0
    if starter_average > 0:
        starter_adjustment = _clamp((starter_average - roster_average) * 0.06, -3.0, 3.0)
    if substitute_average > 0:
        substitute_adjustment = _clamp((substitute_average - roster_average) * 0.03, -1.5, 1.5)
    return starter_adjustment, substitute_adjustment


def _formation_adjustment(update: TeamTournamentUpdate) -> float:
    if not update.formations or update.matches == 0:
        return 0.0
    most_common_count = Counter(update.formations).most_common(1)[0][1]
    return min(0.6, (most_common_count / update.matches) * 0.6)


def _player_score_average(
    rankings: WorldCupRankings,
    player_ids: Iterable[int],
) -> float:
    return _average(
        rankings.players[player_id] for player_id in player_ids if player_id in rankings.players
    )


def _team_update_summary(
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    team_name: str,
) -> str:
    update = dataset.team_updates.get(team_name)
    if update is None or update.matches == 0:
        return f"{team_name} has no completed tournament update."

    starter_adjustment, substitute_adjustment = _lineup_quality_adjustments(
        dataset,
        rankings,
        team_name,
    )
    formation = update.formations[0] if update.formations else "unknown shape"
    return (
        f"{team_name} update: {update.matches} match, {formation}, "
        f"{update.points_per_match:.2f} ppm, {update.goal_difference_per_match:+.1f} GD/m, "
        f"XI {starter_adjustment:+.1f}, subs {substitute_adjustment:+.1f}."
    )


def _recent_team_form(snapshot_dir: Path, team_id: int) -> tuple[int, int, int, int, int]:
    payload = _read_optional_json(snapshot_dir / f"team_{team_id}_recent_fixtures.json")
    wins = draws = losses = goals_for = goals_against = 0
    for item in _response_items(payload):
        teams = _mapping(item.get("teams")) or {}
        goals = _mapping(item.get("goals")) or {}
        home = _mapping(teams.get("home")) or {}
        away = _mapping(teams.get("away")) or {}
        home_id = _int_value(home.get("id"))
        away_id = _int_value(away.get("id"))
        if team_id not in (home_id, away_id):
            continue
        home_goals = _int_value(goals.get("home"))
        away_goals = _int_value(goals.get("away"))
        if home_goals is None or away_goals is None:
            continue
        if home_id == team_id:
            team_goals, opponent_goals = home_goals, away_goals
        else:
            team_goals, opponent_goals = away_goals, home_goals
        goals_for += team_goals
        goals_against += opponent_goals
        if team_goals > opponent_goals:
            wins += 1
        elif team_goals < opponent_goals:
            losses += 1
        else:
            draws += 1
    return wins, draws, losses, goals_for, goals_against


def _squad_players(payload: JsonObject) -> tuple[JsonObject, ...]:
    players: list[JsonObject] = []
    for item in _response_items(payload):
        squad = item.get("players")
        if isinstance(squad, list):
            players.extend(player for player in (_mapping(value) for value in squad) if player)
    return tuple(players)


def _select_player_statistic(payload: JsonObject, national_team_id: int) -> JsonObject:
    statistics: list[JsonObject] = []
    for item in _response_items(payload):
        raw_statistics = item.get("statistics")
        if isinstance(raw_statistics, list):
            statistics.extend(
                statistic
                for statistic in (_mapping(value) for value in raw_statistics)
                if statistic
            )
    if not statistics:
        return {}
    for statistic in statistics:
        team = _mapping(statistic.get("team")) or {}
        league = _mapping(statistic.get("league")) or {}
        if _int_value(team.get("id")) != national_team_id and _int_value(league.get("id")):
            return statistic
    return statistics[0]


def _first_player_mapping(payload: JsonObject) -> JsonObject:
    for item in _response_items(payload):
        player = _mapping(item.get("player"))
        if player:
            return player
    return {}


def _first_response_mapping(payload: JsonObject) -> JsonObject:
    response = _response_items(payload)
    if response:
        return response[0]
    return {}


def _coach_career_teams(coach_item: JsonObject) -> tuple[str, ...]:
    career = coach_item.get("career")
    if not isinstance(career, list):
        return ()
    names: list[str] = []
    for career_item in career:
        item = _mapping(career_item)
        if not item:
            continue
        team = _mapping(item.get("team")) or item
        name = _first_text(team, ("name",))
        if name is not None:
            names.append(name)
    return tuple(names)


def _trophy_count(path: Path) -> int:
    count = 0
    for item in _response_items(_read_optional_json(path)):
        place = _first_text(item, ("place",))
        if place is None or place.lower() in {"winner", "champion", "1st"}:
            count += 1
    return count


def _standing_team_count(payload: JsonObject) -> int:
    count = 0
    for item in _response_items(payload):
        league = _mapping(item.get("league")) or item
        standings = league.get("standings")
        if not isinstance(standings, list):
            continue
        for group in standings:
            if isinstance(group, list):
                count = max(count, len(group))
    return count


def _league_name_from_players(players: Mapping[int, PlayerProfile], league_id: int) -> str:
    for player in players.values():
        if player.league_id == league_id and player.league:
            return player.league
    return f"league-{league_id}"


def _fixture_team(item: JsonObject, side: str) -> tuple[str | None, int | None]:
    teams = _mapping(item.get("teams")) or {}
    team = _mapping(teams.get(side)) or _mapping(item.get(f"{side}_team"))
    if team is None:
        value = item.get(side)
        if isinstance(value, str):
            return value, None
        return None, None
    return _first_text(team, ("name", "country")), _int_value(team.get("id"))


def _is_group_stage_fixture(item: JsonObject, group: str | None) -> bool:
    if group is not None:
        return True
    league = _mapping(item.get("league")) or {}
    round_name = _first_text(league, ("round",)) or _first_text(item, ("round", "stage"))
    return bool(round_name and "group" in round_name.lower())


def _group_from_fixture(item: JsonObject) -> str | None:
    league = _mapping(item.get("league")) or {}
    for value in (
        _first_text(league, ("round", "group")),
        _first_text(item, ("round", "group", "stage")),
    ):
        if value is None:
            continue
        match = re.search(r"\bgroup\s+([A-L])\b", value, flags=re.IGNORECASE)
        if match:
            return f"Group {match.group(1).upper()}"
    return None


def _round_number_from_fixture(item: JsonObject) -> int | None:
    league = _mapping(item.get("league")) or {}
    for value in (
        _first_text(league, ("round",)),
        _first_text(item, ("round", "stage")),
    ):
        if value is None:
            continue
        match = re.search(r"\b(?:stage|round)\s*-\s*(\d+)\b", value, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"\bmatchday\s+(\d+)\b", value, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _specific_group_label(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.search(r"\bgroup\s+([A-L])\b", value, flags=re.IGNORECASE)
    if match is None:
        return None
    return f"Group {match.group(1).upper()}"


def _record_score(wins: int, draws: int, losses: int) -> float:
    total = wins + draws + losses
    if total == 0:
        return 50.0
    return ((wins * 3 + draws) / (total * 3)) * 100


def _national_form_score(team: NationalTeamProfile) -> float:
    record = _record_score(team.recent_wins, team.recent_draws, team.recent_losses)
    matches = team.recent_wins + team.recent_draws + team.recent_losses
    goal_edge = 0.0 if matches == 0 else (team.goals_for - team.goals_against) / matches
    return _clamp_score(record + goal_edge * 5)


def _score_from_expected_goals(expected_goals: float) -> int:
    return max(0, int(expected_goals + 0.35))


def _outcome_for_score(home_score: int, away_score: int) -> Outcome:
    if home_score > away_score:
        return Outcome.HOME_WIN
    if home_score < away_score:
        return Outcome.AWAY_WIN
    return Outcome.DRAW


def _prediction_pick_label(prediction: WorldCupScorePrediction) -> str:
    if prediction.outcome == Outcome.DRAW:
        return "Draw"
    if prediction.outcome == Outcome.HOME_WIN:
        return f"{prediction.home_team} win"
    return f"{prediction.away_team} win"


def _markdown_text(value: str) -> str:
    return value.replace("|", r"\|")


def _group_sort_key(group: str) -> tuple[int, str]:
    match = re.search(r"\bGroup\s+([A-L])\b", group, flags=re.IGNORECASE)
    if match:
        return (0, match.group(1).upper())
    return (1, group)


K = TypeVar("K")


def _scale_values(values: Mapping[K, float]) -> dict[K, float]:
    if not values:
        return {}
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum == minimum:
        midpoint = 50.0 if maximum > 0 else 0.0
        return dict.fromkeys(values, midpoint)
    return {
        key: _clamp_score(((value - minimum) / (maximum - minimum)) * 100)
        for key, value in values.items()
    }


def _average(values: Iterable[float]) -> float:
    items = tuple(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _factor_number(
    factors: JsonObject,
    section: str,
    *keys: str,
    default: float,
) -> float:
    raw_section = _mapping(factors.get(section))
    if raw_section is None:
        return default
    for key in keys:
        value = raw_section.get(key)
        number = _float_value(value)
        if number is not None:
            return number
    return default


def _default_country_history(country: str) -> float:
    return COUNTRY_HISTORY_SCORES.get(_normalized_country_key(country), 50.0)


def _same_country(left: str, right: str) -> bool:
    return _normalized_country_key(left) == _normalized_country_key(right)


def _normalized_country_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    aliases = {
        "Cote dIvoire": "Ivory Coast",
        "IR Iran": "Iran",
        "Korea Republic": "Korea Republic",
        "South Korea": "Korea Republic",
        "Türkiye": "Turkiye",
        "Turkey": "Turkiye",
        "USA": "United States",
        "United States of America": "United States",
    }
    compact = re.sub(r"\s+", " ", ascii_value.replace("'", "")).strip()
    return aliases.get(compact, compact)


def _clamp_score(value: float) -> float:
    return _clamp(value, 0.0, 100.0)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _response_items(payload: JsonObject) -> tuple[JsonObject, ...]:
    response = payload.get("response", payload)
    if isinstance(response, list):
        return tuple(item for item in (_mapping(value) for value in response) if item)
    if isinstance(response, dict):
        return (cast(JsonObject, response),)
    return ()


def _read_json(path: Path) -> JsonObject:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(JsonObject, decoded)


def _read_optional_json(path: Path) -> JsonObject:
    if not path.exists():
        return {}
    return _read_json(path)


def _mapping(value: object) -> JsonObject | None:
    if isinstance(value, dict):
        return cast(JsonObject, value)
    return None


def _first_text(item: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return None


def _string_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int):
        return str(value)
    return None


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _float_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _nested_int(item: Mapping[str, object], *keys: str) -> int | None:
    value: object = item
    for key in keys:
        mapping = _mapping(value)
        if mapping is None:
            return None
        value = mapping.get(key)
    return _int_value(value)


def _datetime_value(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"
