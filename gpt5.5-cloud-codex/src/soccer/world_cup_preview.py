"""Single-match World Cup preview reports."""

from __future__ import annotations

import textwrap
import unicodedata
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

from soccer.world_cup_2026 import (
    DEFAULT_WORLD_CUP_DATA_DIR,
    CoachProfile,
    GroupStageMatch,
    JsonObject,
    PlayerProfile,
    WorldCupDataSet,
    WorldCupRankings,
    WorldCupScorePrediction,
    predict_group_stage_scores,
    prediction_to_json,
)


@dataclass(frozen=True)
class PlayerSelection:
    """A player listed in a preview lineup or bench."""

    player_id: int | None
    name: str
    position: str | None
    number: int | None
    score: float | None


@dataclass(frozen=True)
class TeamMatchPreview:
    """Preview details for one team in a single fixture."""

    team: str
    coach: str
    formation: str
    lineup_source: str
    starters: tuple[PlayerSelection, ...]
    substitutes: tuple[PlayerSelection, ...]


@dataclass(frozen=True)
class WorldCupMatchPreview:
    """A complete single-match preview ready for PDF rendering."""

    match: GroupStageMatch
    prediction: WorldCupScorePrediction
    home: TeamMatchPreview
    away: TeamMatchPreview


@dataclass(frozen=True)
class _LineupSnapshot:
    team: str
    coach: str | None
    formation: str | None
    starters: tuple[PlayerSelection, ...]
    substitutes: tuple[PlayerSelection, ...]
    source: str


@dataclass(frozen=True)
class _PdfLine:
    text: str
    size: int = 10
    gap_after: int = 0


def build_world_cup_match_preview(
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    match_id: str,
    *,
    snapshot_dir: Path = DEFAULT_WORLD_CUP_DATA_DIR,
) -> WorldCupMatchPreview:
    """Build a single-match preview from local snapshots and model rankings."""

    normalized_match_id = _normalize_match_id(match_id)
    if normalized_match_id in dataset.completed_matches:
        raise ValueError(f"{normalized_match_id} is already completed in the loaded dataset")

    match = _match_by_id(dataset, normalized_match_id)
    prediction = _prediction_by_match_id(dataset, rankings, normalized_match_id)
    return WorldCupMatchPreview(
        match=match,
        prediction=prediction,
        home=_team_preview(snapshot_dir, dataset, rankings, match, match.home_team),
        away=_team_preview(snapshot_dir, dataset, rankings, match, match.away_team),
    )


def render_world_cup_match_preview_pdf(
    preview: WorldCupMatchPreview,
    output_path: Path,
) -> Path:
    """Render a single-match preview as a dependency-free PDF."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[_PdfLine] = []
    match = preview.match
    prediction = preview.prediction

    _append(lines, "FIFA 2026 World Cup Match Preview", size=16, gap_after=8)
    _append(lines, f"{match.home_team} vs {match.away_team}", size=14, gap_after=6)
    _append(lines, f"Match: {match.match_id} | {match.group or 'Group unknown'}")
    _append(lines, f"Kickoff: {match.kickoff.astimezone(UTC).isoformat()}")
    _append(lines, f"Venue: {match.venue_name}, {match.venue_city}, {match.venue_country}")
    _append(lines, "", gap_after=4)

    _append(lines, "Prediction", size=13, gap_after=4)
    _append(
        lines,
        (
            f"{prediction.home_team} {prediction.home_score}-"
            f"{prediction.away_score} {prediction.away_team}"
        ),
        size=12,
    )
    _append(
        lines,
        (
            f"Pick: {_pick_label(prediction)} | Confidence: {prediction.confidence:.0%} | "
            f"Expected goals: {prediction.home_expected_goals:.2f}-"
            f"{prediction.away_expected_goals:.2f}"
        ),
    )
    _append(
        lines,
        (
            f"Adjusted ratings: {prediction.home_adjusted_rating:.1f}-"
            f"{prediction.away_adjusted_rating:.1f}; tournament adjustment: "
            f"{prediction.home_tournament_adjustment:+.1f}/"
            f"{prediction.away_tournament_adjustment:+.1f}"
        ),
    )
    _append_wrapped(lines, f"Rationale: {prediction.rationale}", width=92, gap_after=8)

    _append_team(lines, preview.home)
    _append(lines, "", gap_after=4)
    _append_team(lines, preview.away)

    _write_pdf(output_path, lines)
    return output_path


def match_preview_to_json(preview: WorldCupMatchPreview) -> JsonObject:
    """Return a JSON-ready representation of a single-match preview."""

    return {
        "match": {
            "match_id": preview.match.match_id,
            "group": preview.match.group,
            "round_number": preview.match.round_number,
            "home_team": preview.match.home_team,
            "away_team": preview.match.away_team,
            "kickoff": preview.match.kickoff.isoformat(),
            "venue_name": preview.match.venue_name,
            "venue_city": preview.match.venue_city,
            "venue_country": preview.match.venue_country,
        },
        "prediction": prediction_to_json(preview.prediction),
        "home": _team_preview_to_json(preview.home),
        "away": _team_preview_to_json(preview.away),
    }


def _team_preview(
    snapshot_dir: Path,
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    match: GroupStageMatch,
    team_name: str,
) -> TeamMatchPreview:
    fixture_id = _fixture_id_from_match_id(match.match_id)
    announced = _lineup_for_fixture(
        snapshot_dir,
        fixture_id,
        team_name,
        dataset,
        rankings,
        source="announced provider lineup",
    )
    if announced is not None and announced.starters:
        return _team_preview_from_lineup(dataset, rankings, team_name, announced)

    previous = _latest_previous_lineup(
        snapshot_dir,
        dataset,
        rankings,
        match,
        team_name,
    )
    if previous is not None and previous.starters:
        return _team_preview_from_lineup(dataset, rankings, team_name, previous)

    return _projected_team_preview(dataset, rankings, team_name)


def _team_preview_from_lineup(
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    team_name: str,
    lineup: _LineupSnapshot,
) -> TeamMatchPreview:
    coach = lineup.coach or _coach_name_for_team(dataset, rankings, team_name)
    formation = lineup.formation or _latest_formation(dataset, team_name)
    substitutes = lineup.substitutes or _projected_substitutes(
        dataset,
        rankings,
        team_name,
        lineup.starters,
    )
    return TeamMatchPreview(
        team=team_name,
        coach=coach,
        formation=formation,
        lineup_source=lineup.source,
        starters=lineup.starters,
        substitutes=substitutes,
    )


def _projected_team_preview(
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    team_name: str,
) -> TeamMatchPreview:
    players = _ranked_roster_players(dataset, rankings, team_name)
    starters = tuple(players[:11])
    return TeamMatchPreview(
        team=team_name,
        coach=_coach_name_for_team(dataset, rankings, team_name),
        formation=_latest_formation(dataset, team_name),
        lineup_source="projected from squad ranking",
        starters=starters,
        substitutes=tuple(players[11:23]),
    )


def _lineup_for_fixture(
    snapshot_dir: Path,
    fixture_id: int | None,
    team_name: str,
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    *,
    source: str,
) -> _LineupSnapshot | None:
    if fixture_id is None:
        return None
    payload = _read_optional_json(snapshot_dir / f"fixture_{fixture_id}_lineups.json")
    for item in _response_items(payload):
        team = _mapping(item.get("team")) or {}
        provider_team = _first_text(team, ("name",))
        if provider_team is None or _team_key(provider_team) != _team_key(team_name):
            continue
        coach = _mapping(item.get("coach")) or {}
        return _LineupSnapshot(
            team=team_name,
            coach=_first_text(coach, ("name",)),
            formation=_first_text(item, ("formation",)),
            starters=_players_from_lineup_items(
                item.get("startXI"),
                dataset,
                rankings,
            ),
            substitutes=_players_from_lineup_items(
                item.get("substitutes"),
                dataset,
                rankings,
            ),
            source=source,
        )
    return None


def _latest_previous_lineup(
    snapshot_dir: Path,
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    match: GroupStageMatch,
    team_name: str,
) -> _LineupSnapshot | None:
    prior_matches = sorted(
        (
            candidate
            for candidate in dataset.matches
            if candidate.kickoff < match.kickoff
            and candidate.match_id in dataset.completed_matches
            and team_name in {candidate.home_team, candidate.away_team}
        ),
        key=lambda candidate: candidate.kickoff,
        reverse=True,
    )
    for candidate in prior_matches:
        lineup = _lineup_for_fixture(
            snapshot_dir,
            _fixture_id_from_match_id(candidate.match_id),
            team_name,
            dataset,
            rankings,
            source=f"projected from latest lineup in {candidate.match_id}",
        )
        if lineup is not None and lineup.starters:
            return lineup
    return None


def _players_from_lineup_items(
    value: object,
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
) -> tuple[PlayerSelection, ...]:
    if not isinstance(value, list):
        return ()
    selections: list[PlayerSelection] = []
    for raw_item in value:
        item = _mapping(raw_item)
        if item is None:
            continue
        player = _mapping(item.get("player")) or item
        player_id = _int_value(player.get("id"))
        profile = dataset.players.get(player_id or -1)
        selections.append(
            PlayerSelection(
                player_id=player_id,
                name=(
                    _first_text(player, ("name", "firstname"))
                    or (profile.name if profile is not None else None)
                    or "Unknown player"
                ),
                position=_first_text(player, ("pos", "position"))
                or (profile.position if profile is not None else None),
                number=_int_value(player.get("number")),
                score=rankings.players.get(player_id or -1),
            )
        )
    return tuple(selections)


def _ranked_roster_players(
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    team_name: str,
) -> list[PlayerSelection]:
    team = dataset.national_teams.get(team_name)
    if team is None:
        return []

    def sort_key(player_id: int) -> tuple[float, str]:
        player = dataset.players.get(player_id)
        return (rankings.players.get(player_id, 0.0), player.name if player else "")

    players: list[PlayerSelection] = []
    for player_id in sorted(team.roster, key=sort_key, reverse=True):
        player = dataset.players.get(player_id)
        if player is None:
            continue
        players.append(_selection_from_profile(player, rankings.players.get(player_id)))
    return players


def _projected_substitutes(
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    team_name: str,
    starters: tuple[PlayerSelection, ...],
) -> tuple[PlayerSelection, ...]:
    starter_ids = {player.player_id for player in starters if player.player_id is not None}
    return tuple(
        player
        for player in _ranked_roster_players(dataset, rankings, team_name)
        if player.player_id not in starter_ids
    )[:12]


def _selection_from_profile(player: PlayerProfile, score: float | None) -> PlayerSelection:
    return PlayerSelection(
        player_id=player.player_id,
        name=player.name,
        position=player.position,
        number=None,
        score=score,
    )


def _coach_name_for_team(
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    team_name: str,
) -> str:
    team = dataset.national_teams.get(team_name)
    if team is None or not team.coaching_staff:
        return "Unavailable"
    available_coaches = (
        dataset.coaches[coach_id] for coach_id in team.coaching_staff if coach_id in dataset.coaches
    )
    coach = max(
        available_coaches,
        key=lambda item: rankings.coaches.get(item.coach_id, 0.0),
        default=None,
    )
    if isinstance(coach, CoachProfile):
        return coach.name
    return "Unavailable"


def _latest_formation(dataset: WorldCupDataSet, team_name: str) -> str:
    update = dataset.team_updates.get(team_name)
    if update is not None and update.formations:
        return update.formations[0]
    return "Unavailable"


def _match_by_id(dataset: WorldCupDataSet, match_id: str) -> GroupStageMatch:
    for match in dataset.matches:
        if match.match_id == match_id:
            return match
    raise ValueError(f"World Cup match {match_id!r} was not found in the loaded dataset")


def _prediction_by_match_id(
    dataset: WorldCupDataSet,
    rankings: WorldCupRankings,
    match_id: str,
) -> WorldCupScorePrediction:
    for prediction in predict_group_stage_scores(dataset, rankings):
        if prediction.match_id == match_id:
            return prediction
    raise ValueError(f"World Cup match {match_id!r} could not be predicted")


def _append_team(lines: list[_PdfLine], team: TeamMatchPreview) -> None:
    _append(lines, team.team, size=13, gap_after=4)
    _append(lines, f"Coach: {team.coach}")
    _append(lines, f"Formation: {team.formation}")
    _append(lines, f"Lineup source: {team.lineup_source}", gap_after=4)

    _append(lines, "Starting players", size=12)
    if team.starters:
        for index, player in enumerate(team.starters, start=1):
            _append(lines, f"{index}. {_player_label(player)}")
    else:
        _append(lines, "No starting-player information available")

    _append(lines, "Possible substitutes", size=12, gap_after=2)
    if team.substitutes:
        for index, player in enumerate(team.substitutes[:12], start=1):
            _append(lines, f"{index}. {_player_label(player)}")
    else:
        _append(lines, "No substitute information available")


def _append(lines: list[_PdfLine], text: str, *, size: int = 10, gap_after: int = 0) -> None:
    lines.append(_PdfLine(text=text, size=size, gap_after=gap_after))


def _append_wrapped(
    lines: list[_PdfLine],
    text: str,
    *,
    width: int,
    size: int = 10,
    gap_after: int = 0,
) -> None:
    wrapped = textwrap.wrap(text, width=width) or [""]
    for index, chunk in enumerate(wrapped):
        _append(lines, chunk, size=size, gap_after=gap_after if index == len(wrapped) - 1 else 0)


def _player_label(player: PlayerSelection) -> str:
    number = f"#{player.number} " if player.number is not None else ""
    position = f" ({player.position})" if player.position else ""
    score = f" score {player.score:.1f}" if player.score is not None else ""
    return f"{number}{player.name}{position}{score}"


def _pick_label(prediction: WorldCupScorePrediction) -> str:
    if prediction.home_score > prediction.away_score:
        return f"{prediction.home_team} win"
    if prediction.home_score < prediction.away_score:
        return f"{prediction.away_team} win"
    return "Draw"


def _team_preview_to_json(team: TeamMatchPreview) -> JsonObject:
    return {
        "team": team.team,
        "coach": team.coach,
        "formation": team.formation,
        "lineup_source": team.lineup_source,
        "starters": [_player_selection_to_json(player) for player in team.starters],
        "substitutes": [_player_selection_to_json(player) for player in team.substitutes],
    }


def _player_selection_to_json(player: PlayerSelection) -> JsonObject:
    return {
        "player_id": player.player_id,
        "name": player.name,
        "position": player.position,
        "number": player.number,
        "score": player.score,
    }


def _write_pdf(path: Path, lines: list[_PdfLine]) -> None:
    pages: list[list[tuple[_PdfLine, int]]] = []
    current_page: list[tuple[_PdfLine, int]] = []
    y_position = 742
    for line in lines:
        line_height = max(line.size + 4, 13) + line.gap_after
        if y_position - line_height < 50 and current_page:
            pages.append(current_page)
            current_page = []
            y_position = 742
        current_page.append((line, y_position))
        y_position -= line_height
    if current_page:
        pages.append(current_page)

    objects: list[bytes] = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    page_ids: list[int] = []
    for page in pages:
        stream = _page_stream(page)
        content_id = len(objects) + 1
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        page_id = len(objects) + 1
        page_ids.append(page_id)
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for object_id, payload in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(payload)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(pdf))


def _page_stream(page: list[tuple[_PdfLine, int]]) -> bytes:
    commands: list[str] = []
    for line, y_position in page:
        commands.append(f"BT /F1 {line.size} Tf 50 {y_position} Td ({_pdf_text(line.text)}) Tj ET")
    return "\n".join(commands).encode("ascii")


def _pdf_text(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    compact = " ".join(ascii_value.split())
    return compact.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _normalize_match_id(match_id: str) -> str:
    value = match_id.strip()
    if value.startswith("wc-2026-"):
        return value
    return f"wc-2026-{value}"


def _fixture_id_from_match_id(match_id: str) -> int | None:
    raw_value = _normalize_match_id(match_id).removeprefix("wc-2026-")
    return _int_value(raw_value)


def _team_key(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_value.lower().split())


def _read_optional_json(path: Path) -> JsonObject:
    if not path.exists():
        return {}
    import json

    decoded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(decoded, dict):
        return decoded
    raise TypeError(f"{path} must contain a JSON object")


def _response_items(payload: JsonObject) -> tuple[JsonObject, ...]:
    response = payload.get("response", payload)
    if isinstance(response, list):
        return tuple(item for item in (_mapping(value) for value in response) if item)
    if isinstance(response, dict):
        return (response,)
    return ()


def _mapping(value: object) -> JsonObject | None:
    if isinstance(value, dict):
        return value
    return None


def _first_text(item: JsonObject, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
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
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None
