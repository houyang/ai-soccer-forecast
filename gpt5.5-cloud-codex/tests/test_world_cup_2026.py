from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from soccer.models import Outcome
from soccer.world_cup_2026 import (
    CompletedGroupMatch,
    GroupStageMatch,
    NationalTeamProfile,
    WorldCupDataSet,
    WorldCupRankings,
    load_world_cup_dataset,
    predict_elimination_stage_scores,
    predict_full_elimination_bracket,
    predict_group_stage_scores,
    prediction_to_json,
    rank_world_cup_entities,
    render_group_stage_markdown,
)
from soccer.world_cup_preview import (
    build_world_cup_match_preview,
    match_preview_to_json,
    render_world_cup_match_preview_pdf,
)


def test_world_cup_dataset_ranks_entities_and_predicts_scores(tmp_path: Path) -> None:
    _write_world_cup_snapshot(tmp_path)

    dataset = load_world_cup_dataset(tmp_path, expected_team_count=2)
    rankings = rank_world_cup_entities(dataset)
    predictions = predict_group_stage_scores(dataset, rankings)

    assert len(dataset.national_teams) == 2
    assert dataset.national_teams["Alpha"].group == "Group A"
    assert dataset.matches[0].group == "Group A"
    assert dataset.national_teams["Alpha"].roster == (11, 12)
    assert dataset.national_teams["Alpha"].coaching_staff == (901,)
    assert dataset.players[11].age == 25
    assert dataset.players[11].club == "Alpha FC"
    assert dataset.players[11].position == "Attacker"
    assert dataset.players[11].goals == 22
    assert dataset.coaches[901].wins == 8
    assert dataset.clubs[101].country == "Country A"
    assert dataset.leagues[39].total_teams == 2
    assert dataset.leagues[39].matches_played == 2
    assert dataset.leagues[39].average_attendance == 40000

    assert all(0.0 <= value <= 100.0 for value in rankings.leagues.values())
    assert all(0.0 <= value <= 100.0 for value in rankings.clubs.values())
    assert all(0.0 <= value <= 100.0 for value in rankings.players.values())
    assert all(0.0 <= value <= 100.0 for value in rankings.coaches.values())
    assert all(0.0 <= value <= 100.0 for value in rankings.national_teams.values())
    assert rankings.players[11] > rankings.players[21]
    assert rankings.national_teams["Alpha"] > rankings.national_teams["Beta"]

    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.match_id == "wc-2026-1"
    assert prediction.home_team == "Alpha"
    assert prediction.away_team == "Beta"
    assert prediction.home_score > prediction.away_score
    assert prediction.confidence > 0.34

    table = render_group_stage_markdown(predictions)
    assert "## Group A" in table
    assert "| Match | Home | Score | Away | Pick | Confidence |" in table
    assert (
        f"| 1 | Alpha | {prediction.home_score}-{prediction.away_score} | Beta | Alpha win |"
    ) in table


def test_world_cup_predictions_use_first_round_tactical_updates(
    tmp_path: Path,
) -> None:
    _write_world_cup_snapshot(tmp_path)
    _write_json(
        tmp_path / "fixtures_world_cup.json",
        {
            "response": [
                {
                    "fixture": {
                        "id": 1,
                        "date": "2026-06-11T19:00:00Z",
                        "status": {"short": "FT"},
                        "venue": {
                            "name": "Example Stadium",
                            "city": "Dallas",
                            "country": "United States",
                        },
                    },
                    "league": {"round": "Group Stage - 1"},
                    "teams": {
                        "home": {"id": 1, "name": "Alpha"},
                        "away": {"id": 2, "name": "Beta"},
                    },
                    "goals": {"home": 3, "away": 0},
                },
                {
                    "fixture": {
                        "id": 2,
                        "date": "2026-06-18T19:00:00Z",
                        "status": {"short": "NS"},
                        "venue": {
                            "name": "Example Stadium",
                            "city": "Dallas",
                            "country": "United States",
                        },
                    },
                    "league": {"round": "Group Stage - 2"},
                    "teams": {
                        "home": {"id": 2, "name": "Beta"},
                        "away": {"id": 1, "name": "Alpha"},
                    },
                },
            ]
        },
    )
    _write_json(
        tmp_path / "fixture_1_lineups.json",
        {
            "response": [
                {
                    "team": {"id": 1, "name": "Alpha"},
                    "formation": "4-3-3",
                    "startXI": [
                        {"player": {"id": 11, "name": "Alpha Forward"}},
                        {"player": {"id": 12, "name": "Alpha Midfielder"}},
                    ],
                },
                {
                    "team": {"id": 2, "name": "Beta"},
                    "formation": "5-4-1",
                    "startXI": [{"player": {"id": 21, "name": "Beta Forward"}}],
                },
            ]
        },
    )
    _write_json(
        tmp_path / "fixture_1_events.json",
        {
            "response": [
                {
                    "team": {"id": 1, "name": "Alpha"},
                    "type": "subst",
                    "player": {"id": 12, "name": "Alpha Midfielder"},
                }
            ]
        },
    )

    dataset = load_world_cup_dataset(
        tmp_path,
        expected_team_count=2,
        completed_round_limit=1,
    )
    rankings = rank_world_cup_entities(dataset)
    predictions = predict_group_stage_scores(dataset, rankings, remaining_only=True)

    assert set(dataset.completed_matches) == {"wc-2026-1"}
    assert dataset.team_updates["Alpha"].formations == ("4-3-3",)
    assert dataset.team_updates["Alpha"].starter_ids == (11, 12)
    assert dataset.team_updates["Alpha"].substitute_ids == (12,)
    assert len(predictions) == 1
    assert predictions[0].match_id == "wc-2026-2"
    assert predictions[0].away_tournament_adjustment > 0

    serialized = prediction_to_json(predictions[0])
    assert serialized["away_tournament_adjustment"] == predictions[0].away_tournament_adjustment
    assert "4-3-3" in predictions[0].rationale


def test_world_cup_elimination_predictions_use_completed_group_form(
    tmp_path: Path,
) -> None:
    _write_world_cup_snapshot(tmp_path)
    _write_json(
        tmp_path / "fixtures_world_cup.json",
        {
            "response": [
                {
                    "fixture": {
                        "id": 1,
                        "date": "2026-06-11T19:00:00Z",
                        "status": {"short": "FT"},
                        "venue": {
                            "name": "Example Stadium",
                            "city": "Dallas",
                            "country": "United States",
                        },
                    },
                    "league": {"round": "Group Stage - 1"},
                    "teams": {
                        "home": {"id": 1, "name": "Alpha"},
                        "away": {"id": 2, "name": "Beta"},
                    },
                    "goals": {"home": 3, "away": 0},
                },
                {
                    "fixture": {
                        "id": 3,
                        "date": "2026-06-29T19:00:00Z",
                        "status": {"short": "NS"},
                        "venue": {
                            "name": "Example Stadium",
                            "city": "Seattle",
                            "country": "United States",
                        },
                    },
                    "league": {"round": "Round of 32"},
                    "teams": {
                        "home": {"id": 2, "name": "Beta"},
                        "away": {"id": 1, "name": "Alpha"},
                    },
                    "goals": {"home": None, "away": None},
                },
            ]
        },
    )
    _write_json(
        tmp_path / "fixture_1_lineups.json",
        {
            "response": [
                {
                    "team": {"id": 1, "name": "Alpha"},
                    "formation": "4-3-3",
                    "startXI": [
                        {"player": {"id": 11, "name": "Alpha Forward"}},
                        {"player": {"id": 12, "name": "Alpha Midfielder"}},
                    ],
                },
                {
                    "team": {"id": 2, "name": "Beta"},
                    "formation": "5-4-1",
                    "startXI": [{"player": {"id": 21, "name": "Beta Forward"}}],
                },
            ]
        },
    )

    dataset = load_world_cup_dataset(tmp_path, expected_team_count=2)
    rankings = rank_world_cup_entities(dataset)
    group_predictions = predict_group_stage_scores(dataset, rankings, remaining_only=True)
    knockout_predictions = predict_elimination_stage_scores(dataset, rankings)

    assert group_predictions == ()
    assert dataset.matches[1].stage == "Round of 32"
    assert dataset.team_updates["Alpha"].matches == 1
    assert len(knockout_predictions) == 1
    prediction = knockout_predictions[0]
    assert prediction.match_id == "wc-2026-3"
    assert prediction.stage == "Round of 32"
    assert prediction.winner in {"Alpha", "Beta"}
    assert prediction.outcome in {Outcome.HOME_WIN, Outcome.AWAY_WIN}
    assert prediction.decided_by in {"regular_time", "extra_time", "penalties"}
    assert prediction.home_advancement_probability is not None

    serialized = prediction_to_json(prediction)
    assert serialized["stage"] == "Round of 32"
    assert serialized["winner"] == prediction.winner
    table = render_group_stage_markdown(
        knockout_predictions,
        title="FIFA 2026 World Cup Elimination Stage Predictions",
    )
    assert "## Round of 32" in table


def test_world_cup_full_elimination_bracket_projects_final() -> None:
    dataset = _synthetic_bracket_dataset()
    rankings = WorldCupRankings(
        leagues={},
        clubs={},
        players={},
        coaches={},
        national_teams={
            team: 75.0 - int(profile.name[1:]) * 4
            for team, profile in dataset.national_teams.items()
        },
    )

    predictions = predict_full_elimination_bracket(dataset, rankings)

    assert len(predictions) == 32
    assert {prediction.match_number for prediction in predictions} == set(range(73, 105))
    final = predictions[-1]
    assert final.match_number == 104
    assert final.stage == "Final"
    assert final.source_match_numbers == (101, 102)
    assert final.winner is not None
    serialized = prediction_to_json(final)
    assert serialized["match_number"] == 104
    assert serialized["source_match_numbers"] == [101, 102]


def test_world_cup_match_preview_writes_pdf_with_lineups(tmp_path: Path) -> None:
    _write_world_cup_snapshot(tmp_path)
    _write_json(
        tmp_path / "fixtures_world_cup.json",
        {
            "response": [
                {
                    "fixture": {
                        "id": 1,
                        "date": "2026-06-11T19:00:00Z",
                        "status": {"short": "FT"},
                        "venue": {
                            "name": "Example Stadium",
                            "city": "Dallas",
                            "country": "United States",
                        },
                    },
                    "league": {"round": "Group Stage - 1"},
                    "teams": {
                        "home": {"id": 1, "name": "Alpha"},
                        "away": {"id": 2, "name": "Beta"},
                    },
                    "goals": {"home": 2, "away": 0},
                },
                {
                    "fixture": {
                        "id": 2,
                        "date": "2026-06-18T19:00:00Z",
                        "status": {"short": "NS"},
                        "venue": {
                            "name": "Example Stadium",
                            "city": "Dallas",
                            "country": "United States",
                        },
                    },
                    "league": {"round": "Group Stage - 2"},
                    "teams": {
                        "home": {"id": 2, "name": "Beta"},
                        "away": {"id": 1, "name": "Alpha"},
                    },
                },
            ]
        },
    )
    _write_json(
        tmp_path / "fixture_2_lineups.json",
        {
            "response": [
                {
                    "team": {"id": 2, "name": "Beta"},
                    "coach": {"name": "Beta Coach Updated"},
                    "formation": "4-4-2",
                    "startXI": [
                        {"player": {"id": 21, "name": "Beta Forward", "number": 9, "pos": "F"}}
                    ],
                    "substitutes": [
                        {"player": {"id": 22, "name": "Beta Bench", "number": 18, "pos": "M"}}
                    ],
                },
                {
                    "team": {"id": 1, "name": "Alpha"},
                    "coach": {"name": "Alpha Coach Updated"},
                    "formation": "4-3-3",
                    "startXI": [
                        {"player": {"id": 11, "name": "Alpha Forward", "number": 11, "pos": "F"}}
                    ],
                    "substitutes": [
                        {
                            "player": {
                                "id": 12,
                                "name": "Alpha Midfielder",
                                "number": 8,
                                "pos": "M",
                            }
                        }
                    ],
                },
            ]
        },
    )

    base_dataset = load_world_cup_dataset(tmp_path, expected_team_count=2)
    target = next(match for match in base_dataset.matches if match.match_id == "wc-2026-2")
    dataset = load_world_cup_dataset(
        tmp_path,
        expected_team_count=2,
        completed_before=target.kickoff,
    )
    rankings = rank_world_cup_entities(dataset)
    preview = build_world_cup_match_preview(
        dataset,
        rankings,
        "2",
        snapshot_dir=tmp_path,
    )
    pdf_path = render_world_cup_match_preview_pdf(preview, tmp_path / "preview.pdf")

    assert preview.home.coach == "Beta Coach Updated"
    assert preview.home.formation == "4-4-2"
    assert preview.home.starters[0].name == "Beta Forward"
    serialized = match_preview_to_json(preview)
    home = cast(dict[str, object], serialized["home"])
    starters = cast(list[dict[str, object]], home["starters"])
    prediction = cast(dict[str, object], serialized["prediction"])
    assert starters[0]["name"] == "Beta Forward"
    assert prediction["match_id"] == "wc-2026-2"
    assert pdf_path.read_bytes().startswith(b"%PDF-1.4")
    assert b"Beta vs Alpha" in pdf_path.read_bytes()


def _synthetic_bracket_dataset() -> WorldCupDataSet:
    national_teams: dict[str, NationalTeamProfile] = {}
    completed_matches: dict[str, CompletedGroupMatch] = {}
    matches: list[GroupStageMatch] = []
    kickoff = datetime(2026, 6, 11, 19, 0, tzinfo=UTC)
    match_id = 1

    for group in "ABCDEFGHIJKL":
        names = tuple(f"{group}{rank}" for rank in range(1, 5))
        for rank, name in enumerate(names, start=1):
            national_teams[name] = NationalTeamProfile(
                team_id=match_id * 10 + rank,
                name=name,
                group=f"Group {group}",
                roster=(),
                coaching_staff=(),
                recent_wins=4 - rank,
                recent_draws=0,
                recent_losses=rank - 1,
                goals_for=8 - rank,
                goals_against=rank,
                history_score=80.0 - rank * 4,
            )
        for home_index, home in enumerate(names):
            for away in names[home_index + 1 :]:
                home_rank = int(home[1:])
                away_rank = int(away[1:])
                home_score = 2 if home_rank < away_rank else 0
                away_score = 0 if home_rank < away_rank else 2
                normalized_match_id = f"wc-2026-{match_id}"
                completed_matches[normalized_match_id] = CompletedGroupMatch(
                    match_id=normalized_match_id,
                    group=f"Group {group}",
                    round_number=1,
                    home_team=home,
                    away_team=away,
                    home_score=home_score,
                    away_score=away_score,
                    outcome=Outcome.HOME_WIN if home_score > away_score else Outcome.AWAY_WIN,
                )
                matches.append(
                    GroupStageMatch(
                        match_id=normalized_match_id,
                        group=f"Group {group}",
                        round_number=1,
                        home_team=home,
                        away_team=away,
                        kickoff=kickoff,
                        venue_name="Example Stadium",
                        venue_city="Dallas",
                        venue_country="United States",
                        stage="Group Stage - 1",
                    )
                )
                match_id += 1

    round_of_32 = (
        (73, "A2", "B2"),
        (74, "E1", "D3"),
        (75, "F1", "C2"),
        (76, "C1", "F2"),
        (77, "I1", "F3"),
        (78, "E2", "I2"),
        (79, "A1", "E3"),
        (80, "L1", "K3"),
        (81, "D1", "B3"),
        (82, "G1", "I3"),
        (83, "K2", "L2"),
        (84, "H1", "J2"),
        (85, "B1", "J3"),
        (86, "J1", "H2"),
        (87, "K1", "L3"),
        (88, "D2", "G2"),
    )
    for fixture_id, home, away in round_of_32:
        matches.append(
            GroupStageMatch(
                match_id=f"wc-2026-{fixture_id}",
                group=None,
                round_number=None,
                home_team=home,
                away_team=away,
                kickoff=datetime(2026, 6, 28, 19, 0, tzinfo=UTC),
                venue_name="Example Stadium",
                venue_city="Dallas",
                venue_country="United States",
                stage="Round of 32",
            )
        )

    return WorldCupDataSet(
        players={},
        coaches={},
        clubs={},
        leagues={},
        national_teams=national_teams,
        matches=tuple(matches),
        completed_matches=completed_matches,
        team_updates={},
        external_factors={},
    )


def _write_world_cup_snapshot(path: Path) -> None:
    _write_json(
        path / "fixtures_world_cup.json",
        {
            "response": [
                {
                    "fixture": {
                        "id": 1,
                        "date": "2026-06-11T19:00:00Z",
                        "venue": {
                            "name": "Example Stadium",
                            "city": "Dallas",
                            "country": "United States",
                        },
                    },
                    "league": {"round": "Group A - 1"},
                    "teams": {
                        "home": {"id": 1, "name": "Alpha"},
                        "away": {"id": 2, "name": "Beta"},
                    },
                }
            ]
        },
    )
    _write_json(
        path / "teams_world_cup.json",
        {
            "response": [
                {"team": {"id": 1, "name": "Alpha"}},
                {"team": {"id": 2, "name": "Beta"}},
            ]
        },
    )
    _write_json(
        path / "standings_world_cup.json",
        {
            "response": [
                {
                    "league": {
                        "standings": [
                            [
                                {
                                    "team": {"id": 1, "name": "Alpha"},
                                    "group": "Group Stage - Group A",
                                },
                                {
                                    "team": {"id": 2, "name": "Beta"},
                                    "group": "Group Stage - Group A",
                                },
                            ],
                            [
                                {
                                    "team": {"id": 1, "name": "Alpha"},
                                    "group": "Group Stage",
                                },
                            ],
                        ]
                    }
                }
            ]
        },
    )
    _write_json(
        path / "external_factors.json",
        {
            "country_history": {"Alpha": 78, "Beta": 42},
            "country_strength": {"Country A": 85, "Country B": 45},
            "league_average_attendance": {"39": 40000, "140": 12000},
            "club_major_titles": {"101": 30, "102": 8, "201": 1},
        },
    )
    _write_team_snapshot(path, 1, "Alpha", (11, 12), coach_id=901, wins=8, losses=1)
    _write_team_snapshot(path, 2, "Beta", (21,), coach_id=902, wins=2, losses=6)
    _write_player_snapshot(
        path,
        player_id=11,
        name="Alpha Forward",
        age=25,
        position="Attacker",
        club_id=101,
        club="Alpha FC",
        league_id=39,
        league="A League",
        goals=22,
        rating="7.40",
    )
    _write_player_snapshot(
        path,
        player_id=12,
        name="Alpha Midfielder",
        age=28,
        position="Midfielder",
        club_id=102,
        club="Alpha United",
        league_id=39,
        league="A League",
        goals=5,
        rating="6.80",
    )
    _write_player_snapshot(
        path,
        player_id=21,
        name="Beta Forward",
        age=30,
        position="Attacker",
        club_id=201,
        club="Beta FC",
        league_id=140,
        league="B League",
        goals=4,
        rating="6.30",
    )
    _write_club_and_league_snapshots(path)


def _write_team_snapshot(
    path: Path,
    team_id: int,
    team_name: str,
    player_ids: tuple[int, ...],
    *,
    coach_id: int,
    wins: int,
    losses: int,
) -> None:
    _write_json(
        path / f"team_{team_id}_squad.json",
        {
            "response": [
                {
                    "team": {"id": team_id, "name": team_name},
                    "players": [
                        {
                            "id": player_id,
                            "name": f"Player {player_id}",
                            "age": 25,
                            "position": "Attacker",
                        }
                        for player_id in player_ids
                    ],
                }
            ]
        },
    )
    _write_json(
        path / f"team_{team_id}_coaches.json",
        {
            "response": [
                {
                    "id": coach_id,
                    "name": f"{team_name} Coach",
                    "age": 52,
                    "record": {"wins": wins, "draws": 1, "losses": losses},
                    "career": [{"team": {"name": f"{team_name} U23"}}],
                }
            ]
        },
    )
    _write_json(
        path / f"coach_{coach_id}_trophies.json",
        {"response": [{"place": "Winner"}] if wins > losses else []},
    )
    _write_json(
        path / f"team_{team_id}_recent_fixtures.json",
        {
            "response": [
                {
                    "teams": {
                        "home": {"id": team_id, "name": team_name},
                        "away": {"id": 99, "name": "Opponent"},
                    },
                    "goals": {"home": 2 if wins > losses else 0, "away": 0 if wins > losses else 2},
                }
            ]
        },
    )


def _write_player_snapshot(
    path: Path,
    *,
    player_id: int,
    name: str,
    age: int,
    position: str,
    club_id: int,
    club: str,
    league_id: int,
    league: str,
    goals: int,
    rating: str,
) -> None:
    _write_json(
        path / f"player_{player_id}_profile.json",
        {"response": [{"player": {"id": player_id, "name": name, "age": age}}]},
    )
    _write_json(
        path / f"player_{player_id}_statistics.json",
        {
            "response": [
                {
                    "player": {"id": player_id, "name": name, "age": age},
                    "statistics": [
                        {
                            "team": {"id": club_id, "name": club},
                            "league": {"id": league_id, "name": league},
                            "games": {"position": position, "rating": rating},
                            "goals": {"total": goals},
                        }
                    ],
                }
            ]
        },
    )


def _write_club_and_league_snapshots(path: Path) -> None:
    for club_id, league_id, name, wins, draws, losses in (
        (101, 39, "Alpha FC", 24, 5, 3),
        (102, 39, "Alpha United", 18, 7, 7),
        (201, 140, "Beta FC", 9, 8, 15),
    ):
        _write_json(
            path / f"club_{club_id}_statistics_league_{league_id}.json",
            {
                "response": {
                    "team": {"id": club_id, "name": name},
                    "fixtures": {
                        "wins": {"total": wins},
                        "draws": {"total": draws},
                        "loses": {"total": losses},
                    },
                }
            },
        )
    for league_id, name, country in ((39, "A League", "Country A"), (140, "B League", "Country B")):
        _write_json(
            path / f"league_{league_id}.json",
            {
                "response": [
                    {
                        "league": {"id": league_id, "name": name},
                        "country": {"name": country},
                    }
                ]
            },
        )
        _write_json(
            path / f"league_{league_id}_standings.json",
            {
                "response": [
                    {
                        "league": {
                            "standings": [
                                [
                                    {"team": {"id": league_id * 10}},
                                    {"team": {"id": league_id * 10 + 1}},
                                ]
                            ]
                        }
                    }
                ]
            },
        )
        _write_json(
            path / f"league_{league_id}_fixtures.json",
            {"response": [{"fixture": {"id": 1}}, {"fixture": {"id": 2}}]},
        )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
