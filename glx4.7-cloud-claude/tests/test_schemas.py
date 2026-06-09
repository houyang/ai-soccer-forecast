from soccer_agent.tools.schemas import (
    FormSummary, H2HSummary, InjuryReport,
    OddsSummary, WeatherForecast, VenueInfo
)


def test_form_summary_creation():
    form = FormSummary(
        team_id="team_1",
        last_n_matches=5,
        record={"win": 3, "draw": 1, "loss": 1},
        goals_scored=8,
        goals_conceded=4,
        momentum_score=0.6,
        last_5=[
            {"outcome": "win", "score": "2-1", "opponent": "Team B"},
            {"outcome": "draw", "score": "1-1", "opponent": "Team C"},
        ]
    )
    assert form.team_id == "team_1"
    assert form.record["win"] == 3
    assert form.momentum_score == 0.6


def test_odds_summary_value_detection():
    odds = OddsSummary(
        match_id="match_1",
        home_win_odds={"bet365": 2.10, "william_hill": 2.15},
        draw_odds={"bet365": 3.40, "william_hill": 3.45},
        away_win_odds={"bet365": 3.20, "william_hill": 3.25},
        implied_prob_home=0.45,
        value_detected=True
    )
    assert odds.value_detected is True
    assert odds.implied_prob_home == 0.45


def test_injury_report_impact_score():
    injury = InjuryReport(
        team_id="team_1",
        key_out=[
            {"player": "Star Player", "position": "forward", "severity": "high", "return_date": "2025-07-01"}
        ],
        doubtful=[
            {"player": "Midfielder", "position": "midfielder", "severity": "low", "return_date": "2025-06-05"}
        ],
        impact_score=0.7
    )
    assert injury.impact_score == 0.7
    assert len(injury.key_out) == 1