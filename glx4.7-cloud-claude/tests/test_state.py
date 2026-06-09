from soccer_agent.workflows.state import PredictionState, EvaluationState


def test_prediction_state_creation():
    state = PredictionState(
        match_id="match_1",
        competition_id="premier_league",
        stage="group"
    )

    assert state.match_id == "match_1"
    assert state.competition_id == "premier_league"
    assert state.stage == "group"
    assert state.team_a_form is None


def test_prediction_state_with_data():
    from soccer_agent.tools.schemas import FormSummary

    form = FormSummary(
        team_id="team_1",
        last_n_matches=5,
        record={"win": 3, "draw": 1, "loss": 1},
        goals_scored=8,
        goals_conceded=4,
        momentum_score=0.6,
        last_5=[]
    )

    state = PredictionState(
        match_id="match_1",
        competition_id="pl",
        stage="group",
        team_a_form=form
    )

    assert state.team_a_form.momentum_score == 0.6


def test_evaluation_state_creation():
    state = EvaluationState()

    assert state.pending_predictions == []
    assert state.evaluated_predictions == []
    assert not state.metrics_updated