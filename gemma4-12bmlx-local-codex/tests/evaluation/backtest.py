import pytest
from datetime import datetime
from soccer.models.match import Match, Team
from soccer.agent.reasoner import AnalysisAgent

def test_agent_prediction():
    # Setup
    home = Team("Manchester United", "MUN")
    away = Team("Liverpool", "LIV")
    match = Match(home_team=home, away_team=away, timestamp=datetime.now(), venue="Old Trafford")
    
    agent = AnalysisAgent()
    prediction = agent.analyze_match(match)
    
    assert prediction.confidence_score > 0
    assert "rationale" in prediction
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    test_agent_prediction()
