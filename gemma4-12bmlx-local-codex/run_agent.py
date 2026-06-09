from datetime import datetime
from dataclasses import dataclass
from soccer.models.match import Match
from soccer.models.prediction import Prediction
from soccer.agent.reasoner import AnalysisAgent
from soccer.storage.logger import PredictionLogger

@dataclass(frozen=True)
class Team:
    name: str
    short_name: str

def run_demo():
    home = Team("Real Madrid", "RM")
    away = Team("Manchester City", "MC")
    match = Match(home_team=home, away_team=away, timestamp=datetime.now(), venue="Santiago Bernabéu")

    agent = AnalysisAgent()
    logger = PredictionLogger()

    prediction = agent.analyze_match(match)
    print(f"Prediction: {prediction.predicted_outcome}")
    print(f"Rationale: {prediction.rationale}")
    print(f"Confidence: {prediction.confidence_score}")

    # Log it
    logger.log_prediction({
        "match": f"{home.name} vs {away.name}",
        "prediction": prediction.__dict__.copy() # simple serialization
    })

if __name__ == "__main__":
    run_demo()
