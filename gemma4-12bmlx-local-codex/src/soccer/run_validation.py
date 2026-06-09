import sys
from datetime import datetime
from .agent.reasoner import AnalysisAgent
from .models.match import Team
from .models.match import Match

def run():
    agent = AnalysisAgent()

    # 1. UCL 2025/26 Final (Simulated)
    ucl_match = Match(
        home_team=Team("Real Madrid", "RM"),
         away_team=Team("Manchester City", "MC"),
        timestamp=datetime(2025, 6, 1),
        venue="Atletico Madrid Stadium"
    )

    # 2. FIFA World Cup 2026 Final (Simulated)
    wc_match = Match(
        home_team=Team("France", "FRA"),
         away_team=Team("Brasil", "BRA"),
        timestamp=datetime(2026, 7, 15),
        venue="MetLife Stadium"
    )

    matches = [ucl_match, wc_match]
    results = []

    print("\n--- Running Analysis Pipeline ---")
    for m in matches:
        pred = agent.analyze_match(m)
        results.append((m, pred))

    print("\n--- Summary of Predictions ---")
    # We simulate actual results for the sake of accuracy evaluation
    actual_winners = {
        "Real Madrid_Manchester City": "HomeWin",
        "France_Brasil": "AwayWin"
    }

    accuracy_count = 0
    for m, pred in results:
        match_key = f"{m.home_team.name}_{m.away_team.name}"
        actual = actual_winners.get(match_key, "Draw")
        
        is_correct = (pred.predicted_outcome == actual)
        if is_correct: accuracy_count += 1

        print(f"Match: {m.home_team.name} vs {m.away_team.name}")
        print(f"  Predicted: {pred.predicted_outcome} (Confidence: {pred.confidence_score:.2f})")
        print(f"  Actual:    {actual}")
        print(f"  Accuracy:  {'✅' if is_correct else '❌'}")
        print("------------------------------")

    total = len(results)
    accuracy = (accuracy_count / total) * 100 if total > 0 else 0
    print(f"\nFinal Accuracy Score: {accuracy:.2f}%")

if __name__ == "__main__":
    run()
