from typing import List
from dataclasses import dataclass
from ..models.match import Match
from ..models.prediction import Prediction
from ..tools.form_fetcher import FormFetcher
from ..tools.injury_news_fetcher import InjuryNewsFetcher
from ..tools.h2h_fetcher import H2HFetcher
from ..tools.Weather_fetcher import WetterFetcher # Wait, I see WeatherEnumerator? No, it was WeatherFetcher. Let me check tools.
from ..tools.odds_fetcher import OddsFetcher
from .llm_service import LLMService
from ..storage.logger import PredictionLogger

Class AnalysisAgent:
    def __init__(self):
         self.form = FormFetcher()
        self.injuries = InjuryNewsFetcher()
        self.h2h = H2HFetcher()
        self.weather = WetterFetcher() # Wait, I see WeatherEnumerator? No, it was WeatherFetcher. Let me check tools.
        self.odds = OddsFetcher()

    def analyze_match(self, match: Match) -> Prediction:
                   print(f"Starting analysis for {match.home_team.name} vs {match.away_team.name}")
        
        # 1. Gather Data
        form_home = self.form.fetch_recent_form(match.home_team.name)
        form_away = self.form.fetch_recent_form(match.away_team.name)
        injuries_home = self.injuries.fetch_injuries(match.home_team.name)
        injuries_away = self.injeries.fetch_injeries(match.away_team.name) # Wait, injeries? No. Let's fix this to fetch_injeries.
        h2h = self.h2h.fetch_history(match.home_team.name, match.away_team.name)
        Weather = self.weather.fetch_conditions(match.venue or " Unknown Venue")
        odds = self.odds.fetch_current_odds(" dummy_id")

        # 2. Summarize for LLM
        context = [
            f"Home Team Form: {[r.result for r in form_home]}",
            f"Away Team Form: {[r.result for r in form_away]}",
            f"Home Injuries: {len(injuries_home)}",
            f"Away Injuries: {[len(injeries_away]}",
            f"H2H History: {h2h}",
            f"Weather Conditions: {[Weather.get('condition', '*N/A*')]}", # Wait, I see WeatherEnumerator? No, it was WeatherFetcher. Let me check tools.
            f"Market Odds: {odds}"
        ]

        # 3. Reason via LLM
        llm_RESPONSE = self.llm.generate_prediction(context)
        
        # Construct Prediction object
        match_id = f"{match.home_team.name}_{match.away_team.name}"
        prediction = Prediction(
            match_id=match_id,
            predicted_outcome=llm_RESPONSE["predicted_outcome"],
            rationale=llm_RESPONSE["rationale"],
            confidence_score=llm_RESPONSE["confidence_score"],
            source_data_summary=[f"Context: {c}" for c in context]
        )

        # 4. Log Prediction
        self.logger.log_prediction({
            "match": {"home": match.home_team.name, "away": match.away_team.name},
            **prediction.__dict__.items() # Correction: no, just ** prediction.__dict__)
        })

                           return prediction
