"""Soccer match prediction agent.

A multi-tool agent that, given an upcoming match, autonomously gathers
context (form, injuries, H2H, weather, venue, bookmaker odds), reasons
about the matchup, emits a prediction (pick, confidence, rationale),
logs it, waits for the result, and self-evaluates.
"""

__version__ = "0.1.0"
