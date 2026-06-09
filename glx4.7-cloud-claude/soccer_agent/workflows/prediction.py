# soccer_agent/workflows/prediction.py
from typing import Literal
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END

from soccer_agent.workflows.state import PredictionState
from soccer_agent.tools.api_football import FetchTeamFormTool, FetchH2HTool
from soccer_agent.tools.injuries import FetchInjuriesTool
from soccer_agent.tools.odds import FetchOddsTool
from soccer_agent.tools.weather import FetchWeatherTool
from soccer_agent.tools.schemas import FormSummary, H2HSummary, InjuryReport, OddsSummary, WeatherForecast


# Tool instances (these would be injected in production)
_form_tool: FetchTeamFormTool | None = None
_h2h_tool: FetchH2HTool | None = None
_injuries_tool: FetchInjuriesTool | None = None
_odds_tool: FetchOddsTool | None = None
_weather_tool: FetchWeatherTool | None = None
_llm: ChatAnthropic | None = None


def initialize_tools(
    form_tool: FetchTeamFormTool,
    h2h_tool: FetchH2HTool,
    injuries_tool: FetchInjuriesTool,
    odds_tool: FetchOddsTool,
    weather_tool: FetchWeatherTool,
    llm: ChatAnthropic
):
    """Initialize tool instances for the workflow."""
    global _form_tool, _h2h_tool, _injuries_tool, _odds_tool, _weather_tool, _llm
    _form_tool = form_tool
    _h2h_tool = h2h_tool
    _injuries_tool = injuries_tool
    _odds_tool = odds_tool
    _weather_tool = weather_tool
    _llm = llm


async def fetch_team_a_form(state: PredictionState) -> PredictionState:
    """Fetch form data for team A."""
    if _form_tool is None:
        raise RuntimeError("Form tool not initialized")

    # For now, return state unchanged (would fetch team_a_id from match)
    # This is a simplified version
    return state


async def fetch_team_b_form(state: PredictionState) -> PredictionState:
    """Fetch form data for team B."""
    if _form_tool is None:
        raise RuntimeError("Form tool not initialized")
    return state


async def fetch_h2h(state: PredictionState) -> PredictionState:
    """Fetch head-to-head history."""
    if _h2h_tool is None:
        raise RuntimeError("H2H tool not initialized")
    return state


async def fetch_injuries(state: PredictionState) -> PredictionState:
    """Fetch injury reports for both teams."""
    if _injuries_tool is None:
        raise RuntimeError("Injuries tool not initialized")
    return state


async def fetch_odds(state: PredictionState) -> PredictionState:
    """Fetch betting odds."""
    if _odds_tool is None:
        raise RuntimeError("Odds tool not initialized")
    return state


async def fetch_weather(state: PredictionState) -> PredictionState:
    """Fetch weather forecast."""
    if _weather_tool is None:
        raise RuntimeError("Weather tool not initialized")
    return state


async def analyze_context(state: PredictionState) -> PredictionState:
    """Analyze tournament context."""
    if state.stage == "knockout":
        state.context_analysis = "Knockout match: consider aggregate score, away goals rule, fatigue"
    elif state.stage == "final":
        state.context_analysis = "Final: neutral venue, high pressure, rest days critical"
    else:
        state.context_analysis = "Standard league or group stage match"
    return state


async def synthesize_reasoning(state: PredictionState) -> PredictionState:
    """Use LLM to synthesize all data into a prediction."""
    if _llm is None:
        raise RuntimeError("LLM not initialized")

    prompt = _build_reasoning_prompt(state)
    response = await _llm.ainvoke(prompt)

    # Parse response (simplified - would need proper parsing)
    state.predicted_outcome = "home"  # Placeholder
    state.confidence_score = 65.0  # Placeholder
    state.synthesized_rationale = str(response.content)

    return state


def _build_reasoning_prompt(state: PredictionState) -> str:
    """Build the reasoning prompt for the LLM."""
    prompt_parts = [
        "You are a soccer prediction analyst. Predict the outcome of the following match.",
        f"Competition: {state.competition_id}, Stage: {state.stage}",
    ]

    if state.team_a_form:
        prompt_parts.append(f"\nTeam A form: {state.team_a_form.record}")

    if state.team_b_form:
        prompt_parts.append(f"\nTeam B form: {state.team_b_form.record}")

    if state.h2h_history:
        prompt_parts.append(f"H2H: A wins {state.h2h_history.team_a_wins}, "
                           f"B wins {state.h2h_history.team_b_wins}, "
                           f"draws {state.h2h_history.draws}")

    if state.context_analysis:
        prompt_parts.append(f"\nContext: {state.context_analysis}")

    prompt_parts.append(
        "\nOutput JSON with: {\"outcome\": \"home\"|\"draw\"|\"away\", \"confidence\": 0-100, \"rationale\": \"...\"}"
    )

    return "\n".join(prompt_parts)


async def calculate_confidence(state: PredictionState) -> PredictionState:
    """Calculate confidence score based on formula."""
    # Simple formula: base on form momentum + h2h advantage
    confidence = 50.0  # Base

    if state.team_a_form:
        confidence += state.team_a_form.momentum_score * 15

    if state.team_b_form:
        confidence -= state.team_b_form.momentum_score * 15

    # Clamp to 0-100
    state.confidence_score = max(0.0, min(100.0, confidence))
    return state


def build_prediction_graph() -> StateGraph:
    """Build the prediction LangGraph."""
    workflow = StateGraph(PredictionState)

    # Add nodes
    workflow.add_node("fetch_form_a", fetch_team_a_form)
    workflow.add_node("fetch_form_b", fetch_team_b_form)
    workflow.add_node("fetch_h2h", fetch_h2h)
    workflow.add_node("fetch_injuries", fetch_injuries)
    workflow.add_node("fetch_odds", fetch_odds)
    workflow.add_node("fetch_weather", fetch_weather)
    workflow.add_node("analyze_context", analyze_context)
    workflow.add_node("synthesize", synthesize_reasoning)
    workflow.add_node("calculate_confidence", calculate_confidence)

    # Add edges - parallel fetching
    workflow.set_entry_point("fetch_form_a")
    workflow.add_edge("fetch_form_a", "fetch_form_b")
    workflow.add_edge("fetch_form_b", "fetch_h2h")
    workflow.add_edge("fetch_h2h", "fetch_injuries")
    workflow.add_edge("fetch_injuries", "fetch_odds")
    workflow.add_edge("fetch_odds", "fetch_weather")
    workflow.add_edge("fetch_weather", "analyze_context")
    workflow.add_edge("analyze_context", "synthesize")
    workflow.add_edge("synthesize", "calculate_confidence")
    workflow.add_edge("calculate_confidence", END)

    return workflow.compile()