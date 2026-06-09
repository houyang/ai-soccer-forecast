"""Tool package — re-exports each concrete tool for easy registration."""

from .base import BaseTool, ToolError, ToolRegistry, ToolResult
from .form_recent import FormRecentTool
from .h2h_history import H2HHistoryTool
from .injury_news import InjuryNewsTool
from .odds_market import OddsMarketTool
from .venue_info import VenueInfoTool
from .weather_venue import WeatherVenueTool

__all__ = [
    "BaseTool",
    "FormRecentTool",
    "H2HHistoryTool",
    "InjuryNewsTool",
    "OddsMarketTool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "VenueInfoTool",
    "WeatherVenueTool",
]


def default_registry() -> "ToolRegistry":
    """Build a registry pre-loaded with the six Phase-1 tools."""
    reg = ToolRegistry()
    reg.register(FormRecentTool())
    reg.register(InjuryNewsTool())
    reg.register(H2HHistoryTool())
    reg.register(WeatherVenueTool())
    reg.register(OddsMarketTool())
    reg.register(VenueInfoTool())
    return reg
