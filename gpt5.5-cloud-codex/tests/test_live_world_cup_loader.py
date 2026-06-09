from soccer.live_world_cup import (
    DEFAULT_WORLD_CUP_SCHEDULE_URL,
    load_world_cup_catalog,
)


class StaticHttpClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.requested_urls: list[str] = []

    def get_text(self, url: str) -> str:
        self.requested_urls.append(url)
        return self.payload


def test_load_world_cup_catalog_uses_injected_http_client() -> None:
    payload = """
    {
      "matches": [
        {
          "matchId": "10",
          "homeTeam": "Argentina",
          "awayTeam": "Brazil",
          "utcDate": "2026-06-11T19:00:00Z",
          "venue": {
            "name": "MetLife Stadium",
            "city": "East Rutherford",
            "country": "United States"
          }
        }
      ]
    }
    """
    client = StaticHttpClient(payload)

    catalog = load_world_cup_catalog(client=client)

    assert client.requested_urls == [DEFAULT_WORLD_CUP_SCHEDULE_URL]
    request = catalog.requests_for_competition("world-cup-2026")[0]
    assert request.home_team == "Argentina"
    assert request.away_team == "Brazil"


def test_live_loader_rejects_shell_only_payload() -> None:
    client = StaticHttpClient("<html><body>No schedule data here</body></html>")

    try:
        load_world_cup_catalog(client=client)
    except ValueError as exc:
        assert "No World Cup matches found" in str(exc)
    else:
        raise AssertionError("Expected ValueError for shell-only payload")
