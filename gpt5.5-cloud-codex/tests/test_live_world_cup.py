from soccer.live_world_cup import WORLD_CUP_COMPETITION_ID, catalog_from_payload


def test_catalog_from_json_payload_builds_world_cup_competition() -> None:
    payload = """
    {
      "fixtures": [
        {
          "matchNumber": 1,
          "homeTeam": {"name": "Mexico"},
          "awayTeam": {"name": "South Africa"},
          "kickoff": "2026-06-11T19:00:00Z",
          "venue": {
            "name": "Mexico City Stadium",
            "city": "Mexico City",
            "country": "Mexico"
          }
        },
        {
          "matchNumber": 2,
          "homeTeam": {"name": "Korea Republic"},
          "awayTeam": {"name": "Czechia"},
          "kickoff": "2026-06-12T03:00:00+00:00",
          "venueName": "Guadalajara Stadium",
          "city": "Guadalajara",
          "country": "Mexico"
        }
      ]
    }
    """

    catalog = catalog_from_payload(payload)

    requests = catalog.requests_for_competition(WORLD_CUP_COMPETITION_ID)
    assert len(requests) == 2
    assert requests[0].match_id == "wc-2026-1"
    assert requests[0].home_team == "Mexico"
    assert requests[0].away_team == "South Africa"
    assert catalog.venues["wc-2026-2"].name == "Guadalajara Stadium"
    assert catalog.forms["Mexico"].matches == 0


def test_catalog_from_html_payload_reads_embedded_json() -> None:
    payload = """
    <html>
      <body>
        <script type="application/json">
          {
            "props": {
              "pageProps": {
                "matches": [
                  {
                    "id": "match-10",
                    "home": "Germany",
                    "away": "Curacao",
                    "date": "2026-06-14T19:00:00Z",
                    "stadium": "Houston Stadium",
                    "hostCity": "Houston",
                    "countryName": "United States"
                  }
                ]
              }
            }
          }
        </script>
      </body>
    </html>
    """

    catalog = catalog_from_payload(payload)

    request = catalog.requests_for_competition(WORLD_CUP_COMPETITION_ID)[0]
    assert request.match_id == "wc-2026-match-10"
    assert request.home_team == "Germany"
    assert request.away_team == "Curacao"
    assert catalog.venues[request.match_id].city == "Houston"
