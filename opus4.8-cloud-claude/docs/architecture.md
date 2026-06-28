# Architecture

`PredictionAgent` runs a deterministic pipeline over a `ToolRegistry` of single-method
provider Protocols (fixture + HTTP implementations), assembling a `MatchDossier`
(`dossier.build_dossier`, which degrades gracefully on tool failure), then calls a
`Reasoner` once (`reasoning/`: `DeterministicReasoner` or `OllamaReasoner`). Predictions
persist as JSONL via `PredictionStore`. `settle` matches finished results to pending
predictions, scores them (`evaluation.score`), and stores a self-critique. The offline
`harness` runs `Scenario` fixtures with known results and reports accuracy, Brier,
log-loss, calibration, and edge vs the bookmaker baseline. The CLI (`soccer`) exposes
`predict`, `settle`, `eval`, and `report`.

Swap points (all at the registry/config boundary): provider mode (`fixture|http`) and
reasoner (`fake|ollama`). The agent depends only on `ToolRegistry` and the `Reasoner`
protocol. The registry's `as_tools()` view is the seam for a future model-driven
tool-selection loop.

## World Cup pipeline

### Single-match preview card

`soccer wc card <fixture_id>` previews one upcoming match. `lineup.project_lineup` resolves the
most likely XI/formation (confirmed → prior matchday → squad projection); `predict.predict_one`
reuses the Poisson core but applies tournament momentum plus that match's lineup-quality and
formation lean via `adjust.adjustment_for_match`; `card.build_card` packages coaches, lineups,
and the forecast; and `cardpdf.render_card_pdf` renders it with reportlab (the optional `[pdf]`
extra, imported lazily). `--refresh` merges one fixture's latest lineup/result via
`live.refresh_fixture`.

### Knockout-stage forecast

`soccer wc knockout` forecasts the bracket from the Round of 32 to the final. `standings`
computes final group tables from results (FIFA tiebreakers; the API's `rank` field is
unreliable). `bracket` maps the live R32 fixtures onto the official FIFA-2026 slot tree
(matches 73–104), keyed by each team's `(group, rank)` label and cross-checked one-to-one.
`predict.predict_knockout` reuses the Poisson/Dixon-Coles core but resolves no-draw ties with a
reduced-rate extra-time matrix and a rating-edged penalty shootout, exposing an advancement
probability. `simulate.run_modal_bracket` walks the higher-probability side forward to a single
champion and podium; `simulate.run_monte_carlo` re-simulates the whole bracket with an injected,
seeded `random.Random` (pairwise odds cached by team pair) to produce title and round-reach
odds. Knockout matches are modelled at a neutral site (no host bonus; travel/weather retained).
