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
