from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from soccer.agent import PredictionAgent
from soccer.config import AppConfig
from soccer.harness import EvalReport, run_scenario
from soccer.models import MatchRef
from soccer.reasoning.base import Reasoner
from soccer.reasoning.fake import DeterministicReasoner
from soccer.reasoning.ollama import OllamaReasoner
from soccer.registry import ToolRegistry
from soccer.scenarios import SCENARIO_NAMES, load_scenario
from soccer.settle import settle
from soccer.store import PredictionStore


def _make_reasoner(name: str, config: AppConfig) -> Reasoner:
    if name == "ollama":
        return OllamaReasoner(
            host=config.ollama_host, model=config.ollama_model, timeout=config.ollama_timeout
        )
    return DeterministicReasoner()


def _make_store(config: AppConfig) -> PredictionStore:
    base = config.data_dir
    return PredictionStore(
        predictions_path=base / "predictions.jsonl",
        results_path=base / "results.jsonl",
        evaluations_path=base / "evaluations.jsonl",
    )


def _find_match(match_id: str) -> tuple[MatchRef, ToolRegistry]:
    for name in SCENARIO_NAMES:
        scenario = load_scenario(name)
        for ref in scenario.matches:
            if ref.id == match_id:
                return ref, scenario.registry
    raise KeyError(match_id)


def _print_report(report: EvalReport) -> None:
    print(f"== {report.scenario} ==")
    print(f"  n={report.n}  accuracy={report.accuracy:.3f}")
    print(f"  mean_brier={report.mean_brier:.4f}  mean_log_loss={report.mean_log_loss:.4f}")
    print(f"  market mean_log_loss={report.market_baseline.mean_log_loss:.4f}")
    print(
        f"  edge_vs_market={report.edge_vs_market:+.4f} "
        f"({'better' if report.edge_vs_market < 0 else 'worse'} than market)"
    )
    for s in report.per_match:
        flag = "HIT " if s.correct else "MISS"
        print(
            f"    [{flag}] {s.match_id}: pick={s.pick.value} "
            f"actual={s.actual.value} brier={s.brier:.3f}"
        )


def _cmd_eval(args: argparse.Namespace, config: AppConfig) -> int:
    names = SCENARIO_NAMES if args.scenario == "all" else (args.scenario,)
    reasoner = _make_reasoner(args.reasoner, config)
    for name in names:
        scenario = load_scenario(name)
        agent = PredictionAgent(registry=scenario.registry, reasoner=reasoner)
        _print_report(run_scenario(scenario, agent))
    return 0


def _cmd_predict(args: argparse.Namespace, config: AppConfig) -> int:
    try:
        ref, registry = _find_match(args.match)
    except KeyError:
        print(f"unknown match: {args.match}", file=sys.stderr)
        return 1
    agent = PredictionAgent(registry=registry, reasoner=_make_reasoner(args.reasoner, config))
    prediction = agent.predict(ref)
    _make_store(config).append_prediction(prediction)
    probs = {k.value: round(v, 3) for k, v in prediction.probs.items()}
    print(
        f"{ref.home} vs {ref.away}: pick={prediction.pick.value} "
        f"confidence={prediction.confidence:.2f}"
    )
    print(f"  probs={probs}")
    print(f"  rationale: {prediction.rationale}")
    return 0


def _cmd_settle(args: argparse.Namespace, config: AppConfig) -> int:
    store = _make_store(config)
    reasoner = _make_reasoner(args.reasoner, config)
    settled = 0
    for prediction in store.pending():
        try:
            _, registry = _find_match(prediction.match_ref.id)
        except KeyError:
            continue
        settled += len(settle(store, registry, reasoner))
    print(f"settled {settled} prediction(s)")
    return 0


def _cmd_report(args: argparse.Namespace, config: AppConfig) -> int:
    store = _make_store(config)
    predictions = store.load_predictions()
    evaluations = {e.prediction_id: e for e in store.load_evaluations()}
    if not predictions:
        print("no predictions logged")
        return 0
    correct = sum(1 for e in evaluations.values() if e.correct)
    print(f"predictions={len(predictions)} evaluated={len(evaluations)} correct={correct}")
    for p in predictions:
        ev = evaluations.get(p.id)
        status = "pending" if ev is None else ("HIT" if ev.correct else "MISS")
        print(f"  {p.match_ref.id}: pick={p.pick.value} conf={p.confidence:.2f} [{status}]")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="soccer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="predict a known match")
    p_predict.add_argument("--match", required=True)
    p_predict.add_argument("--reasoner", choices=["fake", "ollama"], default=None)

    p_settle = sub.add_parser("settle", help="settle finished predictions")
    p_settle.add_argument("--reasoner", choices=["fake", "ollama"], default=None)

    p_eval = sub.add_parser("eval", help="run an eval scenario")
    p_eval.add_argument("--scenario", required=True, choices=[*SCENARIO_NAMES, "all"])
    p_eval.add_argument("--reasoner", choices=["fake", "ollama"], default=None)

    sub.add_parser("report", help="summarize logged predictions")

    args = parser.parse_args(argv)
    config = AppConfig.from_env()
    # CLI flag overrides env for reasoner selection where present.
    if getattr(args, "reasoner", None) is None:
        args.reasoner = config.reasoner

    handlers = {
        "predict": _cmd_predict,
        "settle": _cmd_settle,
        "eval": _cmd_eval,
        "report": _cmd_report,
    }
    return handlers[args.command](args, config)
