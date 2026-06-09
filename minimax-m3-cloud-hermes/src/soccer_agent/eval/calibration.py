"""Calibration evaluation (Task 28).

Runs the agent over a set of (prediction, actual_winner) pairs,
extracts the stated confidence, converts the multi-class problem to
a 1D "did we pick right" outcome, then measures and tries to
improve calibration.

The reduction we use: for a single prediction with stated
confidence `c` and pick `p`, the corresponding "probability of
being right" is `c` if the pick matched the actual, and
`(1 - c)` if not. This is the standard "one-versus-rest" reduction
for multi-class calibration (see e.g. Guo et al. 2017, "On
Calibration of Modern Neural Networks").

With only 10 eval cases this is a *research* task: the per-bucket
sample size is too small to fit a calibrator without overfitting.
We report raw ECE, Brier, and a leave-one-out cross-validation
score for each calibrator so the user can pick the method that
generalizes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soccer_agent.calibration import (  # noqa: E402
    BinningCalibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    brier,
    ece,
    reliability_table,
)
from soccer_agent.db import Database, init_db  # noqa: E402
from soccer_agent.eval.dataset import EVAL_CASES, EvalCase  # noqa: E402
from soccer_agent.models import Match, Team  # noqa: E402


@dataclass
class CalibSample:
    match_id: str
    pick: str
    actual: str
    confidence: float
    # Reduced: 1 if pick == actual, 0 if not.
    outcome: int
    # Reduced probability of "right": confidence if right, 1-c if wrong.
    p_right: float


def _match_from_case(case: EvalCase) -> Match:
    return Match(
        match_id=case.match_id,
        competition=case.competition,
        round=case.round,
        kickoff=case.kickoff,
        home=Team(id=case.home_id, name=case.home_id.replace("_", " ").title()),
        away=Team(id=case.away_id, name=case.away_id.replace("_", " ").title()),
        venue_id=case.venue_id,
    )


def collect_samples(
    db_path: Path,
    cases: list[EvalCase] | None = None,
) -> list[CalibSample]:
    """Walk the predictions table, join with eval cases by match_id,
    produce CalibSamples."""
    cases = cases if cases is not None else EVAL_CASES
    case_by_id = {c.match_id: c for c in cases}
    with Database(db_path) as db:
        rows = db.execute(
            "SELECT match_id, final_pick, final_confidence "
            "FROM predictions ORDER BY created_at"
        )
    samples: list[CalibSample] = []
    for r in rows:
        mid = r["match_id"]
        if mid not in case_by_id:
            continue
        case = case_by_id[mid]
        pick = r["final_pick"]
        conf = float(r["final_confidence"])
        outcome = 1 if pick == case.actual_winner else 0
        p_right = conf if outcome == 1 else (1.0 - conf)
        samples.append(CalibSample(
            match_id=mid, pick=pick, actual=case.actual_winner,
            confidence=conf, outcome=outcome, p_right=p_right,
        ))
    return samples


def loo_eval(samples: list[CalibSample]) -> dict[str, dict]:
    """Leave-one-out cross-validation for each calibrator.

    For each sample, fit on the rest, apply the calibrator to the
    held-out sample, and record the calibrated probability and the
    outcome. Then compute the post-calibration ECE/Brier over the
    full LOO set.
    """
    methods: dict[str, object] = {
        "identity": IdentityCalibrator,
        "platt": PlattCalibrator,
        "temperature": TemperatureCalibrator,
        "isotonic": IsotonicCalibrator,
        "binning": BinningCalibrator,
    }
    out: dict[str, dict] = {}
    probs_in = [s.p_right for s in samples]
    outs = [s.outcome for s in samples]
    for name, cls in methods.items():
        cal_probs: list[float] = []
        for i in range(len(samples)):
            train_p = probs_in[:i] + probs_in[i + 1:]
            train_y = outs[:i] + outs[i + 1:]
            cal = cls().fit(train_p, train_y)  # type: ignore[attr-defined]
            cal_probs.append(cal.calibrate([probs_in[i]])[0])  # type: ignore[attr-defined]
        out[name] = {
            "calibrated": cal_probs,
            "ece": ece(cal_probs, outs),
            "brier": brier(cal_probs, outs),
        }
    return out


@dataclass
class CalibrationReport:
    n_samples: int
    raw_ece: float
    raw_brier: float
    reliability: list[dict] = field(default_factory=list)
    loo: dict[str, dict] = field(default_factory=dict)
    samples: list[CalibSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "raw": {
                "ece": self.raw_ece,
                "brier": self.raw_brier,
                "reliability": self.reliability,
            },
            "loo": {
                name: {
                    "ece": v["ece"],
                    "brier": v["brier"],
                }
                for name, v in self.loo.items()
            },
            "samples": [
                {
                    "match_id": s.match_id,
                    "pick": s.pick,
                    "actual": s.actual,
                    "confidence": s.confidence,
                    "outcome": s.outcome,
                    "p_right": s.p_right,
                }
                for s in self.samples
            ],
        }

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(f"Calibration report: n={self.n_samples}")
        lines.append(
            f"  raw:        ECE={self.raw_ece:.4f}  "
            f"Brier={self.raw_brier:.4f}"
        )
        for name, v in self.loo.items():
            lines.append(
                f"  LOO-{name:11s}: ECE={v['ece']:.4f}  "
                f"Brier={v['brier']:.4f}"
            )
        # Reliability bucket summary
        lines.append("  Reliability (raw, n_bins=10):")
        for b in self.reliability:
            if b["n"] == 0:
                continue
            lines.append(
                f"    [{b['lo']:.1f}-{b['hi']:.1f}]  "
                f"n={b['n']:2d}  avg_p={b['avg_p']:.3f}  "
                f"avg_y={b['avg_y']:.3f}  gap={b['gap']:+.3f}"
            )
        return "\n".join(lines)


def fit_per_competition_calibrators(
    samples: list[CalibSample],
    *,
    key: str,
    root: Path,
    min_n: int = 20,
    match_to_competition,
) -> dict[str, Path]:
    """Task 35: fit one calibrator per competition and save to disk.

    For each competition with at least ``min_n`` samples, fits a
    calibrator on that subset only and writes it to
    ``<root>/<key>_<COMP>.json`` (e.g. ``isotonic_EPL.json``).
    Competitions with fewer than ``min_n`` samples are skipped —
    the agent's per-comp → global fallback will use the global
    ``isotonic.json`` for them instead.

    The ``match_to_competition`` resolver maps a CalibSample's
    ``match_id`` to its competition name (or None if unknown).
    CalibSamples whose competition can't be resolved are silently
    dropped from the partition.

    Returns: {competition_name: written_path} for the calibrators
    that were actually fit.
    """
    from collections import defaultdict
    from soccer_agent.calibration_store import _CALIBRATORS, save_calibrator
    from soccer_agent.calibration import ece, brier

    by_comp: dict[str, list[CalibSample]] = defaultdict(list)
    for s in samples:
        comp = match_to_competition(s.match_id)
        if comp is None:
            continue
        by_comp[comp].append(s)

    cls = _CALIBRATORS[key]
    written: dict[str, Path] = {}
    for comp, group in sorted(by_comp.items()):
        if len(group) < min_n:
            # Skip — the global fallback covers this competition.
            continue
        probs = [s.p_right for s in group]
        outs = [s.outcome for s in group]
        cal = cls().fit(probs, outs)  # type: ignore[attr-defined]
        cal_probs = cal.calibrate(probs)
        path = save_calibrator(
            cal, key=f"{key}_{comp}", root=root,
            competition=comp, n_samples=len(group),
            ece=ece(cal_probs, outs), brier=brier(cal_probs, outs),
        )
        written[comp] = path
    return written


def run_calibration_report(db_path: Path) -> CalibrationReport:
    init_db(db_path)
    samples = collect_samples(db_path)
    if not samples:
        return CalibrationReport(
            n_samples=0, raw_ece=0.0, raw_brier=0.0,
        )
    probs = [s.p_right for s in samples]
    outs = [s.outcome for s in samples]
    raw_ece = ece(probs, outs)
    raw_brier = brier(probs, outs)
    table = reliability_table(probs, outs, n_bins=10)
    loo = loo_eval(samples)
    return CalibrationReport(
        n_samples=len(samples),
        raw_ece=raw_ece,
        raw_brier=raw_brier,
        reliability=table,
        loo=loo,
        samples=samples,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Measure the agent's calibration on the eval set.",
    )
    p.add_argument(
        "--db", type=Path, required=True,
        help="Path to the predictions database (e.g. data/soccer_agent.db).",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Optional: write the full report (incl. samples) to JSON.",
    )
    p.add_argument(
        "--fixtures", type=Path, default=None,
        help="Optional: a fixtures dir to materialize into before running. "
        "If given AND the harness hasn't been run yet, this script will "
        "run `EvalHarness` with the given `--noise`/`--seed` first.",
    )
    p.add_argument(
        "--noise", type=float, default=0.0,
        help="Fixture noise level (0..1) — see EvalHarness. Default 0.0.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Fixture seed (int) for reproducibility. Default: not pinned.",
    )
    p.add_argument(
        "--save-calibrator", type=Path, default=None,
        help="If given, fit a calibrator on the report's samples and "
        "write it to <path>/<key>.json. The key defaults to 'isotonic' "
        "(the LOO winner at n=34). Combine with --calibrator-key.",
    )
    p.add_argument(
        "--calibrator-key", type=str, default="isotonic",
        choices=["isotonic", "platt", "temperature", "binning"],
        help="Which calibrator to fit/save. Default: isotonic (LOO winner).",
    )
    p.add_argument(
        "--per-competition", action="store_true",
        help="Task 35: in addition to the global calibrator, fit one "
        "<key>_<COMP>.json per competition with at least --min-n samples. "
        "Competitions below the threshold use the global fallback.",
    )
    p.add_argument(
        "--min-n", type=int, default=20,
        help="Minimum samples per competition to fit a per-comp "
        "calibrator. Default 20 (anything smaller is too few to beat "
        "the global calibrator).",
    )
    args = p.parse_args()
    if not args.db.exists() and args.fixtures is None:
        print(
            f"db not found: {args.db}\n"
            "Pass --fixtures <dir> to materialize + run the eval first.",
            file=sys.stderr,
        )
        return 2
    if not args.db.exists() and args.fixtures is not None:
        # Materialize + run the eval harness first.
        from soccer_agent.eval.harness import EvalHarness
        harness = EvalHarness(
            fixtures_dir=args.fixtures,
            db_path=args.db,
            fixture_noise=args.noise,
            fixture_seed=args.seed,
        )
        print(
            f"Running EvalHarness on {len(EVAL_CASES)} cases "
            f"(noise={args.noise}, seed={args.seed})..."
        )
        harness.run()
    report = run_calibration_report(args.db)
    print(report.summary())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nWrote {args.out}")
    # Task 31: optionally save a fitted calibrator for the live
    # predict() path. The calibrator is fit on the *full* sample
    # (not LOO) — for production use we want the most data
    # possible. The LOO report above is the honest CV evaluation.
    if args.save_calibrator is not None:
        from soccer_agent.calibration_store import _CALIBRATORS, save_calibrator
        cls = _CALIBRATORS[args.calibrator_key]
        full = cls().fit(  # type: ignore[attr-defined]
            [s.p_right for s in report.samples],
            [s.outcome for s in report.samples],
        )
        # Pull the LOO metrics for metadata, when available.
        loo_ece = report.loo.get(args.calibrator_key, {}).get("ece", float("nan"))
        loo_brier = report.loo.get(args.calibrator_key, {}).get("brier", float("nan"))
        # The competition label for the metadata block. We don't
        # partition by competition yet (Task 35), so it's "ALL".
        path = save_calibrator(
            full, key=args.calibrator_key, root=args.save_calibrator,
            competition="ALL", n_samples=len(report.samples),
            ece=loo_ece, brier=loo_brier,
        )
        print(
            f"\nSaved {args.calibrator_key} calibrator "
            f"(n={len(report.samples)}, LOO ECE={loo_ece:.4f}) to {path}"
        )

    # Task 35: optionally fit per-competition calibrators.
    if args.per_competition and args.save_calibrator is not None:
        from soccer_agent.eval.dataset import EVAL_CASES
        case_by_id = {c.match_id: c for c in EVAL_CASES}
        def resolver(match_id: str) -> str | None:
            case = case_by_id.get(match_id)
            return case.competition if case else None
        written = fit_per_competition_calibrators(
            report.samples, key=args.calibrator_key,
            root=args.save_calibrator, min_n=args.min_n,
            match_to_competition=resolver,
        )
        if written:
            print(
                f"\nFitted {len(written)} per-competition calibrator(s) "
                f"(min_n={args.min_n}):"
            )
            for comp, p in sorted(written.items()):
                print(f"  {comp:12s} -> {p.name}")
        else:
            print(
                f"\nNo competition had >= {args.min_n} samples; "
                f"per-competition fitting skipped."
            )
    # Return 0 even with bad ECE — the user runs this as a *report*,
    # not a pass/fail gate. They decide what to do with the numbers.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
