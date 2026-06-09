"""One-shot eval driver: run the harness against the 106-case
dataset and print a per-competition summary."""
from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ["SOCCER_AGENT_LLM_PROVIDER"] = "stub"

FX = ROOT / "tmp" / "eval_100cases" / "fixtures"
DB = ROOT / "tmp" / "eval_100cases" / "predictions.db"
OUT = ROOT / "tmp" / "eval_100cases" / "eval.json"

if FX.exists():
    shutil.rmtree(FX)
FX.mkdir(parents=True)
if DB.exists():
    DB.unlink()

from soccer_agent.eval.harness import run_eval  # noqa: E402

result = run_eval(FX, DB, OUT, reasoner="numeric")
print(f"wrote {OUT}, {result.get('n_total')} predictions")

# Per-prediction rows live in the DB; competition comes from the
# case fixture (we join on match_id via the in-memory dataset).
from soccer_agent.db import Database  # noqa: E402
from soccer_agent.eval.dataset import EVAL_CASES  # noqa: E402
case_by_id = {c.match_id: c for c in EVAL_CASES}
with Database(DB) as db:
    db_preds = list(db.execute("SELECT p.*, r.home_goals, r.away_goals "
                               "FROM predictions p "
                               "LEFT JOIN results r ON p.match_id = r.match_id"))
preds = []
for row in db_preds:
    rec = dict(row)
    case = case_by_id.get(rec["match_id"])
    if case is not None:
        rec["competition"] = case.competition
        rec["actual_winner"] = case.actual_winner
    preds.append(rec)

by_comp = defaultdict(list)
for p in preds:
    by_comp[p.get("competition", "?")].append(p)

def brier_of(p, actual):
    """Brier = sum_i (p_i - y_i)^2 over {home, draw, away}.
    final_probs is a JSON string in the DB, e.g. '{"home":0.68,...}'."""
    raw = p.get("final_probs") or "{}"
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode()
    if isinstance(raw, str):
        try:
            probs = json.loads(raw)
        except Exception:
            probs = {}
    else:
        probs = raw
    ph = float(probs.get("home", 0))
    pd = float(probs.get("draw", 0))
    pa = float(probs.get("away", 0))
    yh = 1.0 if actual == "home" else 0.0
    yd = 1.0 if actual == "draw" else 0.0
    ya = 1.0 if actual == "away" else 0.0
    return (ph - yh) ** 2 + (pd - yd) ** 2 + (pa - ya) ** 2

print(f"\n{'Comp':12s}  {'n':>4s}  {'acc':>6s}  {'brier':>6s}  {'log_loss':>9s}  {'correct':>7s}")
print("-" * 56)
all_n = all_correct = 0
weighted_brier = weighted_ll = 0.0
for comp in sorted(by_comp):
    pl = by_comp[comp]
    n = len(pl)
    correct = sum(1 for p in pl if p.get("actual_winner") == p.get("final_pick"))
    brier = sum(brier_of(p, p.get("actual_winner")) for p in pl) / max(n, 1)
    logloss = sum(p.get("log_loss", 0) or 0 for p in pl) / max(n, 1)
    all_n += n
    all_correct += correct
    weighted_brier += brier * n
    weighted_ll += logloss * n
    print(f"{comp:12s}  {n:4d}  {correct / n:6.1%}  {brier:6.3f}  {logloss:9.3f}  {correct:7d}")
print("-" * 56)
print(f"{'TOTAL':12s}  {all_n:4d}  {all_correct / all_n:6.1%}  {weighted_brier / all_n:6.3f}  {weighted_ll / all_n:9.3f}  {all_correct:7d}")
