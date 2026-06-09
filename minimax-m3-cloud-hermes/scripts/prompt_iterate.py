#!/usr/bin/env python
"""scripts/prompt_iterate.py — run a prompt sweep against live ollama.

Sister script to `scripts/e2e_smoke.sh` and the eval harness. Iterates
N prompt candidates over the pinned eval dataset, scores each, and
emits a JSON summary + a markdown leaderboard. The sweep itself is
fast (it just iterates fixtures and calls the LLM); the cost is
entirely in the LLM calls — so we recommend running on a host with a
GPU-backed ollama (Apple Silicon is fine; a 1.5B model does this in
~1-2 minutes total per cycle).

Usage:
    # Default: load candidates from docs/prompts/<name>.md, ollama as backend.
    python scripts/prompt_iterate.py

    # Custom candidate directory:
    python scripts/prompt_iterate.py --candidates-dir docs/prompts

    # Stub LLM (fast, deterministic; useful for CI):
    SOCCER_AGENT_LLM_PROVIDER=stub python scripts/prompt_iterate.py

    # Choose model:
    SOCCER_AGENT_LLM_MODEL=qwen2.5:1.5b python scripts/prompt_iterate.py

Outputs:
    - docs/sweep_results/<timestamp>/per_candidate.json   structured scores
    - docs/sweep_results/<timestamp>/leaderboard.md      human-readable
    - sweep__<candidate>.db                              per-candidate DBs

Exit code 0 if a sweep runs to completion. The script does not gate
on "best candidate is materially better than baseline" — that's a
human review step.
"""

from __future__ import annotations

import argparse
import json as jsonlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure src/ is on the path so this works from the repo root
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from soccer_agent.eval.dataset import EVAL_CASES
from soccer_agent.eval.prompt_sweep import (
    PromptCandidate,
    SweepResult,
    run_prompt_sweep,
)
from soccer_agent.llm import get_client


def _load_candidates(candidates_dir: Path) -> list[PromptCandidate]:
    """Read each `<name>.md` file in `candidates_dir` as a candidate.

    The file's basename (sans `.md`) is the candidate's name; the
    contents are the system prompt.
    """
    if not candidates_dir.exists():
        raise SystemExit(
            f"candidates dir not found: {candidates_dir}. "
            f"Create at least one <name>.md file there."
        )
    files = sorted(candidates_dir.glob("*.md"))
    if not files:
        raise SystemExit(
            f"no .md candidates found in {candidates_dir}"
        )
    return [
        PromptCandidate(name=p.stem, system_prompt=p.read_text())
        for p in files
    ]


def _write_leaderboard_md(result: SweepResult, out: Path) -> None:
    """Write a markdown table ranking the candidates by accuracy."""
    rows = sorted(
        result.per_candidate.items(),
        key=lambda kv: -kv[1].get("accuracy", 0.0),
    )
    lines = [
        "# Prompt Sweep Leaderboard",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "| Rank | Candidate | Accuracy | Brier | n |",
        "|------|-----------|----------|-------|---|",
    ]
    for i, (name, m) in enumerate(rows, 1):
        # `metric_summary` returns both `n_total` and `n`. The leaderboard
        # prefers the resolved count (cases that were scored).
        n = m.get("n_total", m.get("n", 0))
        lines.append(
            f"| {i} | {name} | {m.get('accuracy', 0):.3f} | "
            f"{m.get('brier_mean', 0):.3f} | {n} |"
        )
    if result.best:
        lines += ["", f"**Best:** `{result.best.name}`"]
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a prompt sweep against the pinned eval dataset."
    )
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=ROOT / "docs" / "prompts",
        help="Directory of <name>.md prompt candidates",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write per_candidate.json + leaderboard.md. "
             "Default: docs/sweep_results/<timestamp>/",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=ROOT / "sweep.db",
        help="Base path; per-candidate DBs go alongside with __<name>.db suffix",
    )
    args = parser.parse_args()

    started = time.time()
    candidates = _load_candidates(args.candidates_dir)
    n_cases = len(EVAL_CASES)
    n_cands = len(candidates)
    print(
        f"[prompt_iterate] {n_cands} candidates x {n_cases} eval cases "
        f"= {n_cands * n_cases} LLM calls expected"
    )

    sweep_started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (ROOT / "docs" / "sweep_results" / sweep_started_at)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The LLM client is shared across candidates (it's just a transport
    # layer). If you want a different model per candidate, pass a
    # per-candidate factory to `run_prompt_sweep` directly.
    client_factory = get_client
    result: SweepResult = run_prompt_sweep(
        candidates=candidates,
        eval_cases=EVAL_CASES,
        client_factory=client_factory,
        db_path=args.db_path,
        output=out_dir / "per_candidate.json",
    )

    _write_leaderboard_md(result, out_dir / "leaderboard.md")
    elapsed = time.time() - started
    print(
        f"[prompt_iterate] done in {elapsed:.1f}s. "
        f"Best: {result.best.name if result.best else '<none>'}."
    )
    print(f"[prompt_iterate] results in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
