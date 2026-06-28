from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from soccer.config import AppConfig
from soccer.worldcup.cli import cmd_fetch, cmd_predict, cmd_rank, load_dataset
from soccer.worldcup.entities import WorldCup


def _config(data_dir: Path, key: str | None = None) -> AppConfig:
    return AppConfig(
        data_dir=data_dir,
        ollama_host="",
        ollama_model="",
        ollama_timeout=1.0,
        provider_mode="fixture",
        reasoner="fake",
        api_football_base_url="https://api.test",
        api_football_key=key,
        prediction_dir=data_dir / "prediction",
    )


def _write_dataset(data_dir: Path, wc: WorldCup) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "worldcup-2026.json").write_text(json.dumps(wc.to_dict()), encoding="utf-8")


def test_rank_prints_tables(
    tmp_path: Path, sample_world_cup: WorldCup, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, sample_world_cup)
    rc = cmd_rank(argparse.Namespace(top=5), _config(tmp_path))
    assert rc == 0
    out = capsys.readouterr().out
    assert "National teams" in out
    assert "England" in out


def test_predict_writes_file(
    tmp_path: Path, sample_world_cup: WorldCup, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, sample_world_cup)
    rc = cmd_predict(argparse.Namespace(), _config(tmp_path))
    assert rc == 0
    pred_dir = tmp_path / "prediction"
    written = json.loads((pred_dir / "worldcup-2026-predictions.json").read_text())
    assert len(written) == len(sample_world_cup.matches)
    assert {"score_home", "score_away", "p_home", "kickoff"} <= set(written[0])
    # human-friendly report grouped by group / matchday
    report = (pred_dir / "worldcup-2026-predictions.md").read_text()
    assert "## Group A" in report
    assert "### Matchday 1" in report


def test_predict_remaining_writes_named_files(tmp_path: Path, sample_world_cup: WorldCup) -> None:
    from dataclasses import replace

    from soccer.worldcup.cli import cmd_predict

    # Mark the only match as played so "remaining" has nothing to forecast but a result to show.
    played = replace(sample_world_cup.matches[0], home_goals=2, away_goals=0)
    wc = replace(sample_world_cup, matches=(played,))
    _write_dataset(tmp_path, wc)

    args = argparse.Namespace(
        remaining=True, out_dir=str(tmp_path / "predictions"), name="after1st"
    )
    rc = cmd_predict(args, _config(tmp_path))
    assert rc == 0
    out_dir = tmp_path / "predictions"
    payload = json.loads((out_dir / "after1st.json").read_text())
    assert set(payload) == {"predictions", "results", "adjustments"}
    report = (out_dir / "after1st.md").read_text()
    assert "Completed results" in report


def test_fetch_without_key_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cmd_fetch(argparse.Namespace(throttle=0.0), _config(tmp_path, key=None))
    assert rc == 1
    assert "not set" in capsys.readouterr().out


def test_load_dataset_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.json")


def test_card_writes_json(tmp_path: Path, sample_world_cup: WorldCup) -> None:
    from soccer.worldcup.cli import cmd_card

    _write_dataset(tmp_path, sample_world_cup)
    args = argparse.Namespace(
        fixture_id=9001,
        refresh=False,
        out_dir=None,
        name=None,
        format="json",
        throttle=0.0,
    )
    rc = cmd_card(args, _config(tmp_path))
    assert rc == 0
    data = json.loads((tmp_path / "prediction" / "card-9001.json").read_text())
    assert data["fixture_id"] == 9001
    assert data["home"]["name"] == "England"
    assert "prediction" in data


def test_card_writes_pdf(tmp_path: Path, sample_world_cup: WorldCup) -> None:
    pytest.importorskip("reportlab")
    from soccer.worldcup.cli import cmd_card

    _write_dataset(tmp_path, sample_world_cup)
    args = argparse.Namespace(
        fixture_id=9001, refresh=False, out_dir=None, name=None, format="both", throttle=0.0
    )
    rc = cmd_card(args, _config(tmp_path))
    assert rc == 0
    assert (tmp_path / "prediction" / "card-9001.pdf").read_bytes()[:4] == b"%PDF"


def test_card_unknown_fixture_returns_error(
    tmp_path: Path, sample_world_cup: WorldCup, capsys: pytest.CaptureFixture[str]
) -> None:
    from soccer.worldcup.cli import cmd_card

    _write_dataset(tmp_path, sample_world_cup)
    args = argparse.Namespace(
        fixture_id=4242, refresh=False, out_dir=None, name=None, format="json", throttle=0.0
    )
    rc = cmd_card(args, _config(tmp_path))
    assert rc == 1
    assert "not found" in capsys.readouterr().out


def test_card_refresh_without_key_fails(
    tmp_path: Path, sample_world_cup: WorldCup, capsys: pytest.CaptureFixture[str]
) -> None:
    from soccer.worldcup.cli import cmd_card

    _write_dataset(tmp_path, sample_world_cup)
    args = argparse.Namespace(
        fixture_id=9001, refresh=True, out_dir=None, name=None, format="json", throttle=0.0
    )
    rc = cmd_card(args, _config(tmp_path, key=None))
    assert rc == 1
    assert "not set" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# knockout subcommand tests (Task 8)
# ---------------------------------------------------------------------------

import json as _json  # noqa: E402 (re-import for clarity in appended section)

from soccer.worldcup import cli as wc_cli  # noqa: E402


def test_knockout_errors_without_r32(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # dataset with only a group match -> no R32 -> non-zero exit + hint
    from datetime import UTC, datetime

    from soccer.worldcup.entities import NationalTeam, WcMatch, WorldCup

    wc = WorldCup(
        teams={1: NationalTeam(1, "A", "Group A", "UEFA", False, (), None, 0, 0, 0)},
        matches=(
            WcMatch(
                1,
                1,
                "Group A",
                1,
                1,
                datetime(2026, 6, 11, tzinfo=UTC),
                "v",
                1,
                0,
                "Group Stage - 1",
            ),
        ),
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "worldcup-2026.json").write_text(_json.dumps(wc.to_dict()))
    config = AppConfig(
        data_dir=data_dir,
        ollama_host="",
        ollama_model="",
        ollama_timeout=1.0,
        provider_mode="fixture",
        reasoner="fake",
        api_football_base_url="",
        api_football_key=None,
        prediction_dir=tmp_path / "out",
    )
    args = argparse.Namespace(sims=10, seed=1, out_dir=None, name=None)
    assert wc_cli.cmd_knockout(args, config) == 1
    assert "fetch" in capsys.readouterr().out


def test_knockout_writes_files(tmp_path: Path) -> None:
    # Build a full 32-team dataset with R32 using the modal test's helpers.
    from tests.worldcup.test_simulate_modal import _add_r32, _wc_from_labels

    wc = _add_r32(
        _wc_from_labels(
            [f"{r}{c}" for c in "ABCDEFGHIJKL" for r in (1, 2)] + [f"3{c}" for c in "CDEFGHIJ"]
        )
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "worldcup-2026.json").write_text(_json.dumps(wc.to_dict()))
    out = tmp_path / "out"
    config = AppConfig(
        data_dir=data_dir,
        ollama_host="",
        ollama_model="",
        ollama_timeout=1.0,
        provider_mode="fixture",
        reasoner="fake",
        api_football_base_url="",
        api_football_key=None,
        prediction_dir=out,
    )
    args = argparse.Namespace(sims=50, seed=1, out_dir=None, name=None)
    assert wc_cli.cmd_knockout(args, config) == 0
    assert (out / "worldcup-2026-knockout.json").exists()
    assert (out / "worldcup-2026-knockout.md").exists()
    payload = _json.loads((out / "worldcup-2026-knockout.json").read_text())
    assert "podium" in payload and "title_odds" in payload and "bracket" in payload
