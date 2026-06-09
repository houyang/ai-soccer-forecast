from pathlib import Path

import pytest

from soccer.cli import main


def test_eval_prints_report(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["eval", "--scenario", "wc-2026-final", "--reasoner", "fake"])
    out = capsys.readouterr().out
    assert code == 0
    assert "wc-2026-final" in out
    assert "accuracy" in out.lower()
    assert "edge_vs_market" in out or "edge vs market" in out.lower()


def test_eval_all_scenarios(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["eval", "--scenario", "all", "--reasoner", "fake"])
    out = capsys.readouterr().out
    assert code == 0
    assert "ucl-2025-26" in out and "wc-2026-final" in out


def test_predict_then_report_roundtrip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
    code = main(["predict", "--match", "wc-final", "--reasoner", "fake"])
    assert code == 0
    out = capsys.readouterr().out
    assert "France" in out and "Brazil" in out
    code = main(["report"])
    report_out = capsys.readouterr().out
    assert code == 0
    assert "wc-final" in report_out


def test_predict_unknown_match_errors(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["predict", "--match", "nope", "--reasoner", "fake"])
    assert code == 1
    err = capsys.readouterr().err
    assert "nope" in err


def test_settle_after_predict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
    main(["predict", "--match", "wc-final", "--reasoner", "fake"])
    capsys.readouterr()
    code = main(["settle", "--reasoner", "fake"])
    out = capsys.readouterr().out
    assert code == 0
    assert "1" in out  # one prediction settled
