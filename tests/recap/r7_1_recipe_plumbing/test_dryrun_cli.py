from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from work.recap.r7_1_recipe_plumbing import cli, dryrun
from work.recap.r7_1_recipe_plumbing.dryrun import DryrunRequest, run_dryrun
from work.recap.r7_1_recipe_plumbing.flags import R7BudgetExceeded, R7PlumbingError, RecipeFlags


def _ckpt(tmp_path: Path) -> Path:
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir(exist_ok=True)
    return ckpt.resolve()


def _fake_run(cmd, cwd, env, stdout, stderr, text, timeout, check):  # type: ignore[no-untyped-def]
    output_root = Path(cmd[cmd.index("--output-root") + 1])
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {"loss_finite": True, "loss_value": 1.135, "max_steps": 1}
    (output_root / "dryrun_child_payload.json").write_text(json.dumps(payload), encoding="utf-8")
    stdout.write(json.dumps(payload) + "\n")
    return SimpleNamespace(returncode=0)


def test_cli_rejects_missing_leader_token(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["dryrun", "--ckpt", str(_ckpt(tmp_path)), "--output-root", str(tmp_path), "--gpu", "1"])


def test_gpu_zero_and_two_are_rejected(tmp_path: Path) -> None:
    request_zero = DryrunRequest(str(_ckpt(tmp_path)), RecipeFlags.default(), str(tmp_path / "out0"), 0, "a1")
    request_two = DryrunRequest(str(_ckpt(tmp_path)), RecipeFlags.default(), str(tmp_path / "out2"), 2, "a1")
    with pytest.raises(R7PlumbingError):
        run_dryrun(request_zero)
    with pytest.raises(R7PlumbingError):
        run_dryrun(request_two)


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    request = DryrunRequest(str(tmp_path / "missing"), RecipeFlags.default(), str(tmp_path / "out"), 1, "a1")
    with pytest.raises(FileNotFoundError):
        run_dryrun(request)


def test_budget_above_two_minutes_is_rejected(tmp_path: Path) -> None:
    request = DryrunRequest(
        str(_ckpt(tmp_path)),
        RecipeFlags.default(),
        str(tmp_path / "out"),
        1,
        "a1",
        budget_minutes=3,
    )
    with pytest.raises(R7BudgetExceeded):
        run_dryrun(request)


def test_success_path_writes_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dryrun.subprocess, "run", _fake_run)
    flags = RecipeFlags(
        enable_dual_loss=True,
        dual_loss_alpha=0.5,
        indicator_dropout_p=0.15,
        dual_loss_uses_carrier_text=True,
        carrier_text_field="carrier_text_v1",
    )
    report = run_dryrun(DryrunRequest(str(_ckpt(tmp_path)), flags, str(tmp_path / "out"), 1, "abcdef"))
    assert report.loss_finite is True
    assert (tmp_path / "out" / "dryrun_report.json").is_file()


def test_child_smoke_produces_finite_payload(tmp_path: Path) -> None:
    flags = RecipeFlags(enable_dual_loss=True, dual_loss_alpha=0.5, indicator_dropout_p=0.0)
    rc = dryrun.run_child_smoke(str(_ckpt(tmp_path)), str(tmp_path / "child"), flags)
    payload = json.loads((tmp_path / "child" / "dryrun_child_payload.json").read_text())
    assert rc == 0
    assert payload["loss_finite"] is True


def test_cli_help_contains_recipe_group(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["dryrun", "--help"])
    captured = capsys.readouterr()
    assert "r7.1_recipe_plumbing" in captured.out
