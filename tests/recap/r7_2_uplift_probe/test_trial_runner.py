from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from work.recap.r7_2_uplift_probe.contract import R7BudgetExceeded, R7UpliftError, StepwiseCounterfactual, TrialRequest, preset_to_recipe_flags
from work.recap.r7_2_uplift_probe import trial_runner

TOKEN = "d" * 64


def _request(tmp_path: Path, **overrides: Any) -> TrialRequest:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    values = dict(
        trial_id="trial-1",
        base_ckpt_abs_path=str(ckpt.resolve()),
        recipe_flags=preset_to_recipe_flags("full_C1_C2_C5"),
        recipe_preset="full_C1_C2_C5",
        output_root=str(tmp_path / "out"),
        leader_approval_token=TOKEN,
        gpu_id=1,
    )
    values.update(overrides)
    return TrialRequest(**values)


def test_validate_rejects_existing_output(tmp_path: Path) -> None:
    request = _request(tmp_path)
    Path(request.output_root).mkdir()
    with pytest.raises(R7UpliftError):
        trial_runner._validate_request(request)


def test_validate_rejects_missing_ckpt(tmp_path: Path) -> None:
    request = _request(tmp_path, base_ckpt_abs_path=str((tmp_path / "missing").resolve()))
    with pytest.raises(FileNotFoundError):
        trial_runner._validate_request(request)


def test_validate_rejects_trial_1_gpu_2(tmp_path: Path) -> None:
    with pytest.raises(R7UpliftError):
        trial_runner._validate_request(_request(tmp_path, gpu_id=2))


def test_budget_enforcement_raises(tmp_path: Path) -> None:
    with pytest.raises(R7BudgetExceeded):
        trial_runner._enforce_budget(241 * 60, _request(tmp_path))


def test_parse_event_rejects_non_json() -> None:
    with pytest.raises(Exception):
        trial_runner._parse_event("not-json")


def test_signal_training_to_stop_terminates_fake_process() -> None:
    class FakeProc:
        def __init__(self) -> None:
            self.signals: list[int] = []
        def poll(self) -> None:
            return None
        def send_signal(self, value: int) -> None:
            self.signals.append(value)
        def wait(self, timeout: int) -> int:
            return 0
    proc = FakeProc()
    trial_runner._signal_training_to_stop(proc)  # type: ignore[arg-type]
    assert proc.signals


def test_probe_checkpoint_uses_stepwise_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = tmp_path / "adapter_step_0200"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    request = _request(tmp_path)
    seen = {}
    def fake_probe(adapter_dir: Path, _base: Path, **kwargs: Any) -> StepwiseCounterfactual:
        seen["adapter"] = adapter_dir
        seen.update(kwargs)
        return StepwiseCounterfactual(200, str(adapter_dir), "cf", "INDICATOR_SENSITIVE", False, 0.1)
    monkeypatch.setattr(trial_runner, "probe_adapter", fake_probe)
    item = trial_runner._probe_checkpoint({"step": 200, "path": str(adapter)}, request, tmp_path / "out")
    assert item.counterfactual_verdict == "INDICATOR_SENSITIVE"
    assert seen["gpu_id"] == 1


def test_write_report_serializes_dataclasses(tmp_path: Path) -> None:
    request = _request(tmp_path)
    report = trial_runner.TrialReport(request, "TRAINING_FAILED", 0, (), 1.0, None)
    path = tmp_path / "report.json"
    trial_runner._write_report(path, report)
    assert json.loads(path.read_text())["final_verdict"] == "TRAINING_FAILED"


def test_run_trial_training_failed_with_mock_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path)
    class FakeStdout:
        def __iter__(self):
            return iter(['{"event":"done","reason":"entrypoint_unresolved"}\n'])
    class FakeProc:
        stdout = FakeStdout()
        def wait(self, timeout: int) -> int:
            return 4
        def poll(self) -> int:
            return 4
    monkeypatch.setattr(trial_runner, "_spawn_training_subprocess", lambda *args: FakeProc())
    report = trial_runner.run_trial(request)
    assert report.final_verdict == "TRAINING_FAILED"
    assert (Path(request.output_root) / "trial_report.json").is_file()
