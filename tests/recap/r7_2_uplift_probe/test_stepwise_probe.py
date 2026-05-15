from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from work.recap.r7_2_uplift_probe.contract import R7UpliftError
from work.recap.r7_2_uplift_probe.stepwise_probe import build_probe_command, parse_counterfactual_json, probe_adapter

TOKEN = "c" * 64


def test_build_command_contains_forced_and_adapter(tmp_path: Path) -> None:
    command = build_probe_command(tmp_path / "adapter_step_0200", TOKEN, 1, tmp_path / "probe")
    assert "--forced" in command
    assert command[command.index("--lora-adapter-dir") + 1].endswith("adapter_step_0200")
    assert command[command.index("--gpu") + 1] == "1"


def test_parse_counterfactual_json(tmp_path: Path) -> None:
    path = tmp_path / "counterfactual.json"
    path.write_text(json.dumps({"counterfactual_verdict": "INDICATOR_SENSITIVE", "condition_sha_equal": False, "first_5_actions_l2_diff": [0, 0.2, 0]}))
    item = parse_counterfactual_json(path, adapter_dir=tmp_path / "adapter_step_0400")
    assert item.step == 400
    assert item.first_5_actions_l2_diff_max == 0.2


@pytest.mark.parametrize("payload", [
    {"condition_sha_equal": True, "first_5_actions_l2_diff": [0]},
    {"counterfactual_verdict": "INDICATOR_INVARIANT", "first_5_actions_l2_diff": [0]},
    {"counterfactual_verdict": "INDICATOR_INVARIANT", "condition_sha_equal": True},
])
def test_parse_missing_required_fields_fails(tmp_path: Path, payload: dict[str, Any]) -> None:
    path = tmp_path / "counterfactual.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(R7UpliftError):
        parse_counterfactual_json(path, adapter_dir=tmp_path / "adapter_step_0200")


def test_parse_invalid_verdict_fails(tmp_path: Path) -> None:
    path = tmp_path / "counterfactual.json"
    path.write_text(json.dumps({"counterfactual_verdict": "OTHER", "condition_sha_equal": True, "first_5_actions_l2_diff": [0]}))
    with pytest.raises(R7UpliftError):
        parse_counterfactual_json(path, adapter_dir=tmp_path / "adapter_step_0200")


def test_probe_adapter_copies_counterfactual_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = tmp_path / "adapter_step_0200"
    adapter.mkdir()
    probe_root = tmp_path / "probe_step_0200"
    cf_dir = probe_root / "A.2"
    cf_dir.mkdir(parents=True)
    (cf_dir / "counterfactual.json").write_text(json.dumps({"counterfactual_verdict": "INDICATOR_INVARIANT", "condition_sha_equal": True, "first_5_actions_l2_diff": [0, 0, 0]}))

    def fake_run(*_args: Any, **_kwargs: Any) -> Any:
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("work.recap.r7_2_uplift_probe.stepwise_probe.subprocess.run", fake_run)
    item = probe_adapter(adapter, tmp_path / "ckpt", leader_approval_token=TOKEN, gpu_id=1, output_dir=probe_root, seed=1)
    assert item.counterfactual_verdict == "INDICATOR_INVARIANT"
    assert (tmp_path / "counterfactual_step_0200.json").is_file()
