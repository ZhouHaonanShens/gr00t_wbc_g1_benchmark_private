from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from work.recap.r6_runtime_indicator_probe import cli
from work.recap.r6_runtime_indicator_probe.runtime_probe import ProbeBudget, run_runtime_probe
from work.recap.r6_runtime_indicator_probe import runtime_probe_worker

TOKEN = "e" * 64


def test_cli_default_none_preserves_runtime_call(tmp_path: Path, monkeypatch) -> None:
    seen: list[tuple[object, ...]] = []
    def fake_run(cell, budget, token, *, forced=False, counterfactual=True, lora_adapter_dir=None):
        seen.append((cell, budget.gpu_id, token, forced, counterfactual, lora_adapter_dir))
        from work.recap.r6_runtime_indicator_probe.contract import ProbeCounterfactual, RuntimeTrace
        return RuntimeTrace("A.2", 20000, "Advantage", "p" * 64, "a" * 64, (1, 2, 3, 4, 5), True, "INDICATOR_PRESENT"), ProbeCounterfactual("A.2", 20000, "a" * 64, "b" * 64, False, (0, 0, 0.01, 0, 0), "INDICATOR_SENSITIVE")
    monkeypatch.setattr("work.recap.r6_runtime_indicator_probe.runtime_probe.run_runtime_probe", fake_run)
    assert cli.main(["probe", "--forced", "--cell", "A.2", "--leader-approval-token", TOKEN, "--gpu", "1", "--output-root", str(tmp_path)]) == 0
    assert seen == [("A.2", 1, TOKEN, True, True, None)]


def test_cli_passes_adapter_dir_when_provided(tmp_path: Path, monkeypatch) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    seen = []
    def fake_run(cell, budget, token, *, forced=False, counterfactual=True, lora_adapter_dir=None):
        seen.append(lora_adapter_dir)
        from work.recap.r6_runtime_indicator_probe.contract import ProbeCounterfactual, RuntimeTrace
        return RuntimeTrace("A.2", 20000, "Advantage", "p" * 64, "a" * 64, (1, 2, 3, 4, 5), True, "INDICATOR_PRESENT"), ProbeCounterfactual("A.2", 20000, "a" * 64, "b" * 64, False, (0, 0, 0.01, 0, 0), "INDICATOR_SENSITIVE")
    monkeypatch.setattr("work.recap.r6_runtime_indicator_probe.runtime_probe.run_runtime_probe", fake_run)
    assert cli.main(["probe", "--forced", "--cell", "A.2", "--leader-approval-token", TOKEN, "--gpu", "1", "--lora-adapter-dir", str(adapter), "--output-root", str(tmp_path / "out")]) == 0
    assert seen == [str(adapter)]


def test_runtime_probe_adds_adapter_arg_only_when_present(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    def fake_run(command: list[str], **_: Any) -> Any:
        calls.append(command)
        mode = command[command.index("--force-indicator-mode") + 1]
        payload = {"episode_seed": 20000, "prompt_text_at_tokenizer": "Advantage", "prompt_tokens_sha256": "1" * 64, "action_head_conditioning_sha256": ("a" if mode == "positive" else "b") * 64, "first_5_actions_l2": [1, 2, 3, 4, 5], "indicator_substring_present": True}
        return type("Completed", (), {"stdout": json.dumps(payload)})()
    monkeypatch.setattr("work.recap.r6_runtime_indicator_probe.runtime_probe.subprocess.run", fake_run)
    run_runtime_probe("A.2", ProbeBudget(gpu_id=1), TOKEN, forced=True, lora_adapter_dir=str(tmp_path))
    assert all("--lora-adapter-dir" in call for call in calls)
    assert all(call[2] == sys.executable for call in calls)


def test_worker_parser_accepts_lora_adapter_dir(tmp_path: Path) -> None:
    args = runtime_probe_worker.build_parser().parse_args(["--cell", "A.2", "--max-steps", "1", "--seed", "1", "--force-indicator-mode", "positive", "--lora-adapter-dir", str(tmp_path)])
    assert args.lora_adapter_dir == str(tmp_path)


def test_worker_apply_lora_adapter_noops_when_none() -> None:
    policy = object()
    assert runtime_probe_worker._apply_lora_adapter(policy, None) is policy
