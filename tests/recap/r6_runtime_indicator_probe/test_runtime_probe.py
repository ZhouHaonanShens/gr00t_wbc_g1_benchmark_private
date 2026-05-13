from __future__ import annotations

import json
from typing import Any

import pytest

from work.recap.r6_runtime_indicator_probe.contract import R6BudgetExceeded, R6Error
from work.recap.r6_runtime_indicator_probe.runtime_probe import ProbeBudget, _hash_tensor, run_runtime_probe


def test_runtime_probe_validates_token_gpu_and_budget() -> None:
    with pytest.raises(R6Error):
        run_runtime_probe("A.2", ProbeBudget(), "bad")
    with pytest.raises(R6Error):
        run_runtime_probe("A.2", ProbeBudget(gpu_id=0), "a" * 64)
    with pytest.raises(R6BudgetExceeded):
        run_runtime_probe("A.2", ProbeBudget(max_steps_per_episode=201), "a" * 64)


def test_runtime_probe_builds_env_command_and_runtime_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> Any:
        calls.append(command)
        payload = {"episode_seed": 7, "prompt_text_at_tokenizer": "pick apple\nAdvantage: positive", "first_5_actions_l2": [1, 2, 3, 4, 5], "indicator_substring_present": True}
        return type("Completed", (), {"stdout": json.dumps(payload)})()

    monkeypatch.setattr("work.recap.r6_runtime_indicator_probe.runtime_probe.subprocess.run", fake_run)
    trace = run_runtime_probe("A.3", ProbeBudget(gpu_id=2), "b" * 64)
    assert calls[0][:2] == ["env", "CUDA_VISIBLE_DEVICES=2"]
    assert trace.cell_id == "A.3"
    assert trace.runtime_verdict == "INDICATOR_PRESENT"
    assert trace.first_5_actions_l2 == (1.0, 2.0, 3.0, 4.0, 5.0)


def test_hash_tensor_is_deterministic_without_importing_torch() -> None:
    assert _hash_tensor([1, 2, 3]) == _hash_tensor([1, 2, 3])
