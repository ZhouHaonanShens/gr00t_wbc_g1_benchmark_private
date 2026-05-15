from __future__ import annotations

import json
import sys
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
    with pytest.raises(R6Error):
        run_runtime_probe("A.3", ProbeBudget(gpu_id=1), "a" * 64, forced=True)
    with pytest.raises(R6Error):
        run_runtime_probe("A.2", ProbeBudget(gpu_id=2), "a" * 64, forced=True)


def test_runtime_probe_builds_env_commands_for_positive_and_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> Any:
        calls.append(command)
        mode = command[command.index("--force-indicator-mode") + 1]
        payload = {
            "episode_seed": 20000,
            "prompt_text_at_tokenizer": f"pick apple\nAdvantage: {mode}",
            "prompt_tokens_sha256": ("1" if mode == "positive" else "2") * 64,
            "action_head_conditioning_sha256": ("a" if mode == "positive" else "b") * 64,
            "first_5_actions_l2": [1, 2, 3, 4, 5] if mode == "positive" else [1, 2, 3.01, 4, 5],
            "indicator_substring_present": mode == "positive",
        }
        return type("Completed", (), {"stdout": json.dumps(payload)})()

    monkeypatch.setattr("work.recap.r6_runtime_indicator_probe.runtime_probe.subprocess.run", fake_run)
    trace, cf = run_runtime_probe("A.2", ProbeBudget(gpu_id=1), "b" * 64, forced=True)
    assert [call[call.index("--force-indicator-mode") + 1] for call in calls] == ["positive", "negative"]
    assert all(call[:2] == ["env", "CUDA_VISIBLE_DEVICES=1"] for call in calls)
    assert all(call[2] == sys.executable for call in calls)
    assert all("python3" not in call for call in calls)
    assert trace.cell_id == "A.2"
    assert trace.runtime_verdict == "INDICATOR_PRESENT"
    assert trace.first_5_actions_l2 == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert cf is not None
    assert cf.counterfactual_verdict == "INDICATOR_SENSITIVE"


def test_hash_tensor_is_deterministic_without_importing_torch() -> None:
    assert _hash_tensor([1, 2, 3]) == _hash_tensor([1, 2, 3])
