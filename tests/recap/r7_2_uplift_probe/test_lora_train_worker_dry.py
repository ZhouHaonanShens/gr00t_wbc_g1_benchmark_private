from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from work.recap.r7_2_uplift_probe.contract import R7AdapterTooLargeError, R7TrainingFailedError
from work.recap.r7_2_uplift_probe import lora_train_worker as worker


class FakeModel:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def named_modules(self) -> Iterator[tuple[str, object]]:
        for name in self.names:
            yield name, object()


def test_dry_step_count_avoids_training_loop(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}")
    assert worker.main(["--request-json", str(request), "--output-root", str(tmp_path / "out"), "--dry-step-count", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[0])["event"] == "step"
    assert json.loads(lines[-1])["reason"] == "dry_step_count"


def test_target_modules_enumerate_top_layers_and_action_head() -> None:
    names = [
        "language_model.model.layers.0.self_attn.q_proj",
        "language_model.model.layers.9.self_attn.k_proj",
        "language_model.model.layers.9.self_attn.v_proj",
        "language_model.model.layers.8.self_attn.o_proj",
        "action_head.projector",
    ]
    targets = worker.enumerate_lora_targets(FakeModel(names), 4)
    assert "language_model.model.layers.9.self_attn.k_proj" in targets
    assert "action_head.projector" in targets


def test_target_modules_fail_when_empty() -> None:
    with pytest.raises(R7TrainingFailedError):
        worker.enumerate_lora_targets(FakeModel(["foo.bar"]), 4)


def test_adapter_size_limit_raises(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.bin").write_bytes(b"0" * 1024)
    assert worker.adapter_size_mb(adapter) > 0
    old = worker.ADAPTER_SIZE_LIMIT_MB
    worker.ADAPTER_SIZE_LIMIT_MB = 0.000001
    try:
        with pytest.raises(R7AdapterTooLargeError):
            worker.save_adapter_at_step(type("M", (), {"save_pretrained": lambda self, path: Path(path, "x.bin").write_bytes(b"0" * 1024)})(), 200, tmp_path)
    finally:
        worker.ADAPTER_SIZE_LIMIT_MB = old


def test_loss_nan_detection() -> None:
    assert worker.loss_is_nan(float("nan")) is True
    assert worker.loss_is_nan(1.0) is False


def test_apply_lora_adapter_noop_for_none() -> None:
    policy = object()
    assert worker.apply_lora_adapter_to_policy(policy, None) is policy
