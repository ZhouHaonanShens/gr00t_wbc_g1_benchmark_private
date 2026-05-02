from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.stage_b.execution_boundaries import (  # noqa: E402
    OBJECTIVE_PRIORITY,
    STAGE_B_FORBIDDEN_OPERATIONS,
    build_no_training_contract_markdown,
    classify_stage_b_command,
    main,
    require_stage_b_safe_command,
)


def test_indicator_survival_is_secondary_to_checkpoint_regression() -> None:
    assert (
        OBJECTIVE_PRIORITY[0]
        == "checkpoint_regression_or_policy_collapse_layer_diagnostics"
    )
    assert OBJECTIVE_PRIORITY[1] == "indicator_survival_secondary_axis"

    contract = build_no_training_contract_markdown()

    assert "第二诊断轴" in contract
    assert "不得把它后验改写成 Stage B 主 success metric" in contract


@pytest.mark.parametrize(
    "command",
    [
        (
            "python -m work.recap.stage_b.seam_trace_writer "
            "--self-test --output-dir /tmp/out"
        ),
        [
            "timeout",
            "60",
            "python3",
            "agent/run/stage_b_probe.py",
            "--max-steps",
            "200",
            "--short-diagnostic",
        ],
        "python3 agent/run/stage_b_probe.py --same-observation --max-steps=1",
    ],
)
def test_diagnostic_commands_are_allowed(command: str | list[str]) -> None:
    decision = classify_stage_b_command(command)

    assert decision.allowed, decision.reasons
    assert not any(reason.startswith("forbidden:") for reason in decision.reasons)


@pytest.mark.parametrize(
    "command, expected_reason_fragment",
    [
        ("python work/recap/finetune_full.py --epochs 1", "new_method_training"),
        ("torchrun work/recap/scripts/new_method_train.py", "new_method_training"),
        ("python agent/run/gr00t_formal_eval.py --episodes 30", "gr00t_full_long_run"),
        ("python agent/run/stage_b_probe.py --max-steps 201", "--max-steps>200"),
        ("python agent/run/stage_b_probe.py --episodes=4", "--episodes>3"),
        ("python train.py --lora", "lora"),
        ("python train.py --sft", "sft"),
    ],
)
def test_forbidden_training_and_long_run_commands_are_blocked(
    command: str,
    expected_reason_fragment: str,
) -> None:
    decision = classify_stage_b_command(command)

    assert not decision.allowed
    assert any(expected_reason_fragment in reason for reason in decision.reasons)


def test_require_stage_b_safe_command_raises_on_boundary_violation() -> None:
    with pytest.raises(ValueError, match="Stage B boundary violation"):
        require_stage_b_safe_command("python work/recap/finetune_full.py")


def test_contract_writer_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "STAGE_B_NO_TRAINING_CONTRACT.md"

    exit_code = main(["--write-contract", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    written = output_path.read_text(encoding="utf-8")
    assert "不启动 GR00T full long-run" in written
    assert "new_method_training" in written
    assert "lora" in STAGE_B_FORBIDDEN_OPERATIONS
