from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.stage_b.diagnostic_probe_plan import (  # noqa: E402
    MAX_STAGE_B_DIAGNOSTIC_STEPS,
    build_probe_support_plan,
    main,
)


def test_probe_support_plan_keeps_b1_b2_diagnostic_only() -> None:
    plan = build_probe_support_plan()

    assert plan["diagnostic_only"] is True
    assert plan["formal_benchmark"] is False
    assert plan["method_claim_allowed"] is False
    assert plan["full_long_run_allowed"] is False
    assert plan["training_allowed"] is False

    by_id = {probe["probe_id"]: probe for probe in plan["probes"]}
    b1 = by_id["B1_same_obs_wbc_one_step_triplet"]
    assert b1["max_steps"] == 1
    assert b1["seeds"] == [20000]
    assert b1["requires_same_observation"] is True
    assert b1["requires_chain_action_uuid"] is True
    assert b1["requires_contrast_group_uuid"] is True
    assert b1["boundary_decision"]["allowed"] is True

    b2 = by_id["B2_short_closed_loop_triplet"]
    assert b2["max_steps"] == MAX_STAGE_B_DIAGNOSTIC_STEPS
    assert b2["seeds"] == [20000, 20001, 20002]
    assert b2["default_gpu"] == 1
    assert b2["official_success_flag_role"] == "diagnostic_context_only"
    assert b2["boundary_decision"]["allowed"] is True


def test_probe_support_plan_representative_commands_stay_within_limits() -> None:
    plan = build_probe_support_plan()

    for probe in plan["probes"]:
        command = " ".join(probe["representative_command"])
        assert "timeout " in command
        assert "torchrun" not in command
        assert "train" not in command
        assert "formal_eval" not in command
        assert probe["boundary_decision"]["allowed"] is True


def test_probe_support_plan_writer_cli(tmp_path: Path) -> None:
    output_path = tmp_path / "probe_support_plan.json"

    assert main(["--write", str(output_path)]) == 0
    text = output_path.read_text(encoding="utf-8")
    assert "B1_same_obs_wbc_one_step_triplet" in text
    assert "B2_short_closed_loop_triplet" in text
    assert "diagnostic_context_only" in text

