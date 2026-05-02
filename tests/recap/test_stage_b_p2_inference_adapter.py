from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.stage_b.p2_inference_adapter import (  # noqa: E402
    P2_BLOCKED_UNSAFE,
    P2_NO_ENV_SANITY_PASS,
    P2_READY,
    P2_SKIPPED_P0_EXPLAINS_OR_BLOCKS,
    P2_SKIPPED_P1_NOT_PASS,
    blend_unconditional_swap_prediction,
    build_prediction_contract,
    build_weight_sweep_smoke,
    compare_no_env_action_sanity,
    evaluate_p2_readiness,
    main,
    run_no_env_action_sanity,
)


def test_p2_readiness_requires_p1_pass_and_p0_negative() -> None:
    assert (
        evaluate_p2_readiness(p1_status="PENDING", p0_status="P0_NEGATIVE").status
        == P2_SKIPPED_P1_NOT_PASS
    )
    assert (
        evaluate_p2_readiness(
            p1_status="P1_PASS",
            p0_status="STOP_EVAL_PROTOCOL",
        ).status
        == P2_SKIPPED_P0_EXPLAINS_OR_BLOCKS
    )
    decision = evaluate_p2_readiness(p1_status="P1_PASS", p0_status="P0_NEGATIVE")
    assert decision.status == P2_READY
    assert decision.allowed is True


def test_no_env_action_sanity_passes_only_matching_contracts() -> None:
    result = run_no_env_action_sanity(
        fine_tuned_prediction=np.array([[1.0, 2.0]], dtype=np.float32),
        frozen_unconditional_prediction=np.array([[0.5, 1.0]], dtype=np.float32),
        action_order=("right_arm", "left_arm"),
        normalization_id="norm-v1",
        timestep_schedule_hash="sha256:timestep",
        initial_noise_hash="sha256:noise",
    )

    assert result.status == P2_NO_ENV_SANITY_PASS
    assert result.safe_to_eval is True


def test_no_env_action_sanity_blocks_shape_order_and_noise_mismatch() -> None:
    left = run_no_env_action_sanity(
        fine_tuned_prediction=np.array([[1.0, 2.0]], dtype=np.float32),
        frozen_unconditional_prediction=np.array([[0.5], [1.0]], dtype=np.float32),
        action_order=("right_arm", "left_arm"),
        normalization_id="norm-v1",
        timestep_schedule_hash="sha256:timestep",
        initial_noise_hash="sha256:noise",
    )
    assert left.status == P2_BLOCKED_UNSAFE
    assert any("shape_mismatch" in reason for reason in left.reasons)

    right = run_no_env_action_sanity(
        fine_tuned_prediction=np.array([[1.0, 2.0]], dtype=np.float32),
        frozen_unconditional_prediction=np.array([[0.5, 1.0]], dtype=np.float32),
        action_order=("right_arm", "left_arm"),
        normalization_id="norm-v1",
        timestep_schedule_hash="sha256:timestep",
        initial_noise_hash="sha256:noise",
    )
    altered = compare_no_env_action_sanity(
        fine_tuned_contract=build_prediction_contract(
            label="fine_tuned_positive",
            prediction=np.array([[1.0, 2.0]], dtype=np.float32),
            action_order=("right_arm", "left_arm"),
            normalization_id="norm-v1",
            timestep_schedule_hash="sha256:timestep",
            initial_noise_hash="sha256:noise",
        ),
        frozen_unconditional_contract=build_prediction_contract(
            label="frozen_unconditional",
            prediction=np.array([[0.5, 1.0]], dtype=np.float32),
            action_order=("left_arm", "right_arm"),
            normalization_id="norm-v1",
            timestep_schedule_hash="sha256:timestep",
            initial_noise_hash="sha256:different",
        ),
    )
    assert right.safe_to_eval is True
    assert altered.status == P2_BLOCKED_UNSAFE
    assert "action_order_mismatch" in altered.reasons
    assert "initial_noise_mismatch" in altered.reasons


def test_blend_formula_matches_p2_diagnostic_definition() -> None:
    fine_tuned = np.array([2.0, 4.0], dtype=np.float32)
    frozen = np.array([1.0, 1.5], dtype=np.float32)

    np.testing.assert_allclose(
        blend_unconditional_swap_prediction(
            fine_tuned_conditional_prediction=fine_tuned,
            frozen_unconditional_prediction=frozen,
            weight=0.0,
        ),
        fine_tuned,
    )
    np.testing.assert_allclose(
        blend_unconditional_swap_prediction(
            fine_tuned_conditional_prediction=fine_tuned,
            frozen_unconditional_prediction=frozen,
            weight=2.0,
        ),
        np.array([4.0, 9.0], dtype=np.float32),
    )


def test_weight_sweep_smoke_is_diagnostic_only() -> None:
    payload = build_weight_sweep_smoke(
        fine_tuned_conditional_prediction=np.array([1.0, 2.0]),
        frozen_unconditional_prediction=np.array([0.0, 0.5]),
    )

    assert payload["diagnostic_only"] is True
    assert payload["training_allowed"] is False
    assert payload["checkpoint_update_allowed"] is False
    assert payload["runtime_eval_status"] == "NOT_RUN_SYNTHETIC_SMOKE_ONLY"
    assert [entry["weight"] for entry in payload["sweep"]] == [0.0, 0.5, 1.0, 2.0]


def test_cli_self_test_writes_contract_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "P2_inference_unconditional_swap"

    assert (
        main(
            [
                "--self-test",
                "--output-dir",
                str(out),
                "--p1-status",
                "P1_PASS",
                "--p0-status",
                "P0_NEGATIVE",
            ]
        )
        == 0
    )

    expected = {
        "p2_adapter_contract.md",
        "p2_no_env_action_sanity.json",
        "p2_no_env_action_sanity.md",
        "p2_w_sweep_summary.json",
        "p2_w_sweep_summary.md",
        "p2_gate_decision.json",
        "p2_gate_decision.md",
    }
    assert expected <= {path.name for path in out.iterdir()}

    gate = json.loads((out / "p2_gate_decision.json").read_text(encoding="utf-8"))
    assert gate["gate"]["status"] == P2_READY
    assert gate["gate"]["allowed"] is True
    assert gate["training_allowed"] is False


def test_cli_self_test_skips_sweep_when_p1_p0_gate_pending(tmp_path: Path) -> None:
    out = tmp_path / "P2_inference_unconditional_swap"

    assert main(["--self-test", "--output-dir", str(out)]) == 0

    sweep = json.loads((out / "p2_w_sweep_summary.json").read_text(encoding="utf-8"))
    gate = json.loads((out / "p2_gate_decision.json").read_text(encoding="utf-8"))

    assert sweep["runtime_eval_status"] == "SKIPPED_P1_P0_GATE_OR_SANITY_NOT_READY"
    assert sweep["sweep"] == []
    assert gate["gate"]["allowed"] is False


def test_negative_guidance_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        blend_unconditional_swap_prediction(
            fine_tuned_conditional_prediction=[1.0],
            frozen_unconditional_prediction=[0.0],
            weight=-0.1,
        )
