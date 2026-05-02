from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "payload",
    [
        {"C_success_rate": 0.42},
        {"aggregate": {"variant_code": "X", "success_rate": 0.50}},
        {"per_episode_rows": [{"variant_code": "C", "success": True}]},
        {"pairwise_delta_against_C_or_X": 0.10},
    ],
)
def test_selection_function_refuses_c_or_x_inputs(payload: dict[str, object]) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        validate_no_c_x_selection_inputs,
    )

    with pytest.raises(ValueError, match="selection_using_c_or_x_detected"):
        validate_no_c_x_selection_inputs(payload)


def test_selection_function_accepts_a_and_optional_b_inputs() -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        validate_no_c_x_selection_inputs,
    )

    validate_no_c_x_selection_inputs(
        {
            "stock_A_success_rate": 0.61,
            "stock_A_trace_completeness": 0.98,
            "B_control_sanity_status": "PASS",
            "per_episode_rows": [{"variant_code": "A", "success": True}],
        },
    )


def test_selected_using_c_results_attestation_emits_false(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        selected_using_c_results_attestation,
    )

    attestation_path = tmp_path / "selected_using_c_results_attestation.json"
    selected_using_c_results_attestation(
        attestation_path,
        run_id="stage1_v22_runner_surface_iter7_5_20260426T_nextZ",
        selected_using_c_results=False,
    )

    payload = json.loads(Path(attestation_path).read_text(encoding="utf-8"))
    assert payload["selected_using_c_results"] is False
    assert payload["uses_c_results_for_selection"] is False
    assert payload["forbidden_variant_codes_used"] == []
