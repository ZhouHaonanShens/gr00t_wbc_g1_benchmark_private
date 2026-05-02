from __future__ import annotations

import pytest


def test_contract_helper_rejects_selected_c_or_x_variants() -> None:
    from work.openpi.eval.v22_calibration_contracts import assert_no_c_x_leakage

    with pytest.raises(ValueError, match="BLOCK_C_X_LEAKAGE"):
        assert_no_c_x_leakage(
            calibration_variants=("A",),
            optional_control_variants=("C",),
            forbidden_selection_variants=("C", "X"),
        )

    with pytest.raises(ValueError, match="BLOCK_C_X_LEAKAGE"):
        assert_no_c_x_leakage(
            calibration_variants=("X",),
            optional_control_variants=("B",),
            forbidden_selection_variants=("C", "X"),
        )


def test_contract_helper_requires_c_and_x_forbidden_variants() -> None:
    from work.openpi.eval.v22_calibration_contracts import assert_no_c_x_leakage

    with pytest.raises(ValueError, match="BLOCK_C_X_LEAKAGE"):
        assert_no_c_x_leakage(
            calibration_variants=("A",),
            optional_control_variants=("B",),
            forbidden_selection_variants=("C",),
        )


def test_runtime_payload_scan_rejects_nested_c_or_x_selection_inputs() -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        validate_no_c_x_selection_inputs,
    )

    with pytest.raises(ValueError, match="selection_using_c_or_x_detected"):
        validate_no_c_x_selection_inputs(
            {
                "selection_inputs": {
                    "stock_A_success_rate": 0.5,
                    "pairwise_delta_against_C_or_X": 0.1,
                },
            },
        )

