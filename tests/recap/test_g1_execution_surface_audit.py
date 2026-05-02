from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import audit_g1_execution_surface


def _runtime_trace(
    *,
    decoded: float | None,
    absolute: float | None,
    controller_input: float | None,
    controller_output: float | None = None,
    raw_action_distinct: bool = True,
    raw_or_decoded_distinct: bool = True,
) -> dict[str, object]:
    return {
        "trace_role": "debug_probe",
        "main_gate_eligible": False,
        "status": "READY",
        "normalization_metric": "mean_abs_delta_over_contract_range",
        "controller_output_available": controller_output is not None,
        "controller_output_unavailable_reason": None
        if controller_output is not None
        else "controller_output unavailable in current live seam",
        "terminal_stage_used": "controller_output"
        if controller_output is not None
        else "controller_input",
        "stage_max_mean_abs_delta_over_contract_range": {
            "decoded_action": decoded,
            "absolute_action": absolute,
            "controller_input": controller_input,
            "controller_output": controller_output,
        },
        "upstream_distinction": {
            "prompt_or_token_distinct": True,
            "raw_action_distinct": raw_action_distinct,
            "raw_or_decoded_distinct": raw_or_decoded_distinct,
            "absolute_distinct": bool(absolute and absolute > 0.0),
            "controller_input_distinct": bool(
                controller_input and controller_input > 0.0
            ),
            "controller_output_distinct": bool(
                controller_output is not None and controller_output > 0.0
            ),
        },
    }


def _action_telemetry_group(
    *,
    raw: float,
    decoded: float,
    absolute: float,
    controller: float,
    difference_disappeared_at: str | None,
    controller_absorbed: bool,
    decoded_clip_rate: float = 0.0,
    controller_clip_rate: float = 0.0,
    saturation_rate: float = 0.0,
    zero_all: bool = False,
) -> dict[str, object]:
    return {
        "action_representation": "RELATIVE",
        "reference_state_key": "right_arm",
        "difference_metrics": {
            "raw_action_l2": raw,
            "decoded_action_l2": decoded,
            "absolute_action_l2": absolute,
            "controller_input_l2": controller,
            "difference_disappeared_at": difference_disappeared_at,
            "model_insensitive": False,
            "controller_absorbed_upstream_difference": controller_absorbed,
        },
        "clip_rate": {
            "decoded_action": decoded_clip_rate,
            "controller_input": controller_clip_rate,
        },
        "saturation_rate": saturation_rate,
        "zero_motion_flags": {
            "baseline_controller_input_all_zero": zero_all,
            "probe_controller_input_all_zero": zero_all,
            "all_zero_in_both": zero_all,
        },
        "stages": {
            "controller_input": {
                "baseline": {"zero_output_rate": 1.0 if zero_all else 0.0},
                "probe": {"zero_output_rate": 1.0 if zero_all else 0.0},
            }
        },
    }


def _action_telemetry_payload(group_payload: dict[str, object]) -> dict[str, object]:
    return {
        "per_group_stats": {
            "right_arm": group_payload,
        }
    }


def _action_absorption_payload(strongest: str) -> dict[str, object]:
    return {
        "audit_status": "ready",
        "summary": {
            "strongest_suspected_cause": strongest,
            "absorbed_dimensions_union": ["right_arm"],
            "root_cause_counts": {strongest: 1},
        },
    }


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        audit_g1_execution_surface.main(["--help"])
    assert exc_info.value.code == 0


def test_policy_verdict_when_no_decoded_distinction_survives() -> None:
    report = audit_g1_execution_surface.build_execution_surface_audit(
        runtime_trace=_runtime_trace(
            decoded=0.0,
            absolute=0.0,
            controller_input=0.0,
            raw_action_distinct=False,
            raw_or_decoded_distinct=False,
        )
    )

    assert report["verdict"] == "policy"
    assert report["reason_code"] in {
        "decoded_distinction_absent",
        "raw_action_distinction_absent",
    }


def test_postprocess_verdict_when_difference_dies_before_absolute_action() -> None:
    report = audit_g1_execution_surface.build_execution_surface_audit(
        runtime_trace=_runtime_trace(
            decoded=0.08,
            absolute=0.0,
            controller_input=0.0,
        ),
        action_telemetry=_action_telemetry_payload(
            _action_telemetry_group(
                raw=0.08,
                decoded=0.08,
                absolute=0.0,
                controller=0.0,
                difference_disappeared_at="relative_to_absolute",
                controller_absorbed=False,
            )
        ),
        action_absorption_audit=_action_absorption_payload(
            "relative_to_absolute_scaling"
        ),
    )

    assert report["verdict"] == "postprocess"
    assert report["reason_code"] in {
        "relative_to_absolute_scaling",
        "difference_absorbed_before_absolute_action",
    }


def test_controller_distortion_verdict_fail_softs_to_controller_input_when_output_missing() -> (
    None
):
    report = audit_g1_execution_surface.build_execution_surface_audit(
        runtime_trace=_runtime_trace(
            decoded=0.09,
            absolute=0.08,
            controller_input=0.0,
            controller_output=None,
        ),
        action_telemetry=_action_telemetry_payload(
            _action_telemetry_group(
                raw=0.09,
                decoded=0.09,
                absolute=0.08,
                controller=0.0,
                difference_disappeared_at="controller_input",
                controller_absorbed=True,
                controller_clip_rate=0.6,
                saturation_rate=0.4,
            )
        ),
        action_absorption_audit=_action_absorption_payload("clip_or_saturation"),
    )

    assert report["verdict"] == "controller_distortion"
    assert report["reason_code"] in {
        "controller_clip_or_saturation",
        "difference_absorbed_at_controller_input",
    }
    assert report["controller_output_available"] is False
    assert report["terminal_stage_used"] == "controller_input"
    assert report["controller_output_unavailable_reason"] is not None


def test_unknown_when_distinction_survives_terminal_stage() -> None:
    report = audit_g1_execution_surface.build_execution_surface_audit(
        runtime_trace=_runtime_trace(
            decoded=0.09,
            absolute=0.08,
            controller_input=0.07,
        )
    )

    assert report["verdict"] == "unknown"
    assert report["reason_code"] == "distinction_survives_terminal_stage"


def test_blocked_when_no_usable_core_action_trace_exists() -> None:
    report = audit_g1_execution_surface.build_execution_surface_audit(
        runtime_trace={
            "status": "UNAVAILABLE",
            "controller_output_available": False,
            "controller_output_unavailable_reason": "controller_output unavailable in current live seam",
            "upstream_distinction": {},
            "stage_max_mean_abs_delta_over_contract_range": {
                "decoded_action": None,
                "absolute_action": None,
                "controller_input": None,
                "controller_output": None,
            },
        }
    )

    assert report["verdict"] == "blocked"
    assert report["reason_code"] == "usable_action_trace_missing"
