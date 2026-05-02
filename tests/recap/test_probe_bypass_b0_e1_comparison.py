from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_screening_probe_bypass_diagnostic


def _weak_row_inputs() -> dict[str, object]:
    return {
        "episode_record": {"failure_reason": "terminated_without_success"},
        "step_records": [
            {
                "t": 0,
                "policy_condition": {"phase": "APPROACH"},
                "privileged": {"apple_in_hand": False},
                "apple_to_plate_l2": 0.60,
            },
            {
                "t": 1,
                "policy_condition": {"phase": "APPROACH"},
                "privileged": {"apple_in_hand": False},
                "apple_to_plate_l2": 0.58,
            },
            {
                "t": 2,
                "policy_condition": {"phase": "GRASP"},
                "privileged": {"apple_in_hand": False},
                "apple_to_plate_l2": 0.57,
            },
        ],
    }


def _late_stage_row_inputs() -> dict[str, object]:
    return {
        "episode_record": {"failure_reason": "done_without_success"},
        "step_records": [
            {
                "t": 0,
                "policy_condition": {"phase": "APPROACH"},
                "privileged": {"apple_in_hand": False},
                "apple_to_plate_l2": 0.50,
            },
            {
                "t": 1,
                "policy_condition": {"phase": "GRASP"},
                "privileged": {"apple_in_hand": True},
                "apple_to_plate_l2": 0.35,
            },
            {
                "t": 2,
                "policy_condition": {"phase": "TRANSPORT"},
                "privileged": {"apple_in_hand": True},
                "apple_to_plate_l2": 0.09,
            },
            {
                "t": 3,
                "policy_condition": {"phase": "PLACE"},
                "privileged": {"apple_in_hand": False},
                "apple_to_plate_l2": 0.04,
            },
        ],
    }


def _runtime_distortion_row_inputs() -> dict[str, object]:
    return {
        "episode_record": {
            "failure_reason": "outer_step_budget_exhausted",
            "n_success_steps": 1,
        },
        "step_records": [
            {
                "t": 0,
                "policy_condition": {"phase": "TRANSPORT"},
                "privileged": {"apple_in_hand": True},
                "apple_to_plate_l2": 0.20,
            },
            {
                "t": 1,
                "policy_condition": {"phase": "PLACE"},
                "privileged": {"apple_in_hand": False},
                "apple_to_plate_l2": 0.05,
                "success_step": True,
            },
        ],
        "runtime_trace": {
            "status": "READY",
            "stage_max_mean_abs_delta_over_contract_range": {
                "decoded_action": 0.08,
                "absolute_action": 0.04,
                "controller_input": 0.0,
                "controller_output": None,
            },
            "upstream_distinction": {
                "prompt_or_token_distinct": True,
                "raw_or_decoded_distinct": True,
            },
            "controller_output_available": False,
            "controller_output_unavailable_reason": "controller_output unavailable in current live seam",
        },
    }


def test_default_probe_bypass_materialization_uses_only_b0_and_fences_artifact(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    output_dir = (
        repo_root / "agent/artifacts/apple_recap_flux_graft/diagnostic_probe_bypass"
    )

    payload = (
        gr00t_screening_probe_bypass_diagnostic.materialize_probe_bypass_diagnostic(
            row_evidence_by_label={"B0": _weak_row_inputs()},
            output_dir=output_dir,
            repo_root=repo_root,
        )
    )
    written = json.loads(
        (
            output_dir
            / gr00t_screening_probe_bypass_diagnostic.DIAGNOSTIC_GAP_JSON_NAME
        ).read_text(encoding="utf-8")
    )

    assert written == payload
    assert payload["diagnostic_row_labels"] == ["B0"]
    assert payload["excluded_mainline_rows"] == ["E2"]
    assert payload["screening_mode"] == "diagnostic_probe_bypass"
    assert payload["diagnostic_only"] is True
    assert payload["mainline_authority"] is False
    assert payload["main_verdict_eligible"] is False
    assert payload["external_reference_only"] is True
    assert payload["comparison_verdict"] == "checkpoint_or_policy_likely_weak"
    assert payload["rows"]["B0"]["comparable_to"] is None
    assert payload["rows"]["B0"]["main_verdict_eligible"] is False
    assert payload["rows"]["B0"]["external_reference_only"] is True
    assert payload["rows"]["B0"]["row_signal"] == "screen_negative"
    assert payload["artifact_path"].endswith("diagnostic_probe_vs_screening_gap.json")


def test_include_e1_extends_row_set_to_exact_b0_e1_and_can_flag_probe_too_strict(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"

    payload = (
        gr00t_screening_probe_bypass_diagnostic.build_probe_bypass_diagnostic_payload(
            row_evidence_by_label={
                "B0": _weak_row_inputs(),
                "E1": _late_stage_row_inputs(),
            },
            include_e1=True,
            repo_root=repo_root,
        )
    )

    assert payload["diagnostic_row_labels"] == ["B0", "E1"]
    assert sorted(payload["rows"]) == ["B0", "E1"]
    assert "E2" not in payload["rows"]
    assert payload["rows"]["E1"]["comparable_to"] == "B0"
    assert payload["rows"]["E1"]["row_signal"] == "screen_positive"
    assert payload["comparison_verdict"] == "probe_likely_too_strict"
    assert payload["comparison_verdict"] in payload["allowed_comparison_verdicts"]


def test_runtime_control_distortion_verdict_stays_within_allowed_enum(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    payload = (
        gr00t_screening_probe_bypass_diagnostic.build_probe_bypass_diagnostic_payload(
            row_evidence_by_label={"B0": _runtime_distortion_row_inputs()},
            repo_root=repo_root,
        )
    )

    assert payload["comparison_verdict"] == "runtime_control_distortion_suspected"
    assert payload["comparison_basis"]["trigger"] == (
        "metric_mismatch_after_success_like_behavior"
    )
    assert payload["comparison_verdict"] in list(
        gr00t_screening_probe_bypass_diagnostic.COMPARISON_VERDICT_ENUM
    )
