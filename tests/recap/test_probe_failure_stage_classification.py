from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import analyze_exact_probe_failure
from work.recap.scripts import derive_probe_stage_events


def test_direct_sidecar_signals_prefer_drop_during_transport_taxonomy() -> None:
    step_records = [
        {
            "t": 0,
            "policy_condition": {"phase": "APPROACH"},
            "privileged": {"apple_in_hand": False},
            "semantic_state": "APPLE_VISIBLE_APPROACH",
        },
        {
            "t": 1,
            "policy_condition": {"phase": "GRASP"},
            "privileged": {"apple_in_hand": True},
            "semantic_state": "GRASPING",
        },
        {
            "t": 2,
            "policy_condition": {"phase": "TRANSPORT"},
            "privileged": {"apple_in_hand": False},
            "drop_during_transport": True,
            "semantic_state": "TRANSPORTING",
        },
    ]
    stage_payload = derive_probe_stage_events.derive_probe_stage_events(
        step_records=step_records,
    )
    analysis = analyze_exact_probe_failure.analyze_exact_probe_failure(
        episode_record={"failure_reason": "outer_step_budget_exhausted"},
        step_records=step_records,
        stage_payload=stage_payload,
    )

    assert stage_payload["progress_flags"]["drop_during_transport_seen"] is True
    assert analysis["high_level_reason"] == "task_stage_failure"
    assert analysis["stage_level_reason"] == "drop_during_transport"
    assert analysis["confidence"] >= 0.9
    assert "apple_in_hand" not in analysis["missing_signals"]


def test_geometry_fallback_degrades_gracefully_when_direct_signals_are_absent() -> None:
    step_records = [
        {
            "outer_step": 1,
            "apple_to_right_eef_l2": 0.30,
            "apple_height_z": 0.20,
            "apple_to_plate_l2": 0.60,
        },
        {
            "outer_step": 2,
            "apple_to_right_eef_l2": 0.06,
            "apple_height_z": 0.27,
            "apple_to_plate_l2": 0.50,
        },
        {
            "outer_step": 3,
            "apple_to_right_eef_l2": 0.05,
            "apple_height_z": 0.25,
            "apple_to_plate_l2": 0.48,
        },
    ]

    analysis = analyze_exact_probe_failure.analyze_exact_probe_failure(
        episode_record={"failure_reason": "terminated_without_success"},
        step_records=step_records,
    )

    assert analysis["high_level_reason"] == "task_stage_failure"
    assert analysis["stage_level_reason"] == "transport_failed"
    assert 0.5 <= analysis["confidence"] <= 0.8
    assert "policy_condition.phase_or_semantic_state" in analysis["missing_signals"]
    assert "apple_in_hand" in analysis["missing_signals"]


def test_failure_stage_guess_normalizes_string_and_mapping_shapes() -> None:
    string_guess_analysis = analyze_exact_probe_failure.analyze_exact_probe_failure(
        episode_record={
            "failure_reason": "done_without_success",
            "failure_stage_guess": "PLACE",
        },
        step_records=[],
    )
    mapping_guess_analysis = analyze_exact_probe_failure.analyze_exact_probe_failure(
        episode_record={
            "failure_reason": "done_without_success",
            "failure_stage_guess": {"label": "near_plate_but_not_success"},
        },
        step_records=[],
    )

    assert string_guess_analysis["stage_level_reason"] == "place_failed"
    assert mapping_guess_analysis["stage_level_reason"] == "place_failed"
    assert string_guess_analysis["normalized_failure_stage_guess"]["label"] == "PLACE"
    assert (
        mapping_guess_analysis["normalized_failure_stage_guess"]["label"]
        == "near_plate_but_not_success"
    )


def test_metric_mismatch_after_success_like_behavior_is_separate_from_stage_failure() -> (
    None
):
    step_records = [
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
    ]
    runtime_trace = {
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
    }

    analysis = analyze_exact_probe_failure.analyze_exact_probe_failure(
        episode_record={
            "failure_reason": "outer_step_budget_exhausted",
            "n_success_steps": 1,
        },
        step_records=step_records,
        runtime_trace=runtime_trace,
    )

    assert (
        analysis["high_level_reason"] == "metric_mismatch_after_success_like_behavior"
    )
    assert (
        analysis["stage_level_reason"] == "metric_mismatch_after_success_like_behavior"
    )
    assert analysis["confidence"] >= 0.85


def test_preflight_blocker_keeps_stage_taxonomy_distinct() -> None:
    analysis = analyze_exact_probe_failure.analyze_exact_probe_failure(
        triplet_gate={
            "formal_eligibility": "BLOCK",
            "reason_code": "triplet_binding_blocked",
            "issues": [
                {
                    "code": "binding_missing",
                    "field_path": "$.run_manifest",
                    "message": "same-checkpoint binding missing",
                }
            ],
        },
        episode_record={"failure_reason": "triplet_binding_blocked"},
    )

    assert analysis["high_level_reason"] == "preflight_blocker"
    assert analysis["stage_level_reason"] is None
    assert analysis["confidence"] >= 0.95
