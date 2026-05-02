from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_dual_branch_scorecard


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _artifact_payloads(tmp_path: Path) -> dict[str, Path]:
    formal_public_anchor = _write_json(
        tmp_path / "public_anchor_formal.json",
        {
            "schema_version": "gr00t_public_anchor_formal_v1",
            "artifact_kind": "gr00t_public_anchor_formal",
            "success_rate": 0.5,
            "success_count": 5,
            "systemic_break_flags": [],
        },
    )
    public_anchor_gate = _write_json(
        tmp_path / "public_anchor_sanity_gate.json",
        {
            "schema_version": "gr00t_public_anchor_sanity_gate_v1",
            "artifact_kind": "gr00t_public_anchor_sanity_gate",
            "public_anchor_comparable": True,
            "continue_to_audit": True,
            "sanity_status": "PASS",
            "gate_reason": "non_zero_success_and_no_systemic_breaks",
        },
    )
    unitree_controller = _write_json(
        tmp_path / "controller_audit_unitree_g1.json",
        {
            "schema_version": "gr00t_controller_audit_unitree_g1_v1",
            "artifact_kind": "gr00t_controller_audit_unitree_g1",
            "equivalent_to_official_unitree_g1": True,
            "mismatch_fields": [],
            "policy_horizon_expected": 30,
            "policy_horizon_runtime": 30,
        },
    )
    new_embodiment_controller = _write_json(
        tmp_path / "controller_audit_new_embodiment.json",
        {
            "schema_version": "gr00t_controller_audit_new_embodiment_v1",
            "artifact_kind": "gr00t_controller_audit_new_embodiment",
            "formal_branch_eligibility": "ALLOW",
            "formal_branch_blockers": [],
            "branch_manifest_path": str(tmp_path / "branch_manifest.json"),
            "reason_code": "OK",
            "public_anchor_comparable": False,
        },
    )
    unitree_action = _write_json(
        tmp_path / "action_chain_telemetry_unitree_g1.json",
        {
            "schema_version": "gr00t_action_chain_telemetry_v1",
            "artifact_kind": "gr00t_action_chain_telemetry",
            "public_anchor_comparable": True,
            "controller_absorbed_upstream_difference": True,
            "controller_absorbed_groups": ["left_arm", "waist"],
            "model_insensitive_groups": ["right_hand"],
            "zero_motion_flags": {"all_zero_in_both_groups": ["right_hand"]},
        },
    )
    new_embodiment_action = _write_json(
        tmp_path / "action_chain_telemetry_new_embodiment.json",
        {
            "schema_version": "gr00t_action_chain_telemetry_v1",
            "artifact_kind": "gr00t_action_chain_telemetry",
            "public_anchor_comparable": False,
            "controller_absorbed_upstream_difference": True,
            "controller_absorbed_groups": ["left_arm"],
            "model_insensitive_groups": ["right_hand"],
            "zero_motion_flags": {"all_zero_in_both_groups": ["right_hand"]},
        },
    )
    unitree_condition = _write_json(
        tmp_path / "condition_flip_scorecard_unitree_g1.json",
        {
            "schema_version": "gr00t_condition_flip_scorecard_v1",
            "artifact_kind": "gr00t_condition_flip_scorecard",
            "public_anchor_comparable": True,
            "paired_scene_id": "unitree_g1::S_drop",
            "pass_fail_gate": "PASS",
            "response_ratio": {"min_ratio_across_semantic_flips": 0.18},
            "gate_details": {
                "status": "PASS",
                "reason_code": "semantic_condition_branching_detected",
                "passing_variants": ["blank", "target_swapped", "contradictory"],
            },
        },
    )
    new_embodiment_condition = _write_json(
        tmp_path / "condition_flip_scorecard_new_embodiment.json",
        {
            "schema_version": "gr00t_condition_flip_scorecard_v1",
            "artifact_kind": "gr00t_condition_flip_scorecard",
            "public_anchor_comparable": False,
            "paired_scene_id": "new_embodiment::S_drop",
            "pass_fail_gate": "PASS",
            "response_ratio": {"min_ratio_across_semantic_flips": 0.15},
            "gate_details": {
                "status": "PASS",
                "reason_code": "semantic_condition_branching_detected",
                "passing_variants": ["blank", "target_swapped", "contradictory"],
            },
        },
    )
    unitree_gap = _write_json(
        tmp_path / "teacher_student_gap_scorecard_unitree_g1.json",
        {
            "schema_version": "gr00t_teacher_student_gap_scorecard_v1",
            "artifact_kind": "gr00t_teacher_student_gap_scorecard",
            "public_anchor_comparable": True,
            "status": "ALLOW",
            "case_code": "teacher_reachable_student_zero_branch_consistent",
            "teacher_case_code": "teacher_reachable_student_currently_zero",
            "scene_pool_status": "formal_teacher_replay_reachable_pool_materialized",
            "student_branch_match_rate": 0.74489796,
            "teacher_reachable_families": ["S_drop", "S_lost"],
            "teacher_unreachable_families": ["S_transport_mid", "S_pre_place"],
            "summary": {"action_group_gap_count": 3},
        },
    )
    new_embodiment_gap = _write_json(
        tmp_path / "teacher_student_gap_scorecard_new_embodiment.json",
        {
            "schema_version": "gr00t_teacher_student_gap_scorecard_v1",
            "artifact_kind": "gr00t_teacher_student_gap_scorecard",
            "public_anchor_comparable": False,
            "status": "ALLOW",
            "case_code": "teacher_reachable_student_zero_branch_consistent",
            "teacher_case_code": "teacher_reachable_student_currently_zero",
            "scene_pool_status": "formal_teacher_replay_reachable_pool_materialized",
            "student_branch_match_rate": 0.74489796,
            "teacher_reachable_families": ["S_drop", "S_lost"],
            "teacher_unreachable_families": ["S_transport_mid", "S_pre_place"],
            "summary": {"action_group_gap_count": 3},
        },
    )
    unitree_reachability = _write_json(
        tmp_path / "teacher_reachability_gate_unitree_g1.json",
        {
            "schema_version": "gr00t_teacher_reachability_gate_v1",
            "artifact_kind": "gr00t_teacher_reachability_gate",
            "public_anchor_comparable": True,
            "status": "ALLOW",
            "allow_formal_ladders": True,
            "scene_pool_status": "formal_teacher_replay_reachable_pool_materialized",
            "teacher_case_code": "teacher_reachable_student_currently_zero",
            "replay_case_code": "replay_high_public_anchor_nonzero_student_zero_training_or_data_issue",
            "reachable_scene_ids": ["unitree_g1::S_drop", "unitree_g1::S_lost"],
            "blocking_reasons": [],
            "teacher_upper_bound": {
                "teacher_reachable_rate": 0.75,
                "teacher_success_count": 36,
            },
            "branch_prerequisites": {
                "branch_prerequisite_ok": True,
                "branch_blockers": [],
            },
        },
    )
    new_embodiment_reachability = _write_json(
        tmp_path / "teacher_reachability_gate_new_embodiment.json",
        {
            "schema_version": "gr00t_teacher_reachability_gate_v1",
            "artifact_kind": "gr00t_teacher_reachability_gate",
            "public_anchor_comparable": False,
            "status": "ALLOW",
            "allow_formal_ladders": True,
            "scene_pool_status": "formal_teacher_replay_reachable_pool_materialized",
            "teacher_case_code": "teacher_reachable_student_currently_zero",
            "replay_case_code": "replay_high_branch_local_stack_healthy_student_zero",
            "reachable_scene_ids": [
                "new_embodiment::S_drop",
                "new_embodiment::S_lost",
            ],
            "blocking_reasons": [],
            "teacher_upper_bound": {
                "teacher_reachable_rate": 0.75,
                "teacher_success_count": 36,
            },
            "branch_prerequisites": {
                "branch_prerequisite_ok": True,
                "branch_blockers": [],
            },
        },
    )
    return {
        "formal_public_anchor": formal_public_anchor,
        "public_anchor_gate": public_anchor_gate,
        "unitree_controller": unitree_controller,
        "new_embodiment_controller": new_embodiment_controller,
        "unitree_action": unitree_action,
        "new_embodiment_action": new_embodiment_action,
        "unitree_condition": unitree_condition,
        "new_embodiment_condition": new_embodiment_condition,
        "unitree_gap": unitree_gap,
        "new_embodiment_gap": new_embodiment_gap,
        "unitree_reachability": unitree_reachability,
        "new_embodiment_reachability": new_embodiment_reachability,
    }


def _evidence_payloads(tmp_path: Path, artifacts: dict[str, Path]) -> dict[str, Path]:
    task4 = _write_json(
        tmp_path / "task-4-public-anchor.json",
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_4_public_anchor_evidence",
            "implementation": {
                "formal_artifact": str(artifacts["formal_public_anchor"]),
                "sanity_gate_artifact": str(artifacts["public_anchor_gate"]),
            },
        },
    )
    task5 = _write_json(
        tmp_path / "task-5-unitree-audit.json",
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_5_unitree_controller_audit_evidence",
            "implementation": {"report_artifact": str(artifacts["unitree_controller"])},
        },
    )
    task6 = _write_json(
        tmp_path / "task-6-new-embodiment-audit.json",
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_6_new_embodiment_audit_evidence",
            "implementation": {
                "report_artifact": str(artifacts["new_embodiment_controller"])
            },
        },
    )
    task7 = _write_json(
        tmp_path / "task-7-action-telemetry.json",
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_7_action_chain_telemetry_evidence",
            "implementation": {
                "unitree_output": str(artifacts["unitree_action"]),
                "new_embodiment_output": str(artifacts["new_embodiment_action"]),
            },
        },
    )
    task8 = _write_json(
        tmp_path / "task-8-condition-flip.json",
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_8_condition_flip_evidence",
            "implementation": {
                "unitree_output": str(artifacts["unitree_condition"]),
                "new_embodiment_output": str(artifacts["new_embodiment_condition"]),
            },
        },
    )
    task9 = _write_json(
        tmp_path / "task-9-teacher-student-gap.json",
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_9_teacher_student_gap_evidence",
            "implementation": {
                "unitree_output": str(artifacts["unitree_gap"]),
                "new_embodiment_output": str(artifacts["new_embodiment_gap"]),
            },
        },
    )
    task10 = _write_json(
        tmp_path / "task-10-teacher-reachability.json",
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_10_teacher_reachability_evidence",
            "implementation": {
                "unitree_output": str(artifacts["unitree_reachability"]),
                "new_embodiment_output": str(artifacts["new_embodiment_reachability"]),
            },
        },
    )
    return {
        "task4": task4,
        "task5": task5,
        "task6": task6,
        "task7": task7,
        "task8": task8,
        "task9": task9,
        "task10": task10,
    }


def _build_fixture_bundle(tmp_path: Path) -> dict[str, Path]:
    artifacts = _artifact_payloads(tmp_path)
    return {**artifacts, **_evidence_payloads(tmp_path, artifacts)}


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = gr00t_dual_branch_scorecard.main(["--help"])
    assert exc_info.value.code == 0


def test_main_writes_dual_branch_scorecard_from_repo_defaults(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "dual_branch_scorecard.json"

    exit_code = gr00t_dual_branch_scorecard.main(["--output", str(output_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = _read_json(output_path)

    assert exit_code == 0
    assert captured.err == ""
    assert payload == written
    assert (
        payload["schema_version"] == gr00t_dual_branch_scorecard.REPORT_SCHEMA_VERSION
    )
    assert payload["artifact_kind"] == gr00t_dual_branch_scorecard.REPORT_ARTIFACT_KIND
    assert payload["branch_order"] == ["unitree_g1", "new_embodiment"]
    assert payload["official_comparable_line"] == "unitree_g1"
    assert payload["internal_only_comparable_line"] == "new_embodiment"
    assert payload["allow_p_ladder"] == {"unitree_g1": True, "new_embodiment": True}
    assert payload["allow_d_ladder"] == {"unitree_g1": True, "new_embodiment": True}
    assert payload["branches"][0]["official_comparable_line"] is True
    assert payload["branches"][1]["internal_only_comparable_line"] is True
    assert payload["branches"][1]["public_anchor_status"]["status"] == "NOT_APPLICABLE"


def test_build_scorecard_keeps_dual_branch_comparability_explicit(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture_bundle(tmp_path)
    output_path = tmp_path / "dual_branch_scorecard.json"

    report = gr00t_dual_branch_scorecard.build_dual_branch_scorecard(
        output_path=output_path,
        task4_public_anchor_evidence=fixture["task4"],
        task5_unitree_audit_evidence=fixture["task5"],
        task6_new_embodiment_audit_evidence=fixture["task6"],
        task7_action_telemetry_evidence=fixture["task7"],
        task8_condition_flip_evidence=fixture["task8"],
        task9_teacher_student_gap_evidence=fixture["task9"],
        task10_teacher_reachability_evidence=fixture["task10"],
    )

    assert report["branch_order"] == ["unitree_g1", "new_embodiment"]
    assert report["official_comparable_line"] == "unitree_g1"
    assert report["internal_only_comparable_line"] == "new_embodiment"
    assert report["prerequisite_status"]["unitree_g1"]["status"] == "PASS"
    assert report["prerequisite_status"]["new_embodiment"]["status"] == "PASS"
    assert report["allow_p_ladder"] == {"unitree_g1": True, "new_embodiment": True}
    assert report["allow_d_ladder"] == {"unitree_g1": True, "new_embodiment": True}
    assert report["recommended_next_step"]["unitree_g1"].startswith(
        "proceed_to_task_12"
    )
    assert report["recommended_next_step"]["new_embodiment"].startswith(
        "proceed_to_task_12"
    )

    unitree = report["branches"][0]
    new_embodiment = report["branches"][1]
    assert unitree["official_comparable_line"] is True
    assert unitree["public_anchor_status"]["status"] == "PASS"
    assert unitree["diagnostic_summary"]["public_anchor_success_rate"] == pytest.approx(
        0.5, rel=1e-6
    )
    assert unitree["diagnostic_summary"][
        "teacher_student_branch_match_rate"
    ] == pytest.approx(0.74489796, rel=1e-6)
    assert unitree["diagnostic_summary"]["teacher_reachable_scene_count"] == 2
    assert unitree["diagnostic_summary"][
        "action_telemetry_controller_absorbed_groups"
    ] == [
        "left_arm",
        "waist",
    ]

    assert new_embodiment["internal_only_comparable_line"] is True
    assert new_embodiment["public_anchor_status"]["status"] == "NOT_APPLICABLE"
    assert new_embodiment["public_anchor_comparable"] is False
    assert new_embodiment["diagnostic_summary"][
        "condition_flip_min_response_ratio"
    ] == pytest.approx(0.15, rel=1e-6)


def test_missing_condition_flip_blocks_corresponding_branch_and_writes_failure_note(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture_bundle(tmp_path)
    broken_task8 = _write_json(
        tmp_path / "task-8-condition-flip-broken.json",
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_8_condition_flip_evidence",
            "implementation": {
                "unitree_output": str(fixture["unitree_condition"]),
                "new_embodiment_output": str(
                    tmp_path / "missing_new_embodiment_condition.json"
                ),
            },
        },
    )
    output_path = tmp_path / "dual_branch_scorecard.json"

    report = gr00t_dual_branch_scorecard.build_dual_branch_scorecard(
        output_path=output_path,
        task4_public_anchor_evidence=fixture["task4"],
        task5_unitree_audit_evidence=fixture["task5"],
        task6_new_embodiment_audit_evidence=fixture["task6"],
        task7_action_telemetry_evidence=fixture["task7"],
        task8_condition_flip_evidence=broken_task8,
        task9_teacher_student_gap_evidence=fixture["task9"],
        task10_teacher_reachability_evidence=fixture["task10"],
    )
    _ = gr00t_dual_branch_scorecard.write_scorecard_artifacts(
        report, output_path=output_path
    )

    written = _read_json(output_path)
    failure_note_path = output_path.with_name(
        gr00t_dual_branch_scorecard.FAILURE_NOTE_MARKDOWN_NAME
    )
    new_embodiment = written["branches"][1]

    assert written["allow_p_ladder"]["unitree_g1"] is True
    assert written["allow_p_ladder"]["new_embodiment"] is False
    assert written["allow_d_ladder"]["new_embodiment"] is False
    assert written["prerequisite_status"]["new_embodiment"]["status"] == "BLOCK"
    assert written["prerequisite_status"]["new_embodiment"]["missing_checks"] == [
        "condition_flip"
    ]
    assert (
        new_embodiment["recommended_next_step"]
        == "restore_missing_prerequisite_artifacts_before_task_12"
    )
    assert failure_note_path.is_file()
    assert "new_embodiment" in failure_note_path.read_text(encoding="utf-8")
