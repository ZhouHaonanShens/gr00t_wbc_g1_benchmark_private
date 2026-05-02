from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_teacher_student_gap


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _open_loop_pass_fixture(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "g1_state_conditioned_open_loop_agreement_v1",
        "artifact_kind": "state_conditioned_open_loop_agreement_report",
        "status": "PASS",
        "checks": {
            "history_condition_response": {
                "status": "PASS",
                "passed": True,
                "response_ratio": 1.25,
                "min_response_ratio": 0.001,
            },
            "valid_mask_effectiveness": {
                "status": "PASS",
                "passed": True,
                "max_abs_prediction_delta": 0.0,
            },
            "action_range": {
                "status": "PASS",
                "passed": True,
                "range_check": {"allowed_abs_limit": 2.22, "violation_count": 0},
            },
        },
        "telemetry": {"allowed_abs_limit": 2.22},
        "summary": {"passed_check_count": 3, "total_check_count": 3},
        "failure": None,
    }
    return _write_json(tmp_path / "open_loop_pass.json", payload)


def _action_telemetry_fixture(tmp_path: Path, *, branch: str) -> Path:
    payload = {
        "schema_version": "gr00t_action_chain_telemetry_v1",
        "artifact_kind": "gr00t_action_chain_telemetry",
        "branch": branch,
        "public_anchor_comparable": branch == "UNITREE_G1",
        "action_order": ["left_arm", "right_hand", "navigate_command"],
        "per_group_stats": {
            "left_arm": {
                "difference_metrics": {
                    "raw_action_l2": 0.9,
                    "decoded_action_l2": 0.8,
                    "absolute_action_l2": 0.7,
                    "controller_input_l2": 0.0,
                    "difference_disappeared_at": "controller_input",
                    "model_insensitive": False,
                    "controller_absorbed_upstream_difference": True,
                },
                "clip_rate": {"decoded_action": 0.0, "controller_input": 0.0},
                "saturation_rate": 0.0,
                "zero_motion_flags": {
                    "baseline_controller_input_all_zero": False,
                    "probe_controller_input_all_zero": False,
                    "all_zero_in_both": False,
                },
            },
            "right_hand": {
                "difference_metrics": {
                    "raw_action_l2": 0.0,
                    "decoded_action_l2": 0.0,
                    "absolute_action_l2": 0.0,
                    "controller_input_l2": 0.0,
                    "difference_disappeared_at": "raw_action",
                    "model_insensitive": True,
                    "controller_absorbed_upstream_difference": False,
                },
                "clip_rate": {"decoded_action": 0.0, "controller_input": 0.0},
                "saturation_rate": 0.0,
                "zero_motion_flags": {
                    "baseline_controller_input_all_zero": True,
                    "probe_controller_input_all_zero": True,
                    "all_zero_in_both": True,
                },
            },
            "navigate_command": {
                "difference_metrics": {
                    "raw_action_l2": 0.3,
                    "decoded_action_l2": 0.3,
                    "absolute_action_l2": 0.3,
                    "controller_input_l2": 0.3,
                    "difference_disappeared_at": None,
                    "model_insensitive": False,
                    "controller_absorbed_upstream_difference": False,
                },
                "clip_rate": {"decoded_action": 0.0, "controller_input": 0.0},
                "saturation_rate": 0.0,
                "zero_motion_flags": {
                    "baseline_controller_input_all_zero": False,
                    "probe_controller_input_all_zero": False,
                    "all_zero_in_both": False,
                },
            },
        },
        "controller_absorbed_groups": ["left_arm"],
        "model_insensitive_groups": ["right_hand"],
        "clip_rate": {"decoded_action_overall": 0.0, "controller_input_overall": 0.0},
        "saturation_rate": {"overall": 0.0},
    }
    return _write_json(tmp_path / f"action_telemetry_{branch.lower()}.json", payload)


def _teacher_unreachable_fixture(tmp_path: Path, *, branch: str) -> Path:
    branch_slug = branch.lower()
    payload = {
        "schema_version": "gr00t_teacher_reachability_gate_v1",
        "artifact_kind": "gr00t_teacher_reachability_gate",
        "branch": branch,
        "branch_scope": (
            "official_public_anchor_line"
            if branch == "UNITREE_G1"
            else "branch_internal_only"
        ),
        "public_anchor_comparable": branch == "UNITREE_G1",
        "status": "BLOCK",
        "allow_formal_ladders": False,
        "scene_pool_status": "blocked_teacher_all_zero",
        "teacher_case_code": "teacher_all_zero_block",
        "teacher_case": "Teacher stayed at zero on every formal family.",
        "replay_case_code": "replay_all_zero_stack_or_env_risk",
        "replay_case": "Replay is also zero.",
        "reachable_scene_ids": [],
        "teacher_reachable_scene_ids": [],
        "replay_reachable_scene_ids": [],
        "blocking_reasons": ["teacher_all_zero"],
        "teacher_upper_bound": {
            "artifact_kind": "state_conditioned_teacher_upper_bound_gate",
            "teacher_success_count": 0,
            "snapshot_baseline_success_count": 0,
            "teacher_reachable_rate": 0.0,
        },
        "branch_prerequisites": {
            "branch_prerequisite_ok": True,
            "branch_prerequisite_code": "fixture_ok",
            "branch_blockers": [],
            "controller_audit_path": None,
            "branch_manifest_path": None,
        },
        "scene_pool": {
            "scene_rows": [
                {
                    "scene_id": f"{branch_slug}::S_drop",
                    "family": "S_drop",
                    "priority": "high",
                    "teacher_reachable": False,
                    "teacher_success_count": 0,
                    "teacher_interpretation_code": "teacher_unreachable_on_snapshots_no_progress",
                    "snapshot_baseline_success_count": 0,
                    "replay_success": False,
                    "included_in_formal_scene_pool": False,
                },
                {
                    "scene_id": f"{branch_slug}::S_lost",
                    "family": "S_lost",
                    "priority": "high",
                    "teacher_reachable": False,
                    "teacher_success_count": 0,
                    "teacher_interpretation_code": "teacher_unreachable_on_snapshots_no_progress",
                    "snapshot_baseline_success_count": 0,
                    "replay_success": False,
                    "included_in_formal_scene_pool": False,
                },
            ]
        },
    }
    return _write_json(tmp_path / f"teacher_unreachable_{branch.lower()}.json", payload)


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = gr00t_teacher_student_gap.main(["--help"])
    assert exc_info.value.code == 0


@pytest.mark.parametrize(
    ("branch", "expected_name", "expected_scope"),
    [
        (
            "UNITREE_G1",
            "teacher_student_gap_scorecard_unitree_g1.json",
            "official_public_anchor_line",
        ),
        (
            "NEW_EMBODIMENT",
            "teacher_student_gap_scorecard_new_embodiment.json",
            "branch_internal_only",
        ),
    ],
)
def test_main_writes_branch_gap_scorecard(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    branch: str,
    expected_name: str,
    expected_scope: str,
) -> None:
    output_path = tmp_path / expected_name

    exit_code = gr00t_teacher_student_gap.main(
        ["--branch", branch, "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = _read_json(output_path)

    assert exit_code == 0
    assert captured.err == ""
    assert payload == written
    assert payload["schema_version"] == gr00t_teacher_student_gap.REPORT_SCHEMA_VERSION
    assert payload["artifact_kind"] == gr00t_teacher_student_gap.REPORT_ARTIFACT_KIND
    assert payload["branch"] == branch
    assert payload["branch_scope"] == expected_scope
    assert payload["output_path"].endswith(expected_name)
    assert payload["failure_note_path"] is None
    assert payload["status"] == "ALLOW"
    assert (
        payload["scene_pool_status"]
        == "formal_teacher_replay_reachable_pool_materialized"
    )
    assert payload["teacher_reachable_families"] == ["S_drop", "S_lost"]
    assert payload["teacher_unreachable_families"] == [
        "S_transport_mid",
        "S_pre_place",
    ]
    assert payload["case_code"] == "teacher_reachable_student_zero_branch_consistent"
    assert payload["student_branch_match_rate"] >= 0.60
    assert set(payload["per_action_group_gap"].keys()) == {
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "waist",
        "base_height_command",
        "navigate_command",
    }
    assert payload["per_action_group_gap"]["left_arm"]["telemetry_case_code"] == (
        "student_controller_absorbed_group"
    )
    assert payload["per_action_group_gap"]["right_hand"]["telemetry_case_code"] == (
        "student_zero_motion_group"
    )
    assert (
        payload["per_action_group_gap"]["right_hand"]["gap_score"]
        > payload["per_action_group_gap"]["navigate_command"]["gap_score"]
    )
    assert payload["per_family_gap"][0]["family"] == "S_drop"
    assert payload["per_family_gap"][0]["case_code"] == (
        "teacher_reachable_student_zero_branch_consistent"
    )
    assert payload["per_family_gap"][0]["blame_bucket"] == "student_not_learned"


def test_default_output_path_matches_branch_contract() -> None:
    assert (
        gr00t_teacher_student_gap.default_output_path_for_branch("UNITREE_G1").name
        == "teacher_student_gap_scorecard_unitree_g1.json"
    )
    assert (
        gr00t_teacher_student_gap.default_output_path_for_branch("NEW_EMBODIMENT").name
        == "teacher_student_gap_scorecard_new_embodiment.json"
    )


def test_teacher_unreachable_block_isolated_from_student_blame(tmp_path: Path) -> None:
    branch = "UNITREE_G1"
    output_path = tmp_path / "teacher_student_gap_scorecard_unitree_g1.json"
    teacher_reachability = _teacher_unreachable_fixture(tmp_path, branch=branch)
    action_telemetry = _action_telemetry_fixture(tmp_path, branch=branch)
    open_loop = _open_loop_pass_fixture(tmp_path)

    report = gr00t_teacher_student_gap.build_teacher_student_gap_scorecard(
        branch,
        output_path=output_path,
        teacher_reachability_json=teacher_reachability,
        action_telemetry_json=action_telemetry,
        open_loop_agreement_json=open_loop,
    )
    _ = gr00t_teacher_student_gap.write_scorecard_artifacts(
        report, output_path=output_path
    )

    failure_note_path = output_path.with_name(
        gr00t_teacher_student_gap.FAILURE_NOTE_MARKDOWN_NAME_BY_BRANCH[branch]
    )
    written = _read_json(output_path)

    assert written == report
    assert report["status"] == "BLOCK"
    assert report["case_code"] == "teacher_unreachable_gap_block"
    assert report["teacher_reachable_families"] == []
    assert report["teacher_unreachable_families"] == ["S_drop", "S_lost"]
    assert report["failure_note_path"] == str(failure_note_path)
    assert all(
        row["blame_bucket"] == "teacher_unreachable" for row in report["per_family_gap"]
    )
    assert all(
        row["teacher_student_success_gap_rate"] is None
        for row in report["per_family_gap"]
    )
    assert failure_note_path.is_file()
    assert "teacher_unreachable_gap_block" in failure_note_path.read_text(
        encoding="utf-8"
    )
