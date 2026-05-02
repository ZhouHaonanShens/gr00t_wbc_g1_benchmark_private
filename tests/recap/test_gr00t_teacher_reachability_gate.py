from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_teacher_reachability_gate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _teacher_all_zero_fixture(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "state_conditioned_teacher_upper_bound_gate_v1",
        "artifact_kind": "state_conditioned_teacher_upper_bound_gate",
        "mapping": {
            "teacher_success_count": 0,
            "teacher_reachable_rate": 0.0,
            "current_model_baseline_success_count": 0,
        },
        "families": [
            {
                "family": "S_drop",
                "priority": "high",
                "success_count": 0,
                "interpretation_code": "teacher_unreachable_on_snapshots_no_progress",
                "current_model_baseline": {"success_count": 0},
            },
            {
                "family": "S_lost",
                "priority": "high",
                "success_count": 0,
                "interpretation_code": "teacher_unreachable_on_snapshots_no_progress",
                "current_model_baseline": {"success_count": 0},
            },
        ],
    }
    return _write_json(tmp_path / "teacher_all_zero.json", payload)


def _replay_all_zero_fixture(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "gr00t_replay_upper_bound_report_v1",
        "artifact_kind": gr00t_teacher_reachability_gate.REPLAY_UPPER_BOUND_ARTIFACT_KIND,
        "source_kind": "test_fixture",
        "policy_role": "stack_integration_debug_upper_bound",
        "attempt_count": 2,
        "success_count": 0,
        "success_rate": 0.0,
        "scene_results": [
            {
                "scene_id": "unitree_g1::S_drop",
                "family": "S_drop",
                "success": False,
                "source_kind": "test_fixture",
            },
            {
                "scene_id": "unitree_g1::S_lost",
                "family": "S_lost",
                "success": False,
                "source_kind": "test_fixture",
            },
        ],
    }
    return _write_json(tmp_path / "replay_all_zero.json", payload)


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_teacher_reachability_gate.main(["--help"])
    assert exc_info.value.code == 0


def test_unitree_cli_writes_teacher_reachability_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "teacher_gate_unitree"

    exit_code = gr00t_teacher_reachability_gate.main(
        ["--branch", "UNITREE_G1", "--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact = _read_json(
        output_dir
        / gr00t_teacher_reachability_gate.REPORT_JSON_NAME_BY_BRANCH["UNITREE_G1"]
    )

    assert exit_code == 0
    assert captured.err == ""
    assert artifact == payload
    assert (
        payload["artifact_kind"] == gr00t_teacher_reachability_gate.REPORT_ARTIFACT_KIND
    )
    assert (
        payload["scene_pool_status"]
        == "formal_teacher_replay_reachable_pool_materialized"
    )
    assert payload["teacher_case_code"] == "teacher_reachable_student_currently_zero"
    assert (
        payload["replay_case_code"]
        == "replay_high_public_anchor_nonzero_student_zero_training_or_data_issue"
    )
    assert payload["allow_formal_ladders"] is True
    assert payload["reachable_scene_ids"] == [
        "unitree_g1::S_drop",
        "unitree_g1::S_lost",
    ]
    assert payload["failure_note_path"] is None


def test_new_embodiment_cli_writes_branch_local_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "teacher_gate_new_embodiment"

    exit_code = gr00t_teacher_reachability_gate.main(
        ["--branch", "NEW_EMBODIMENT", "--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact = _read_json(
        output_dir
        / gr00t_teacher_reachability_gate.REPORT_JSON_NAME_BY_BRANCH["NEW_EMBODIMENT"]
    )

    assert exit_code == 0
    assert captured.err == ""
    assert artifact == payload
    assert payload["branch_scope"] == "branch_internal_only"
    assert payload["public_anchor_comparable"] is False
    assert (
        payload["scene_pool_status"]
        == "formal_teacher_replay_reachable_pool_materialized"
    )
    assert payload["teacher_case_code"] == "teacher_reachable_student_currently_zero"
    assert (
        payload["replay_case_code"]
        == "replay_high_branch_local_stack_healthy_student_zero"
    )
    assert payload["allow_formal_ladders"] is True
    assert payload["reachable_scene_ids"] == [
        "new_embodiment::S_drop",
        "new_embodiment::S_lost",
    ]


def test_teacher_all_zero_blocks_strong_attribution_and_writes_failure_note(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "teacher_gate_blocked"
    teacher_fixture = _teacher_all_zero_fixture(tmp_path)

    exit_code = gr00t_teacher_reachability_gate.main(
        [
            "--branch",
            "UNITREE_G1",
            "--output-dir",
            str(output_dir),
            "--teacher-gate-json",
            str(teacher_fixture),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact = _read_json(
        output_dir
        / gr00t_teacher_reachability_gate.REPORT_JSON_NAME_BY_BRANCH["UNITREE_G1"]
    )
    failure_note_path = (
        output_dir
        / gr00t_teacher_reachability_gate.FAILURE_NOTE_MARKDOWN_NAME_BY_BRANCH[
            "UNITREE_G1"
        ]
    )

    assert exit_code == 0
    assert captured.err == ""
    assert artifact == payload
    assert payload["scene_pool_status"] == "blocked_teacher_all_zero"
    assert payload["teacher_case_code"] == "teacher_all_zero_block"
    assert payload["allow_formal_ladders"] is False
    assert payload["reachable_scene_ids"] == []
    assert payload["failure_note_path"] == str(failure_note_path)
    assert failure_note_path.is_file()
    assert "teacher_all_zero_block" in failure_note_path.read_text(encoding="utf-8")


def test_replay_all_zero_blocks_formal_scene_pool_even_when_teacher_reachable(
    tmp_path: Path,
) -> None:
    teacher_summary = gr00t_teacher_reachability_gate.load_teacher_gate_summary(
        gr00t_teacher_reachability_gate.DEFAULT_TEACHER_GATE_JSON
    )
    teacher_summary["families"] = [
        {**row, "scene_id": f"unitree_g1::{row['family']}"}
        for row in teacher_summary["families"]
    ]
    current_baseline = gr00t_teacher_reachability_gate.load_current_baseline_summary(
        "UNITREE_G1",
        gr00t_teacher_reachability_gate.DEFAULT_UNITREE_PUBLIC_ANCHOR_JSON,
    )
    branch_context = gr00t_teacher_reachability_gate.load_branch_context(
        "UNITREE_G1",
        controller_audit_path=gr00t_teacher_reachability_gate.DEFAULT_UNITREE_CONTROLLER_AUDIT_JSON,
        branch_manifest_path=None,
    )
    replay_upper_bound = {
        "artifact_kind": gr00t_teacher_reachability_gate.REPLAY_UPPER_BOUND_ARTIFACT_KIND,
        "source_kind": "test_fixture",
        "policy_role": "stack_integration_debug_upper_bound",
        "attempt_count": 2,
        "success_count": 0,
        "success_rate": 0.0,
        "scene_results": [
            {"scene_id": "unitree_g1::S_drop", "family": "S_drop", "success": False},
            {"scene_id": "unitree_g1::S_lost", "family": "S_lost", "success": False},
        ],
    }

    payload = gr00t_teacher_reachability_gate.build_teacher_reachability_gate_payload(
        branch="UNITREE_G1",
        teacher_summary=teacher_summary,
        current_baseline=current_baseline,
        replay_upper_bound=replay_upper_bound,
        branch_context=branch_context,
        output_path=tmp_path / "report.json",
    )

    assert payload["teacher_case_code"] == "teacher_reachable_student_currently_zero"
    assert payload["replay_case_code"] == "replay_all_zero_stack_or_env_risk"
    assert payload["scene_pool_status"] == "blocked_replay_all_zero"
    assert payload["allow_formal_ladders"] is False
    assert payload["reachable_scene_ids"] == []
