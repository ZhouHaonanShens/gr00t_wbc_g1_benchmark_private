from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_controller_audit_unitree_g1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_runtime_log() -> Path:
    return (
        REPO_ROOT
        / "agent"
        / "runtime_logs"
        / "policy_modality_probe"
        / "00_smoke_eval_g1_once.log"
    )


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_controller_audit_unitree_g1.main(["--help"])
    assert exc_info.value.code == 0


def test_main_writes_happy_path_audit_from_repo_runtime_log(
    tmp_path: Path, capsys
) -> None:
    output_path = tmp_path / "controller_audit_unitree_g1.json"

    exit_code = gr00t_controller_audit_unitree_g1.main(
        ["--runtime-log", str(_repo_runtime_log()), "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = _read_json(output_path)

    assert exit_code == 0
    assert captured.err == ""
    assert payload == written
    assert (
        payload["schema_version"]
        == gr00t_controller_audit_unitree_g1.REPORT_SCHEMA_VERSION
    )
    assert (
        payload["artifact_kind"]
        == gr00t_controller_audit_unitree_g1.REPORT_ARTIFACT_KIND
    )
    assert payload["state_order_expected"] == [
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    ]
    assert payload["state_order_runtime"] == payload["state_order_expected"]
    assert payload["action_order_expected"] == [
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "waist",
        "base_height_command",
        "navigate_command",
    ]
    assert payload["action_order_runtime"] == payload["action_order_expected"]
    assert payload["relative_action_keys"] == ["left_arm", "right_arm"]
    assert payload["absolute_action_keys"] == [
        "left_hand",
        "right_hand",
        "waist",
        "base_height_command",
        "navigate_command",
    ]
    assert payload["timebase"]["policy_horizon_expected"] == 30
    assert payload["timebase"]["n_action_steps_expected"] == 20
    assert payload["execution_surface_contract"] == {
        "policy_horizon_expected": 30,
        "n_action_steps_expected": 20,
        "relative_action_keys": ["left_arm", "right_arm"],
        "absolute_action_keys": [
            "left_hand",
            "right_hand",
            "waist",
            "base_height_command",
            "navigate_command",
        ],
        "action_representation_by_key": {
            "left_arm": "relative",
            "right_arm": "relative",
            "left_hand": "absolute",
            "right_hand": "absolute",
            "waist": "absolute",
            "base_height_command": "absolute",
            "navigate_command": "absolute",
        },
        "relative_to_absolute_rule": {
            "enabled_for_relative_action_keys": True,
            "reference_state_timestep": "last",
            "reference_state_keys": {
                "left_arm": "left_arm",
                "right_arm": "right_arm",
            },
        },
        "must_not_conflate_horizon_and_execution": True,
        "repo_local_formalization": {
            "field_names_are_repo_local": True,
            "upstream_policy_horizon_authority": "action.delta_indices",
            "upstream_execution_steps_authority": "rollout --n_action_steps",
            "note": (
                "This report freezes repo-local contract field names for comparability; "
                "they summarize upstream semantics but are not upstream official JSON field names."
            ),
        },
    }
    assert payload["policy_horizon_runtime"] == 30
    assert payload["equivalent_to_official_unitree_g1"] is True
    assert payload["mismatch_fields"] == []
    assert not output_path.with_name(
        gr00t_controller_audit_unitree_g1.FAILURE_NOTE_MARKDOWN_NAME
    ).exists()


def test_main_writes_failure_note_when_runtime_horizon_mismatches(
    tmp_path: Path, capsys
) -> None:
    runtime_log = tmp_path / "runtime_mismatch.log"
    output_path = tmp_path / "controller_audit_unitree_g1.json"
    original = _repo_runtime_log().read_text(encoding="utf-8")
    mutated = original.replace("SERVER action_horizon: 30", "SERVER action_horizon: 16")
    mutated = mutated.replace("shape=(1, 30, 1)", "shape=(1, 16, 1)")
    mutated = mutated.replace("shape=(1, 30, 7)", "shape=(1, 16, 7)")
    mutated = mutated.replace("shape=(1, 30, 3)", "shape=(1, 16, 3)")
    runtime_log.write_text(mutated, encoding="utf-8")

    exit_code = gr00t_controller_audit_unitree_g1.main(
        ["--runtime-log", str(runtime_log), "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    failure_note_path = output_path.with_name(
        gr00t_controller_audit_unitree_g1.FAILURE_NOTE_MARKDOWN_NAME
    )
    failure_note = failure_note_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert captured.err == ""
    assert payload["equivalent_to_official_unitree_g1"] is False
    assert "policy_horizon_runtime" in payload["mismatch_fields"]
    assert "action_chunk_horizon_runtime" in payload["mismatch_fields"]
    assert failure_note_path.exists()
    assert "policy_horizon_runtime" in failure_note
    assert "actual_rollout_side_runtime_log" not in failure_note


def test_build_audit_report_flags_action_dim_drift() -> None:
    runtime = gr00t_controller_audit_unitree_g1.parse_runtime_sample(
        _repo_runtime_log()
    )
    runtime["action_dims_runtime"] = dict(runtime["action_dims_runtime"])
    runtime["action_dims_runtime"]["left_arm"] = 6

    expected = gr00t_controller_audit_unitree_g1._load_unitree_g1_config()
    expected_state_order = list(expected["state_order_expected"])
    observed_state_key_set = set(runtime["reset_state_keys_raw"])
    state_order_runtime = [
        key for key in expected_state_order if key in observed_state_key_set
    ]
    mismatches: list[str] = []
    gr00t_controller_audit_unitree_g1._mismatch_field(
        mismatches,
        name="action_dims_runtime",
        expected=gr00t_controller_audit_unitree_g1.EXPECTED_ACTION_DIMS,
        actual=runtime["action_dims_runtime"],
    )

    assert state_order_runtime == expected_state_order
    assert mismatches == ["action_dims_runtime"]
