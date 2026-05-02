from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_eval_contract_gate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_eval_contract_gate.main(["--help"])
    assert exc_info.value.code == 0


def test_main_materializes_formal_eval_contract_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "freeze"

    exit_code = gr00t_eval_contract_gate.main(["--output-dir", str(output_dir)])

    assert exit_code == 0
    freeze = _read_json(output_dir / gr00t_eval_contract_gate.FREEZE_JSON_NAME)
    report = _read_json(
        output_dir / gr00t_eval_contract_gate.COMPARABILITY_REPORT_JSON_NAME
    )

    assert freeze["schema_version"] == gr00t_eval_contract_gate.FREEZE_SCHEMA_VERSION
    assert freeze["artifact_kind"] == gr00t_eval_contract_gate.FREEZE_ARTIFACT_KIND
    assert freeze["env_name"] == gr00t_eval_contract_gate.DEFAULT_ENV_NAME
    assert (
        freeze["wrapper_parameters"]
        == gr00t_eval_contract_gate.DEFAULT_WRAPPER_PARAMETERS
    )
    assert freeze["camera_config"] == gr00t_eval_contract_gate.DEFAULT_CAMERA_CONFIG
    assert (
        freeze["max_episode_steps"]
        == gr00t_eval_contract_gate.DEFAULT_MAX_EPISODE_STEPS
    )
    assert freeze["n_action_steps"] == gr00t_eval_contract_gate.DEFAULT_N_ACTION_STEPS
    assert (
        freeze["policy_horizon_expected"]
        == gr00t_eval_contract_gate.DEFAULT_POLICY_HORIZON_EXPECTED
    )
    assert freeze["action_semantics"] == {
        "policy_horizon_expected": gr00t_eval_contract_gate.DEFAULT_POLICY_HORIZON_EXPECTED,
        "n_action_steps": gr00t_eval_contract_gate.DEFAULT_N_ACTION_STEPS,
        "relative_action_keys": ["left_arm", "right_arm"],
        "absolute_action_keys": [
            "left_hand",
            "right_hand",
            "waist",
            "base_height_command",
            "navigate_command",
        ],
        "action_representation_by_key": {
            "left_arm": "RELATIVE",
            "right_arm": "RELATIVE",
            "left_hand": "ABSOLUTE",
            "right_hand": "ABSOLUTE",
            "waist": "ABSOLUTE",
            "base_height_command": "ABSOLUTE",
            "navigate_command": "ABSOLUTE",
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
                "This repo freezes local contract field names for comparability; "
                "they summarize upstream semantics but are not upstream official JSON field names."
            ),
        },
    }
    assert (
        freeze["scene_pool_identifier"]
        == gr00t_eval_contract_gate.DEFAULT_SCENE_POOL_IDENTIFIER
    )
    assert (
        freeze["prompt_template_id"]
        == gr00t_eval_contract_gate.DEFAULT_PROMPT_TEMPLATE_ID
    )
    assert freeze["server_contract"]["use_sim_policy_wrapper"] is True
    assert freeze["seed_manifest"]["required_fields"] == [
        "python",
        "numpy",
        "torch",
        "env",
        "rollout_episode_order",
    ]
    assert freeze["seed_manifest"]["seed_values"] == list(
        gr00t_eval_contract_gate.DEFAULT_FORMAL_SEED_VALUES
    )
    assert freeze["seed_manifest"]["seed_values_origin"] == "repo_local_formal_protocol"
    assert freeze["normalization_policy"]["cross_branch_reuse_allowed"] is False
    assert (
        freeze["branch_comparability_tag"]
        == gr00t_eval_contract_gate.DEFAULT_BRANCH_COMPARABILITY_TAG
    )
    assert (
        freeze["embodiment_branches"]["UNITREE_G1"]["public_anchor_comparable"] is True
    )
    assert (
        freeze["embodiment_branches"]["NEW_EMBODIMENT"]["public_anchor_comparable"]
        is False
    )
    assert (
        freeze["checkpoint_provenance_schema"]["required_use_sim_policy_wrapper"]
        is True
    )
    assert freeze["allowed_change_surface"]["policy"] == "BLOCK_ALL_FORMAL_DRIFT"
    assert freeze["allowed_change_surface"]["protected_field_paths"] == list(
        gr00t_eval_contract_gate.PROTECTED_FIELD_PATHS
    )

    assert report["artifact_kind"] == gr00t_eval_contract_gate.REPORT_ARTIFACT_KIND
    assert report["contract_gate"] == {
        "name": gr00t_eval_contract_gate.CONTRACT_GATE_NAME,
        "passed": True,
        "status": "PASS",
    }
    assert report["counts"] == {
        "protected_field_count": len(gr00t_eval_contract_gate.PROTECTED_FIELD_PATHS),
        "drift_count": 0,
    }
    assert report["digest_basis"]["selectors"] == list(
        gr00t_eval_contract_gate.PROTECTED_FIELD_PATHS
    )
    assert report["canonical_contract_digest"] == report["candidate_contract_digest"]
    assert report["checks"]["freeze_payload_schema"]["passed"] is True
    assert report["checks"]["protected_field_freeze"]["passed"] is True
    assert report["checks"]["protected_field_freeze"]["offending_field_paths"] == []
    assert report["checks"]["branch_comparability_rules"]["passed"] is True
    assert not (
        output_dir / gr00t_eval_contract_gate.FAILURE_NOTE_MARKDOWN_NAME
    ).exists()


def test_build_comparability_report_rejects_illegal_formal_drift() -> None:
    freeze = gr00t_eval_contract_gate.build_eval_contract_freeze()
    candidate = copy.deepcopy(freeze)
    candidate["env_name"] = "gr00tlocomanip_g1_sim/AnotherEnv"
    candidate["wrapper_parameters"]["timebase"]["control_frequency_hz"] = 60
    candidate["camera_config"]["video_observation_keys"] = ["ego_view"]
    candidate["n_action_steps"] = 16
    candidate["action_semantics"]["relative_action_keys"] = ["left_arm"]
    candidate["scene_pool_identifier"] = "repo_local::different_scene_pool"
    candidate["prompt_template_id"] = "different_prompt"
    candidate["seed_manifest"]["seed_values"] = [31001, 31002]
    candidate["branch_comparability_tag"] = "different_branch_tag"
    candidate["embodiment_branches"]["NEW_EMBODIMENT"]["public_anchor_comparable"] = (
        True
    )
    candidate["normalization_policy"]["cross_branch_reuse_allowed"] = True
    candidate["checkpoint_provenance_schema"]["required_use_sim_policy_wrapper"] = False

    report = gr00t_eval_contract_gate.build_comparability_report(freeze, candidate)

    assert report["contract_gate"]["passed"] is False
    assert report["contract_gate"]["status"] == "FAIL"
    assert report["counts"]["drift_count"] >= 7
    assert report["checks"]["protected_field_freeze"]["passed"] is False
    assert report["checks"]["protected_field_freeze"]["status"] == "FAIL"
    assert report["checks"]["protected_field_freeze"]["offending_field_paths"] == [
        "env_name",
        "wrapper_parameters",
        "camera_config",
        "n_action_steps",
        "action_semantics",
        "scene_pool_identifier",
        "prompt_template_id",
        "seed_manifest",
        "normalization_policy",
        "branch_comparability_tag",
        "checkpoint_provenance_schema",
        "embodiment_branches.NEW_EMBODIMENT.public_anchor_comparable",
    ]
    assert report["checks"]["branch_comparability_rules"]["passed"] is False
    assert report["checks"]["branch_comparability_rules"]["offending_field_paths"] == [
        "branch_comparability_tag",
        "embodiment_branches.NEW_EMBODIMENT.public_anchor_comparable",
    ]
    assert report["canonical_contract_digest"] != report["candidate_contract_digest"]
    assert all(
        drift["reason"] == "formal eval contract drift is forbidden"
        for drift in report["checks"]["protected_field_freeze"]["drifts"]
    )


def test_main_rejects_candidate_drift_cleanly_and_writes_failure_note(
    tmp_path: Path,
    capsys,
) -> None:
    output_dir = tmp_path / "freeze"
    candidate_path = tmp_path / "candidate.json"
    candidate = gr00t_eval_contract_gate.build_eval_contract_freeze()
    candidate["camera_config"]["view_count"] = 1
    candidate_path.write_text(
        json.dumps(candidate, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    exit_code = gr00t_eval_contract_gate.main(
        ["--output-dir", str(output_dir), "--candidate-json", str(candidate_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "comparability gate failed" in captured.err
    assert "Traceback" not in captured.err

    report = _read_json(
        output_dir / gr00t_eval_contract_gate.COMPARABILITY_REPORT_JSON_NAME
    )
    failure_note = (
        output_dir / gr00t_eval_contract_gate.FAILURE_NOTE_MARKDOWN_NAME
    ).read_text(encoding="utf-8")
    assert report["contract_gate"]["passed"] is False
    assert report["checks"]["protected_field_freeze"]["offending_field_paths"] == [
        "camera_config"
    ]
    assert "camera_config" in failure_note


def test_main_rejects_non_directory_output_path_cleanly(
    tmp_path: Path,
    capsys,
) -> None:
    bad_output_path = tmp_path / "freeze.json"
    bad_output_path.write_text("{}\n", encoding="utf-8")

    exit_code = gr00t_eval_contract_gate.main(["--output-dir", str(bad_output_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "output-dir must be a directory path" in captured.err
    assert "Traceback" not in captured.err
