from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import interface_localization_action_roundtrip
from work.recap.scripts import gr00t_action_chain_telemetry
from work.recap.scripts import gr00t_controller_audit_unitree_g1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        interface_localization_action_roundtrip.main(["--help"])
    assert exc_info.value.code == 0


def test_main_writes_action_roundtrip_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "interface_localization_sprint"

    exit_code = interface_localization_action_roundtrip.main(
        ["--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    payload = _read_json(
        output_dir / interface_localization_action_roundtrip.ACTION_ROUNDTRIP_JSON_NAME
    )

    assert (
        payload
        == interface_localization_action_roundtrip.build_action_semantics_roundtrip(
            REPO_ROOT,
            output_dir=output_dir,
        )
    )
    assert (
        payload["schema_version"]
        == interface_localization_action_roundtrip.ACTION_ROUNDTRIP_SCHEMA_VERSION
    )
    assert (
        payload["artifact_kind"]
        == interface_localization_action_roundtrip.ACTION_ROUNDTRIP_ARTIFACT_KIND
    )
    assert payload["backpointer"]["writer_script"] == (
        "work/recap/scripts/interface_localization_action_roundtrip.py"
    )


def test_roundtrip_freezes_canonical_space_and_split_buckets() -> None:
    payload = interface_localization_action_roundtrip.build_action_semantics_roundtrip(
        REPO_ROOT
    )

    assert payload["canonical_space"]["canonical_space_name"] == (
        interface_localization_action_roundtrip.CANONICAL_SPACE_NAME
    )
    assert payload["canonical_space"]["comparison_stage"] == "absolute_action"
    assert payload["watch_bucket_order"] == [
        "body_wrist_upper_limb_chain",
        "dex3_finger_hand_path",
    ]

    body_bucket = payload["watch_buckets"]["body_wrist_upper_limb_chain"]
    dex_bucket = payload["watch_buckets"]["dex3_finger_hand_path"]

    assert body_bucket["source_mapping"]["original_action_key"] == "right_arm"
    assert body_bucket["source_mapping"]["action_representation"] == "RELATIVE"
    assert body_bucket["source_mapping"]["reference_state_key"] == "right_arm"
    assert body_bucket["source_mapping"]["joint_order"][-3:] == [
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    assert set(body_bucket["checkpoints"].keys()) == {"produced", "adapted", "consumed"}
    assert body_bucket["checkpoints"]["adapted"]["status"] == "mutated"
    assert body_bucket["conclusions"]["difference_disappeared_at"] is None
    assert (
        body_bucket["conclusions"]["controller_absorbed_upstream_difference"] is False
    )
    assert body_bucket["conclusions"]["model_insensitive"] is False
    assert body_bucket["conclusions"]["watch_bucket_classification"] == (
        "live_difference_persists"
    )

    assert dex_bucket["source_mapping"]["original_action_key"] == "right_hand"
    assert dex_bucket["source_mapping"]["action_representation"] == "ABSOLUTE"
    assert dex_bucket["source_mapping"]["reference_state_key"] is None
    assert all(
        joint.startswith("right_hand_")
        for joint in dex_bucket["source_mapping"]["joint_order"]
    )
    assert all(
        "wrist" not in joint for joint in dex_bucket["source_mapping"]["joint_order"]
    )
    assert dex_bucket["checkpoints"]["adapted"]["status"] == "survived"
    assert dex_bucket["conclusions"]["difference_disappeared_at"] == "model"
    assert dex_bucket["conclusions"]["controller_absorbed_upstream_difference"] is False
    assert dex_bucket["conclusions"]["model_insensitive"] is True

    assert payload["summary"]["controller_absorbed_watch_buckets"] == []
    assert payload["summary"]["model_insensitive_watch_buckets"] == [
        "dex3_finger_hand_path"
    ]
    assert payload["summary"]["watch_bucket_classification_by_bucket"] == {
        "body_wrist_upper_limb_chain": "live_difference_persists",
        "dex3_finger_hand_path": "model_insensitive",
    }
    assert payload["summary"]["explicit_split"] == {
        "body_wrist_upper_limb_chain": "right_arm",
        "dex3_finger_hand_path": "right_hand",
        "must_not_collapse_back_to_right_hand": True,
    }


def test_relative_absolute_semantics_drift_is_rejected() -> None:
    watch_bucket_specs = (
        interface_localization_action_roundtrip.default_watch_bucket_specs()
    )
    watch_bucket_specs["body_wrist_upper_limb_chain"]["source_mapping"][
        "action_representation"
    ] = "ABSOLUTE"

    with pytest.raises(ValueError, match="relative/absolute semantics"):
        interface_localization_action_roundtrip.build_action_semantics_roundtrip(
            REPO_ROOT,
            watch_bucket_specs=watch_bucket_specs,
        )


def test_joint_order_drift_is_rejected() -> None:
    watch_bucket_specs = (
        interface_localization_action_roundtrip.default_watch_bucket_specs()
    )
    watch_bucket_specs["dex3_finger_hand_path"]["source_mapping"]["joint_order"] = [
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    ]

    with pytest.raises(ValueError, match="joint order drift"):
        interface_localization_action_roundtrip.build_action_semantics_roundtrip(
            REPO_ROOT,
            watch_bucket_specs=watch_bucket_specs,
        )


def test_split_bucket_collapse_is_rejected() -> None:
    watch_bucket_specs = (
        interface_localization_action_roundtrip.default_watch_bucket_specs()
    )
    watch_bucket_specs["body_wrist_upper_limb_chain"]["source_mapping"][
        "original_action_key"
    ] = "right_hand"

    with pytest.raises(ValueError, match="must stay bound to right_arm"):
        interface_localization_action_roundtrip.build_action_semantics_roundtrip(
            REPO_ROOT,
            watch_bucket_specs=watch_bucket_specs,
        )


def test_roundtrip_stays_aligned_with_frozen_g1_execution_surface() -> None:
    payload = interface_localization_action_roundtrip.build_action_semantics_roundtrip(
        REPO_ROOT
    )
    audit = gr00t_controller_audit_unitree_g1.build_audit_report(
        runtime_log=(
            REPO_ROOT
            / "agent"
            / "runtime_logs"
            / "policy_modality_probe"
            / "00_smoke_eval_g1_once.log"
        )
    )

    body_bucket = payload["watch_buckets"]["body_wrist_upper_limb_chain"]
    dex_bucket = payload["watch_buckets"]["dex3_finger_hand_path"]
    execution_surface_contract = audit["execution_surface_contract"]

    assert execution_surface_contract["policy_horizon_expected"] == 30
    assert execution_surface_contract["n_action_steps_expected"] == 20
    assert body_bucket["source_mapping"]["original_action_key"] in set(
        execution_surface_contract["relative_action_keys"]
    )
    assert dex_bucket["source_mapping"]["original_action_key"] in set(
        execution_surface_contract["absolute_action_keys"]
    )
    assert body_bucket["source_mapping"]["reference_state_key"] == "right_arm"
    assert dex_bucket["source_mapping"]["reference_state_key"] is None


def test_action_delta_sidecar_respects_roundtrip_arm_hand_split() -> None:
    payload = interface_localization_action_roundtrip.build_action_semantics_roundtrip(
        REPO_ROOT
    )
    sidecar = gr00t_action_chain_telemetry.build_grouped_action_chain_sidecar(
        "UNITREE_G1",
        stage_group_values={
            "decoded_action": {
                "right_arm": [0.1] * 7,
                "right_hand": [0.2] * 7,
            },
            "absolute_action": {
                "right_arm": [0.3] * 7,
                "right_hand": [0.4] * 7,
            },
            "controller_input": {
                "right_arm": [0.5] * 7,
                "right_hand": [0.6] * 7,
            },
        },
        stage_unavailable_reasons={
            "raw_action": "raw action stage intentionally omitted in this roundtrip alignment test"
        },
    )

    body_bucket = payload["watch_buckets"]["body_wrist_upper_limb_chain"]
    dex_bucket = payload["watch_buckets"]["dex3_finger_hand_path"]
    grouped = sidecar["per_group_stage_surfaces"]

    assert body_bucket["source_mapping"]["original_action_key"] == "right_arm"
    assert dex_bucket["source_mapping"]["original_action_key"] == "right_hand"
    assert (
        grouped["right_arm"]["action_representation"]
        == body_bucket["source_mapping"]["action_representation"]
    )
    assert (
        grouped["right_hand"]["action_representation"]
        == dex_bucket["source_mapping"]["action_representation"]
    )
    assert grouped["right_arm"]["stages"]["decoded_action"]["available"] is True
    assert grouped["right_hand"]["stages"]["decoded_action"]["available"] is True


def test_action_delta_sidecar_accepts_singleton_batch_policy_server_surface() -> None:
    horizon = 30
    right_arm_surface = [[[0.1] * 7 for _ in range(horizon)]]
    sidecar = gr00t_action_chain_telemetry.build_grouped_action_chain_sidecar(
        "UNITREE_G1",
        stage_group_values={
            "decoded_action": {
                "action.right_arm": right_arm_surface,
            },
        },
    )

    right_arm = sidecar["per_group_stage_surfaces"]["right_arm"]
    decoded_stage = right_arm["stages"]["decoded_action"]

    assert decoded_stage["available"] is True
    assert decoded_stage["shape"] == [horizon, 7]
    assert len(decoded_stage["values"]) == horizon
    assert decoded_stage["values"][0] == pytest.approx([0.1] * 7)
    assert (
        sidecar["stage_group_coverage"]["decoded_action"]["available_group_count"]
        == 1
    )
